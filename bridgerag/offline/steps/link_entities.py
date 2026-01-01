import logging
import json
from itertools import combinations
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import numpy as np
from neo4j import Driver
from jinja2 import Template
import re
from pathlib import Path

from bridgerag.database.graph_ops import create_same_as_links
from bridgerag.database.vector_ops import get_vectors_by_ids
from bridgerag.core.llm_client import LLMClient
from bridgerag.prompts.prompts import ENTITY_SIMILARITY_PROMPT

logger = logging.getLogger(__name__)
# 调高httpx的日志级别，以避免打印过多的请求信息
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- 常量定义 ---
# 仅将在2到10个文档中出现的实体视为链接候选者。
# 这可以避免处理过于罕见或过于普遍的实体。
MIN_DOCS = 2
MAX_DOCS = 10
# 创建SAME_AS链接的相似度分数阈值。
SIMILARITY_THRESHOLD = 7.0
# 混合评分模型的权重。
LLM_SCORE_WEIGHT = 0.7
EMBEDDING_SCORE_WEIGHT = 0.3
# 批处理写入数据库的链接数量
LINK_BATCH_SIZE = 50


def _get_candidate_entities(driver: Driver) -> Dict[str, List[Dict[str, Any]]]:
    """
    从Neo4j中获取候选实体，并按实体名称分组。
    候选者是在适度数量文档中出现的命名实体。
    注意：此查询不再获取 embedding，因为它将从 Milvus 中批量检索。
    """
    query = f"""
    MATCH (e:Entity)
    WHERE e.is_named_entity = true
    WITH e.name as name, collect(e) as entities, count(e) as count
    WHERE count >= {MIN_DOCS} AND count <= {MAX_DOCS}
    RETURN name, [entity in entities | {{
        entity_id: entity.entity_id,
        summary: entity.summary
    }}] as entity_list
    """
    logger.info("正在获取用于链接的候选实体...")
    records = {}
    try:
        with driver.session() as session:
            results = session.run(query)
            for record in results:
                records[record["name"]] = record["entity_list"]
        logger.info(f"找到了 {len(records)} 个唯一的实体名称作为候选。")
        return records
    except Exception as e:
        logger.error(f"获取候选实体失败: {e}")
        raise


def _get_llm_similarity_score(
    llm_client: LLMClient, entity1: Dict[str, Any], entity2: Dict[str, Any]
) -> Tuple[int, str]:
    """
    使用LLM为两个实体获取语义相似度分数。
    """
    prompt = ENTITY_SIMILARITY_PROMPT.render(
        entity_1_name=entity1["name"],
        entity_1_description=entity1["summary"],
        entity_2_name=entity2["name"],
        entity_2_description=entity2["summary"],
    )
    try:
        response = llm_client.generate(prompt)
        
        # 增加对空响应的检查
        if not response or not response.strip():
            logger.warning(
                f"LLM for entities '{entity1.get('entity_id')}' and '{entity2.get('entity_id')}' "
                f"returned an empty or whitespace response."
            )
            return 0, "LLM returned empty response"

        # 从Markdown代码块中提取JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            logger.warning(
                f"Could not find a JSON object in the LLM response for entities "
                f"'{entity1.get('entity_id')}' and '{entity2.get('entity_id')}'. "
                f"Raw response was: '{response}'"
            )
            return 0, "No JSON object found in LLM response"
        
        json_string = json_match.group(0)
        result = json.loads(json_string)
        return int(result.get("score", 0)), result.get("reasoning", "")
    except json.JSONDecodeError as e:
        # 在日志中记录原始响应以供调试
        logger.warning(
            f"Failed to decode LLM JSON response for entities "
            f"'{entity1.get('entity_id')}' and '{entity2.get('entity_id')}'. "
            f"Error: {e}. Raw response was: '{response}'"
        )
        return 0, f"JSON decode error: {e}"
    except (openai.APIError, TypeError) as e:
        logger.warning(
            f"无法获取实体 "
            f"{entity1.get('entity_id')} 和 {entity2.get('entity_id')} 的LLM相似度: {e}"
        )
        return 0, f"错误: {e}"


def _get_embedding_similarity_score(
    embedding1: List[float], embedding2: List[float]
) -> float:
    """
    计算两个实体嵌入向量之间的余弦相似度。

    参数:
        embedding1 (List[float]): 第一个实体的嵌入向量。
        embedding2 (List[float]): 第二个实体的嵌入向量。

    返回:
        float: 介于 0 和 1 之间的余弦相似度分数。如果输入无效则返回 0.0。
    """
    if not embedding1 or not embedding2:
        return 0.0

    try:
        vec1 = np.array(embedding1, dtype=np.float32)
        vec2 = np.array(embedding2, dtype=np.float32)

        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
        # 将 NaN（如果范数为0）转换为0.0
        return float(np.nan_to_num(similarity))
    except (ValueError, TypeError) as e:
        logger.warning(f"计算余弦相似度时出错: {e}")
        return 0.0


def _score_candidate_pair(
    pair: Tuple[Dict[str, Any], Dict[str, Any]],
    llm_client: LLMClient,
) -> Dict[str, Any]:
    """
    为一个实体对计算混合相似度分数。
    """
    entity1, entity2 = pair

    # 1. 获取基于LLM的语义分数
    llm_score, reasoning = _get_llm_similarity_score(llm_client, entity1, entity2)

    # 2. 获取基于Embedding的余弦相似度分数
    embedding_score = 0.0
    if "embedding" in entity1 and "embedding" in entity2:
        embedding_score = _get_embedding_similarity_score(
            entity1["embedding"], entity2["embedding"]
        )
    
    # 3. 计算最终的混合分数
    # 将余弦相似度分数 (0-1) 放大到与LLM分数相同的 10 分制
    final_score = (llm_score * LLM_SCORE_WEIGHT) + (
        (embedding_score * 10) * EMBEDDING_SCORE_WEIGHT
    )

    logger.debug(
        f"评分对 ({entity1['entity_id']}, {entity2['entity_id']}): "
        f"LLM={llm_score}, EMB={embedding_score:.4f}, Final={final_score:.2f}"
    )

    return {
        "entity_1_id": entity1["entity_id"],
        "entity_2_id": entity2["entity_id"],
        "entity_1_name": entity1["name"],
        "entity_2_name": entity2["name"],
        "llm_score": llm_score,
        "llm_reasoning": reasoning,
        "embedding_score": round(embedding_score, 4),
        "final_score": round(final_score, 2),
    }


def run_entity_linking(
    driver: Driver,
    llm_client: LLMClient,
    milvus_collection_name: str,
    fast_link: bool = False,
    max_workers: int = 8,
    details_output_path: str = "logs/entity_linking_details.jsonl"
) -> None:
    """
    跨文档实体链接流程的主协调函数。
    """
    logger.info("--- 开始跨文档实体链接 ---")

    # 确保日志文件目录存在
    output_path = Path(details_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    candidate_groups = _get_candidate_entities(driver)

    if not candidate_groups:
        logger.info("未找到候选实体组。跳过实体链接。")
        return

    links_to_create = []
    detailed_results = []
    total_links_found = 0

    if fast_link:
        logger.info("快速链接模式已启用。将直接链接所有同名候选实体。")
        for name, entities in candidate_groups.items():
            if len(entities) > 1:
                for entity1, entity2 in combinations(entities, 2):
                    links_to_create.append({
                        "entity_1_id": entity1["entity_id"],
                        "entity_2_id": entity2["entity_id"],
                        "score": 10.0,
                        "reasoning": "Fast link mode enabled",
                    })
    else:
        # 1. 从 Milvus 批量获取所有候选实体的向量
        all_entity_ids = list(set([
            entity["entity_id"]
            for entities in candidate_groups.values()
            for entity in entities
        ]))

        logger.info(f"正在从 Milvus 批量获取 {len(all_entity_ids)} 个实体的向量...")
        id_to_vector_map = get_vectors_by_ids(
            collection_name=milvus_collection_name,
            ids=all_entity_ids,
            id_field="entity_id",
        )
        logger.info(f"成功从 Milvus 获取了 {len(id_to_vector_map)} 个向量。")

        # 2. 将向量注入回候选实体数据中
        for entities in candidate_groups.values():
            for entity in entities:
                entity["embedding"] = id_to_vector_map.get(entity["entity_id"])

        # 3. 生成所有有效的候选对
        all_candidate_pairs = []
        for name, entities in candidate_groups.items():
            if len(entities) > 1:
                for entity1, entity2 in combinations(entities, 2):
                    # 确保两个实体都成功获取了向量
                    if entity1.get("embedding") and entity2.get("embedding"):
                        entity1['name'] = name
                        entity2['name'] = name
                        all_candidate_pairs.append((entity1, entity2))
        
        if not all_candidate_pairs:
            logger.info("没有可链接的候选对。跳过此步骤。")
            return

        logger.info(f"已生成 {len(all_candidate_pairs)} 个候选对进行评分。")
        logger.info(f"详细评分结果将保存至: {details_output_path}")

        # 4. 并行评分和批处理写入
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(
                    _score_candidate_pair, pair, llm_client
                ): pair
                for pair in all_candidate_pairs
            }
            for future in as_completed(future_to_pair):
                try:
                    result = future.result()
                    
                    is_linked = result and result.get("final_score", 0) >= SIMILARITY_THRESHOLD
                    result["is_linked"] = is_linked
                    detailed_results.append(result)
                    
                    if is_linked:
                        link_data = {
                            "entity_1_id": result["entity_1_id"],
                            "entity_2_id": result["entity_2_id"],
                            "score": result["final_score"],
                            "reasoning": result["llm_reasoning"],
                        }
                        links_to_create.append(link_data)
                        
                        if len(links_to_create) >= LINK_BATCH_SIZE:
                            total_links_found += len(links_to_create)
                            logger.info(f"达到批处理大小，正在将 {len(links_to_create)} 个链接写入数据库...")
                            create_same_as_links(driver, links_to_create)
                            links_to_create.clear()
                except Exception as e:
                    pair_ids = (
                        future_to_pair[future][0]["entity_id"],
                        future_to_pair[future][1]["entity_id"],
                    )
                    logger.error(f"评分候选对 {pair_ids} 时出错: {e}")

    # 5. 创建链接（处理快速模式和最后一个批次）
    if not links_to_create:
        if total_links_found == 0:
            logger.info("没有链接满足相似度阈值或在快速模式下生成。")
    else:
        total_links_found += len(links_to_create)
        logger.info(f"正在将最后 {len(links_to_create)} 个链接写入数据库...")
        create_same_as_links(driver, links_to_create)

    if detailed_results:
        logger.info(f"正在将 {len(detailed_results)} 条详细评分结果写入文件...")
        try:
            with open(details_output_path, "w", encoding="utf-8") as f:
                for item in detailed_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"详细结果已成功保存到 {details_output_path}")
        except Exception as e:
            logger.error(f"保存详细结果到文件时失败: {e}")

    logger.info(f"总共找到并创建了 {total_links_found} 个高可信度链接。")
    logger.info("--- 跨文档实体链接完成 ---")
