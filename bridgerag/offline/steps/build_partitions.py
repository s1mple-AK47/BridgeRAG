from collections import defaultdict
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Tuple
from transformers import PreTrainedTokenizer
import logging
import re
import json
import os # Added for saving failed chunks
from neo4j import Driver

from bridgerag.utils.text_processing import chunk_text_by_tokens, normalize_entity_name
from bridgerag.config import settings
from bridgerag.prompts.prompts import (
    ENTITY_EXTRACTION_ONLY_PROMPT,
    ENTITY_GLEANING_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    ENTITY_SUMMARY_PROMPT_TEMPLATE,
    DOCUMENT_SUMMARY_PROMPT_TEMPLATE
)
from bridgerag.utils.json_parser import robust_json_parser
from bridgerag.database import graph_ops, vector_ops

# TYPE_CHECKING block is only evaluated by static type checkers, not at runtime.
# This prevents circular import errors.
if TYPE_CHECKING:
    from bridgerag.core.llm_client import LLMClient
    from bridgerag.core.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


def _create_chunks(
    doc_id: str,
    doc_content: str,
    tokenizer: "PreTrainedTokenizer"
) -> List[Dict[str, Any]]:
    """
    接收单个文档的内容，将其分块，并为每个块生成一个全局唯一的ID。

    这个函数是离线处理管道中的一个基础步骤。

    参数:
        doc_id (str): 文档的唯一标识符，将用于构成块ID。
        doc_content (str): 文档的原始文本内容。
        tokenizer (PreTrainedTokenizer): 用于文本分块的分词器实例。

    返回:
        List[Dict[str, Any]]: 一个包含文本块信息的列表。每个字典都增加了
                              一个 'chunk_id' 键，其值为 'doc_id_chunk_order_index'。
    """
    logger.info(f"开始为文档 '{doc_id}' 创建文本块...")

    # 从配置中加载分块参数
    # 注意: 我们在这里可以不使用 settings.text_chunk_size 的默认值，
    # 而是直接从配置加载，保证了配置的单一来源。
    max_tokens = settings.text_chunk_size
    overlap_tokens = settings.overlap_token_size

    # 调用通用的分块工具函数
    chunks = chunk_text_by_tokens(
        tokenizer=tokenizer,
        content=doc_content,
        max_token_size=max_tokens,
        overlap_token_size=overlap_tokens,
        split_by_character="\\n\\n"  # 优先按段落切分
    )

    # --- 新增：安全检查和硬截断 ---
    # 这是一个安全网，用于处理 chunk_text_by_tokens 无法按预期分割的罕见情况
    # （例如，一个没有空格或换行符的超长字符串）。
    final_chunks = []
    for chunk in chunks:
        token_ids = tokenizer.encode(chunk["content"])
        if len(token_ids) > max_tokens+400:
            logger.warning(
                f"文档 '{doc_id}' 的一个块（索引 {chunk['chunk_order_index']}）"
                f"在初始分块后仍然超长（{len(token_ids)} > {max_tokens} tokens）。"
                f"这可能由一个无法分割的长字符串引起。将对其进行硬截断。"
            )
            # 硬截断 token 列表
            truncated_token_ids = token_ids[:max_tokens]
            # 将截断后的 token ID 解码回文本
            chunk["content"] = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
            logger.info(f"块已被截断为 {len(tokenizer.encode(chunk['content']))} 个 token。")
        final_chunks.append(chunk)


    # 为每个块生成并添加全局唯一的 chunk_id
    for chunk in final_chunks:
        chunk_order_index = chunk["chunk_order_index"]
        chunk_id = f"{doc_id}_{chunk_order_index}"
        chunk["chunk_id"] = chunk_id
        chunk["doc_id"] = doc_id  # 修正：为每个块添加 doc_id
        logger.debug(f"为文档 '{doc_id}' 生成了块 ID: {chunk_id}")

    if not final_chunks:
        logger.warning(f"文档 '{doc_id}' 未能生成任何文本块。")
    else:
        logger.info(f"成功为文档 '{doc_id}' 创建了 {len(final_chunks)} 个文本块。")

    return final_chunks


def _clean_and_validate_entity(entity_data: Dict[str, Any], chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    清洗并验证从LLM提取的单个实体字典。
    """
    raw_entity_name = entity_data.get("entity_name", "").strip()
    
    # 新增：硬编码规则，拒绝仅由数字组成的实体名（例如年份）
    if raw_entity_name.isdigit():
        logger.info(f"在块 {chunk_id} 中发现无效的纯数字实体名 '{raw_entity_name}'，已忽略。")
        return None
        
    description = entity_data.get("entity_description", "").strip()
    is_named_entity = entity_data.get("is_named_entity", False)

    if not raw_entity_name:
        logger.warning(f"在块 {chunk_id} 中发现无效实体：缺少 entity_name。原始数据: {entity_data}")
        return None

    entity_name = normalize_entity_name(raw_entity_name)
    if not entity_name:
        logger.warning(f"实体名称 '{raw_entity_name}' 规范化后为空，已忽略。")
        return None

    if not description or description == "文本中无可用描述。" or description == "No description available in the text.":
        logger.debug(f"在块 {chunk_id} 中发现无描述的实体 '{entity_name}'。")
        description = ""

    if not isinstance(is_named_entity, bool):
        logger.warning(f"在块 {chunk_id} 中发现无效的 is_named_entity 类型 (值: {is_named_entity})，将默认为 False。")
        is_named_entity = False

    return {
        "entity_name": entity_name,
        "description": description,
        "is_named_entity": is_named_entity,
        "source_chunk_id": chunk_id
    }


def _clean_relation_format(relation_data: Dict[str, Any], chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    对LLM抽取的关系进行初步的格式清洗和验证。
    """
    source_entity = relation_data.get("source_entity")
    target_entity = relation_data.get("target_entity")
    description = relation_data.get("relationship_description")

    if not all([source_entity, target_entity, description]):
        return None

    relation_data["source_entity"] = normalize_entity_name(source_entity)
    relation_data["target_entity"] = normalize_entity_name(target_entity)
    return relation_data


def _extract_entities_and_relations(
    chunk_id: str,
    chunk_content: str,
    llm_client: "LLMClient"
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    从单个文本块中，通过“两步提取法”提取实体和关系，并加入“拾遗”步骤。
    
    步骤:
    1.  **实体提取**: 首次调用LLM，只提取实体。
    2.  **实体拾遗**: 再次调用LLM，检查是否有遗漏的实体。
    3.  **关系提取**: 基于前两步的完整实体列表，调用LLM提取它们之间的关系。
    """
    logger.warning(f"开始为块 '{chunk_id}' 执行三阶段提取...")

    def call_llm_with_retry(
        llm_client: "LLMClient", 
        prompt: str, 
        chunk_id: str, 
        max_retries: int = 3,
        allow_empty_result: bool = False
    ) -> Optional[Dict[str, Any]]:
        """一个内部辅助函数，用于带重试逻辑地调用LLM并解析JSON。"""
        for attempt in range(max_retries):
            try:
                # 修正: 只有在重试时才打印 "将重试" 日志
                if attempt > 0:
                    logger.warning(f"[{chunk_id}] 第 {attempt + 1}/{max_retries} 次尝试:将重试...")

                # --- 新增日志：打印完整的Prompt ---
                logger.debug(f"[{chunk_id}] --- LLM PROMPT (Attempt {attempt + 1}) ---\n{prompt}")
                
                response_text = llm_client.generate(prompt)
                
                # --- 新增日志：打印LLM返回的原始文本 ---
                logger.debug(f"[{chunk_id}] --- LLM RAW RESPONSE (Attempt {attempt + 1}) ---\n{response_text}")

                data = robust_json_parser(response_text)
                
                # 新增: 强制要求解析结果必须是字典，否则视为解析失败并重试
                if not isinstance(data, dict):
                    raise json.JSONDecodeError(f"robust_json_parser未能返回有效的JSON字典，返回类型为 {type(data)}。", response_text, 0)

                # 如果允许空结果，只要解析成功就返回
                if allow_empty_result:
                    if attempt != 0:
                        logger.warning(f"[{chunk_id}] 第 {attempt + 1}/{max_retries} 次尝试: JSON解析成功但内容可能为空")
                    return data

                # 如果不允许空结果（默认），则需要内容非空
                if data and (data.get("entities") or data.get("relationships")):
                    if attempt != 0:
                        logger.warning(f"[{chunk_id}] 第 {attempt + 1}/{max_retries} 次尝试成功")
                    return data
                else:
                    logger.warning(f"[{chunk_id}] 第 {attempt + 1}/{max_retries} 次尝试: JSON解析成功但内容为空，将重试...")

            except Exception as e:
                logger.warning(f"[{chunk_id}] 第 {attempt + 1}/{max_retries} 次尝试时发生异常: {e}")
        
        logger.error(f"[{chunk_id}] LLM调用在 {max_retries} 次尝试后仍然失败。")
        return None # 返回 None 表示最终失败


    # --- 阶段 1: 仅提取实体 ---
    logger.debug(f"[{chunk_id}] 阶段 1: 首次实体提取...")
    prompt_entities = ENTITY_EXTRACTION_ONLY_PROMPT.render(language="English", input_text=chunk_content)
    data_stage1 = call_llm_with_retry(llm_client, prompt_entities, chunk_id)
    if data_stage1 is None:
        logger.error(f"[{chunk_id}] 阶段 1 实体提取失败，中止处理该块。")
        return None, None
    initial_entities_raw = data_stage1.get("entities", [])

    valid_entities = []
    for entity_data in initial_entities_raw:
        clean_entity = _clean_and_validate_entity(entity_data, chunk_id)
        if clean_entity:
            valid_entities.append(clean_entity)
    
    logger.info(f"[{chunk_id}] 阶段 1: 提取了 {len(valid_entities)} 个初始实体。")

    # --- 阶段 2: 实体拾遗 ---
    logger.debug(f"[{chunk_id}] 阶段 2: 实体拾遗...")
    prompt_gleaning = ENTITY_GLEANING_PROMPT.render(
        language="English",
        input_text=chunk_content,
        previously_extracted_entities=json.dumps(initial_entities_raw, indent=2, ensure_ascii=False)
    )
    data_gleaning = call_llm_with_retry(llm_client, prompt_gleaning, chunk_id, allow_empty_result=True)
    if data_gleaning is None:
        logger.error(f"[{chunk_id}] 阶段 2 实体拾遗失败，中止处理该块。")
        return None, None
    gleaned_entities_raw = data_gleaning.get("entities", [])

    new_entities_count = 0
    existing_entity_names = {e["entity_name"] for e in valid_entities}
    for entity_data in gleaned_entities_raw:
        clean_entity = _clean_and_validate_entity(entity_data, chunk_id)
        if clean_entity and clean_entity["entity_name"] not in existing_entity_names:
            valid_entities.append(clean_entity)
            existing_entity_names.add(clean_entity["entity_name"])
            new_entities_count += 1
            
    logger.info(f"[{chunk_id}] 阶段 2: 拾遗了 {new_entities_count} 个新实体。总实体数: {len(valid_entities)}")

    if not valid_entities:
        logger.warning(f"块 '{chunk_id}' 在实体提取和拾遗后未发现任何有效实体。")
        return [], []

    # --- 阶段 3: 关系提取 ---
    logger.debug(f"[{chunk_id}] 阶段 3: 基于 {len(valid_entities)} 个实体提取关系...")
    final_entity_names_for_prompt = [e["entity_name"] for e in valid_entities]
    prompt_relations = RELATION_EXTRACTION_PROMPT.render(
        language="English",
        input_text=chunk_content,
        entity_list=json.dumps(final_entity_names_for_prompt, indent=2, ensure_ascii=False)
    )
    data_relations = call_llm_with_retry(llm_client, prompt_relations, chunk_id)
    if data_relations is None:
        logger.error(f"[{chunk_id}] 阶段 3 关系提取失败，中止处理该块。")
        return None, None
    relations_raw = data_relations.get("relationships", [])

    valid_relations = []
    final_entity_names = {e["entity_name"] for e in valid_entities}
    for rel_data in relations_raw:
        clean_rel = _clean_relation_format(rel_data, chunk_id)
        if clean_rel:
            # 最终验证，确保关系中的实体确实在我们的最终列表中
            if clean_rel["source_entity"] in final_entity_names and clean_rel["target_entity"] in final_entity_names:
                valid_relations.append(clean_rel)
            else:
                logger.info(f"在块 {chunk_id} 中发现悬空关系（阶段3），已忽略。关系: {clean_rel}")
    
    logger.info(f"[{chunk_id}] 阶段 3: 提取了 {len(valid_relations)} 个有效关系。")
    logger.info(f"块 '{chunk_id}' 提取完成。总计: {len(valid_entities)} 实体, {len(valid_relations)} 关系。")

    return valid_entities, valid_relations


def _summarize_entity_context(
    doc_id: str,
    entity_name: str,
    descriptions: List[str],
    relations: List[str],
    llm_client: "LLMClient",
    tokenizer: "PreTrainedTokenizer"
) -> str:
    """
    使用LLM为一个实体生成最终的描述性摘要，或在上下文足够短时直接合并。
    这个摘要融合了该实体的所有直接描述和其关系信息。
    """
    meaningful_descriptions = [desc for desc in descriptions if desc and not desc.isspace()]
    # 修正: 统一使用真正的换行符 \n
    full_context = "\n".join(meaningful_descriptions)
    relations_str = "\n".join(relations) if relations else ""

    # 将上下文和关系合并为一个单独的文本块用于评估和可能的返回
    combined_text_parts = [full_context]
    if relations_str:
        combined_text_parts.append("\n\n关系:\n" + relations_str)
    combined_text = "".join(combined_text_parts)

    # --- START DIAGNOSTIC LOGGING ---
    token_ids = tokenizer.encode(combined_text)
    token_count = len(token_ids)
    # --- END DIAGNOSTIC LOGGING ---

    TOKEN_THRESHOLD = 300

    # 如果 token 数量低于阈值，直接返回合并后的文本
    if token_count <= TOKEN_THRESHOLD:
        logger.info(f"实体 '{entity_name}' 的上下文长度 ({token_count} tokens) <= {TOKEN_THRESHOLD}，跳过 LLM 摘要。")
        return combined_text.strip()
    
    logger.info(f"实体 '{entity_name}' 的上下文长度 ({token_count} tokens) > {TOKEN_THRESHOLD}，将使用 LLM 生成摘要。")

    # 只有当 token 数量超过阈值时，才渲染完整的 prompt
    # 核心修正：修复了传递给模板的变量名，使其与 prompts.py 中的定义完全匹配
    prompt = ENTITY_SUMMARY_PROMPT_TEMPLATE.render(
        entity_name=entity_name,
        language="English", # 修正：添加缺失的 language 参数
        direct_descriptions=full_context,
        relations=relations_str if relations_str else "无"
    )

    try:
        summary = llm_client.generate(prompt)
        logger.info(f"成功为实体 '{entity_name}' 生成了摘要。")
        return summary.strip()
    except Exception as e:
        logger.error(f"为实体 '{entity_name}' 生成摘要时出错: {e}")
        # 在出错的情况下，优雅地降级，返回拼接后的原始描述
        return combined_text.strip()


def _merge_and_summarize_doc_knowledge(
    doc_id: str,
    all_raw_entities: List[Dict[str, Any]],
    all_raw_relations: List[Dict[str, Any]],
    llm_client: "LLMClient",
    tokenizer: "PreTrainedTokenizer",
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    聚合、合并、并为文档中的每个实体生成包含其关系信息的最终摘要。
    """
    # 第一步：聚合实体和关系
    aggregated_entities = defaultdict(lambda: {"descriptions": [], "source_chunk_ids": [], "is_named_entity_votes": []})
    for entity in all_raw_entities:
        name = entity["entity_name"]
        # 恢复代码：无条件聚合，让下游处理
        aggregated_entities[name]["descriptions"].append(entity["description"])
        aggregated_entities[name]["source_chunk_ids"].append(entity["source_chunk_id"])
        aggregated_entities[name]["is_named_entity_votes"].append(entity["is_named_entity"])

    aggregated_relations = defaultdict(list)
    for rel in all_raw_relations:
        # 使用 (源, 目标) 元组作为键，以合并具有相同方向的重复关系
        # 注意：我们不再对元组进行排序，以保留关系的方向性
        key = (rel["source_entity"], rel["target_entity"])
        aggregated_relations[key].append(rel)

    # 第二步：合并关系
    merged_relations = []
    for key, rel_list in aggregated_relations.items():
        # 使用 .get() 并提供默认值来安全地访问键，防止KeyError
        all_descriptions = [r.get("relationship_description", "") for r in rel_list]
        merged_description = " | ".join(sorted(set(filter(None, all_descriptions))))

        # 计算平均强度，如果'relationship_strength'不存在，则默认为5
        strengths = [r.get("relationship_strength", 5) for r in rel_list]
        avg_strength = sum(strengths) / len(strengths) if strengths else 5

        # 合并关键词
        all_keywords = set()
        for r in rel_list:
            # 使用 .get() 并提供空列表作为默认值
            all_keywords.update(r.get("relationship_keywords", []))

        merged_relations.append({
            "source": key[0],
            "target": key[1],
            "description": merged_description,
            "strength": avg_strength,
            "keywords": sorted(list(all_keywords))
        })

    # 第三步：为每个实体构建增强上下文并生成最终摘要
    final_entities = []
    entity_relations_map = defaultdict(list)
    for rel in merged_relations:
        # 为源实体和目标实体都添加关系描述，以便生成摘要
        relation_for_source = f"与实体 [{rel['target']}] 的关系: {rel['description']}"
        relation_for_target = f"与实体 [{rel['source']}] 的关系: {rel['description']}"
        entity_relations_map[rel['source']].append(relation_for_source)
        entity_relations_map[rel['target']].append(relation_for_target)

    for name, data in aggregated_entities.items():
        # 提前进行命名实体检查
        # 如果一个实体的所有实例都被标记为非命名实体，则直接跳过，不进行摘要
        is_named = any(data["is_named_entity_votes"])
        if not is_named:
            logger.info(f"实体 '{name}' 被识别为非命名实体，将跳过摘要和最终收录。")
            continue

        # 获取该实体的所有关系描述
        relations_context = []
        # 查找以此实体为源的所有关系
        for (source, target), rel_list in aggregated_relations.items():
            if source == name:
                # 修正 KeyError: 使用 'relationship_description'
                all_descriptions = [r.get("relationship_description") for r in rel_list]
                # 使用 filter(None, ...) 确保在合并前移除空描述
                combined_description = " | ".join(filter(None, all_descriptions))
                if combined_description:
                    relations_context.append(f"{source} -> {target}: {combined_description}")
        
        # 查找以此实体为目标的所有关系
        for (source, target), rel_list in aggregated_relations.items():
            if target == name:
                # 修正 KeyError: 使用 'relationship_description'
                all_descriptions = [r.get("relationship_description") for r in rel_list]
                # 使用 filter(None, ...) 确保在合并前移除空描述
                combined_description = " | ".join(filter(None, all_descriptions))
                if combined_description:
                    relations_context.append(f"{source} -> {target}: {combined_description}")
        
        # 新增：前置检查，确保实体有实质内容才进行摘要
        meaningful_descriptions = [d for d in data["descriptions"] if d and d.strip()]
        if not meaningful_descriptions and not relations_context:
            final_summary = f"实体 '{name}' 在文档 '{doc_id}' 中被提及，但无详细上下文。"
        else:
            # 调用LLM生成最终摘要
            final_summary = _summarize_entity_context(
                doc_id=doc_id,
                entity_name=name,
                descriptions=meaningful_descriptions,
                relations=relations_context,
                llm_client=llm_client,
                tokenizer=tokenizer
            )

        # 确定最终的 is_named_entity 状态 (全真则真)
        is_named_final = all(data["is_named_entity_votes"]) if data["is_named_entity_votes"] else False

        # 为实体生成全局唯一的 ID
        entity_id = f"{doc_id}_{name}"

        # 此处的 is_named 判断现在作为双重保险，但主要过滤逻辑已前置
        # 只有被判断为命名实体的才会被保留
        final_entities.append(
            {
                "entity_id": entity_id,
                "name": name,
                "summary": final_summary,
                "type": "named_entity",
                "doc_id": doc_id,
                # 修正: 将键名从 'chunk_ids' 改为 'source_chunk_ids'，以匹配 graph_ops.py 中的 Cypher 查询
                "source_chunk_ids": sorted(list(set(data["source_chunk_ids"]))),
                # 保留最终的命名实体状态
                "is_named_entity": is_named_final
            }
        )

    logger.info(f"文档 '{doc_id}' 的最终有效命名实体数量: {len(final_entities)}")
    
    # ----------------- 第3步：使用最终实体列表验证关系 -----------------
    final_entity_names = {e["name"] for e in final_entities}
    final_relations = []
    
    logger.info(f"开始使用最终实体列表验证 {len(all_raw_relations)} 条原始关系...")
    for relation in all_raw_relations:
        source_name = relation.get("source_entity")
        target_name = relation.get("target_entity")
        
        if source_name in final_entity_names and target_name in final_entity_names:
            # 获取源实体和目标实体的ID
            source_entity_id = next((e["entity_id"] for e in final_entities if e["name"] == source_name), None)
            target_entity_id = next((e["entity_id"] for e in final_entities if e["name"] == target_name), None)

            if source_entity_id and target_entity_id:
                final_relations.append({
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                    "source_entity_name": source_name,
                    "target_entity_name": target_name,
                    "description": relation.get("relationship_description"),
                    "doc_id": doc_id,
                })

    logger.info(f"文档 '{doc_id}' 的最终有效关系数量: {len(final_relations)}")

    return final_entities, final_relations


def _generate_document_summary(
    doc_id: str, 
    final_entities: List[Dict[str, Any]],
    llm_client: "LLMClient", 
    embedding_client: "EmbeddingClient",
    tokenizer: "PreTrainedTokenizer"
) -> tuple[str, list[float]]:
    """使用LLM和嵌入模型为文档生成摘要和向量。"""
    logger.info(f"[{doc_id}] 正在为文档生成最终摘要...")

    # --- Token 限制配置 ---
    # 模型最大上下文长度: 32768
    # 预留给 completion 的 token: 8192 (max_tokens)
    # 预留给 prompt 模板本身的 token: ~500
    # 安全的实体列表最大 token 数
    MAX_ENTITY_LIST_TOKENS = 20000

    # 1. 格式化实体列表以注入 Prompt，并进行 token 截断
    entity_entries = []
    for entity in final_entities:
        # 只包含命名实体以获得更高质量的摘要
        if entity.get("is_named_entity"):
            entity_entries.append(f"- {entity['name']}: {entity['summary']}")
    
    if not entity_entries:
        logger.warning(f"文档 '{doc_id}' 中没有找到有效的命名实体来生成摘要，将返回空摘要。")
        return "No named entities found to generate a summary.", []

    # 逐条添加实体，直到达到 token 限制
    entity_list_str = ""
    current_tokens = 0
    included_count = 0
    for entry in entity_entries:
        entry_with_newline = entry + "\n"
        entry_tokens = len(tokenizer.encode(entry_with_newline, add_special_tokens=False))
        
        if current_tokens + entry_tokens > MAX_ENTITY_LIST_TOKENS:
            logger.warning(
                f"[{doc_id}] 实体列表 token 数 ({current_tokens + entry_tokens}) 将超过限制 "
                f"({MAX_ENTITY_LIST_TOKENS})。已包含 {included_count}/{len(entity_entries)} 个实体，"
                f"剩余 {len(entity_entries) - included_count} 个实体将被截断。"
            )
            break
        
        entity_list_str += entry_with_newline
        current_tokens += entry_tokens
        included_count += 1
    
    logger.info(f"[{doc_id}] 实体列表包含 {included_count} 个实体，共 {current_tokens} 个 token。")

    # 2. 使用LLM生成文档摘要
    prompt = DOCUMENT_SUMMARY_PROMPT_TEMPLATE.render(entity_list=entity_list_str, language="en")
    try:
        summary = llm_client.generate(prompt)
        logger.info(f"[{doc_id}] LLM成功为文档生成摘要。")
    except Exception as e:
        logger.error(f"[{doc_id}] LLM未能为文档生成摘要: {e}", exc_info=True)
        summary = ""

    if not summary:
        logger.warning(f"[{doc_id}] LLM未能为文档生成摘要。")
        return "", []

    logger.info(f"[{doc_id}] 正在为文档摘要生成嵌入向量...")
    try:
        # 定义一个安全的最大 token 长度，略小于模型的绝对最大值
        MAX_EMBEDDING_TOKENS = 2048

        # 检查 token 长度
        token_ids = tokenizer.encode(summary, add_special_tokens=False)
        if len(token_ids) > MAX_EMBEDDING_TOKENS:
            logger.warning(
                f"[{doc_id}] 生成的摘要 token 长度 ({len(token_ids)}) 超过了嵌入模型的最大限制 "
                f"({MAX_EMBEDDING_TOKENS})。将进行截断。"
            )
            # 截断 token ID 列表
            truncated_token_ids = token_ids[:MAX_EMBEDDING_TOKENS]
            # 将截断后的 token ID 解码回文本
            summary = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
            logger.info(f"[{doc_id}] 摘要已被截断为 {len(tokenizer.encode(summary, add_special_tokens=False))} 个 token。")

        embedding = embedding_client.get_embeddings([summary])[0]
        logger.info(f"[{doc_id}] 文档摘要的嵌入向量生成成功。")
        return summary, embedding
    except Exception as e:
        logger.error(f"[{doc_id}] 为文档摘要生成嵌入向量失败: {e}", exc_info=True)
        return summary, []


def process_document(
    doc_id: str,
    doc_content: str,
    llm_client: "LLMClient",
    embedding_client: "EmbeddingClient",
    tokenizer: "PreTrainedTokenizer",
) -> (List[Dict[str, Any]], List[Dict[str, Any]], str, List[Dict[str, Any]]):
    """
    处理单个文档的完整端到端流程。

    这个函数是本模块的唯一公共接口，它协调了所有内部辅助函数
    来完成从原始文本到结构化、已合并知识的转化，并最终生成一份摘要。

    流程:
    1.  将文档内容分块并生成唯一ID。
    2.  对每个块并行地提取实体和关系。
    3.  在文档级别对所有提取出的信息进行聚合、合并和摘要。
    4.  使用最终的具名实体生成一份全面的文档摘要。

    参数:
        doc_id (str): 文档的唯一标识符。
        doc_content (str): 文档的原始文本内容。
        llm_client ("LLMClient"): 用于所有AI推理任务的客户端实例。
        embedding_client ("EmbeddingClient"): 用于生成文档摘要嵌入的嵌入客户端实例。
        tokenizer ("PreTrainedTokenizer"): 用于文本分块的分词器实例。

    返回:
        一个元组，包含四个元素：
        - final_entities: 文档中最终的、唯一的、经过摘要的实体列表。
        - final_relations: 文档中最终的、唯一的关系列表。
        - document_summary: 一份包含所有具名实体的文档摘要。
        - chunks: 此次处理过程中生成的文本块列表。
    """
    import concurrent.futures

    logger.info(f"========== 开始处理文档: {doc_id} ==========")

    # 1. 文本分块与ID生成
    chunks = _create_chunks(doc_id, doc_content, tokenizer)

    if not chunks:
        logger.warning(f"文档 '{doc_id}' 未能生成任何文本块，处理中止。")
        return [], [], "Document is empty or could not be chunked.", []

    # 新增：预处理步骤，过滤掉过短的、无意义的块
    MIN_CHUNK_TOKENS = 10
    meaningful_chunks = []
    for chunk in chunks:
        # 使用分词器来精确计算token数量
        if len(tokenizer.encode(chunk["content"])) > MIN_CHUNK_TOKENS:
            meaningful_chunks.append(chunk)
        else:
            logger.debug(f"文档 '{doc_id}' 的一个块因过短被跳过: '{chunk['content'][:50]}...'")
    
    if not meaningful_chunks:
        logger.warning(f"文档 '{doc_id}' 在过滤掉短块后，未能剩下任何有意义的文本块，处理中止。")
        return [], [], "Document contains no meaningful chunks after filtering.", chunks


    # 2. 并行提取实体和关系
    all_raw_entities = []
    all_raw_relations = []
    failed_chunks = []

    # 使用线程池来实现简单的并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # 为每个块提交一个任务
        future_to_chunk = {
            executor.submit(recursively_process_failed_chunk, chunk, llm_client): chunk
            for chunk in meaningful_chunks
        }
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                entities, relations = future.result()
                if entities is None and relations is None:
                    logger.error(f"块 '{chunk['chunk_id']}' 最终提取失败，已记录以供后续处理。")
                    failed_chunks.append(chunk)
                else:
                    all_raw_entities.extend(entities)
                    all_raw_relations.extend(relations)
            except Exception as exc:
                logger.error(f"块 '{chunk['chunk_id']}' 在提取过程中产生未捕获的异常: {exc}")
                failed_chunks.append(chunk)

    if not all_raw_entities:
        logger.warning(f"文档 '{doc_id}' 未能提取出任何实体，处理中止。")
        return [], [], "No entities were extracted from the document.", chunks

    # 3. 在文档级别融合知识
    final_entities, final_relations = _merge_and_summarize_doc_knowledge(
        doc_id=doc_id,
        all_raw_entities=all_raw_entities,
        all_raw_relations=all_raw_relations,
        llm_client=llm_client,
        tokenizer=tokenizer
    )

    # 4. 生成文档摘要和向量
    document_summary, _ = _generate_document_summary(
        doc_id, 
        final_entities, # 修正: 传递 final_entities 而不是 doc_content
        llm_client, 
        embedding_client,
        tokenizer
    )

    # 如果存在处理失败的块，则将其保存到日志文件中
    if failed_chunks:
        log_dir = "logs/failed_chunks"
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"failed_chunks_{doc_id}.json")
        logger.warning(f"文档 '{doc_id}' 中有 {len(failed_chunks)} 个块处理失败，详细信息已保存到: {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(failed_chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存失败的块到文件 {file_path} 时出错: {e}")


    logger.info(f"========== 完成处理文档: {doc_id} ==========")
    logger.info(f"最终生成 {len(final_entities)} 个实体、{len(final_relations)} 个关系以及一份文档摘要。")

    return final_entities, final_relations, document_summary, chunks


def recursively_process_failed_chunk(
    chunk: Dict[str, Any],
    llm_client: "LLMClient",
    recursion_depth: int = 0
) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    一个专门为处理顽固失败块而设计的函数，它采用“分而治之”的递归策略。

    流程:
    1. 首先，尝试使用标准的 `_extract_entities_and_relations` 函数处理整个块。
    2. 如果失败，则检查块是否满足切分条件（足够大且未超过递归深度）。
    3. 如果满足条件，则将块一分为二，并对每个子块递归调用本函数。
    4. 如果两个子块都成功，则合并并返回结果。
    5. 如果任何一步最终失败，则返回 None。
    """
    MAX_RECURSION_DEPTH = 1  # 最多允许一次切分

    chunk_id = chunk["chunk_id"]
    chunk_content = chunk["content"]

    # 1. 尝试使用标准流程处理
    initial_result = _extract_entities_and_relations(
        chunk_id, chunk_content, llm_client
    )

    if initial_result[0] is not None or initial_result[1] is not None:
        # 标准流程成功（即使结果为空列表也算成功）
        return initial_result
    else:
        # 标准流程彻底失败，尝试分而治之
        if recursion_depth < MAX_RECURSION_DEPTH:
            logger.warning(f"[{chunk_id}] 标准提取失败，尝试切分块并递归处理 (深度: {recursion_depth + 1})")
            
            midpoint = len(chunk_content) // 2
            
            chunk1 = {"chunk_id": chunk_id + "_p1", "content": chunk_content[:midpoint], "doc_id": chunk["doc_id"]}
            chunk2 = {"chunk_id": chunk_id + "_p2", "content": chunk_content[midpoint:], "doc_id": chunk["doc_id"]}

            # 递归调用
            result1 = recursively_process_failed_chunk(
                chunk1, llm_client, recursion_depth + 1
            )
            result2 = recursively_process_failed_chunk(
                chunk2, llm_client, recursion_depth + 1
            )

            if result1 is not None and result2 is not None:
                # 两个子块都成功，合并结果
                entities1, relations1 = result1
                entities2, relations2 = result2
                # 修正：确保为合并后的实体附加正确的原始来源块ID
                for e in entities1: e["source_chunk_id"] = chunk_id
                for e in entities2: e["source_chunk_id"] = chunk_id
                
                logger.info(f"[{chunk_id}] 子块 p1 和 p2 均递归成功，正在合并结果。")
                return entities1 + entities2, relations1 + relations2
            else:
                logger.error(f"[{chunk_id}] 递归处理子块失败，该块最终处理失败。")
                return None, None
        else:
            # 无法或不再进行切分，最终失败
            logger.error(f"[{chunk_id}] 提取失败且已达到最大递归深度，该块最终处理失败。")
            return None, None


def update_document_summary(
    doc_id: str,
    neo4j_driver: Driver,
    llm_client: "LLMClient",
    embedding_client: "EmbeddingClient",
    tokenizer: "PreTrainedTokenizer"
):
    """
    为一个已经补充了新知识的文档，重新生成并更新其总摘要。
    这是一个独立的、用于“收尾”的函数。

    流程:
    1. 从 Neo4j 获取该文档当前最完整的实体列表。
    2. 调用 LLM，基于这些实体生成一个新的、高质量的文档摘要。
    3. 将新的摘要和其向量更新到 Neo4j 和 Milvus 中。
    """
    logger.info(f"[{doc_id}] 开始为文档更新总摘要...")
    
    # 1. 从 Neo4j 获取最新、最全的实体知识
    logger.debug(f"[{doc_id}] 正在从 Neo4j 获取最新的实体列表...")
    knowledge = graph_ops.get_knowledge_for_document(neo4j_driver, doc_id)
    if not knowledge or not knowledge.get("entities"):
        logger.warning(f"[{doc_id}] 在 Neo4j 中未找到实体，无法更新文档摘要。")
        return
    
    final_entities = knowledge["entities"]

    # 2. (新增) 批量更新 Milvus 中的实体摘要和向量
    logger.debug(f"[{doc_id}] 步骤 2/4: 批量更新 Milvus 中的实体...")
    try:
        entity_summaries = [e['summary'] for e in final_entities]
        if entity_summaries:
            logger.info(f"[{doc_id}] 正在为 {len(entity_summaries)} 个实体摘要重新生成嵌入向量...")
            entity_embeddings = embedding_client.get_embeddings(entity_summaries)
            
            # 准备要 upsert 的记录
            entity_records = [
                {
                    "entity_id": entity["entity_id"],
                    "doc_id": doc_id,
                    "name": entity["name"],
                    "description": summary,
                    "is_named_entity": entity.get("is_named_entity", True),
                    "dense_vector": embedding,
                }
                for entity, summary, embedding in zip(final_entities, entity_summaries, entity_embeddings)
            ]

            # a. 先删除旧的实体向量
            vector_ops.delete_vectors_by_filter(
                collection_name=settings.entity_collection_name,
                filter_expr=f'doc_id == "{doc_id}"'
            )
            # b. 再插入新的实体向量
            vector_ops.upsert_vectors(
                collection_name=settings.entity_collection_name,
                data=entity_records
            )
            logger.info(f"[{doc_id}] 成功更新了 {len(entity_records)} 个实体在 Milvus 中的向量。")
    except Exception as e:
        logger.error(f"[{doc_id}] 在更新 Milvus 实体时发生错误: {e}", exc_info=True)
        # 即使实体更新失败，也继续尝试更新文档摘要
    
    # 3. 生成新的文档摘要和向量
    logger.debug(f"[{doc_id}] 步骤 3/4: 正在调用 LLM 生成新的文档摘要...")
    new_summary, new_embedding = _generate_document_summary(
        doc_id,
        final_entities,
        llm_client,
        embedding_client,
        tokenizer
    )

    if not new_summary:
        logger.error(f"[{doc_id}] 未能生成新的文档摘要，更新中止。")
        return

    # 4. 更新数据库中的文档摘要
    try:
        # 更新 Neo4j
        logger.debug(f"[{doc_id}] 步骤 4/4: 正在更新 Neo4j 和 Milvus 中的文档摘要...")
        with neo4j_driver.session() as session:
            query = "MERGE (d:Document {doc_id: $doc_id}) SET d.summary = $summary"
            session.run(query, doc_id=doc_id, summary=new_summary)
        
        # 更新 Milvus
        # a. 先删除旧的
        vector_ops.delete_vectors_by_filter(
            collection_name=settings.summary_collection_name,
            filter_expr=f'doc_id == "{doc_id}"'
        )
        # b. 再插入新的
        vector_ops.upsert_vectors(
            collection_name=settings.summary_collection_name,
            data=[{
                "doc_id": doc_id,
                "summary": new_summary,
                "dense_vector": new_embedding
            }]
        )
        logger.info(f"[{doc_id}] 成功更新文档总摘要。")

    except Exception as e:
        logger.error(f"[{doc_id}] 在更新数据库中的文档摘要时发生错误: {e}", exc_info=True)
        raise
