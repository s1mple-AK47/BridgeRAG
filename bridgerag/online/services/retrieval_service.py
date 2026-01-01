from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

from bridgerag.core.llm_client import LLMClient
from bridgerag.core.embedding_client import EmbeddingClient
from bridgerag.database.graph_db import GraphDBConnection
import bridgerag.database.graph_ops as graph_ops
from bridgerag.database.vector_db import VectorDBConnection
import bridgerag.database.vector_ops as vector_ops
from bridgerag.config import settings
from bridgerag.prompts import PROMPTS
from bridgerag.utils.json_parser import parse_llm_json_output

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    负责在线推理流程的第三阶段：执行计划与证据收集。

    该服务接收由 ReasoningService 生成的检索计划，并精准地从各个数据库中
    提取用于最终答案生成的上下文信息（证据）。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        graph_db: GraphDBConnection,
        vector_db: VectorDBConnection,
    ):
        """
        初始化检索服务。

        参数:
            llm_client: LLM 客户端。
            embedding_client: 向量化客户端。
            graph_db: 图数据库连接。
            vector_db: 向量数据库连接。
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.chunk_collection_name = settings.chunk_collection_name
        logger.info("RetrievalService 初始化完成。")


    def _select_entities_with_llm(self, question: str, doc_id: str, doc_summary: str, doc_entities: List[str]) -> List[str]:
        """
        (鲁棒性增强的核心)
        利用 LLM 从文档的实体列表中，挑选出与用户问题最相关的实体。

        参数:
            question: 用户的问题。
            doc_id: 文档的 ID。
            doc_summary: 文档摘要。
            doc_entities: 从文档中提取的所有实体的列表。

        返回:
            一个只包含被 LLM 挑选出的相关实体的列表。
        """
        logger.debug(f"开始为问题 '{question[:30]}...' 使用 LLM 挑选实体。")

        
        if not doc_entities:
            logger.warning("实体列表为空，无法进行挑选。")
            return []

        try:
            prompt_template = PROMPTS["entity_selection"]
            # 将实体列表转换为字符串格式以便填入 prompt
            entities_list_str = json.dumps(doc_entities)
            
            prompt = prompt_template.render(
                question=question,
                doc_summary=doc_summary,
                entities_list_str=entities_list_str
            )
            response_text = self.llm_client.generate(prompt)
            
            # 修正: 使用稳健的JSON解析器，并使用正确的键 "selected_entities"
            selected_entities_json = parse_llm_json_output(response_text, expected_type=dict)
            if not selected_entities_json or "selected_entities" not in selected_entities_json:
                logger.warning(
                    f"LLM未能从文档 '{doc_id}' 中选择出有效的实体。返回: {response_text}"
                )
                return []
            
            return selected_entities_json.get("selected_entities", [])

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"解析LLM对于实体选择的返回时出错: {e}")
            return []
        except Exception as e:
            logger.error(f"使用 LLM 挑选实体时发生未知错误: {e}")
            return []


    def _retrieve_from_main_doc(self, question: str, question_vector: List[float], doc_id: str) -> str:
        """
        从主文档中检索与问题相关的信息。
        """
        logger.debug(f"开始从主文档 '{doc_id}' 检索信息...")
        try:
            # 1. 获取文档元数据（摘要、实体列表）
            with self.graph_db._driver.session() as session:
                doc_metadata = session.read_transaction(
                    graph_ops.get_document_metadata,
                    doc_id=doc_id
                )
            
            if not doc_metadata:
                return f"Source: Document {doc_id}\nError: Document metadata not found."

            doc_summary = doc_metadata.get("summary", "")
            doc_entities = doc_metadata.get("entities", [])
            
            # 2. LLM选择相关实体
            selected_entities = self._select_entities_with_llm(question, doc_id, doc_summary, doc_entities)

            # 3. 扩展实体列表，包含 `SAME_AS` 关联的实体
            expanded_entities = self._expand_and_filter_entities(
                question_vector,
                doc_id, # 修正: 传入 doc_id
                selected_entities, 
                top_k=settings.SAME_AS_ENTITY_EXPANSION_LIMIT
            )

            all_relevant_entities = list(set(selected_entities + expanded_entities))
            logger.debug(f"文档 '{doc_id}' 的最终相关实体列表: {all_relevant_entities}")

            # 4. 获取实体摘要和相关文本块
            evidence_parts = [f"Source: Document {doc_id}"]
            
            if all_relevant_entities:
                with self.graph_db._driver.session() as session:
                    entity_summaries_list = session.read_transaction(
                        graph_ops.get_entity_summaries_by_name,
                        doc_id=doc_id,
                        entity_names=all_relevant_entities
                    )

                if entity_summaries_list:
                    summary_texts = [f"- {s['name']}: {s['summary']}" for s in entity_summaries_list]
                    evidence_parts.append("Relevant Entity Summaries:\n" + "\n".join(summary_texts))

                retrieved_chunks = vector_ops.search_chunks_by_vector(
                    collection_name=self.chunk_collection_name,
                    query_vector=question_vector,
                    doc_id=doc_id,
                    top_k=settings.CHUNK_RETRIEVAL_LIMIT
                )
                if retrieved_chunks:
                    chunk_texts = [f'- "{chunk["content"]}"' for chunk in retrieved_chunks]
                    evidence_parts.append("Relevant Document Chunks:\n" + "\n".join(chunk_texts))
            
            return "\n".join(evidence_parts)

        except Exception as e:
            logger.error(f"从主文档 '{doc_id}' 检索时出错: {e}", exc_info=True)
            return f"Source: Document {doc_id}\nError: Failed to retrieve details."

    def _retrieve_from_assist_doc(self, doc_id: str, entity_names: List[str]) -> str:
        """
        从辅助文档中检索特定实体的信息。
        """
        logger.debug(f"开始从辅助文档 '{doc_id}' 检索实体 {entity_names} 的信息...")
        if not entity_names:
            return ""
        
        try:
            with self.graph_db._driver.session() as session:
                entity_summaries = session.read_transaction(
                    graph_ops.get_entity_summaries_by_name,
                    doc_id=doc_id,
                    entity_names=entity_names
                )

            if not entity_summaries:
                return f"Source: Document {doc_id}\n- No information found for entities: {entity_names}"

            summary_texts = [
                f"- Entity: {s['name']}\n  Summary: {s['summary']}"
                for s in entity_summaries
            ]
            
            return f"Source: Document {doc_id}\n" + "\n".join(summary_texts)

        except Exception as e:
            logger.error(f"从辅助文档 '{doc_id}' 检索时出错: {e}", exc_info=True)
            return f"Source: Document {doc_id}\nError: Failed to retrieve entity summaries."

    def _expand_and_filter_entities(self, question_vector: List[float], doc_id: str, entity_names: List[str], top_k: int = 5) -> List[str]:
        """
        对于给定的实体列表，查找其 `SAME_AS` 关联的实体，
        然后根据与问题的向量相似度对它们进行排序和筛选。
        """
        if not entity_names:
            return []

        logger.debug(f"开始为实体 {entity_names} 扩展和筛选...")

        try:
            # 1. 从图数据库获取所有相关的 `SAME_AS` 实体及其摘要
            # 注意：这里需要传入原始实体所在的 doc_id 列表，以避免关联到自身文档中的实体
            # 由于当前函数签名没有 doc_id，我们假设 get_linked_entities 能处理
            with self.graph_db._driver.session() as session:
                linked_entities = session.read_transaction(
                    graph_ops.get_linked_entities,
                    doc_id=doc_id, # 修正: 传入 doc_id
                    entity_names=entity_names
                )
            
            if not linked_entities:
                logger.debug(f"实体 {entity_names} 没有找到 'SAME_AS' 关联实体。")
                return []

            # 2. 获取问题向量
            question_embedding = np.array(question_vector).reshape(1, -1)
            
            # 3. 批量获取实体摘要的向量
            summaries = [e['summary'] for e in linked_entities if e and e.get('summary')]
            if not summaries:
                 logger.warning(f"找到 {len(linked_entities)} 个关联实体，但它们都没有摘要，无法进行向量相似度筛选。")
                 return []
            
            valid_linked_entities = [e for e in linked_entities if e and e.get('summary')]

            entity_embeddings = np.array(self.embedding_client.get_embeddings(summaries))

            # 4. 计算问题向量与每个实体摘要向量的余弦相似度
            if entity_embeddings.ndim == 1:
                entity_embeddings = entity_embeddings.reshape(1, -1)

            similarities = cosine_similarity(question_embedding, entity_embeddings)[0]

            # 5. 排序并筛选 Top-K
            entity_similarity_pairs = sorted(
                zip(valid_linked_entities, similarities), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            top_k_entities = [entity['name'] for entity, score in entity_similarity_pairs[:top_k]]
            
            logger.debug(f"为实体 {entity_names} 扩展并筛选出 top {top_k} 个相关实体: {top_k_entities}")
            return top_k_entities
        except Exception as e:
            logger.error(f"在扩展和筛选实体时发生错误: {e}", exc_info=True)
            return []


    def execute_retrieval_plan(self, question: str, plan: Dict) -> List[str]:
        """
        根据检索计划，并行地从主文档和辅助文档中检索信息。
        """
        logger.info(f"开始执行检索计划")
        main_docs = plan.get("main_documents", [])
        assist_docs_map = plan.get("assist_documents", {})
        
        # 优化：在流程开始时只对问题进行一次向量化
        question_vector = self.embedding_client.get_embeddings([question])[0]
    
        tasks = []
        results = []

        with ThreadPoolExecutor() as executor:
            # 为每个主文档创建一个检索任务
            main_doc_futures = {
                executor.submit(self._retrieve_from_main_doc, question, question_vector, doc_id): doc_id 
                for doc_id in main_docs
            }
            
            # 为每个辅助文档创建一个检索任务
            assist_doc_futures = {executor.submit(self._retrieve_from_assist_doc, doc_id, entities): doc_id for doc_id, entities in assist_docs_map.items()}

            # 合并future字典
            all_futures = {**main_doc_futures, **assist_doc_futures}

            for future in as_completed(all_futures):
                doc_id = all_futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"执行文档 '{doc_id}' 的检索任务时捕获到异常: {e}", exc_info=True)
                    results.append(f"Source: Document {doc_id}\nError: Failed to retrieve details due to an exception.")

        logger.info(f"检索计划执行完毕，共收集到 {len(results)} 条证据。")
        return results
