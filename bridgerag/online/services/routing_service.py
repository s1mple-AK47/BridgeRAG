from typing import List, Dict, Any
from collections import defaultdict
import logging
import json

from bridgerag.core.llm_client import LLMClient
from bridgerag.core.embedding_client import EmbeddingClient
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.vector_db import VectorDBConnection
import bridgerag.database.graph_ops as graph_ops
import bridgerag.database.vector_ops as vector_ops
from bridgerag.prompts.prompts import QUESTION_ENTITY_EXTRACTION_PROMPT
from bridgerag.utils.text_processing import normalize_entity_name
from bridgerag.config import settings
from bridgerag.prompts import PROMPTS

logger = logging.getLogger(__name__)

class RoutingService:
    """
    负责在线推理流程的第一阶段：候选文档路由。

    该服务接收用户问题，通过一系列步骤从海量文档中筛选出最可能包含答案的
    Top-K 个候选文档，为后续的精细化检索和推理做准备。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        graph_db: GraphDBConnection,
        vector_db: VectorDBConnection,
        embedding_client: EmbeddingClient
    ):
        """
        初始化路由服务。

        参数:
            llm_client: LLM 客户端。
            graph_db: 图数据库连接。
            vector_db: 向量数据库连接。
            embedding_client: 向量化客户端。
        """
        self.llm_client = llm_client
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.embedding_client = embedding_client
        self.summary_collection_name = settings.summary_collection_name
        logger.info("RoutingService 初始化完成。")

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """
        使用 LLM 从用户问题中抽取出关键实体。
        """
        logger.debug(f"正在从问题中提取实体: '{question[:50]}...'")
        prompt_template = PROMPTS["question_entity_extraction"]
        # 修正: Jinja2 模板中的占位符是 {{user_question}}，因此这里必须使用 user_question 关键字
        prompt = prompt_template.render(user_question=question)
        
        try:
            response_text = self.llm_client.generate(prompt)
            data = json.loads(response_text)
            entities = data.get("entities", [])
            
            # 规范化实体名称
            normalized_entities = [normalize_entity_name(entity) for entity in entities]
            
            logger.info(f"从问题中提取并规范化实体: {normalized_entities}")
            return normalized_entities
        except json.JSONDecodeError:
            logger.error(f"无法解析从LLM返回的实体JSON: {response_text}")
            return []
        except Exception as e:
            logger.error(f"从问题中提取实体时发生错误: {e}")
            return []

    def _recall_from_graph(self, entities: List[str], limit: int) -> List[Dict]:
        """
        基于实体列表，在知识图谱中召回相关文档，并按实体匹配度排序。
        """
        if not entities:
            return []

        logger.info(f"开始在图数据库中为实体 {entities} 召回文档...")
        
        try:
            # 修正: 使用正确的 session.read_transaction 模式
            with self.graph_db._driver.session() as session:
                recalled_docs = session.read_transaction(
                    graph_ops.get_documents_by_entities,
                    entity_names=entities,
                    limit=limit
                )
            
            logger.info(f"成功从图数据库中召回 {len(recalled_docs)} 个文档。")
            return recalled_docs
        except Exception as e:
            logger.error(f"从图数据库召回文档时出错: {e}")
            return []

    def _recall_from_vector(self, question: str, top_k: int) -> List[Dict]:
        """
        基于文档摘要的向量相似度，从 Milvus 中召回 Top-K 文档。
        """
        logger.debug("开始从向量数据库召回文档...")
        try:
            # 1. 将问题向量化
            query_vector = self.embedding_client.get_embeddings([question])[0]

            # 2. 在摘要集合中进行搜索
            search_results = vector_ops.search_summaries_by_vector(
                collection_name=self.summary_collection_name,
                query_vector=query_vector,
                top_k=top_k
            )
            
            logger.info(f"成功从向量数据库召回 {len(search_results)} 个文档。")
            return search_results
        except Exception as e:
            logger.error(f"从向量数据库召回文档时出错: {e}")
            return []

    def _fuse_rankings(
        self, 
        ranked_lists: List[List[Dict[str, Any]]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        使用倒数排序融合 (RRF) 算法合并多个排序列表。

        RRF 是一种无需调参的、效果优秀的排序融合方法。
        它根据文档在各个列表中的排名来计算最终分数，而不是原始分数。
        公式: RRF_Score(d) = sum(1 / (k + rank))

        参数:
            ranked_lists (List[List[Dict]]): 一个包含多个召回结果列表的列表。
                                              每个内部列表都是一个排序后的字典列表，
                                              且每个字典必须包含 'doc_id' 键。
            k (int): RRF算法中的一个常数，用于降低排名靠后文档的权重。
                     通常建议值为60。

        返回:
            一个按RRF分数降序排列的、融合后的文档列表。
        """
        logger.info(f"开始对 {len(ranked_lists)} 个召回列表进行排序融合...")
        
        fused_scores = defaultdict(float)
        
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                doc_id = doc.get("doc_id")
                if not doc_id:
                    continue
                # RRF 公式
                fused_scores[doc_id] += 1.0 / (k + rank + 1) # rank 从0开始，所以+1

        if not fused_scores:
            logger.warning("排序融合未能产生任何结果。")
            return []

        # 重新排序
        reranked_results = [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in fused_scores.items()
        ]
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"排序融合完成，共得到 {len(reranked_results)} 个唯一文档。")
        return reranked_results


    def route(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        执行完整的文档路由流程：提取实体、双路召回、排序融合。
        """
        logger.info(f"开始为问题 '{question[:50]}...' 进行路由...")
        
        # 步骤 1: 从问题中提取实体
        entities = self._extract_entities_from_question(question)
        
        # 步骤 2: 并行执行两种召回策略
        # 为了让 RRF 融合效果更好，我们从每个源召回更多的文档
        recall_limit = max(top_k * 2, 10)
        
        graph_recalled_docs = self._recall_from_graph(entities, limit=recall_limit)
        vector_recalled_docs = self._recall_from_vector(question, top_k=recall_limit)
        
        # 步骤 3: 使用 RRF 融合和重排结果
        fused_docs = self._fuse_rankings([graph_recalled_docs, vector_recalled_docs])
        
        # 步骤 4: 返回 Top-K 结果
        final_docs = fused_docs[:top_k]
        logger.info(f"路由完成，返回 {len(final_docs)} 个文档: {[d['doc_id'] for d in final_docs]}")
        
        return final_docs
