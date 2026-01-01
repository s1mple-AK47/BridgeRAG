import logging
from typing import Dict, Any, List
from urllib.parse import urlparse

# 数据库连接
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.vector_db import VectorDBConnection
from bridgerag.database.object_storage import ObjectStorageConnection

# 核心客户端
from bridgerag.core.llm_client import LLMClient
from bridgerag.core.embedding_client import EmbeddingClient

# 在线服务
from bridgerag.online.services.routing_service import RoutingService
from bridgerag.online.services.reasoning_service import ReasoningService
from bridgerag.online.services.retrieval_service import RetrievalService
from bridgerag.online.services.synthesis_service import SynthesisService
from bridgerag.online.schemas import QueryResponse, SynthesisDecision, QueryResult
from bridgerag.config import settings

logger = logging.getLogger(__name__)

class OnlineQueryProcessor:
    """
    在线查询处理的总指挥官。
    负责初始化所有服务并编排完整的、迭代式的端到端查询流程。
    """
    def __init__(self):
        """
        初始化所有必要的客户端、数据库连接和服务。
        """
        self.logger = logger
        self.logger.info("正在初始化 OnlineQueryProcessor...")
        
        # 从配置中加载参数，正确初始化所有客户端和数据库连接
        self.graph_db = GraphDBConnection(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        parsed_milvus_uri = urlparse(settings.milvus_uri)
        milvus_host = parsed_milvus_uri.hostname
        milvus_port = parsed_milvus_uri.port

        self.vector_db = VectorDBConnection(
            host=milvus_host,
            port=milvus_port
        )
        self.llm_client = LLMClient()
        self.embedding_client = EmbeddingClient()

        # 注入依赖，初始化所有服务
        self.routing_service = RoutingService(self.llm_client, self.graph_db, self.vector_db, self.embedding_client)
        self.reasoning_service = ReasoningService(self.llm_client, self.graph_db)
        self.retrieval_service = RetrievalService(self.llm_client, self.embedding_client, self.graph_db, self.vector_db)
        self.synthesis_service = SynthesisService(self.llm_client)
        self.logger.info("OnlineQueryProcessor 初始化完成。")

    def close(self):
        """
        关闭并清理所有数据库连接。
        """
        self.logger.info("正在关闭 OnlineQueryProcessor 的数据库连接...")
        self.graph_db.close()
        self.vector_db.close()
        self.logger.info("数据库连接已关闭。")

    def process_query(self, question: str, max_turns: int = 3) -> QueryResult:
        self.logger.info(f"开始处理问题: '{question}'")
        
        conversation_history = []
        all_main_documents = set()
        current_question = question
        final_decision = None

        for current_turn in range(max_turns):
            self.logger.info(f"--- 第 {current_turn + 1} 轮查询 ---")
            self.logger.info(f"当前问题: '{current_question}'")

            # 1. 路由
            self.logger.info("步骤 1: 路由 - 为问题寻找最相关的文档...")
            candidate_docs = self.routing_service.route(current_question, top_k=settings.RERANK_TOP_K)
            doc_ids = [doc['doc_id'] for doc in candidate_docs]
            all_main_documents.update(doc_ids)
            self.logger.info(f"  - 找到 {len(doc_ids)} 个候选文档: {doc_ids}")

            # 2. 推理
            self.logger.info("步骤 2: 推理 - 生成检索计划...")
            retrieval_plan = self.reasoning_service.generate_retrieval_plan(current_question, candidate_docs)
            self.logger.info(f"  - 推理内容: {retrieval_plan.get('reasoning', 'N/A')}")
            self.logger.info(f"  - 主文档: {retrieval_plan.get('main_documents', [])}")
            self.logger.info(f"  - 辅助文档: {retrieval_plan.get('assist_documents', {})}")

            # 3. 检索
            self.logger.info("步骤 3: 检索 - 从文档中收集证据...")
            evidence = self.retrieval_service.execute_retrieval_plan(current_question, retrieval_plan)
            self.logger.info(f"  - 收集到 {len(evidence)} 条证据。")

            # 4. 综合
            self.logger.info("步骤 4: 综合 - 基于证据生成决策...")
            history_str = "\n".join(conversation_history)
            final_decision = self.synthesis_service.generate_answer(
                question=current_question, 
                history=history_str,
                evidences=evidence
            )
            self.logger.info(f"  - 决策: {final_decision.decision}")
            self.logger.info(f"  - 内容: {final_decision.content}")
            self.logger.info(f"  - 摘要: {final_decision.summary}")

            # 更新历史记录
            conversation_history.append(f"User Question: {current_question}")
            conversation_history.append(f"LLM Summary: {final_decision.summary}")

            if final_decision.decision == "ANSWER":
                self.logger.info("决策为 'ANSWER'，查询结束。")
                break
            else:
                current_question = final_decision.content
                self.logger.info(f"决策为 'SUB_QUESTION'，生成新问题: '{current_question}'")
        else:
             self.logger.warning(f"已达到最大查询轮次 ({max_turns})，终止查询。")

        return QueryResult(
            question=question,
            answer=final_decision.content if final_decision else "No answer could be generated.",
            main_documents=sorted(list(all_main_documents)),
            conversation_history="\n".join(conversation_history)
        )
