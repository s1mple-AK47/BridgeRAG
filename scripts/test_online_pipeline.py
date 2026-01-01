import logging
import sys
from pathlib import Path
import asyncio

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import settings
from bridgerag.utils.logging_config import setup_logging
from bridgerag.core.llm_client import LLMClient
from bridgerag.core.embedding_client import EmbeddingClient
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.vector_db import VectorDBConnection

# 导入所有在线服务
from bridgerag.online.services.routing_service import RoutingService
from bridgerag.online.services.reasoning_service import ReasoningService
from bridgerag.online.services.retrieval_service import RetrievalService
from bridgerag.online.services.synthesis_service import SynthesisService

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)


async def main():
    """
    一个用于测试和演示完整在线查询流程的脚本。

    该脚本会按顺序执行以下步骤:
    1. 初始化所有必要的客户端和服务 (LLM, Embedding, 数据库连接, 以及四个在线服务)。
    2. 定义一个测试问题。
    3. 依次调用在线服务的核心方法，模拟一个完整的 RAG 查询流程:
        a. RoutingService: 对问题进行路由，获取候选文档。
        b. ReasoningService: 基于候选文档生成检索计划。
        c. RetrievalService: 执行检索计划，搜集详细的上下文证据。
        d. SynthesisService: 基于所有信息，生成最终答案或决策。
    4. 打印每一步的详细输入和输出，以展示数据在流程中的变化。
    """
    logger.info("--- 启动在线查询流程测试脚本 ---")

    # --- 1. 初始化所有客户端和服务 ---
    try:
        logger.info("正在初始化所有客户端和服务...")
        # 数据库和AI客户端
        llm_client = LLMClient()
        embedding_client = EmbeddingClient()
        
        graph_db_conn = GraphDBConnection(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )

        milvus_uri_parts = settings.milvus_uri.replace("http://", "").replace("https://", "").split(":")
        milvus_host = milvus_uri_parts[0]
        milvus_port = milvus_uri_parts[1] if len(milvus_uri_parts) > 1 else "19530"
        vector_db_conn = VectorDBConnection(host=milvus_host, port=milvus_port)

        # 在线服务
        routing_service = RoutingService(llm_client, graph_db_conn, vector_db_conn, embedding_client)
        reasoning_service = ReasoningService(llm_client, graph_db_conn)
        retrieval_service = RetrievalService(llm_client, embedding_client, graph_db_conn, vector_db_conn)
        synthesis_service = SynthesisService(llm_client)
        
        logger.info("所有客户端和服务已成功初始化。")
    except Exception as e:
        logger.error(f"客户端或服务初始化失败: {e}", exc_info=True)
        return
    
    # --- 2. 定义测试问题 ---
    question = "Who created Star Wars?"
    logger.info(f"测试问题: \"{question}\"")

    try:
        # --- 3. 完整在线流程调用 ---
        
        # a. Routing
        logger.info("\n--- 步骤 1: 调用 RoutingService ---")
        candidate_docs = routing_service.route(question=question, top_k=3)
        logger.info(f"RoutingService 输出 (候选文档): {candidate_docs}")
        if not candidate_docs:
            logger.warning("路由阶段未能找到任何候选文档，流程中止。")
            return

        # b. Reasoning (Planning)
        logger.info("\n--- 步骤 2: 调用 ReasoningService ---")
        retrieval_plan = reasoning_service.generate_retrieval_plan(question=question, candidate_docs=candidate_docs)
        logger.info(f"ReasoningService 输出 (检索计划): {retrieval_plan}")
        if not retrieval_plan.get("main_documents") and not retrieval_plan.get("assist_documents"):
            logger.warning("推理阶段未能生成有效的检索计划，流程中止。")
            return
            
        # c. Retrieval
        logger.info("\n--- 步骤 3: 调用 RetrievalService ---")
        evidence = retrieval_service.execute_retrieval_plan(question=question, plan=retrieval_plan)
        evidence_str = "\n\n".join(evidence)
        logger.info(f"RetrievalService 输出 (证据):\n{evidence_str}")
        logger.info("")
        if not evidence:
            logger.warning("检索阶段未能收集到任何证据，流程中止。")
            return

        # d. Synthesis
        logger.info("\n--- 步骤 4: 调用 SynthesisService ---")
        # 修正: 移除 await，并使用正确的参数名 'evidences'
        final_decision = synthesis_service.generate_answer(
            question=question,
            history="",  # 第一轮查询，无历史
            evidences=evidence
        )

        logger.info("SynthesisService 输出 (最终决策):")
        # 修正: SynthesisDecision 是一个Pydantic对象，应直接访问其属性，而不是使用 .get()
        logger.info(f"  - Decision: {final_decision.decision}")
        logger.info(f"  - Content: {final_decision.content}")
        logger.info(f"  - Summary: {final_decision.summary}")

    except Exception as e:
        logger.error(f"在线查询流程执行期间发生未知错误: {e}", exc_info=True)
    finally:
        # --- 清理和关闭连接 ---
        graph_db_conn.close()
        vector_db_conn.close()
        logger.info("\n--- 在线查询流程测试脚本执行完毕 ---")


if __name__ == "__main__":
    asyncio.run(main())
