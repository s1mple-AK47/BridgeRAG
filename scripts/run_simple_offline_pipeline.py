import logging
import os
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import settings
from bridgerag.utils.logging_config import setup_logging
from bridgerag.core.llm_client import LLMClient
from bridgerag.core.embedding_client import EmbeddingClient
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.object_storage import ObjectStorageConnection
from bridgerag.database.vector_db import VectorDBConnection

# 导入离线处理流程的核心步骤函数
from bridgerag.offline.steps.build_partitions import process_document
from bridgerag.offline.steps.save_knowledge_graph import save_partition
from bridgerag.offline.steps.link_entities import run_entity_linking

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """
    一个简单、同步的端到端离线处理流水线脚本。

    该脚本会执行以下步骤:
    1. 初始化所有必要的客户端和服务连接 (LLM, Embedding, Neo4j, MinIO, Milvus)。
    2. 读取 'data/sample_docs' 目录下的所有 Markdown (.md) 文件。
    3. 对每个文档执行完整的处理流程:
        a. 调用 `process_document` 进行文本分块、实体与关系提取、知识融合和摘要生成。
        b. 调用 `save_partition` 将处理结果存入 Neo4j, Milvus, 和 MinIO。
           - 使用 `force_rewrite=True` 来确保脚本的可重复运行性。
    4. 在所有文档处理并保存完毕后，调用 `run_entity_linking` 执行跨文档的实体链接。
    """
    logger.info("--- 启动简单离线处理流水线 ---")

    # 根据 tasks.py 中的实践，milvus_alias 硬编码为 "default"
    MILVUS_ALIAS = "default"

    # --- 1. 初始化所有客户端和服务 ---
    try:
        logger.info("正在初始化所有客户端...")
        llm_client = LLMClient()
        embedding_client = EmbeddingClient()
        
        # 从 LLMClient 中获取 Tokenizer
        if not llm_client.tokenizer:
            raise ValueError("Tokenizer 未在 LLMClient 中成功初始化。")
        tokenizer = llm_client.tokenizer

        neo4j_conn = GraphDBConnection(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        minio_conn = ObjectStorageConnection(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key
        )
        
        # 确保 Milvus 连接已建立
        # 从 MILVUS_URI (e.g., "http://localhost:19530") 中解析 host 和 port
        milvus_uri_parts = settings.milvus_uri.replace("http://", "").replace("https://", "").split(":")
        milvus_host = milvus_uri_parts[0]
        milvus_port = milvus_uri_parts[1] if len(milvus_uri_parts) > 1 else "19530"

        milvus_conn = VectorDBConnection(
            host=milvus_host,
            port=milvus_port,
            alias=MILVUS_ALIAS
        )
        
        logger.info("所有客户端和服务已成功初始化。")
    except Exception as e:
        logger.error(f"客户端初始化失败: {e}", exc_info=True)
        return

    # --- 2. 加载示例文档 ---
    sample_docs_dir = project_root / "data" / "sample_docs"
    if not sample_docs_dir.exists():
        logger.error(f"示例文件目录不存在: {sample_docs_dir}")
        return
        
    documents_to_process = list(sample_docs_dir.glob("*.md"))
    if not documents_to_process:
        logger.warning(f"在目录 {sample_docs_dir} 中未找到任何 .md 文件。")
        return

    logger.info(f"发现 {len(documents_to_process)} 个示例文档待处理。")

    # --- 3. 依次处理每个文档 ---
    for doc_path in documents_to_process:
        doc_id = doc_path.stem  # 使用文件名（不含扩展名）作为文档ID
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_content = f.read()

            logger.info(f"--- 开始处理文档: {doc_id} ---")
            
            # 3a. 提取信息
            final_entities, final_relations, document_summary, chunks = process_document(
                doc_id=doc_id,
                doc_content=doc_content,
                llm_client=llm_client,
                embedding_client=embedding_client,
                tokenizer=tokenizer,
            )



            # 3b. 保存处理结果
            if final_entities:
                save_partition(
                    doc_id=doc_id,
                    chunks=chunks,
                    entities=final_entities,
                    relations=final_relations,
                    document_summary=document_summary,
                    neo4j_driver=neo4j_conn._driver,
                    minio_client=minio_conn.client,
                    embedding_client=embedding_client,
                    milvus_alias=MILVUS_ALIAS,
                    minio_configs={"bucket_name": settings.minio_bucket_name},
                    embedding_configs={"embedding_model": settings.vllm_embedding_model_name},
                    milvus_collection_names={
                        "chunk_collection_name": settings.chunk_collection_name,
                        "entity_collection_name": settings.entity_collection_name,
                        "summary_collection_name": settings.summary_collection_name
                    },
                    force_rewrite=True  # 确保幂等性
                )
                logger.info(f"--- 文档 {doc_id} 处理和保存成功 ---")
            else:
                logger.warning(f"文档 {doc_id} 未能提取出任何实体，跳过保存步骤。")

        except Exception as e:
            logger.error(f"处理文档 {doc_id} 时发生未知错误: {e}", exc_info=True)
            continue # 继续处理下一个文档

    # --- 4. 执行实体链接 ---
    logger.info("--- 所有文档处理完毕，开始执行实体链接 ---")
    try:
        run_entity_linking(
            driver=neo4j_conn._driver,
            llm_client=llm_client,
            milvus_collection_name=settings.entity_collection_name
        )
        logger.info("--- 实体链接完成 ---")
    except Exception as e:
        logger.error(f"实体链接过程中发生错误: {e}", exc_info=True)

    # --- 清理和关闭连接 ---
    neo4j_conn.close()
    milvus_conn.close()
    logger.info("--- 简单离线处理流水线执行完毕 ---")


if __name__ == "__main__":
    main()
