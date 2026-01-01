import logging
import signal
from celery.exceptions import WorkerTerminate
from celery.signals import task_failure
from celery.platforms import signals as celery_platforms
from typing import Any, Dict, List
from urllib.parse import urlparse

from bridgerag.config import Settings
from bridgerag.core.llm_client import LLMClient
from bridgerag.core.embedding_client import EmbeddingClient
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.object_storage import ObjectStorageConnection
from bridgerag.database.vector_db import VectorDBConnection
from bridgerag.utils.text_processing import load_tokenizer
from bridgerag.utils.logging_config import setup_logging
from bridgerag.celery_app import celery_app

# 导入原始的业务逻辑函数
from .steps import build_partitions, save_knowledge_graph

logger = logging.getLogger(__name__)


# --- Celery Signal Handler for graceful shutdown on task failure ---

# @task_failure.connect
# def on_task_failure_shutdown_worker(sender=None, task_id=None, exception=None, **kwargs):
#     """
#     Celery signal handler that shuts down the worker when any task fails.
#     This is useful for debugging to prevent a flood of failed tasks.
#     """
#     logging.error(f"Task {getattr(sender, 'name', 'unknown')} [{task_id}] failed with exception: {exception}")
#     logging.error("A task failed. Shutting down the worker to stop the pipeline.")
#     # Use the recommended way to stop the worker from within a signal handler.
#     # This will raise an exception that the worker catches to perform a warm shutdown.
#     raise WorkerTerminate()


# --- Client Initialization ---

# This dictionary will cache client instances within a single worker process
# to avoid re-initialization for every task.
_worker_clients = None


def _initialize_clients():
    """
    Initializes and caches all necessary clients within a Celery worker process.
    This avoids passing non-serializable client objects between tasks and
    prevents re-creating clients for every single task.
    """
    global _worker_clients
    if _worker_clients is not None:
        return _worker_clients

    setup_logging()

    # LLM and Embedding clients read their config from the global `settings` object.
    llm_client = LLMClient()
    embedding_client = EmbeddingClient()

    # Database clients require connection parameters.
    settings = Settings()
    graph_ops_client = GraphDBConnection(
        uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password
    )
    tokenizer = load_tokenizer()

    # Parse Milvus URI to get host and port for the VectorDB connection.
    parsed_milvus_uri = urlparse(settings.milvus_uri)
    vector_ops_client = VectorDBConnection(
        host=parsed_milvus_uri.hostname, port=parsed_milvus_uri.port
    )

    _worker_clients = {
        "llm": llm_client,
        "embedding": embedding_client,
        "graph_ops": graph_ops_client,
        "vector_ops": vector_ops_client,
        "tokenizer": tokenizer,
    }
    logging.info("Clients initialized successfully in worker process.")
    return _worker_clients


# --- Celery Tasks ---

@celery_app.task(name="offline.process_document")
def process_document_task(document_data: dict, doc_id: str):
    """Celery task to process a single document."""
    logger = logging.getLogger(__name__)
    logger.info(f"[TASK] Starting to process document: {doc_id}")
    try:
        clients = _initialize_clients()
        entities, relations, summary, chunks = build_partitions.process_document(
            doc_id=doc_id,
            doc_content=document_data["text"],
            llm_client=clients["llm"],
            embedding_client=clients["embedding"],
            tokenizer=clients["tokenizer"],
        )
        
        # 将元组打包成字典以适配下游任务
        processed_data = {
            "doc_id": doc_id,
            "entities": entities,
            "relations": relations,
            "document_summary": summary,
            "chunks": chunks,
        }
        
        logger.info(f"Successfully processed document: {doc_id}")
        
        # 增加详细日志，用于调试
        import json
        logger.debug(f"即将返回的处理后数据 for doc {doc_id}: {json.dumps(processed_data, indent=2, ensure_ascii=False)}")
        
        return processed_data
    except Exception as e:
        logger.error(f"[TASK] 处理文档 {doc_id} 失败: {e}", exc_info=True)
        # 抛出异常，Celery可以根据配置进行重试
        raise


@celery_app.task(name="offline.save_knowledge_graph")
def save_knowledge_graph_task(processed_data: dict):
    """Celery task to save the processed knowledge graph partition to databases."""
    # 增加详细日志，用于调试
    import json
    logger.debug(f"收到的待保存数据: {json.dumps(processed_data, indent=2, ensure_ascii=False)}")
    
    if not processed_data:
        logger.warning("Received empty processed data. Skipping save.")
        return None
    try:
        clients = _initialize_clients()
        settings = Settings()
        
        object_storage_conn = ObjectStorageConnection(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
        )

        try:
            save_knowledge_graph.save_partition(
                **processed_data,
                neo4j_driver=clients["graph_ops"]._driver,
                minio_client=object_storage_conn.client,
                embedding_client=clients["embedding"],
                milvus_alias="default",
                minio_configs={"bucket_name": settings.minio_bucket_name},
                embedding_configs={"embedding_model": settings.vllm_embedding_model_name},
                milvus_collection_names={
                    "chunk_collection_name": settings.chunk_collection_name,
                    "entity_collection_name": settings.entity_collection_name,
                    "summary_collection_name": settings.summary_collection_name,
                },
                force_rewrite=False
            )
        except Exception as e:
            doc_id_for_error = processed_data.get("doc_id", "Unknown")
            logger.error(f"调用 save_partition 时出错, doc_id: {doc_id_for_error}. 错误: {e}", exc_info=True)
            # 关键：让 Celery worker 优雅地关闭
            logger.error("一个任务失败。正在关闭 worker 以停止整个管道。")
            # 使用 app.control.shutdown() 来请求关闭所有 worker
            # 这是一个更激进的停止方式，适用于一个失败就意味着整个批次失败的场景
            # self.app.control.shutdown()
            raise e # 重新抛出异常，让 Celery 知道任务失败了

        doc_id = processed_data.get("doc_id", "Unknown")
        logger.info(f"Successfully saved knowledge graph for document: {doc_id}")
        # Return only the necessary data for the next step
        return {
            "doc_id": doc_id,
            "entity_count": len(processed_data.get("entities", [])),
        }
    except Exception as e:
        logger.error(
            f"An error occurred during save_knowledge_graph_task for doc "
            f"'{processed_data.get('doc_id', 'Unknown')}': {e}",
            exc_info=True
        )
        raise
