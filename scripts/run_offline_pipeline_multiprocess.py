"""
HotpotQA 数据集的离线处理流水线 - 多进程版本（不依赖 Celery）

使用 Python multiprocessing 实现并发处理，每个进程独立初始化客户端。
"""

import json
import logging
import multiprocessing as mp
from multiprocessing import Pool, Manager
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import Settings
from bridgerag.utils.logging_config import setup_logging

# 配置日志
setup_logging()
logger = logging.getLogger(__name__)

# ============== 配置 ==============
NUM_WORKERS = 8  # 并发进程数，根据你的 CPU 和 GPU 资源调整
# ==================================


# 全局变量，用于在每个 worker 进程中缓存客户端
_worker_clients = None


def init_worker():
    """
    Worker 进程初始化函数。
    在每个 worker 进程启动时调用一次，初始化所有客户端。
    """
    global _worker_clients
    
    # 重新设置日志（每个进程需要独立设置）
    setup_logging()
    
    from bridgerag.core.llm_client import LLMClient
    from bridgerag.core.embedding_client import EmbeddingClient
    from bridgerag.database.graph_db import GraphDBConnection
    from bridgerag.database.vector_db import VectorDBConnection
    from bridgerag.database.object_storage import ObjectStorageConnection
    from bridgerag.utils.text_processing import load_tokenizer
    
    settings = Settings()
    
    # 初始化 LLM 和 Embedding 客户端
    llm_client = LLMClient()
    embedding_client = EmbeddingClient()
    
    # 初始化数据库连接
    graph_conn = GraphDBConnection(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password
    )
    
    parsed_milvus_uri = urlparse(settings.milvus_uri)
    vector_conn = VectorDBConnection(
        host=parsed_milvus_uri.hostname,
        port=parsed_milvus_uri.port
    )
    
    object_storage_conn = ObjectStorageConnection(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key
    )
    
    tokenizer = load_tokenizer()
    
    _worker_clients = {
        "llm": llm_client,
        "embedding": embedding_client,
        "graph_conn": graph_conn,
        "vector_conn": vector_conn,
        "object_storage_conn": object_storage_conn,
        "tokenizer": tokenizer,
        "settings": settings,
    }
    
    logger.info(f"Worker 进程 {mp.current_process().name} 初始化完成")


def process_single_document(doc_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个文档的完整流程：提取知识 -> 保存到数据库
    
    Args:
        doc_data: {"id": doc_id, "text": doc_content}
    
    Returns:
        {"doc_id": str, "success": bool, "error": str or None, "entity_count": int}
    """
    global _worker_clients
    
    doc_id = doc_data["id"]
    doc_content = doc_data["text"]
    
    result = {
        "doc_id": doc_id,
        "success": False,
        "error": None,
        "entity_count": 0
    }
    
    try:
        # 延迟导入，避免在主进程中导入
        from bridgerag.offline.steps import build_partitions, save_knowledge_graph
        
        clients = _worker_clients
        settings = clients["settings"]
        
        logger.info(f"[{doc_id}] 开始处理文档...")
        
        # Step 1: 提取实体、关系、摘要
        entities, relations, summary, chunks = build_partitions.process_document(
            doc_id=doc_id,
            doc_content=doc_content,
            llm_client=clients["llm"],
            embedding_client=clients["embedding"],
            tokenizer=clients["tokenizer"],
        )
        
        logger.info(f"[{doc_id}] 提取完成: {len(entities)} 实体, {len(relations)} 关系")
        
        # Step 2: 保存到数据库
        save_knowledge_graph.save_partition(
            doc_id=doc_id,
            entities=entities,
            relations=relations,
            document_summary=summary,
            chunks=chunks,
            neo4j_driver=clients["graph_conn"]._driver,
            minio_client=clients["object_storage_conn"].client,
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
        
        result["success"] = True
        result["entity_count"] = len(entities)
        logger.info(f"[{doc_id}] 处理完成并保存成功")
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[{doc_id}] 处理失败: {e}", exc_info=True)
    
    return result


def get_processed_doc_ids() -> set:
    """从 MinIO 和本地日志获取已处理的文档ID"""
    from bridgerag.database.object_storage import ObjectStorageConnection
    import bridgerag.database.object_storage_ops as storage_ops
    
    settings = Settings()
    processed_ids = set()
    
    # 从 MinIO 获取
    try:
        minio_conn = ObjectStorageConnection(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key
        )
        bucket_name = settings.minio_bucket_name
        
        if minio_conn.client.bucket_exists(bucket_name):
            object_names = storage_ops.list_objects(minio_conn.client, bucket_name)
            minio_ids = {Path(obj_name).stem for obj_name in object_names}
            processed_ids.update(minio_ids)
            logger.info(f"从 MinIO 发现 {len(minio_ids)} 个已处理文档")
    except Exception as e:
        logger.warning(f"从 MinIO 获取已处理文档失败: {e}")
    
    # 从本地日志获取
    log_file = project_root / "logs" / "successful_documents.log"
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_ids = {line.strip() for line in f if line.strip()}
            processed_ids.update(log_ids)
            logger.info(f"从日志文件发现 {len(log_ids)} 个已处理文档")
        except Exception as e:
            logger.warning(f"读取日志文件失败: {e}")
    
    return processed_ids


def load_documents(file_path: Path) -> list:
    """从 JSONL 文件加载文档"""
    documents = []
    seen_ids = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                doc_id = data.get("id", "").strip()
                text = data.get("text", "").strip()
                
                if doc_id and text and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    documents.append({"id": doc_id, "text": text})
            except json.JSONDecodeError:
                continue
    
    return documents


def append_to_success_log(doc_id: str):
    """将成功处理的文档ID追加到日志文件"""
    log_file = project_root / "logs" / "successful_documents.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{doc_id}\n")


def main():
    """主函数"""
    data_file = project_root / "hotpot_docs.jsonl"
    
    if not data_file.exists():
        logger.error(f"数据文件不存在: {data_file}")
        logger.error("请先运行 python scripts/convert_hotpot_data.py 转换数据格式")
        return
    
    # 获取已处理的文档
    processed_ids = get_processed_doc_ids()
    logger.info(f"共 {len(processed_ids)} 个文档已处理，将被跳过")
    
    # 加载待处理文档
    all_docs = load_documents(data_file)
    docs_to_process = [doc for doc in all_docs if doc["id"] not in processed_ids]
    
    if not docs_to_process:
        logger.info("所有文档均已处理完毕！")
        return
    
    logger.info(f"共 {len(all_docs)} 篇文档，{len(docs_to_process)} 篇待处理")
    logger.info(f"使用 {NUM_WORKERS} 个并发进程")
    
    # 统计
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    # 创建进程池
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        # 使用 imap_unordered 实现流式处理，边处理边返回结果
        for i, result in enumerate(pool.imap_unordered(process_single_document, docs_to_process)):
            if result["success"]:
                success_count += 1
                append_to_success_log(result["doc_id"])
            else:
                fail_count += 1
                logger.error(f"文档 {result['doc_id']} 处理失败: {result['error']}")
            
            # 进度报告
            total_done = success_count + fail_count
            elapsed = time.time() - start_time
            avg_time = elapsed / total_done if total_done > 0 else 0
            remaining = len(docs_to_process) - total_done
            eta = avg_time * remaining
            
            if total_done % 10 == 0 or total_done == len(docs_to_process):
                logger.info(
                    f"进度: {total_done}/{len(docs_to_process)} "
                    f"(成功: {success_count}, 失败: {fail_count}) "
                    f"| 平均: {avg_time:.1f}s/doc | ETA: {eta/60:.1f}min"
                )
    
    # 最终统计
    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"处理完成！")
    logger.info(f"成功: {success_count}, 失败: {fail_count}")
    logger.info(f"总耗时: {total_time/60:.1f} 分钟")
    logger.info(f"平均每篇: {total_time/(success_count+fail_count):.1f} 秒")


if __name__ == "__main__":
    # 设置 multiprocessing 启动方式
    mp.set_start_method('spawn', force=True)
    main()
