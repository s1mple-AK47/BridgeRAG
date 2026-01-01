import glob
import json
import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Tuple, List, Dict, Any
from urllib.parse import urlparse

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bridgerag.config import settings
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database import graph_ops
from bridgerag.offline.steps import build_partitions
from bridgerag.core.llm_client import LLMClient
from bridgerag.core.embedding_client import EmbeddingClient
from bridgerag.utils.logging_config import setup_logging
from transformers import AutoTokenizer
from bridgerag.database.vector_db import VectorDBConnection

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


def get_failed_docs_info(failed_chunks_dir: str) -> Dict[str, str]:
    """扫描 failed_chunks 目录，提取唯一的文档 ID 及其对应的日志文件路径。"""
    logger.info(f"正在扫描失败块日志目录: {failed_chunks_dir}")
    failed_files = glob.glob(os.path.join(failed_chunks_dir, "failed_chunks_*.json"))
    docs_info = {}
    for f_path in failed_files:
        basename = os.path.basename(f_path)
        doc_id = basename.replace("failed_chunks_", "").replace(".json", "")
        if doc_id not in docs_info:
            docs_info[doc_id] = f_path
    return docs_info


def process_and_supplement_chunk(
    chunk: Dict[str, Any], llm_client: LLMClient, neo4j_driver
) -> Tuple[str, bool, str]:
    """处理单个失败的块并将其知识补充到数据库中。"""
    doc_id = chunk["doc_id"]
    chunk_id = chunk["chunk_id"]
    try:
        # 使用新的、带递归能力的函数来处理失败的块
        result = build_partitions.recursively_process_failed_chunk(
            chunk=chunk,
            llm_client=llm_client
        )
        
        if result is None:
             raise ValueError("递归提取最终失败。")
        
        entities, relations = result

        graph_ops.supplement_knowledge(
            driver=neo4j_driver,
            doc_id=doc_id,
            new_entities=entities or [],
            new_relations=relations or [],
        )
        return chunk_id, True, ""
    except Exception as e:
        logger.error(f"处理块 '{chunk_id}' 时失败: {e}", exc_info=True)
        return chunk_id, False, str(e)


def main():
    """运行增量重新处理流水线的主函数。"""
    logger.info("--- 开始增量重新处理失败的文档 ---")

    config = settings
    failed_chunks_dir = config.failed_chunks_dir
    max_workers = config.max_workers

    # 1. 获取所有需要处理的文档及其失败日志
    failed_docs_info = get_failed_docs_info(failed_chunks_dir)
    if not failed_docs_info:
        logger.info("未找到需要重新处理的失败文档。正在退出。")
        return

    logger.info(f"发现 {len(failed_docs_info)} 个文档包含失败的块，准备处理。")
    
    # 初始化所有客户端和连接
    llm_client = LLMClient()
    embedding_client = EmbeddingClient()
    tokenizer = AutoTokenizer.from_pretrained(config.vllm_tokenizer_path or config.vllm_generation_model_name)
    gdb_conn = GraphDBConnection(uri=config.neo4j_uri, user=config.neo4j_user, password=config.neo4j_password)
    neo4j_driver = gdb_conn._driver
    
    # 修复：从URI中解析host和port，并为Milvus建立一个持久的连接
    parsed_milvus_uri = urlparse(f"//{config.milvus_uri}" if "://" not in config.milvus_uri else config.milvus_uri)
    milvus_host = parsed_milvus_uri.hostname
    milvus_port = parsed_milvus_uri.port
    VectorDBConnection(host=milvus_host, port=milvus_port)

    successful_docs = set()
    failed_docs = set()

    try:
        # --- 阶段 1: 并发补充所有失败块的知识 ---
        logger.info("\n--- 阶段 1: 开始并发补充所有失败块的知识 ---")
        all_failed_chunks = []
        for doc_id, f_path in failed_docs_info.items():
            with open(f_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                all_failed_chunks.extend(chunks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(process_and_supplement_chunk, chunk, llm_client, neo4j_driver): chunk
                for chunk in all_failed_chunks
            }
            
            progress = tqdm(as_completed(future_to_chunk), total=len(all_failed_chunks), desc="补充知识块")
            for future in progress:
                chunk = future_to_chunk[future]
                chunk_id, success, error_msg = future.result()
                doc_id = chunk["doc_id"]
                if success:
                    successful_docs.add(doc_id)
                else:
                    failed_docs.add(doc_id)

        # --- 阶段 2: 并发更新受影响文档的总摘要 ---
        logger.info("\n--- 阶段 2: 开始为已补充知识的文档并发更新总摘要 ---")
        docs_to_update = successful_docs - failed_docs
        
        if not docs_to_update:
            logger.warning("没有文档成功完成知识补充，跳过摘要更新阶段。")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_doc = {
                    executor.submit(
                        build_partitions.update_document_summary,
                        doc_id=doc_id,
                        neo4j_driver=neo4j_driver,
                        llm_client=llm_client,
                        embedding_client=embedding_client,
                        tokenizer=tokenizer,
                    ): doc_id
                    for doc_id in docs_to_update
                }
                
                progress_summary = tqdm(as_completed(future_to_doc), total=len(docs_to_update), desc="更新文档摘要")
                for future in progress_summary:
                    doc_id = future_to_doc[future]
                    try:
                        # 检查future是否在执行过程中抛出了异常
                        future.result()
                    except Exception as e:
                        logger.error(f"更新文档 '{doc_id}' 的摘要时失败: {e}", exc_info=True)
                        failed_docs.add(doc_id)
                        successful_docs.discard(doc_id)
    
    finally:
        # 修复：使用正确的属性 `_closed`
        if neo4j_driver and not neo4j_driver._closed:
            gdb_conn.close()

    # --- 阶段 3: 清理日志并打印摘要 ---
    logger.info("\n--- 阶段 3: 清理日志文件并生成报告 ---")
    final_successful_ids = successful_docs - failed_docs
    for doc_id in final_successful_ids:
        f_path = failed_docs_info[doc_id]
        try:
            os.remove(f_path)
            logger.info(f"成功处理并删除日志: {f_path}")
        except OSError as e:
            logger.error(f"删除日志文件 {f_path} 时出错: {e}")

    logger.info("\n--- 增量重新处理摘要 ---")
    logger.info(f"成功修复的文档: {len(final_successful_ids)} -> {list(final_successful_ids)}")
    logger.info(f"仍然失败的文档: {len(failed_docs)} -> {list(failed_docs)}")
    logger.info("\n--- 增量重新处理完成 ---")


if __name__ == "__main__":
    main()
