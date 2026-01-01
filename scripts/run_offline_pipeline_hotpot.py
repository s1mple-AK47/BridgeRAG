"""
HotpotQA 数据集的离线处理流水线。

读取 hotpot_docs.jsonl 文件，格式为: {"id": "<title>", "text": "<content>"}
"""

import json
import logging
from pathlib import Path
import sys

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.offline.pipeline import trigger_batch_processing
from bridgerag.utils.logging_config import setup_logging
from bridgerag.database.object_storage import ObjectStorageConnection
import bridgerag.database.object_storage_ops as storage_ops
from bridgerag.config import settings

# 配置日志记录器
setup_logging()
logger = logging.getLogger(__name__)


def get_processed_doc_ids_from_minio() -> set:
    """从 MinIO 中获取所有已处理并存档的文档ID。"""
    logger.info("正在连接到 MinIO 以检查已处理的文档...")
    processed_ids = set()
    try:
        minio_conn = ObjectStorageConnection(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key
        )
        bucket_name = settings.minio_bucket_name
        
        if not minio_conn.client.bucket_exists(bucket_name):
            logger.warning(f"MinIO 存储桶 '{bucket_name}' 不存在，假设所有文档都未被处理。")
            return processed_ids

        object_names = storage_ops.list_objects(minio_conn.client, bucket_name)
        processed_ids = {Path(obj_name).stem for obj_name in object_names}
        
        logger.info(f"从 MinIO 中发现 {len(processed_ids)} 个已处理的文档。")
        return processed_ids
    except Exception as e:
        logger.error(f"连接到 MinIO 或列出对象时发生错误: {e}", exc_info=True)
        return set()


def get_processed_doc_ids_from_log(log_file_path: Path) -> set:
    """从本地成功日志文件中读取已处理的文档ID。"""
    if not log_file_path.exists():
        return set()
    
    logger.info(f"正在从 {log_file_path} 读取已处理的文档ID...")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            processed_ids = {line.strip() for line in f if line.strip()}
        logger.info(f"从日志文件中发现了 {len(processed_ids)} 个已处理的文档。")
        return processed_ids
    except Exception as e:
        logger.error(f"读取日志文件时发生错误: {e}", exc_info=True)
        return set()


def load_documents_from_jsonl(file_path: Path) -> list[dict]:
    """
    从 JSONL 文件中加载文档。
    
    期望格式: {"id": "<title>", "text": "<content>"}
    """
    documents = []
    seen_ids = set()
    
    logger.info(f"开始从 {file_path} 加载文档...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    doc_id = data.get("id", "").strip()
                    text = data.get("text", "").strip()
                    
                    if not doc_id or not text:
                        logger.warning(f"第 {line_num} 行: 缺少 id 或 text，已跳过")
                        continue
                    
                    if doc_id in seen_ids:
                        continue
                    
                    seen_ids.add(doc_id)
                    documents.append({"id": doc_id, "text": text})
                    
                except json.JSONDecodeError:
                    logger.warning(f"第 {line_num} 行: JSON 解析错误，已跳过")
                    continue
        
        logger.info(f"文件加载完成，共 {len(documents)} 篇唯一文档。")
        return documents
        
    except FileNotFoundError:
        logger.error(f"数据文件未找到: {file_path}")
        return []
    except Exception as e:
        logger.error(f"加载文档时发生错误: {e}", exc_info=True)
        return []


def main():
    """主函数"""
    BATCH_SIZE = 36
    
    # 数据文件路径
    data_file = project_root / "hotpot_docs.jsonl"
    success_log_file = project_root / "logs" / "successful_documents.log"
    
    if not data_file.exists():
        logger.error(f"数据文件不存在: {data_file}")
        logger.error("请先运行 python scripts/convert_hotpot_data.py 转换数据格式")
        return
    
    # 断点续传：获取已处理的文档ID
    processed_in_minio = get_processed_doc_ids_from_minio()
    processed_in_log = get_processed_doc_ids_from_log(success_log_file)
    processed_doc_ids = processed_in_minio.union(processed_in_log)
    logger.info(f"合并后共计 {len(processed_doc_ids)} 个已处理文档将被跳过。")
    
    # 加载所有文档
    all_documents = load_documents_from_jsonl(data_file)
    
    # 过滤已处理的文档
    documents_to_process = [
        doc for doc in all_documents if doc['id'] not in processed_doc_ids
    ]
    
    if not documents_to_process:
        logger.info("所有文档均已处理完毕，无需启动新的流水线。")
        return
    
    logger.info(f"共加载 {len(all_documents)} 篇文档，其中 {len(documents_to_process)} 篇待处理。")
    
    total_docs = len(documents_to_process)
    processed_count = 0
    
    for i in range(0, total_docs, BATCH_SIZE):
        batch_docs = documents_to_process[i:i + BATCH_SIZE]
        batch_for_pipeline = [(doc["id"], {"text": doc["text"]}) for doc in batch_docs]
        
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"正在提交批次 {batch_num}，包含 {len(batch_for_pipeline)} 个文档...")
        
        task_group_result = trigger_batch_processing(batch_for_pipeline)
        
        if task_group_result:
            try:
                logger.info(f"批次提交成功，任务组 ID: {task_group_result.id}。等待处理完成...")
                # 增加超时时间到 2 小时
                task_group_result.get(timeout=7200)
                logger.info(f"批次 {batch_num} 已处理完毕。")
            except TimeoutError:
                logger.error(f"批次 {batch_num} 处理超时（2小时）。流水线中止，请检查 Worker 状态。")
                logger.error("提示：超时不代表任务失败，Worker 可能仍在处理。请等待 Worker 完成后再重新运行脚本。")
                break  # 超时时停止，不继续提交新批次
            except Exception as e:
                logger.warning(f"批次 {batch_num} 处理时遇到错误: {e}")
                logger.warning("将继续等待 30 秒后提交下一批次...")
                import time
                time.sleep(30)  # 给 Worker 一些恢复时间
        else:
            logger.error(f"批次 {batch_num} 提交失败，流水线中止。")
            break
        
        processed_count += len(batch_for_pipeline)
        logger.info(f"===> 当前进度: {processed_count}/{total_docs}")
    
    logger.info("所有批次均已提交并处理完毕。")


if __name__ == "__main__":
    main()
