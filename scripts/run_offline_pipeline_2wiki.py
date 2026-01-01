import json
import logging
from pathlib import Path
import sys
import html

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
        # We only care about the filename without extension (stem)
        processed_ids = {Path(obj_name).stem for obj_name in object_names}
        
        logger.info(f"从 MinIO 中发现 {len(processed_ids)} 个已处理的文档。")
        return processed_ids
    except Exception as e:
        logger.error(f"连接到 MinIO 或列出对象时发生错误: {e}", exc_info=True)
        return set()


def get_processed_doc_ids_from_log(log_file_path: Path) -> set:
    """从本地成功日志文件中读取已处理的文档ID。"""
    if not log_file_path.exists():
        logger.warning(f"成功日志文件 {log_file_path} 不存在，假设没有从日志中恢复的文档。")
        return set()
    
    logger.info(f"正在从 {log_file_path} 读取已处理的文档ID...")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            processed_ids = {line.strip() for line in f if line.strip()}
        logger.info(f"从日志文件中发现了 {len(processed_ids)} 个已处理的文档。")
        return processed_ids
    except Exception as e:
        logger.error(f"读取日志文件 {log_file_path} 时发生错误: {e}", exc_info=True)
        return set()


def load_documents_from_2wiki_docs(file_path: Path) -> list[dict]:
    """
    Loads documents directly from a JSONL file where each line is a document.
    Example line: {"id": "Document Title", "text": "Document content..."}
    """
    documents = []
    logger.info(f"开始从 {file_path} 加载所有文档...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if "id" in data and "text" in data:
                        documents.append({
                            "id": html.unescape(data["id"]),
                            "text": html.unescape(data["text"])
                        })
                    else:
                        logger.warning(f"在第 {i+1} 行发现一个格式不正确的文档，已跳过。")
                except json.JSONDecodeError:
                    logger.warning(f"无法解析第 {i+1} 行的JSON，已跳过。")
        
        logger.info(f"文件加载完成。共加载了 {len(documents)} 篇文档。")
        return documents
    except FileNotFoundError:
        logger.error(f"错误：数据文件未找到于路径 {file_path}")
        return []
    except Exception as e:
        logger.error(f"加载文档时发生未知错误: {e}", exc_info=True)
        return []

def main():
    """
    主函数，用于加载 2WikiMultihop 数据集文档并触发离线处理流水线。
    """
    BATCH_SIZE = 20

    # Correct path for the new dataset
    data_file = project_root / "2wikimultihop_docs_final.jsonl"
    success_log_file = project_root / "logs" / "successful_documents.log"
    
    # --- Resume Logic ---
    processed_in_minio = get_processed_doc_ids_from_minio()
    processed_in_log = get_processed_doc_ids_from_log(success_log_file)
    processed_doc_ids = processed_in_minio.union(processed_in_log)
    logger.info(f"合并 MinIO 和本地日志后，共计 {len(processed_doc_ids)} 个唯一的已处理文档将被跳过。")

    # Load all documents from the new source file
    all_documents = load_documents_from_2wiki_docs(data_file)
    
    documents_to_process = [
        doc for doc in all_documents if doc['id'] not in processed_doc_ids
    ]
    
    if not documents_to_process:
        logger.info("所有文档均已处理完毕，无需启动新的流水线。")
        return

    logger.info(f"共加载 {len(all_documents)} 篇文档，其中 {len(documents_to_process)} 篇是新的，将分批提交处理。")
    
    total_docs = len(documents_to_process)
    processed_count = 0

    for i in range(0, total_docs, BATCH_SIZE):
        batch_docs = documents_to_process[i:i + BATCH_SIZE]
        
        # Convert list of dicts to the list of tuples (id, data) expected by the pipeline
        batch_for_pipeline = [(doc["id"], {"text": doc["text"]}) for doc in batch_docs]

        logger.info(f"正在提交批次 {i // BATCH_SIZE + 1}，包含 {len(batch_for_pipeline)} 个文档...")
        
        task_group_result = trigger_batch_processing(batch_for_pipeline)
        
        if task_group_result:
            try:
                logger.info(f"批次提交成功，任务组 ID: {task_group_result.id}。现在开始等待批次处理完成...")
                # .get() will block until all tasks in this batch are complete.
                # The timeout is set to 1 hour (3600 seconds).
                task_group_result.get(timeout=3600) 
                logger.info(f"批次 {i // BATCH_SIZE + 1} 已处理完毕。")
            except TimeoutError:
                logger.error(f"批次 {i // BATCH_SIZE + 1} 处理超时（超过1小时）。将继续处理下一个批次。")
            except Exception as e:
                logger.warning(f"批次 {i // BATCH_SIZE + 1} 在处理时遇到错误。这可能表示该批次中有部分任务失败。将继续处理下一个批次。详细错误: {e}")
        else:
            logger.error(f"批次 {i // BATCH_SIZE + 1} 提交失败，流水线中止。")
            break
            
        processed_count += len(batch_for_pipeline)
        logger.info(f"===> 当前已提交并等待完成的文档总数: {processed_count}/{total_docs}")
            
    logger.info("所有批次均已提交并处理完毕。")

if __name__ == "__main__":
    main()
