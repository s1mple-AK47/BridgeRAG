import json
import logging
from pathlib import Path
import sys
import argparse

# 将项目根目录添加到 sys.path
# 这使得脚本可以从任何位置运行，并且能够正确地找到 bridgerag 模块
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
        
        # 确保存储桶存在，如果不存在，则认为没有任何文档被处理过
        if not minio_conn.client.bucket_exists(bucket_name):
            logger.warning(f"MinIO 存储桶 '{bucket_name}' 不存在，假设所有文档都未被处理。")
            return processed_ids

        # 列出所有已存在的知识分区对象
        object_names = storage_ops.list_objects(minio_conn.client, bucket_name)
        
        # 从对象名 (e.g., "doc-abc.json") 中提取文档ID
        processed_ids = {Path(obj_name).stem for obj_name in object_names}
        
        logger.info(f"从 MinIO 中发现 {len(processed_ids)} 个已处理的文档。")
        return processed_ids
    except Exception as e:
        logger.error(f"连接到 MinIO 或列出对象时发生错误: {e}", exc_info=True)
        # 在出错的情况下，返回一个空集合，以防止意外跳过所有文档
        return set()
    finally:
        # No need to explicitly close the connection, MinIO client handles it.
        pass


def load_documents_from_jsonl(file_path: Path) -> list[dict]:
    """
    从 JSONL 文件中加载、去重并转换文档。

    参数:
        file_path: JSONL 文件的路径。

    返回:
        一个字典列表，每个字典代表一个待处理的文档，
        包含 'id' 和 'text' 键。
    """
    documents = []
    seen_doc_ids = set()  # 用于跟踪已添加的 doc_id，实现去重
    total_paragraphs = 0
    logger.info(f"开始从 {file_path} 加载并去重文档...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 提取 paragraphs 列表
                paragraphs = data.get("paragraphs", [])
                for para in paragraphs:
                    total_paragraphs += 1
                    # 确保每个段落都有 id 和 text
                    if "id" in para and "text" in para:
                        doc_id = para["id"]
                        # 检查 doc_id 是否已经存在
                        if doc_id not in seen_doc_ids:
                            documents.append({
                                "id": doc_id,
                                "text": para["text"]
                            })
                            seen_doc_ids.add(doc_id)
                    else:
                        logger.warning(f"在文件 {file_path} 的某行中发现一个格式不正确的段落，已跳过。")

        unique_docs_count = len(documents)
        # 注意：这里的重复项计数也包含了格式不正确的段落
        duplicate_count = total_paragraphs - unique_docs_count
        logger.info(f"文件加载完成。共扫描 {total_paragraphs} 个段落，去重和过滤后剩余 {unique_docs_count} 篇唯一有效文档。移除了 {duplicate_count} 个重复项或无效项。")
        return documents
    except FileNotFoundError:
        logger.error(f"错误：数据文件未找到于路径 {file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"错误：无法解析文件 {file_path} 中的 JSON。请检查文件格式。")
        return []
    except Exception as e:
        logger.error(f"加载文档时发生未知错误: {e}", exc_info=True)
        return []

def main():
    """
    主函数，用于加载文档并触发离线处理流水线。
    """
    # 定义每个批次处理的文档数量
    BATCH_SIZE = 16

    # 构建数据文件的绝对路径
    data_file = project_root / "data.jsonl"
    
    # 1. (新增) 从 MinIO 获取已处理的文档ID，实现断点续传
    processed_doc_ids = get_processed_doc_ids_from_minio()

    # 2. 从 data.jsonl 加载所有文档
    all_documents = load_documents_from_jsonl(data_file)
    
    # 3. 过滤掉已处理的文档
    documents_to_process = [
        doc for doc in all_documents if doc['id'] not in processed_doc_ids
    ]
    
    if not documents_to_process:
        logger.info("所有文档均已处理完毕，无需启动新的流水线。")
        return

    logger.info(f"共发现 {len(all_documents)} 篇文档，其中 {len(documents_to_process)} 篇是新的，将分批提交处理。")
    
    total_docs = len(documents_to_process)
    processed_count = 0

    for i in range(0, total_docs, BATCH_SIZE):
        batch_docs = documents_to_process[i:i + BATCH_SIZE]
        
        # 将字典列表转换为管道期望的（id, data）元组列表
        batch_for_pipeline = [(doc["id"], {"text": doc["text"]}) for doc in batch_docs]

        logger.info(f"正在提交批次 {i // BATCH_SIZE + 1}，包含 {len(batch_for_pipeline)} 个文档...")
        
        task_group_id = trigger_batch_processing(batch_for_pipeline)
        
        if task_group_id:
            processed_count += len(batch_for_pipeline)
            logger.info(f"批次提交成功。任务组 ID: {task_group_id}")
            logger.info(f"===> 当前总进度: {processed_count}/{total_docs}")
        else:
            logger.error(f"批次 {i // BATCH_SIZE + 1} 提交失败，流水线中止。")
            break
            
    logger.info("所有批次均已提交完毕。")

if __name__ == "__main__":
    main()
