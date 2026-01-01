import logging
import sys
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from tqdm import tqdm

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import settings
from bridgerag.utils.logging_config import setup_logging
from bridgerag.online.main import OnlineQueryProcessor

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)

# --- 日志级别控制 ---
# 将核心库的日志级别调高，以保持基准测试输出的整洁
logging.getLogger("bridgerag.online").setLevel(logging.WARNING)
logging.getLogger("bridgerag.database").setLevel(logging.WARNING)
logging.getLogger("bridgerag.core").setLevel(logging.WARNING)


def load_questions_from_file(file_path: Path, limit: int) -> List[Dict]:
    """从指定的 jsonl 文件中加载问题数据。"""
    questions_data = []
    logger.info(f"开始从 {file_path} 加载问题数据，上限为 {limit} 个问题...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                data = json.loads(line)
                # 兼容性修改：如果 "id" 字段不存在，则使用行号作为 ID
                if "id" not in data:
                    data["id"] = f"2wiki_q_{i}"
                
                # 直接使用文件中的数据结构
                if all(k in data for k in ["id", "question", "answer", "ids"]):
                    questions_data.append(data)
        logger.info(f"成功加载 {len(questions_data)} 条问答数据。")
        return questions_data
    except FileNotFoundError:
        logger.error(f"错误：数据文件未找到于路径 {file_path}")
        return []
    except Exception as e:
        logger.error(f"加载问答数据时发生未知错误: {e}", exc_info=True)
        return []

def process_single_query(query_processor: OnlineQueryProcessor, question_data: Dict):
    """处理单个查询并返回一个用于保存的完整结果字典。"""
    question_id = question_data["id"]
    question_text = question_data["question"]
    
    logger.info(f"\n--- 开始处理问题 ID: {question_id} ---")
    logger.info(f"问题: '{question_text}'")
    
    start_time = time.time()
    try:
        query_result = query_processor.process_query(question_text)
        
        end_time = time.time()
        duration = end_time - start_time
        
        output_record = {
            "id": question_id,
            "question": question_text,
            "answer": question_data["answer"],
            "ids": question_data["ids"],
            "LLM_answer": query_result.answer,
            "LLM_ids": query_result.main_documents, # main_documents 包含所有相关文档
        }
        
        logger.info(f"--- 问题 ID: {question_id} 处理完成 (耗时: {duration:.2f}s) ---")
        return question_id, True, output_record
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"处理问题 ID: {question_id} 时发生错误 (耗时: {duration:.2f}s): {e}", exc_info=True)
        return question_id, False, str(e)

def get_processed_question_ids(result_file: Path) -> set:
    """从已有的结果文件中读取所有已经处理过的问题ID。"""
    if not result_file.exists():
        return set()
    
    processed_ids = set()
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if "id" in data:
                    processed_ids.add(data["id"])
            except json.JSONDecodeError:
                logger.warning(f"结果文件中发现格式错误的行，已跳过: {line.strip()}")
    logger.info(f"从现有结果文件中恢复了 {len(processed_ids)} 个已处理的问题ID。")
    return processed_ids

def main(output_file_path: str = None):
    """一个用于并发测试和基准测试完整在线查询流程，并将结果保存到文件的脚本。"""
    script_start_time = time.time()
    logger.info("--- 启动在线查询流程基准测试脚本 (2WikiMultiHop) ---")
    QUESTION_LIMIT = 10000
    MAX_WORKERS = 8

    try:
        logger.info("正在初始化所有共享客户端和服务...")
        query_processor = OnlineQueryProcessor()
        logger.info("所有客户端和服务已成功初始化。")
    except Exception as e:
        logger.error(f"共享客户端或服务初始化失败: {e}", exc_info=True)
        return

    # 更改输入文件为 2wikimultihop_150_questions_final.jsonl
    data_file = project_root / "2wikimultihop_150_questions_final.jsonl"
    
    # Use provided output file path or default
    if output_file_path:
        output_file = Path(output_file_path)
    else:
        # 更改默认输出文件
        output_file = project_root / "logs" / "online_query_results_2wiki.jsonl"
        
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 断点续传：获取已处理的ID
    processed_ids = get_processed_question_ids(output_file)
    
    # 使用通用的数据加载函数
    all_questions_data = load_questions_from_file(data_file, limit=QUESTION_LIMIT)
    
    # 过滤掉已处理的问题
    questions_to_process = [
        q for q in all_questions_data if q["id"] not in processed_ids
    ]

    if not questions_to_process:
        logger.info(f"所有问题均已在 '{output_file.name}' 中处理完毕，无需启动新的基准测试。")
        return

    success_count = 0
    fail_count = 0
    
    total_processed_before = len(processed_ids)
    
    logger.info(f"\n--- 本次需要处理 {len(questions_to_process)} 个新问题 (之前已处理 {total_processed_before} 个) ---")
    logger.info(f"--- 将使用 {MAX_WORKERS} 个工作线程 ---")
    
    # 使用追加模式 'a' 打开文件，并在循环中实时写入
    with open(output_file, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_question = {
                executor.submit(process_single_query, query_processor, q_data): q_data 
                for q_data in questions_to_process
            }
            
            progress = tqdm(as_completed(future_to_question), total=len(questions_to_process), desc="2WikiMultiHop 在线查询基准测试")
            for future in progress:
                q_id, success, result = future.result()
                if success:
                    success_count += 1
                    # 每处理完一个就立即写入文件
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                else:
                    fail_count += 1

    total_after = total_processed_before + success_count
    logger.info(f"\n--- 本次运行处理完毕 ---")
    logger.info(f"{success_count} 条新结果已追加到: {output_file}")
    logger.info(f"文件中总结果数: {total_after} / {len(all_questions_data)}")
    logger.info(f"本次成功: {success_count}, 本次失败: {fail_count}")
    
    try:
        query_processor.close()
        logger.info("数据库连接已成功关闭。")
    except Exception as e:
        logger.error(f"关闭数据库连接时出错: {e}", exc_info=True)
        
    logger.info("--- 在线查询流程基准测试脚本 (2WikiMultiHop) 执行完毕 ---")

    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    logger.info(f"--- 脚本总执行耗时: {total_duration // 60:.0f} 分 {total_duration % 60:.2f} 秒 ---")


if __name__ == "__main__":
    main()
