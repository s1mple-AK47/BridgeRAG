"""
HotpotQA 数据集的在线查询基准测试脚本。

读取 hotpot_questions.jsonl 文件，格式为:
{"id": "q_<index>", "question": "...", "answer": "...", "ids": ["title1", "title2"]}

使用多进程实现并发，每个进程独立初始化查询处理器。
"""

import logging
import sys
import json
from pathlib import Path
import time
import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Dict
from tqdm import tqdm

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import settings
from bridgerag.utils.logging_config import setup_logging

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)

# 日志级别控制
logging.getLogger("bridgerag.online").setLevel(logging.WARNING)
logging.getLogger("bridgerag.database").setLevel(logging.WARNING)
logging.getLogger("bridgerag.core").setLevel(logging.WARNING)

# 全局变量，用于在每个 worker 进程中缓存查询处理器
_worker_processor = None


def init_worker():
    """Worker 进程初始化函数，初始化查询处理器。"""
    global _worker_processor
    
    setup_logging()
    logging.getLogger("bridgerag.online").setLevel(logging.WARNING)
    logging.getLogger("bridgerag.database").setLevel(logging.WARNING)
    logging.getLogger("bridgerag.core").setLevel(logging.WARNING)
    
    from bridgerag.online.main import OnlineQueryProcessor
    _worker_processor = OnlineQueryProcessor()
    logger.info(f"Worker 进程 {mp.current_process().name} 初始化完成")


def load_questions(file_path: Path, limit: int) -> List[Dict]:
    """从 hotpot_questions.jsonl 加载问题数据。"""
    questions_data = []
    logger.info(f"开始从 {file_path} 加载问题数据，上限为 {limit} 个...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                data = json.loads(line)
                if all(k in data for k in ["id", "question", "answer", "ids"]):
                    questions_data.append(data)
        
        logger.info(f"成功加载 {len(questions_data)} 条问答数据。")
        return questions_data
    except FileNotFoundError:
        logger.error(f"数据文件未找到: {file_path}")
        return []
    except Exception as e:
        logger.error(f"加载问答数据时发生错误: {e}", exc_info=True)
        return []


def process_single_query(question_data: Dict) -> tuple:
    """处理单个查询并返回结果。"""
    global _worker_processor
    
    question_id = question_data["id"]
    question_text = question_data["question"]
    
    start_time = time.time()
    try:
        query_result = _worker_processor.process_query(question_text)
        duration = time.time() - start_time
        
        output_record = {
            "id": question_id,
            "question": question_text,
            "answer": question_data["answer"],
            "ids": question_data["ids"],
            "LLM_answer": query_result.answer,
            "LLM_ids": query_result.main_documents,
            "duration": round(duration, 2)
        }
        
        return question_id, True, output_record
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"处理问题 ID: {question_id} 时发生错误 (耗时: {duration:.2f}s): {e}")
        return question_id, False, str(e)


def get_processed_question_ids(result_file: Path) -> set:
    """从已有的结果文件中读取已处理的问题ID。"""
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
                continue
    
    logger.info(f"从结果文件中恢复了 {len(processed_ids)} 个已处理的问题ID。")
    return processed_ids


def main(output_file_path: str = None):
    """主函数"""
    logger.info("--- 启动 HotpotQA 在线查询基准测试 ---")
    
    QUESTION_LIMIT = 10000
    NUM_WORKERS = 16  # 并发进程数
    
    # 文件路径
    data_file = project_root / "hotpot_questions.jsonl"
    
    if not data_file.exists():
        logger.error(f"问答文件不存在: {data_file}")
        logger.error("请先运行 python scripts/convert_hotpot_data.py 转换数据格式")
        return
    
    if output_file_path:
        output_file = Path(output_file_path)
    else:
        output_file = project_root / "logs" / "online_query_results_hotpot.jsonl"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 断点续传
    processed_ids = get_processed_question_ids(output_file)
    
    # 加载问题
    all_questions = load_questions(data_file, limit=QUESTION_LIMIT)
    
    # 过滤已处理的问题
    questions_to_process = [
        q for q in all_questions if q["id"] not in processed_ids
    ]
    
    if not questions_to_process:
        logger.info("所有问题均已处理完毕。")
        return
    
    success_count = 0
    fail_count = 0
    total_processed_before = len(processed_ids)
    
    logger.info(f"本次需要处理 {len(questions_to_process)} 个新问题 (之前已处理 {total_processed_before} 个)")
    logger.info(f"使用 {NUM_WORKERS} 个工作进程")
    
    # 多进程并发处理
    with open(output_file, 'a', encoding='utf-8') as f:
        with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
            results = pool.imap_unordered(process_single_query, questions_to_process)
            
            progress = tqdm(
                results,
                total=len(questions_to_process),
                desc="HotpotQA 在线查询"
            )
            
            for q_id, success, result in progress:
                if success:
                    success_count += 1
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()  # 立即写入磁盘
                else:
                    fail_count += 1
    
    # 汇总
    total_after = total_processed_before + success_count
    logger.info(f"\n--- 本次运行完成 ---")
    logger.info(f"结果已保存到: {output_file}")
    logger.info(f"文件中总结果数: {total_after} / {len(all_questions)}")
    logger.info(f"本次成功: {success_count}, 失败: {fail_count}")
    
    logger.info("--- HotpotQA 在线查询基准测试完成 ---")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
