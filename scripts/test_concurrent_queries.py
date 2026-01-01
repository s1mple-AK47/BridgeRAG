import json
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time
import threading

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.utils.logging_config import configure_logging

# 配置日志
configure_logging()
logger = logging.getLogger(__name__)

# API 服务器的地址
API_URL = "http://localhost:8000/query"

def load_tasks_from_jsonl(file_path: Path) -> list[dict]:
    """从 JSONL 文件中加载所有任务。"""
    tasks = []
    logger.info(f"开始从 {file_path} 加载任务...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 提取 paragraphs 中的 id 列表
                para_ids = [p.get("id") for p in data.get("paragraphs", []) if p.get("id")]
                
                # 确保关键字段存在
                if all(k in data for k in ["id", "question", "answer"]):
                    tasks.append({
                        "id": data["id"],
                        "question": data["question"],
                        "answer": data["answer"],
                        "ids": para_ids,
                    })
    except Exception as e:
        logger.error(f"加载任务时出错: {e}", exc_info=True)
    logger.info(f"成功加载了 {len(tasks)} 个任务。")
    return tasks

def send_query_request(question: str) -> dict:
    """向 API 服务器发送单个查询请求。"""
    payload = {"question": question, "max_turns": 3}
    try:
        # 增加超时以应对可能的长时间处理
        response = requests.post(API_URL, json=payload, timeout=300) # 5分钟超时
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"请求失败 (问题: '{question[:30]}...'): {e}")
        return {"error": str(e)}

def main():
    """主函数，并发地发送所有问题，将结果保存到文件，并支持断点续传。"""
    data_file = project_root / "data.jsonl"
    output_file = project_root / "results.jsonl"
    
    # --- 断点续传逻辑 ---
    processed_ids = set()
    if output_file.exists():
        logger.info(f"发现已存在的结果文件: {output_file}，将进行断点续传。")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"结果文件中有损坏的行，已跳过: {line.strip()}")
        logger.info(f"已找到 {len(processed_ids)} 个已处理的问题。")

    all_tasks = load_tasks_from_jsonl(data_file)
    tasks_to_process = [task for task in all_tasks if task['id'] not in processed_ids]

    if not tasks_to_process:
        logger.info("所有问题均已处理完毕。")
        return

    MAX_WORKERS = 10 
    logger.info(f"准备使用 {MAX_WORKERS} 个并发线程，处理 {len(tasks_to_process)} 个新问题...")
    
    start_time = time.time()
    file_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(send_query_request, task['question']): task for task in tasks_to_process}
        
        for i, future in enumerate(as_completed(future_to_task)):
            task = future_to_task[future]
            try:
                api_result = future.result()
                
                final_result = {
                    "id": task["id"],
                    "question": task["question"],
                    "answer": task["answer"],
                    "ids": task["ids"],
                    "LLM_answer": "Error: " + api_result.get("error") if "error" in api_result else api_result.get('answer', 'N/A'),
                    "LLM_ids": [] if "error" in api_result else api_result.get('main_documents', [])
                }
                
                with file_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(final_result, ensure_ascii=False) + '\n')
                
                logger.info(f"({i+1}/{len(tasks_to_process)}) 问题 ID: {task['id']} 已处理并保存。")

            except Exception as e:
                logger.error(f"处理任务 ID '{task['id']}' 的结果时发生未知异常: {e}", exc_info=True)
    
    end_time = time.time()
    logger.info(f"本次运行处理了 {len(tasks_to_process)} 个查询。")
    logger.info(f"总耗时: {end_time - start_time:.2f} 秒。")
    logger.info(f"所有结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
