"""
将 HotpotQA 数据格式转换为 BridgeRAG 项目所需的格式。

输入文件:
- datas/wiki_pages_final.jsonl: {"query_title": "...", "title": "...", "content": "...", "url": "..."}
- datas/hotpot_qa_filtered.jsonl: {"question": "...", "answer": "...", "title": ["...", "..."]}

输出文件:
- hotpot_docs.jsonl: {"id": "<title>", "text": "<content>"}  (用于离线流水线)
- hotpot_questions.jsonl: {"id": "q_<index>", "question": "...", "answer": "...", "ids": ["title1", "title2"]}  (用于在线基准测试)
"""

import json
import logging
import os
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_wiki_pages(input_file: Path, output_file: Path):
    """
    将 wiki_pages_final.jsonl 转换为离线流水线所需的格式。
    
    输入格式: {"query_title": "...", "title": "...", "content": "...", "url": "..."}
    输出格式: {"id": "<title>", "text": "<content>"}
    """
    logger.info(f"开始转换文档文件: {input_file}")
    
    seen_ids = set()
    doc_count = 0
    duplicate_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # 使用 title 作为文档 ID
                doc_id = data.get("title", "").strip()
                content = data.get("content", "").strip()
                
                if not doc_id or not content:
                    logger.warning(f"第 {line_num} 行: 缺少 title 或 content，已跳过")
                    continue
                
                # 去重
                if doc_id in seen_ids:
                    duplicate_count += 1
                    continue
                
                seen_ids.add(doc_id)
                
                # 写入转换后的格式
                output_record = {
                    "id": doc_id,
                    "text": content
                }
                outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                doc_count += 1
                
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行: JSON 解析错误 - {e}")
                continue
    
    logger.info(f"文档转换完成: 共 {doc_count} 篇文档，跳过 {duplicate_count} 个重复项")
    logger.info(f"输出文件: {output_file}")
    return doc_count


def convert_questions(input_file: Path, output_file: Path):
    """
    将 hotpot_qa_filtered.jsonl 转换为在线基准测试所需的格式。
    
    输入格式: {"question": "...", "answer": "...", "title": ["...", "..."]}
    输出格式: {"id": "q_<index>", "question": "...", "answer": "...", "ids": ["title1", "title2"]}
    """
    logger.info(f"开始转换问答文件: {input_file}")
    
    question_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                question = data.get("question", "").strip()
                answer = data.get("answer", "").strip()
                titles = data.get("title", [])
                
                if not question or not answer:
                    logger.warning(f"第 {line_num} 行: 缺少 question 或 answer，已跳过")
                    continue
                
                # 去重 titles 列表，保持顺序
                unique_titles = list(dict.fromkeys(titles))
                
                # 写入转换后的格式
                output_record = {
                    "id": f"q_{line_num}",
                    "question": question,
                    "answer": answer,
                    "ids": unique_titles
                }
                outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                question_count += 1
                
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行: JSON 解析错误 - {e}")
                continue
    
    logger.info(f"问答转换完成: 共 {question_count} 个问题")
    logger.info(f"输出文件: {output_file}")
    return question_count


def main():
    """主函数"""
    # 输入文件路径
    wiki_input = project_root / "datas" / "wiki_pages_final.jsonl"
    qa_input = project_root / "datas" / "hotpot_qa_filtered.jsonl"
    
    # 输出文件路径 (放在项目根目录，与现有脚本兼容)
    docs_output = project_root / "hotpot_docs.jsonl"
    questions_output = project_root / "hotpot_questions.jsonl"
    
    # 检查输入文件是否存在
    if not wiki_input.exists():
        logger.error(f"文档文件不存在: {wiki_input}")
        sys.exit(1)
    
    if not qa_input.exists():
        logger.error(f"问答文件不存在: {qa_input}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("开始数据格式转换")
    logger.info("=" * 60)
    
    # 转换文档
    doc_count = convert_wiki_pages(wiki_input, docs_output)
    
    logger.info("-" * 60)
    
    # 转换问答
    question_count = convert_questions(qa_input, questions_output)
    
    logger.info("=" * 60)
    logger.info("转换完成!")
    logger.info(f"  - 文档文件: {docs_output} ({doc_count} 篇)")
    logger.info(f"  - 问答文件: {questions_output} ({question_count} 个问题)")
    logger.info("=" * 60)
    logger.info("")
    logger.info("接下来你可以:")
    logger.info("  1. 运行离线流水线: python scripts/run_offline_pipeline_hotpot.py")
    logger.info("  2. 运行实体链接: python scripts/run_entity_linking.py")
    logger.info("  3. 运行在线查询: python scripts/run_online_query_benchmark_hotpot.py")


if __name__ == "__main__":
    main()
