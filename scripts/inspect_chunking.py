import sys
import os
import logging
import json
from pathlib import Path

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridgerag.utils.text_processing import chunk_text_by_tokens, load_tokenizer
from bridgerag.config import settings

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents_from_jsonl(file_path: Path, question_limit: int = None) -> list[dict]:
    """
    从 JSONL 文件中加载、去重并转换文档。
    此函数逻辑与 run_offline_pipeline_small.py 保持一致。
    """
    documents = []
    seen_doc_ids = set()
    questions_processed = 0
    logger.info(f"开始从 {file_path} 加载文档，最多处理前 {question_limit or 'all'} 个问题...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if question_limit and questions_processed >= question_limit:
                    logger.info(f"已达到 {question_limit} 个问题的加载上限。")
                    break
                
                data = json.loads(line)
                paragraphs = data.get("paragraphs", [])
                for para in paragraphs:
                    if "id" in para and "text" in para:
                        doc_id = para["id"]
                        if doc_id not in seen_doc_ids:
                            documents.append({
                                "id": doc_id,
                                "text": para["text"]
                            })
                            seen_doc_ids.add(doc_id)
                
                questions_processed += 1
        
        logger.info(f"文件加载完成。从 {questions_processed} 个问题中，共加载了 {len(documents)} 篇唯一有效文档。")
        return documents
    except FileNotFoundError:
        logger.error(f"错误：数据文件未找到于路径 {file_path}")
        return []
    except Exception as e:
        logger.error(f"加载文档时发生未知错误: {e}", exc_info=True)
        return []

def inspect_all_documents():
    """
    加载与前N个问题相关的文档，对它们进行分块，并报告找到的最大字符数。
    """
    logger.info("开始对来自 data.jsonl 的多个文档进行分块检查...")
    project_root = Path(__file__).resolve().parent.parent

    # 1. 加载分词器
    try:
        tokenizer = load_tokenizer()
    except Exception as e:
        logger.error(f"加载分词器失败: {e}")
        return

    # 2. 按照与真实场景相同的逻辑加载文档
    QUESTION_LIMIT = 20
    data_file = project_root / "data.jsonl"
    documents_to_inspect = load_documents_from_jsonl(data_file, question_limit=QUESTION_LIMIT)

    if not documents_to_inspect:
        logger.warning("未能加载任何文档进行检查。")
        return

    logger.info(f"成功加载 {len(documents_to_inspect)} 篇文档，现在开始逐一分块并检查...")

    # 3. 遍历所有文档，处理并追踪全局最大块
    overall_max_chars = 0
    max_info = {}

    for doc in documents_to_inspect:
        doc_id = doc.get("id")
        content = doc.get("text", "")

        if not content.strip():
            logger.warning(f"文档 {doc_id} 内容为空，已跳过。")
            continue

        chunk_size = settings.text_chunk_size
        overlap_size = settings.overlap_token_size
        
        chunks = chunk_text_by_tokens(
            tokenizer=tokenizer,
            content=content,
            max_token_size=chunk_size,
            overlap_token_size=overlap_size
        )

        for chunk in chunks:
            char_count = len(chunk.get("content", ""))
            if char_count > overall_max_chars:
                overall_max_chars = char_count
                max_info = {
                    "doc_id": doc_id,
                    "chunk_index": chunk.get("chunk_order_index", "N/A"),
                    "token_count": chunk.get("tokens", 0),
                    "char_count": char_count,
                }

    # 4. 打印最终的综合报告
    logger.info("-" * 60)
    logger.info("分块检查综合报告 (扫描了所有文档)")
    logger.info("-" * 60)
    if max_info:
        logger.info(
            f"在所有文档中，发现的最大文本块位于文档 ID: '{max_info['doc_id']}'"
        )
        logger.info(
            f"  - 块索引: {max_info['chunk_index']}"
        )
        logger.info(
            f"  - Token 数量: {max_info['token_count']}"
        )
        logger.info(
            f"  - 字符数量: {max_info['char_count']} <--- 全局最大值"
        )
    else:
        logger.info("未能处理任何文本块。")
    logger.info("-" * 60)


if __name__ == "__main__":
    inspect_all_documents()
