from typing import Any, List, Optional
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

def chunk_text_by_tokens(
    tokenizer: PreTrainedTokenizer,
    content: str,
    max_token_size: int = 1024,
    overlap_token_size: int = 64,
    split_by_character: Optional[str] = None,
    split_by_character_only: bool = False,
) -> List[dict[str, Any]]:
    """
    一个灵活的文本分块函数，可以根据语义字符和token数量进行层级化切分。

    该函数的设计目标是：
    1. 优先按照指定的语义字符（如换行符）进行分割，以保持段落的完整性。
    2. 对于分割后依然过长的文本块，再使用基于token的滑动窗口方法进行二次切分。
    3. 提供一个纯粹的、只按字符分割的模式，用于特殊场景。

    参数:
        tokenizer (PreTrainedTokenizer): 用于编码文本和计算token数量的分词器实例。
        content (str): 需要被切分的原始文本内容。
        max_token_size (int): 每个文本块的最大token数量。
        overlap_token_size (int): 在使用滑动窗口切分时，两个相邻块之间的重叠token数量。
        split_by_character (Optional[str]): 用于第一轮语义分割的字符，例如 "\\n\\n"。
                                             如果为 None，则直接使用滑动窗口。默认为 None。
        split_by_character_only (bool): 如果为 True，则仅按指定字符分割，不对超长块做进一步处理。
                                        默认为 False。

    返回:
        List[dict[str, Any]]: 一个包含多个块信息的列表。每个块是一个字典，包含:
                              - "tokens": 该块的token数量。
                              - "content": 该块的文本内容。
                              - "chunk_order_index": 该块在文档中的原始顺序索引。
    """
    if not content:
        return []

    tokens = tokenizer.encode(content, add_special_tokens=False)
    results: List[dict[str, Any]] = []

    if split_by_character:
        raw_chunks = content.split(split_by_character)
        processed_chunks = []

        if split_by_character_only:
            # 仅按字符分割模式
            for chunk in raw_chunks:
                if not chunk.strip(): continue
                _tokens = tokenizer.encode(chunk, add_special_tokens=False)
                processed_chunks.append((len(_tokens), chunk))
        else:
            # 混合模式：先按字符分割，再对超长块进行滑动窗口切分
            for chunk in raw_chunks:
                if not chunk.strip(): continue
                _tokens = tokenizer.encode(chunk, add_special_tokens=False)
                if len(_tokens) > max_token_size:
                    # 如果块太长，则使用滑动窗口进行切分
                    for start in range(0, len(_tokens), max_token_size - overlap_token_size):
                        chunk_tokens = _tokens[start : start + max_token_size]
                        chunk_content = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                        processed_chunks.append((len(chunk_tokens), chunk_content))
                else:
                    processed_chunks.append((len(_tokens), chunk))
        
        for index, (_len, chunk_content) in enumerate(processed_chunks):
            results.append({
                "tokens": _len,
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            })
    else:
        # 纯滑动窗口模式
        for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
            chunk_tokens = tokens[start : start + max_token_size]
            chunk_content = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            results.append({
                "tokens": len(chunk_tokens),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            })
            
    logger.info(f"成功将文本切分为 {len(results)} 个块。")

    # --- 新增逻辑：合并过小的尾部块 ---
    MIN_CHUNK_TOKENS = 300
    if len(results) > 1:
        last_chunk = results[-1]
        if last_chunk["tokens"] < MIN_CHUNK_TOKENS:
            logger.info(
                f"最后一个文本块 (索引: {last_chunk['chunk_order_index']}) 的 token 数量 "
                f"({last_chunk['tokens']}) 小于阈值 {MIN_CHUNK_TOKENS}，将进行合并。"
            )
            
            # 将最后一个块的内容合并到倒数第二个块
            second_last_chunk = results[-2]
            second_last_chunk["content"] += "\\n\\n" + last_chunk["content"]
            
            # 重新计算合并后块的 token 信息
            # 注意: 这里我们只更新 token 数量，不再回溯更新原始的 token 索引，
            # 因为 chunk_order_index 已经能保证其相对位置。
            merged_tokens = tokenizer.encode(second_last_chunk["content"], add_special_tokens=False)
            second_last_chunk["tokens"] = len(merged_tokens)

            # 移除最后一个块
            results.pop()
            logger.info(f"合并完成。现在共有 {len(results)} 个块。")

    return results

def normalize_entity_name(name: str) -> str:
    """
    对实体名称进行规范化处理，以确保数据一致性。
    流程:
    1. 转换为小写。
    2. 去除大部分标点符号，但保留对名称有意义的连字符和空格。
    3. 将多个连续的空格合并为单个空格。
    4. 去除首尾的空格。
    """
    import re
    if not name:
        return ""
    name = name.lower()
    # 保留字母、数字、空格和连字符
    name = re.sub(r'[^a-z0-9\s-]', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def load_tokenizer() -> PreTrainedTokenizer:
    """
    从配置中指定的本地路径加载并返回一个 PreTrainedTokenizer。
    """
    from transformers import AutoTokenizer
    from bridgerag.config import settings

    model_path = settings.local_embedding_model_path
    logger.info(f"正在从本地路径加载分词器: {model_path}")
    
    try:
        # trust_remote_code=True 对于某些模型是必需的
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("分词器加载成功。")
        return tokenizer
    except Exception as e:
        logger.error(f"从路径 '{model_path}' 加载分词器失败: {e}")
        raise
