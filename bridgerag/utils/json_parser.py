import json
import re
from typing import Dict, Any, List, Union
import logging
import os
import time

logger = logging.getLogger(__name__)

def robust_json_parser(json_string: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    一个健壮的JSON解析器，能够处理LLM可能返回的、被Markdown包裹或不完整的JSON字符串。

    处理流程:
    1. 尝试直接解析整个字符串。
    2. 如果失败，尝试从 "```json ... ```" markdown代码块中提取并解析。
    3. 如果再次失败，则尝试寻找从第一个 '{' 到最后一个 '}' 的最大JSON对象进行解析。
    4. 如果所有尝试都失败，则记录错误，将原始响应保存到文件，并返回一个默认的空结构。
    """
    # 1. 尝试直接解析
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass  # 继续尝试其他方法

    # 2. 尝试从Markdown代码块中提取
    json_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = json_pattern.search(json_string)
    if match:
        json_string_in_block = match.group(1).strip()
        try:
            return json.loads(json_string_in_block)
        except json.JSONDecodeError:
            json_string = json_string_in_block  # 使用块内内容进行下一步的回退解析

    # 3. 尝试回退解析不完整的JSON
    try:
        # 寻找第一个 '{' 和最后一个 '}' 之间的内容
        start_index = json_string.find('{')
        end_index = json_string.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            potential_json = json_string[start_index : end_index + 1]
            data = json.loads(potential_json)
            logger.warning("成功解析了一个部分或不完整的JSON响应。")
            return data
    except json.JSONDecodeError:
        pass  # 最终的失败处理

    # 4. 保存错误响应以供调试
    log_dir = "logs/failed_json"
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(log_dir, f"failed_json_{timestamp}.log")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_string)
        logger.error(f"无法解析JSON响应。原始响应已保存到: {file_path}")
    except Exception as e:
        logger.error(f"尝试保存失败的JSON响应时发生错误: {e}")

    logger.error(f"即使使用所有回退策略，也无法将LLM响应解析为JSON...")
    # 抛出异常而不是返回默认值，以便上层重试逻辑可以捕获它
    raise json.JSONDecodeError("无法将LLM响应解析为JSON", json_string, 0)


def parse_llm_json_output(
    llm_output: str, 
    expected_type: Union[type[list], type[dict]] = dict
) -> Union[Dict[str, Any], List[Any]]:
    """
    从LLM的文本输出中稳健地解析JSON对象或列表。
    该函数能够处理被Markdown代码块 (```json ... ```) 包裹的JSON。

    参数:
        llm_output: LLM返回的原始字符串。
        expected_type: 期望的返回类型，可以是 dict 或 list。

    返回:
        解析后的字典或列表。如果解析失败，则返回一个空的字典或列表。
    """
    if not llm_output or not llm_output.strip():
        logger.warning("LLM output is empty or contains only whitespace.")
        return [] if expected_type is list else {}

    # 1. 尝试使用正则表达式从Markdown代码块中提取JSON
    match = re.search(r"```(?:json)?\s*\n({.*?}|\[.*?\])\n\s*```", llm_output, re.DOTALL)
    
    json_str = ""
    if match:
        json_str = match.group(1).strip()
    else:
        # 2. 如果没有找到代码块，则采用贪婪策略寻找最后一个 "{" 和 "}" 之间的内容
        try:
            last_brace_pos = llm_output.rindex('{')
            last_bracket_pos = llm_output.rindex('}')
            
            if last_brace_pos < last_bracket_pos:
                json_str = llm_output[last_brace_pos:last_bracket_pos+1]
            else:
                json_str = llm_output # Fallback to the whole string
        except ValueError:
            # 如果找不到 "{" 或 "}"，则直接使用原始字符串
            json_str = llm_output


    # 3. 解析提取出的JSON字符串
    try:
        parsed_json = json.loads(json_str)
        if isinstance(parsed_json, expected_type):
            return parsed_json
        else:
            logger.warning(f"Parsed JSON is of type {type(parsed_json)}, but expected {expected_type}.")
            return [] if expected_type is list else {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from string: '{json_str}'. Error: {e}")
        return [] if expected_type is list else {}
