from typing import Dict, Any, List
import logging
import json
from pydantic import ValidationError
import re

from bridgerag.core.llm_client import LLMClient
from bridgerag.prompts import PROMPTS
from bridgerag.online.schemas import SynthesisDecision
from bridgerag.utils.json_parser import parse_llm_json_output

logger = logging.getLogger(__name__)

class SynthesisService:
    """
    负责在线推理流程的第四阶段：综合与生成。

    该服务接收由 RetrievalService 搜集并整合好的上下文信息（证据），
    并利用 LLM 生成最终的、综合性的答案。
    """

    def __init__(self, llm_client: LLMClient):
        """
        初始化答案生成服务。

        参数:
            llm_client: LLM 客户端。
        """
        self.llm_client = llm_client
        self.synthesis_prompt_template = PROMPTS["synthesis"]
        logger.info("SynthesisService 初始化完成。")

    def _format_synthesis_prompt(self, question: str, history: str, evidence: str) -> str:
        """
        格式化综合推理提示。

        参数:
            question: 当前需要回答的问题。
            history: 到目前为止的对话历史和思考过程。
            evidence: 从数据库中检索到的上下文信息。

        返回:
            格式化后的提示字符串。
        """
        # 修正: 确保 render 时使用的变量名与 prompt 模板中的占位符完全一致
        return self.synthesis_prompt_template.render(
            user_question=question,
            history=history,
            evidence_list=evidence
        )

    def generate_answer(self, question: str, history: str, evidences: List[str]) -> Dict[str, Any]:
        """
        Based on the user's question and the retrieved evidence, generate the final answer.
        This includes deciding whether to answer directly, admit inability to answer, or ask clarifying sub-questions.
        """
        logger.debug("开始生成最终答案或子问题...")

        # Convert evidences to a string for the prompt
        evidence_str = "\n\n".join(evidences)

        # 修正: 确保调用 _format_synthesis_prompt 时，将 evidence_str 作为 'evidence' 参数传入
        prompt = self._format_synthesis_prompt(question=question, history=history, evidence=evidence_str)

        try:
            response_text = self.llm_client.generate(prompt)
            # 修正: 使用我们为LLM输出定制的、更健壮的JSON解析器
            response_data = parse_llm_json_output(response_text)

            if response_data:
                decision = SynthesisDecision(**response_data)
                logger.info(f"LLM决策: {decision.decision}")
                return decision
            else:
                logger.error(f"无法从LLM响应中提取JSON: '{response_text}'")
                # 出错时，默认认为需要更多信息，并以原始问题重试
                return SynthesisDecision(
                    decision="SUB_QUESTION",
                    content=question,
                    summary=f"Error: Failed to extract JSON from LLM response. Response: '{response_text}'"
                )

        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"解析或验证LLM的JSON输出时出错: {e}. Raw response: '{response_text}'")
            return SynthesisDecision(
                decision="SUB_QUESTION",
                content=question,
                summary=f"Error: Failed to validate JSON. Details: {e}"
            )
        except Exception as e:
            logger.error(f"生成答案时发生未知错误: {e}", exc_info=True)
            return SynthesisDecision(
                decision="SUB_QUESTION",
                content=question,
                summary=f"Error: An unexpected error occurred during synthesis. Details: {e}"
            )
