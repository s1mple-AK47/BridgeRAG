from typing import List, Dict, Any
import logging
import json
import re # 导入 re 模块

from bridgerag.core.llm_client import LLMClient
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.prompts.prompts import RETRIEVAL_PLAN_PROMPT
import bridgerag.database.graph_ops as graph_ops
from bridgerag.prompts import PROMPTS
from bridgerag.config import settings

logger = logging.getLogger(__name__)

class ReasoningService:
    """
    负责在线推理流程的“大脑”，编排整个迭代式问答循环。

    核心职责包括：
    1. 根据候选文档生成检索计划。
    2. (后续) 管理迭代循环，分发检索和生成任务。
    3. (后续) 判断何时结束循环或返回最终答案。
    """

    def __init__(self, llm_client: LLMClient, graph_db: GraphDBConnection):
        """
        初始化推理服务。

        参数:
            llm_client (LLMClient): 用于与语言模型交互的客户端实例。
            graph_db (GraphDBConnection): 图数据库的连接实例，用于获取规划所需上下文。
        """
        self.llm_client = llm_client
        self.graph_db = graph_db
        logger.info("ReasoningService 初始化完成。")

    def _get_planning_context(self, candidate_docs: List[Dict]) -> str:
        """
        从候选文档中提取元数据，构建规划阶段的上下文。
        """
        logger.debug(f"为 {len(candidate_docs)} 个候选文档获取规划上下文...")
        doc_ids = [doc["doc_id"] for doc in candidate_docs]
        if not doc_ids:
            return "没有找到候选文档。"

        try:
            # 修正: 使用正确的 session.read_transaction 模式
            with self.graph_db._driver.session() as session:
                docs_metadata = session.read_transaction(
                    graph_ops.get_documents_metadata_batch,
                    doc_ids=doc_ids
                )
            

            context_parts = []
            for meta in docs_metadata:
                if meta and meta.get("summary"): # 检查 summary 是否存在且不为 None
                    doc_id = meta['doc_id']
                    summary = meta['summary']
                    entities = meta['entities']
                    # 过滤掉 None 或空字符串的实体
                    valid_entities = [e for e in entities if e]
                    context_parts.append(
                        f"- Document ID: {doc_id}\n  Summary: {summary}\n  Entities: {valid_entities}"
                    )
                else:
                    logger.warning(f"未能获取到文档 {meta.get('doc_id', 'Unknown')} 的有效元数据。")

            context_str = "\n\n".join(context_parts)
            logger.debug(f"构建的规划上下文: \n{context_str[:500]}...") # 打印部分上下文
            return context_str
        except Exception as e:
            logger.error(f"从图数据库获取规划上下文时出错: {e}", exc_info=True)
            return "从数据库获取文档信息时发生错误。"

    def generate_retrieval_plan(self, question: str, candidate_docs: List[Dict]) -> Dict:
        """
        接收用户问题和候选文档，调用LLM生成一个结构化的检索计划。
        """
        if not candidate_docs:
            logger.warning("没有候选文档，无法生成检索计划。")
            return {"main_documents": [], "assist_documents": {}}

        doc_ids = [doc["doc_id"] for doc in candidate_docs]
        logger.debug(f"为问题 '{question[:30]}...' 和文档 {doc_ids} 生成检索计划...")

        # 步骤 1: 获取规划所需的上下文
        context = self._get_planning_context(candidate_docs)

        # 步骤 2: 填充 Prompt 模板
        prompt_template = PROMPTS["retrieval_plan"]
        prompt = prompt_template.render(user_question=question, candidate_documents_context=context)

        try:
            # 步骤 3: 调用 LLM
            response_text = self.llm_client.generate(prompt)

            # 从Markdown代码块中提取JSON
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                logger.error(f"无法在LLM响应中找到检索计划JSON: {response_text}")
                # 返回一个默认的“安全”计划，而不是让流程崩溃
                return self._fallback_plan(candidate_docs)

            json_string = json_match.group(0)
            plan = json.loads(json_string)
            logger.info("成功生成检索计划。")
            return plan

        except json.JSONDecodeError:
            logger.error(f"无法解析LLM返回的检索计划JSON: {response_text}")
            return self._fallback_plan(candidate_docs)
        except Exception as e:
            logger.error(f"生成检索计划时发生未知错误: {e}", exc_info=True)
            return self._fallback_plan(candidate_docs)

    def _fallback_plan(self, candidate_docs: List[Dict]) -> Dict:
        """
        当LLM无法生成有效JSON时，提供一个默认的检索计划。
        """
        logger.warning("LLM未能生成有效的检索计划，使用默认计划。")
        doc_ids = [doc["doc_id"] for doc in candidate_docs]
        return {"main_documents": doc_ids, "assist_documents": {}}
