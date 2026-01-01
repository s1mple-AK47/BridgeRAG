import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json

from bridgerag.online.services.reasoning_service import ReasoningService

class TestReasoningService(unittest.TestCase):

    def setUp(self):
        """
        初始化 ReasoningService 及其模拟依赖。
        """
        self.mock_llm_client = MagicMock()
        self.mock_graph_db = MagicMock()
        
        # 模拟 graph_db 的 session 上下文管理器
        self.mock_graph_session = self.mock_graph_db._driver.session.return_value.__enter__.return_value
        self.mock_graph_session.read_transaction.side_effect = lambda func, *args, **kwargs: func(MagicMock(), *args, **kwargs)

        self.reasoning_service = ReasoningService(
            llm_client=self.mock_llm_client,
            graph_db=self.mock_graph_db
        )

    @patch('bridgerag.online.services.reasoning_service.graph_ops')
    def test_get_planning_context_success(self, mock_graph_ops):
        """
        测试 _get_planning_context 方法能否成功获取并格式化上下文。
        """
        # 模拟数据库返回
        mock_db_return = [
            {"doc_id": "doc1", "summary": "Summary of doc1.", "entities": ["Entity A", "Entity B"]},
            {"doc_id": "doc2", "summary": "Summary of doc2.", "entities": ["Entity C"]},
        ]
        
        # 更新模拟以匹配新的调用方式
        self.mock_graph_session.read_transaction.return_value = mock_db_return
        
        candidate_docs = [{"doc_id": "doc1"}, {"doc_id": "doc2"}]
        context_list = self.reasoning_service._get_planning_context(candidate_docs)
        
        self.mock_graph_session.read_transaction.assert_called_once()
        self.assertEqual(len(context_list), 2)
        self.assertEqual(context_list[0]['doc_id'], 'doc1')

    @patch('bridgerag.online.services.reasoning_service.parse_llm_json_output')
    def test_generate_retrieval_plan_success(self, mock_parser):
        """
        测试 generate_retrieval_plan 在 LLM 返回有效 JSON 时能否成功解析。
        """
        # 模拟依赖方法的返回
        with patch.object(self.reasoning_service, '_get_planning_context', return_value=[{"doc_id": "doc1"}]):
            # 模拟解析器返回
            mock_plan = {
                "reasoning": "mock reasoning",
                "main_documents": ["doc1"],
                "assist_documents": {"doc2": ["Entity C"]}
            }
            mock_parser.return_value = mock_plan
            
            plan = self.reasoning_service.generate_retrieval_plan("test question", [{"doc_id": "doc1"}])
            
            self.mock_llm_client.generate.assert_called_once()
            self.assertEqual(plan, mock_plan)

    @patch('bridgerag.online.services.reasoning_service.parse_llm_json_output')
    def test_generate_retrieval_plan_json_error(self, mock_parser):
        """
        测试 generate_retrieval_plan 在 LLM 返回无效 JSON 时的容错。
        """
        with patch.object(self.reasoning_service, '_get_planning_context', return_value=[{"doc_id": "doc1"}]):
            mock_parser.return_value = {} # 模拟解析失败
            
            candidate_docs = [{"doc_id": "doc1"}]
            plan = self.reasoning_service.generate_retrieval_plan("test question", candidate_docs)
            
            # 在出错时，应该返回一个包含候选文档的、安全的计划
            self.assertEqual(plan, {"reasoning": "Failed to parse LLM output.", "main_documents": ["doc1"], "assist_documents": {}})

if __name__ == '__main__':
    unittest.main()
