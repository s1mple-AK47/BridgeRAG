import unittest
from unittest.mock import MagicMock, patch

from bridgerag.online.main import OnlineQueryProcessor
from bridgerag.online.schemas import SynthesisDecision, QueryResult

class TestOnlineQueryProcessor(unittest.TestCase):

    def setUp(self):
        """
        初始化 OnlineQueryProcessor 并模拟所有底层服务和客户端。
        """
        patchers = {
            'graph_db': patch('bridgerag.online.main.GraphDBConnection'),
            'vector_db': patch('bridgerag.online.main.VectorDBConnection'),
            'llm_client': patch('bridgerag.online.main.LLMClient'),
            'embedding_client': patch('bridgerag.online.main.EmbeddingClient'),
            'routing': patch('bridgerag.online.main.RoutingService'),
            'reasoning': patch('bridgerag.online.main.ReasoningService'),
            'retrieval': patch('bridgerag.online.main.RetrievalService'),
            'synthesis': patch('bridgerag.online.main.SynthesisService'),
        }
        
        self.mocks = {name: patcher.start() for name, patcher in patchers.items()}
        
        self.mock_routing_service = self.mocks['routing'].return_value
        self.mock_reasoning_service = self.mocks['reasoning'].return_value
        self.mock_retrieval_service = self.mocks['retrieval'].return_value
        self.mock_synthesis_service = self.mocks['synthesis'].return_value

        self.processor = OnlineQueryProcessor()

    def tearDown(self):
        """在每个测试后停止所有 patcher。"""
        patch.stopall()

    def test_process_query_single_turn_answer(self):
        """测试单轮查询成功返回答案的端到端流程。"""
        self.mock_routing_service.route.return_value = [{'doc_id': 'doc1'}]
        self.mock_reasoning_service.generate_retrieval_plan.return_value = {"main_documents": ["doc1"]}
        self.mock_retrieval_service.execute_retrieval_plan.return_value = ["Evidence"]
        self.mock_synthesis_service.generate_answer.return_value = SynthesisDecision(
            decision="ANSWER", content="Final answer.", summary="Summary."
        )

        result = self.processor.process_query("What is A?")

        self.mock_synthesis_service.generate_answer.assert_called_once()
        self.assertEqual(result.answer, "Final answer.")
        self.assertIn("User Question: What is A?", result.conversation_history)

    def test_process_query_multi_turn(self):
        """测试多轮迭代查询的流程。"""
        # --- 准备 ---
        self.mock_routing_service.route.return_value = [{'doc_id': 'doc1'}]
        self.mock_reasoning_service.generate_retrieval_plan.return_value = {"main_documents": ["doc1"]}
        self.mock_retrieval_service.execute_retrieval_plan.return_value = ["Evidence"]
        
        self.mock_synthesis_service.generate_answer.side_effect = [
            SynthesisDecision(decision="SUB_QUESTION", content="What about B?", summary="Need info on B."),
            SynthesisDecision(decision="ANSWER", content="Final answer about A and B.", summary="Final summary.")
        ]

        # --- 执行 ---
        result = self.processor.process_query("What is A?", max_turns=3)

        # --- 断言 ---
        self.assertEqual(self.mock_routing_service.route.call_count, 2)
        self.assertEqual(self.mock_synthesis_service.generate_answer.call_count, 2)
        
        # 验证第二次 routing 的输入是第一次 synthesis 返回的子问题
        self.mock_routing_service.route.assert_called_with("What about B?", top_k=5)
        
        # 验证历史记录的正确性
        self.assertIn("User Question: What is A?", result.conversation_history)
        self.assertIn("LLM Summary: Need info on B.", result.conversation_history)
        self.assertIn("User Question: What about B?", result.conversation_history)
        self.assertIn("LLM Summary: Final summary.", result.conversation_history)

        self.assertEqual(result.answer, "Final answer about A and B.")

    def test_process_query_max_turns_exceeded(self):
        """测试达到最大迭代次数时，系统能正确终止并返回当前结果。"""
        self.mock_routing_service.route.return_value = [{'doc_id': 'doc1'}]
        self.mock_reasoning_service.generate_retrieval_plan.return_value = {"main_documents": ["doc1"]}
        self.mock_retrieval_service.execute_retrieval_plan.return_value = ["Evidence"]
        self.mock_synthesis_service.generate_answer.return_value = SynthesisDecision(
            decision="SUB_QUESTION", content="Another sub-question?", summary="Still not enough."
        )

        result = self.processor.process_query("Initial Question", max_turns=3)

        self.assertEqual(self.mock_synthesis_service.generate_answer.call_count, 3)
        self.assertEqual(result.answer, "Another sub-question?")

if __name__ == '__main__':
    unittest.main()
