import unittest
from unittest.mock import MagicMock, patch
import json

from bridgerag.online.services.routing_service import RoutingService
# 相互依赖，暂时无法直接导入，在测试时使用 mock
# from bridgerag.database.graph_db import GraphDBConnection
# from bridgerag.database.vector_db import VectorDBConnection
# from bridgerag.core.llm_client import LLMClient
# from bridgerag.core.embedding_client import EmbeddingClient

class TestRoutingService(unittest.TestCase):

    def setUp(self):
        """
        在每个测试用例前，初始化 RoutingService 及其模拟依赖。
        """
        self.mock_llm_client = MagicMock()
        self.mock_graph_db = MagicMock()
        self.mock_vector_db = MagicMock()
        self.mock_embedding_client = MagicMock()

        # 模拟 VectorDBConnection 的 close 方法
        self.mock_vector_db.close = MagicMock()

        # 模拟 graph_db 的 session 上下文管理器
        self.mock_graph_session = self.mock_graph_db._driver.session.return_value.__enter__.return_value
        self.mock_graph_session.read_transaction.side_effect = lambda func, *args, **kwargs: func(MagicMock(), *args, **kwargs)

        self.routing_service = RoutingService(
            llm_client=self.mock_llm_client,
            graph_db=self.mock_graph_db,
            vector_db=self.mock_vector_db,
            embedding_client=self.mock_embedding_client
        )

    @patch('bridgerag.online.services.routing_service.parse_llm_json_output')
    @patch('bridgerag.online.services.routing_service.PROMPTS')
    def test_extract_entities_from_question_success(self, mock_prompts, mock_parser):
        """
        测试 _extract_entities_from_question 在 LLM 返回有效 JSON 时能否成功解析。
        """
        # 模拟
        mock_template = MagicMock()
        mock_prompts.__getitem__.return_value = mock_template
        mock_parser.return_value = {"entities": ["Neo4j", "Milvus", "  Test Space  "]}
        
        question = "What is Neo4j and Milvus?"
        entities = self.routing_service._extract_entities_from_question(question)
        
        # 断言
        mock_template.render.assert_called_once_with(user_question=question)
        self.mock_llm_client.generate.assert_called_once()
        self.assertEqual(entities, ["neo4j", "milvus", "test space"]) # 验证是否被正确规范化

    @patch('bridgerag.online.services.routing_service.parse_llm_json_output')
    def test_extract_entities_from_question_json_error(self, mock_parser):
        """
        测试 _extract_entities_from_question 在 LLM 返回无效 JSON 时的容错能力。
        """
        mock_parser.return_value = {} # 模拟解析失败
        entities = self.routing_service._extract_entities_from_question("test question")
        self.assertEqual(entities, [])

    @patch('bridgerag.online.services.routing_service.parse_llm_json_output')
    def test_extract_entities_from_question_no_entities(self, mock_parser):
        """
        测试 _extract_entities_from_question 在 LLM 返回空列表时的行为。
        """
        mock_parser.return_value = {"entities": []}
        entities = self.routing_service._extract_entities_from_question("A question with no entities")
        self.assertEqual(entities, [])

    def test_fuse_rankings_rrf(self):
        """
        测试 _fuse_rankings 方法是否能正确实现 RRF 算法。
        """
        graph_results = [{"doc_id": "doc1", "score": 1.0}, {"doc_id": "doc2", "score": 0.8}]
        vector_results = [{"doc_id": "doc2", "score": 0.9}, {"doc_id": "doc3", "score": 0.85}]

        fused_results = self.routing_service._fuse_rankings([graph_results, vector_results])

        self.assertEqual(len(fused_results), 3)
        self.assertEqual(fused_results[0]["doc_id"], "doc2")
        self.assertEqual(fused_results[1]["doc_id"], "doc1")
        self.assertEqual(fused_results[2]["doc_id"], "doc3")
        self.assertIn("score", fused_results[0])

    def test_fuse_rankings_empty_inputs(self):
        """
        测试 _fuse_rankings 在输入为空列表时的行为。
        """
        self.assertEqual(self.routing_service._fuse_rankings([[], []]), [])
        graph_results = [{"doc_id": "doc1", "score": 1.0}]
        fused = self.routing_service._fuse_rankings([graph_results, []])
        self.assertEqual(fused[0]["doc_id"], "doc1")
        self.assertAlmostEqual(fused[0]["score"], 1 / 61)


    @patch('bridgerag.online.services.routing_service.graph_ops')
    def test_recall_from_graph_success(self, mock_graph_ops):
        """
        测试 _recall_from_graph 成功执行的情况。
        """
        mock_graph_ops.get_documents_by_entities.return_value = [{"doc_id": "doc_graph", "entity_count": 1}]

        results = self.routing_service._recall_from_graph(["test entity"], limit=10)
        
        self.mock_graph_session.read_transaction.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['doc_id'], 'doc_graph')

    @patch('bridgerag.online.services.routing_service.vector_ops')
    def test_recall_from_vector_success(self, mock_vector_ops):
        """
        测试 _recall_from_vector 成功执行的情况。
        """
        self.mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_ops.search_summaries_by_vector.return_value = [{"doc_id": "doc_vector", "score": 0.9}]

        results = self.routing_service._recall_from_vector("some question", top_k=1)
        
        self.mock_embedding_client.get_embeddings.assert_called_once_with(["some question"])
        mock_vector_ops.search_summaries_by_vector.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['doc_id'], 'doc_vector')


    @patch('bridgerag.online.services.routing_service.RoutingService._recall_from_graph')
    @patch('bridgerag.online.services.routing_service.RoutingService._recall_from_vector')
    @patch('bridgerag.online.services.routing_service.RoutingService._extract_entities_from_question')
    def test_route_integration(self, mock_extract, mock_recall_vector, mock_recall_graph):
        """
        测试 route 方法的集成逻辑，模拟子方法的调用。
        """
        # 模拟设置
        mock_extract.return_value = ["test entity"]
        mock_recall_graph.return_value = [{"doc_id": "doc_graph", "score": 1.0}]
        mock_recall_vector.return_value = [{"doc_id": "doc_vector", "score": 0.9}]

        # 执行
        top_docs = self.routing_service.route("some question", top_k=2)

        # 断言
        mock_extract.assert_called_once_with("some question")
        # 当 top_k=2, recall_limit = max(2*2, 20) = 20
        mock_recall_graph.assert_called_once_with(["test entity"], limit=20)
        mock_recall_vector.assert_called_once_with("some question", top_k=20)
        self.assertEqual(len(top_docs), 2)


if __name__ == '__main__':
    unittest.main()
