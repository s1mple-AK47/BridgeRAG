import unittest
from unittest.mock import MagicMock, patch, call
import json
import numpy as np

from bridgerag.online.services.retrieval_service import RetrievalService

class TestRetrievalService(unittest.TestCase):

    def setUp(self):
        """
        初始化 RetrievalService 及其模拟依赖。
        """
        self.mock_llm_client = MagicMock()
        self.mock_embedding_client = MagicMock()
        self.mock_graph_db = MagicMock()
        self.mock_vector_db = MagicMock()
        
        # 模拟 graph_db 的 session 上下文管理器
        self.mock_graph_session = self.mock_graph_db._driver.session.return_value.__enter__.return_value
        # 模拟事务的 read_transaction 方法
        self.mock_graph_session.read_transaction.side_effect = lambda func, *args, **kwargs: func(MagicMock(), *args, **kwargs)

        self.retrieval_service = RetrievalService(
            llm_client=self.mock_llm_client,
            embedding_client=self.mock_embedding_client,
            graph_db=self.mock_graph_db,
            vector_db=self.mock_vector_db
        )

    def test_select_entities_with_llm_success(self):
        """
        测试 _select_entities_with_llm 在 LLM 返回有效 JSON 时能否成功解析。
        """
        mock_response = json.dumps({"selected_entities": ["Entity A", "Entity B"]})
        self.mock_llm_client.generate.return_value = mock_response
        
        # 验证返回的实体必须是原始实体列表的子集
        all_entities = ["Entity A", "Entity B", "Entity C"]
        selected_entities = self.retrieval_service._select_entities_with_llm(
            "question", "doc1", "summary", all_entities
        )
        
        self.mock_llm_client.generate.assert_called_once()
        self.assertEqual(selected_entities, ["Entity A", "Entity B"])

    def test_select_entities_with_llm_json_error(self):
        """
        测试 _select_entities_with_llm 在 LLM 返回无效 JSON 时的容错。
        """
        self.mock_llm_client.generate.return_value = "Invalid JSON"
        selected_entities = self.retrieval_service._select_entities_with_llm("question", "doc1", "summary", ["A"])
        self.assertEqual(selected_entities, [])

    @patch('bridgerag.online.services.retrieval_service.graph_ops')
    def test_expand_and_filter_entities_success(self, mock_graph_ops):
        """
        测试 _expand_and_filter_entities 的成功路径。
        """
        # 模拟输入
        question_vector = [1.0, 0.0, 0.0]
        entity_names = ["Entity A"]
        
        # 模拟依赖返回
        self.mock_embedding_client.get_embeddings.return_value = [[0.5, 0.866, 0], [0.866, 0.5, 0]]

        # 更新模拟以匹配新的调用方式
        def mock_read_transaction(func, *args, **kwargs):
            # 模拟 get_linked_entities 的返回
            if func == mock_graph_ops.get_linked_entities:
                return [
                    {'name': 'Entity B', 'summary': 'Summary B'},
                    {'name': 'Entity C', 'summary': 'Summary C'}
                ]
            return None
        self.mock_graph_session.read_transaction.side_effect = mock_read_transaction

        # 执行
        expanded_entities = self.retrieval_service._expand_and_filter_entities(
            question_vector, "doc1", entity_names, top_k=1
        )
        
        # 断言
        self.mock_graph_session.read_transaction.assert_called_once()
        self.mock_embedding_client.get_embeddings.assert_called_once_with(['Summary B', 'Summary C'])
        self.assertEqual(len(expanded_entities), 1)
        self.assertEqual(expanded_entities[0], 'Entity C') # Cosine similarity of [1,0,0] with [0.866, 0.5, 0] is higher


    def test_retrieve_from_assist_doc_success(self):
        """
        测试 _retrieve_from_assist_doc 的成功路径。
        """
        # 模拟
        def mock_read_transaction(func, *args, **kwargs):
             return [
                {'name': 'Entity A', 'summary': 'Summary A'},
                {'name': 'Entity B', 'summary': 'Summary B'}
            ]
        self.mock_graph_session.read_transaction.side_effect = mock_read_transaction
        
        # 执行
        evidence = self.retrieval_service._retrieve_from_assist_doc("doc_assist", ["Entity A", "Entity B"])
        
        # 断言
        self.mock_graph_session.read_transaction.assert_called_once()
        self.assertIn("Source: Document doc_assist", evidence)
        self.assertIn("Entity: Entity A", evidence)
        self.assertIn("Summary: Summary A", evidence)
        self.assertIn("Entity: Entity B", evidence)

    @patch('bridgerag.online.services.retrieval_service.RetrievalService._retrieve_from_assist_doc')
    @patch('bridgerag.online.services.retrieval_service.RetrievalService._retrieve_from_main_doc')
    def test_execute_retrieval_plan_concurrency(self, mock_main_retriever, mock_assist_retriever):
        """
        测试 execute_retrieval_plan 是否正确使用 ThreadPoolExecutor 并行处理。
        """
        # 模拟
        question = "test question"
        plan = {
            "main_documents": ["main_doc1", "main_doc2"],
            "assist_documents": {"assist_doc1": ["Entity A"]}
        }
        mock_question_vector = [0.1, 0.2, 0.3]
        self.mock_embedding_client.get_embeddings.return_value = [mock_question_vector]

        mock_main_retriever.return_value = "Main evidence"
        mock_assist_retriever.return_value = "Assist evidence"
        
        # 执行
        evidence_list = self.retrieval_service.execute_retrieval_plan(question, plan)
        
        # 断言
        # 验证问题只被向量化一次
        self.mock_embedding_client.get_embeddings.assert_called_once_with([question])

        # 验证主文档检索被正确调用
        self.assertEqual(mock_main_retriever.call_count, 2)
        mock_main_retriever.assert_has_calls([
            call(question, mock_question_vector, "main_doc1"),
            call(question, mock_question_vector, "main_doc2")
        ], any_order=True)

        # 验证辅助文档检索被正确调用
        mock_assist_retriever.assert_called_once_with("assist_doc1", ["Entity A"])
        
        self.assertEqual(len(evidence_list), 3)
        self.assertIn("Main evidence", evidence_list)
        self.assertIn("Assist evidence", evidence_list)

if __name__ == '__main__':
    unittest.main()
