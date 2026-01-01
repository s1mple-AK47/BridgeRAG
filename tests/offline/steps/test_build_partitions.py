import unittest
from unittest.mock import MagicMock, patch, call
import json
from bridgerag.config import settings
from bridgerag.offline.steps.build_partitions import (
    _create_chunks, 
    _extract_knowledge_from_chunk, 
    _link_and_summarize_entities,
    _generate_document_summary,
    process_document
)
from bridgerag.utils.text_processing import chunk_text_by_tokens

# 由于 PreTrainedTokenizer 依赖 torch 等重型库，我们在这里完全模拟它
class MockTokenizer:
    def __init__(self, model_name):
        pass

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, tokens, **kwargs):
        return "".join([chr(i) for i in tokens])

@patch('bridgerag.utils.text_processing.get_tokenizer', lambda model_name: MockTokenizer(model_name))
class TestBuildPartitions(unittest.TestCase):

    def setUp(self):
        """在每个测试用例运行前设置模拟对象"""
        self.mock_llm_client = MagicMock()
        self.mock_embedding_client = MagicMock()
        self.mock_tokenizer = MockTokenizer("mock-model")
        
        self.chunk_patcher = patch('bridgerag.offline.steps.build_partitions.chunk_text_by_tokens')
        self.mock_chunk_function = self.chunk_patcher.start()
        
    def tearDown(self):
        """在测试用例运行后停止 patcher"""
        self.chunk_patcher.stop()

    def test_create_chunks_basic(self):
        """测试 _create_chunks 函数能否正确处理基本输入"""
        self.mock_chunk_function.return_value = [{'content': 'Test document.', 'chunk_order_index': 0}]
        chunks = _create_chunks("doc1", "Test document.", self.mock_tokenizer)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['chunk_id'], 'doc1_0')
        self.mock_chunk_function.assert_called_once()

    @patch('bridgerag.utils.json_parser.parse_llm_json_output')
    def test_extract_knowledge_from_chunk(self, mock_parser):
        """测试 _extract_knowledge_from_chunk 函数"""
        mock_parser.return_value = {
            "entities": [{"name": "E1", "summary": "S1"}],
            "relationships": [{"source": "E1", "target": "E2"}]
        }
        entities, relations = _extract_knowledge_from_chunk("c1", "content", self.mock_llm_client)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['name'], 'e1') # Normalized
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]['source'], 'e1')

    @patch('bridgerag.offline.steps.build_partitions._get_llm_similarity_score', return_value=(8.5, "reason"))
    def test_link_and_summarize_entities(self, mock_llm_score):
        """测试 _link_and_summarize_entities 的实体链接和摘要逻辑"""
        all_raw_entities = {
            "rag": [
                {'name': 'rag', 'summary': 'Summary A', 'is_named_entity': True, 'source_chunk_ids': ['c1']},
                {'name': 'rag', 'summary': 'Summary B', 'is_named_entity': True, 'source_chunk_ids': ['c2']}
            ],
            "llm": [{'name': 'llm', 'summary': 'Summary C', 'is_named_entity': True, 'source_chunk_ids': ['c1']}]
        }
        final_entities, same_as_links = _link_and_summarize_entities(
            "doc1", all_raw_entities, self.mock_llm_client, self.mock_tokenizer
        )
        self.assertEqual(len(final_entities), 2)
        self.assertEqual(len(same_as_links), 0) # No cross-document links in this test
        rag_entity = next(e for e in final_entities if e['name'] == 'rag')
        self.assertEqual(rag_entity['entity_id'], 'doc1_rag')
        # Check if descriptions are merged for summary
        self.assertIn("Summary A", rag_entity['summary'])
        self.assertIn("Summary B", rag_entity['summary'])
        
    def test_generate_document_summary(self):
        """测试 _generate_document_summary 函数"""
        self.mock_llm_client.generate.return_value = "Final document summary."
        self.mock_embedding_client.get_embeddings.return_value = [[0.1]]
        
        final_entities = [{'name': 'E1', 'summary': 'S1', 'is_named_entity': True}]
        summary, vector = _generate_document_summary("doc1", final_entities, self.mock_llm_client, self.mock_embedding_client, self.mock_tokenizer)
        
        self.assertEqual(summary, "Final document summary.")
        self.assertEqual(vector, [0.1])
        self.mock_llm_client.generate.assert_called_once()
        self.mock_embedding_client.get_embeddings.assert_called_once_with(["Final document summary."])

    @patch('bridgerag.offline.steps.build_partitions._generate_document_summary')
    @patch('bridgerag.offline.steps.build_partitions._link_and_summarize_entities')
    @patch('bridgerag.offline.steps.build_partitions._extract_knowledge_from_chunk')
    @patch('bridgerag.offline.steps.build_partitions._create_chunks')
    def test_process_document_workflow(self, mock_create_chunks, mock_extract, mock_link, mock_gen_summary):
        """测试 process_document 函数的集成工作流"""
        mock_create_chunks.return_value = [{'chunk_id': 'c1', 'content': '...'}]
        mock_extract.return_value = ([{'name': 'e1', 'summary': 's1', 'is_named_entity': True, 'source_chunk_ids': ['c1']}], [])
        mock_link.return_value = ([{'name': 'FinalE1', 'entity_id': 'd1_e1'}], [])
        mock_gen_summary.return_value = ("Final summary.", [0.1])

        result = process_document(
            "d1", "content", self.mock_llm_client, self.mock_embedding_client, self.mock_tokenizer
        )
        
        self.assertIn('final_entities', result)
        self.assertEqual(result['document_summary'], "Final summary.")
        mock_link.assert_called_once()
        mock_gen_summary.assert_called_once()

if __name__ == '__main__':
    unittest.main()
