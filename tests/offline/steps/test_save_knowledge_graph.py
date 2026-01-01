import unittest
from unittest.mock import MagicMock, patch, ANY
import numpy as np

# 模拟 save_knowledge_graph 模块的依赖项
@patch('bridgerag.offline.steps.save_knowledge_graph.graph_ops', new_callable=MagicMock)
@patch('bridgerag.offline.steps.save_knowledge_graph.vector_ops', new_callable=MagicMock)
@patch('bridgerag.offline.steps.save_knowledge_graph.object_storage_ops', new_callable=MagicMock)
class TestSaveKnowledgeGraph(unittest.TestCase):

    def setUp(self):
        """在每个测试用例运行前准备模拟数据和客户端"""
        self.mock_embedding_client = MagicMock()
        self.mock_embedding_client.get_embeddings.return_value = [np.random.rand(128).tolist() for _ in range(2)]

        self.doc_id = "test_doc_001"
        
        # 准备模拟的 Settings 对象
        self.mock_config = MagicMock()
        self.mock_config.minio_bucket_name = "test-bucket"
        self.mock_config.chunk_collection_name = "test_chunks"
        self.mock_config.entity_collection_name = "test_entities"
        self.mock_config.summary_collection_name = "test_summaries"
        
        self.partition_data = {
            "chunks": [
                {'chunk_id': 'test_doc_001_0', 'doc_id': self.doc_id, 'content': 'Apple is a fruit.'},
            ],
            "final_entities": [
                {'entity_id': 'test_doc_001_apple', 'name': 'apple', 'is_named_entity': True, 'summary': 'A sweet fruit.', 'doc_id': self.doc_id, 'type': 'named_entity', 'source_chunk_ids': ['c1']},
            ],
            "final_relations": [
                {'source_entity_id': 'test_doc_001_apple', 'target_entity_id': 'test_doc_001_banana', 
                 'description': 'is different from', 'strength': 1.0, 'keywords': []}
            ],
            "document_summary": "This document is about fruits.",
            "same_as_links": []
        }
        
        self.mock_neo4j_driver = MagicMock()
        self.mock_minio_client = MagicMock()
        self.milvus_alias = "default"

    def test_save_partition_success(self, mock_obj_ops, mock_vec_ops, mock_graph_ops):
        """测试 save_partition 函数能否在成功执行时正确调用所有依赖项"""
        from bridgerag.offline.steps.save_knowledge_graph import save_partition

        # --- 执行 ---
        save_partition(
            partition_data=self.partition_data,
            doc_id=self.doc_id,
            config=self.mock_config,
            neo4j_driver=self.mock_neo4j_driver,
            object_storage_conn=self.mock_minio_client,
            embedding_client=self.mock_embedding_client,
            milvus_alias=self.milvus_alias,
            force_rewrite=False
        )

        # --- 验证 ---
        self.assertEqual(self.mock_embedding_client.get_embeddings.call_count, 3)
        mock_graph_ops.upsert_graph_structure.assert_called_once()
        mock_graph_ops.upsert_relations.assert_called_once()
        self.assertEqual(mock_vec_ops.upsert_vectors.call_count, 3)
        mock_obj_ops.upload_text_as_object.assert_called_once()
        mock_vec_ops.delete_vectors_by_filter.assert_not_called()

    def test_delete_partition_data_success(self, mock_obj_ops, mock_vec_ops, mock_graph_ops):
        """测试 _delete_partition_data 函数能否正确调用所有删除操作"""
        from bridgerag.offline.steps.save_knowledge_graph import _delete_partition_data
        
        # 模拟 query 返回找到数据
        mock_vec_ops.query_by_filter.return_value = [{'some_id': 1}]
        mock_vec_ops.get_pk_field.return_value = 'summary_id'

        _delete_partition_data(
            doc_id=self.doc_id,
            config=self.mock_config,
            neo4j_driver=self.mock_neo4j_driver,
            object_storage_conn=self.mock_minio_client,
            milvus_alias=self.milvus_alias,
        )

        # --- 验证 ---
        mock_graph_ops.delete_document_graph.assert_called_once()
        
        # Milvus: 验证 query 被调用了三次来检查数据
        self.assertEqual(mock_vec_ops.query_by_filter.call_count, 3)
        # 验证 delete 被调用
        self.assertEqual(mock_vec_ops.delete_vectors_by_filter.call_count, 2)
        mock_vec_ops.delete_vectors_by_pks.assert_called_once()
        
        mock_obj_ops.remove_objects.assert_called_once()


if __name__ == '__main__':
    unittest.main()
