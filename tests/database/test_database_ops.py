import unittest
from unittest.mock import MagicMock, patch, call
from io import BytesIO

from bridgerag.database import graph_ops, vector_ops, object_storage_ops

# ==================================
#  Tests for Graph Operations (Neo4j)
# ==================================
class TestGraphOps(unittest.TestCase):

    def setUp(self):
        """为每个测试设置一个模拟的 Neo4j driver 和 session。"""
        self.mock_driver = MagicMock()
        # 模拟 driver.session() 上下文管理器
        self.mock_session = self.mock_driver.session.return_value.__enter__.return_value
        # 模拟事务对象 tx
        self.mock_tx = self.mock_session.begin_transaction.return_value.__enter__.return_value
        # 模拟 execute_read/execute_write 的行为
        self.mock_session.execute_read.side_effect = lambda func, *args, **kwargs: func(self.mock_tx, *args, **kwargs)
        self.mock_session.execute_write.side_effect = lambda func, *args, **kwargs: func(self.mock_tx, *args, **kwargs)
        
    def test_upsert_graph_structure(self):
        """测试 upsert_graph_structure 函数能否生成正确的 Cypher 查询和参数。"""
        graph_ops.upsert_graph_structure(
            self.mock_driver,
            doc_id="doc1",
            doc_summary="Test summary",
            chunks=[{"chunk_id": "c1", "content": "Chunk content"}],
            entities=[{
                "entity_id": "e1", 
                "name": "Entity One", # 修正: entity_name -> name
                "summary": "Desc",     # 修正: description -> summary
                "doc_id": "doc1",
                "is_named_entity": True, 
                "type": "named_entity",
                "source_chunk_ids": ["c1"]
            }]
        )
        
        self.mock_session.run.assert_called_once()
        args, kwargs = self.mock_session.run.call_args
        query = args[0]
        self.assertIn("MERGE (d:Document {doc_id: $doc_id})", query)
        self.assertIn("UNWIND $chunks as chunk_props", query)
        self.assertIn("UNWIND $entities as entity_props", query)
        params = kwargs
        self.assertEqual(params['doc_id'], "doc1")
        self.assertEqual(params['doc_summary'], "Test summary")

    def test_upsert_relations(self):
        """测试 upsert_relations 函数能否生成正确的 Cypher 查询和参数。"""
        relations = [{
            "source_entity_id": "e1", 
            "target_entity_id": "e2", 
            "description": "is related to", 
            "strength": 5.0,
            "keywords": ["test"]
        }]
        
        graph_ops.upsert_relations(self.mock_driver, relations)
        
        self.mock_session.run.assert_called_once()
        args, kwargs = self.mock_session.run.call_args
        query = args[0]
        self.assertIn("UNWIND $batch as relation", query)
        self.assertIn("MATCH (source:Entity {entity_id: relation.source_entity_id})", query)
        self.assertIn("MATCH (target:Entity {entity_id: relation.target_entity_id})", query)
        self.assertIn("MERGE (source)-[r:RELATED_TO]->(target)", query)
        self.assertEqual(kwargs.get('batch'), relations)

    def test_create_same_as_links(self):
        """测试 create_same_as_links 函数。"""
        links = [{"entity_1_id": "e1", "entity_2_id": "e2", "score": 0.88}]
        
        graph_ops.create_same_as_links(self.mock_driver, links)
        
        self.mock_session.run.assert_called_once()
        args, kwargs = self.mock_session.run.call_args
        self.assertIn("UNWIND $links as link", args[0])
        self.assertIn("MERGE (a)-[r:SAME_AS]->(b)", args[0])
        self.assertEqual(kwargs.get('links'), links)

    def test_delete_document_graph(self):
        """测试 delete_document_graph 函数。"""
        graph_ops.delete_document_graph(self.mock_tx, "doc1")
        self.mock_tx.run.assert_called_once()
        query = self.mock_tx.run.call_args.args[0]
        self.assertIn("MATCH (d:Document {doc_id: $doc_id})", query)
        self.assertIn("DETACH DELETE", query)

    def test_get_document_metadata_returns_data(self):
        """测试 get_document_metadata 在找到数据时能否正确返回。"""
        # 模拟 tx.run().single() 返回一个包含 data 方法的模拟记录
        mock_record = MagicMock()
        mock_record.data.return_value = {
            "doc_id": "doc1",
            "summary": "s1",
            "entities": ["e1", None] # 包含 None 以测试清理逻辑
        }
        mock_cursor = MagicMock()
        mock_cursor.single.return_value = mock_record
        self.mock_tx.run.return_value = mock_cursor

        result = graph_ops.get_document_metadata(self.mock_tx, "doc1")
        
        self.mock_tx.run.assert_called_once()
        self.assertEqual(result['summary'], "s1")
        self.assertEqual(result['entities'], ["e1"]) # 确认 None 被移除

    def test_get_document_metadata_returns_none(self):
        """测试 get_document_metadata 在未找到数据时能否正确返回 None。"""
        mock_cursor = MagicMock()
        mock_cursor.single.return_value = None
        self.mock_tx.run.return_value = mock_cursor
        
        result = graph_ops.get_document_metadata(self.mock_tx, "doc1")
        self.assertIsNone(result)

    def test_get_documents_by_entities(self):
        """测试 get_documents_by_entities 函数。"""
        graph_ops.get_documents_by_entities(self.mock_tx, ["e1"], limit=5)
        self.mock_tx.run.assert_called_once()
        query = self.mock_tx.run.call_args.args[0]
        self.assertIn("ORDER BY overlap_score DESC", query)
        # 验证关键字参数
        kwargs = self.mock_tx.run.call_args.kwargs
        self.assertEqual(kwargs['limit'], 5)
        self.assertEqual(kwargs['entity_names'], ["e1"])

# ===================================
#  Tests for Vector Operations (Milvus)
# ===================================
@patch('bridgerag.database.vector_ops.Collection')
@patch('bridgerag.database.vector_ops.utility')
class TestVectorOps(unittest.TestCase):

    def test_create_collection_if_not_exists_creates_new(self, mock_utility, mock_collection_cls):
        """测试当集合不存在时，是否会创建集合和索引。"""
        mock_utility.has_collection.return_value = False
        mock_collection_instance = mock_collection_cls.return_value
        
        vector_ops._create_collection_if_not_exists(
            collection_name="test_coll",
            schema="mock_schema",
            index_params=[{"field_name": "vector", "index_params": {}}]
        )
        
        mock_utility.has_collection.assert_called_with("test_coll", using="default")
        mock_collection_cls.assert_called_with(name="test_coll", schema="mock_schema", using="default", consistency_level="Strong")
        mock_collection_instance.create_index.assert_called_once()

    def test_upsert_vectors(self, mock_utility, mock_collection_cls):
        """测试 upsert_vectors 是否正确调用了 collection.upsert。"""
        mock_collection_instance = mock_collection_cls.return_value
        data = [{"id": 1, "vector": [0.1, 0.2]}]
        
        vector_ops.upsert_vectors("test_coll", data)
        
        mock_collection_cls.assert_called_with("test_coll", using="default")
        mock_collection_instance.upsert.assert_called_with(data)
        mock_collection_instance.flush.assert_called_once()

    def test_delete_vectors_by_filter(self, mock_utility, mock_collection_cls):
        """测试 delete_vectors_by_filter 是否使用了正确的过滤表达式。"""
        mock_collection_instance = mock_collection_cls.return_value
        expr = "doc_id == 'doc1'"
        vector_ops.delete_vectors_by_filter("test_coll", expr)
        mock_collection_instance.delete.assert_called_with(expr)
        mock_collection_instance.flush.assert_called_once()

    def test_search_chunks_by_vector(self, mock_utility, mock_collection_cls):
        """测试 search_chunks_by_vector 是否使用了正确的过滤表达式和参数。"""
        mock_collection_instance = mock_collection_cls.return_value
        vector_ops.search_chunks_by_vector("chunk_coll", [0.1], "doc1", top_k=3)
        
        mock_collection_instance.search.assert_called_once()
        kwargs = mock_collection_instance.search.call_args.kwargs
        self.assertEqual(kwargs['limit'], 3)
        self.assertEqual(kwargs['expr'], "doc_id == 'doc1'")

    def test_get_vectors_by_ids(self, mock_utility, mock_collection_cls):
        """测试 get_vectors_by_ids 是否生成了正确的过滤表达式。"""
        mock_collection_instance = mock_collection_cls.return_value
        vector_ops.get_vectors_by_ids("entity_coll", ["e1", "e2"], "entity_id")
        
        mock_collection_instance.query.assert_called_once()
        expr = mock_collection_instance.query.call_args.kwargs['expr']
        self.assertEqual(expr, "entity_id in ['e1', 'e2']")


# ==========================================
#  Tests for Object Storage Operations (MinIO)
# ==========================================
class TestObjectStorageOps(unittest.TestCase):

    def setUp(self):
        """为每个测试设置一个模拟的 MinIO client。"""
        self.mock_client = MagicMock()

    def test_upload_text_as_object(self):
        """测试 upload_text_as_object 是否正确调用了 put_object。"""
        self.mock_client.bucket_exists.return_value = True
        
        object_storage_ops.upload_text_as_object(
            self.mock_client, "test-bucket", "test.txt", "hello world"
        )
        
        # 验证 put_object 被调用，并且 data 参数是 BytesIO 类型
        self.mock_client.put_object.assert_called_once()
        call_kwargs = self.mock_client.put_object.call_args.kwargs
        self.assertEqual(call_kwargs['bucket_name'], "test-bucket")
        self.assertEqual(call_kwargs['object_name'], "test.txt")
        self.assertIsInstance(call_kwargs['data'], BytesIO)
        self.assertEqual(call_kwargs['length'], 11)

    def test_object_exists(self):
        """测试 object_exists 函数。"""
        self.mock_client.stat_object.return_value = True
        exists = object_storage_ops.object_exists(self.mock_client, "b", "o")
        self.assertTrue(exists)
        self.mock_client.stat_object.assert_called_once_with("b", "o")
        
    def test_download_object_as_text(self):
        """测试 download_object_as_text 是否能正确解码返回内容。"""
        # 模拟 get_object 返回一个包含字节流的响应对象
        mock_response = MagicMock()
        mock_response.read.return_value = "你好世界".encode('utf-8')
        mock_response.__enter__.return_value = mock_response # For 'with' statement
        self.mock_client.get_object.return_value = mock_response
        
        content = object_storage_ops.download_object_as_text(
            self.mock_client, "test-bucket", "test.txt"
        )
        
        self.mock_client.get_object.assert_called_with("test-bucket", "test.txt")
        self.assertEqual(content, "你好世界")
        # 验证连接被释放
        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_called_once()
        
    def test_list_objects(self):
        """测试 list_objects 函数。"""
        mock_obj = MagicMock()
        mock_obj.object_name = "test.txt"
        self.mock_client.list_objects.return_value = [mock_obj]
        
        names = object_storage_ops.list_objects(self.mock_client, "b", prefix="p")
        self.mock_client.list_objects.assert_called_with("b", prefix="p", recursive=True)
        self.assertEqual(names, ["test.txt"])

    def test_delete_object(self):
        """测试 delete_object 是否正确调用了 remove_object。"""
        object_storage_ops.delete_object(self.mock_client, "test-bucket", "test.txt")
        self.mock_client.remove_object.assert_called_with("test-bucket", "test.txt")


if __name__ == '__main__':
    unittest.main()
