import unittest
from unittest.mock import patch, MagicMock, call

from bridgerag.offline.pipeline import trigger_batch_processing

# 关键修正：我们必须 patch `chain` 和 `group` 被 *使用* 的地方，
# 也就是 `pipeline` 模块的命名空间。
@patch('bridgerag.offline.pipeline.save_knowledge_graph_task')
@patch('bridgerag.offline.pipeline.process_document_task')
@patch('bridgerag.offline.pipeline.group') 
@patch('bridgerag.offline.pipeline.chain') 
class TestCeleryWorkflow(unittest.TestCase):

    def test_trigger_batch_processing_workflow(
        self, 
        mock_chain, 
        mock_group,
        mock_process_task, 
        mock_save_task
    ):
        """
        测试 trigger_batch_processing 函数是否能构建并提交
        一个正确的 Map-Reduce (group) 工作流。
        """
        # --- 准备 ---
        documents = {
            "doc1": "content1",
            "doc2": "content2"
        }

        # 模拟 .s() 调用返回可识别的签名对象
        p1_sig = MagicMock(name="process_doc1_sig")
        p2_sig = MagicMock(name="process_doc2_sig")
        s_sig = MagicMock(name="save_sig")
        
        mock_process_task.s.side_effect = [p1_sig, p2_sig]
        mock_save_task.s.return_value = s_sig

        # 模拟 chain 和 group 实例的返回
        sub_chain_1 = MagicMock(name="sub_chain_1")
        sub_chain_2 = MagicMock(name="sub_chain_2")
        mock_group_instance = MagicMock(name="group_instance")
        
        mock_chain.side_effect = [sub_chain_1, sub_chain_2]
        mock_group.return_value = mock_group_instance

        # --- 执行 ---
        trigger_batch_processing(documents)

        # --- 验证 ---
        
        # 1. 验证为每个文档的任务签名被创建
        mock_process_task.s.assert_has_calls([
            call(doc_id='doc1', file_content='content1'),
            call(doc_id='doc2', file_content='content2')
        ], any_order=True)
        # 每个子链都需要一个 save 签名
        self.assertEqual(mock_save_task.s.call_count, 2)

        # 2. 验证为每个文档创建了 "process | save" 子链
        mock_chain.assert_has_calls([
            call(p1_sig, s_sig),
            call(p2_sig, s_sig),
        ], any_order=True)

        # 3. 验证所有子链被组合成一个 group
        mock_group.assert_called_once_with([sub_chain_1, sub_chain_2])

        # 4. 验证最终的 group 被异步提交
        mock_group_instance.apply_async.assert_called_once()


if __name__ == '__main__':
    unittest.main()
