import unittest
from unittest.mock import MagicMock, patch
import json

from pydantic import ValidationError

from bridgerag.online.services.synthesis_service import SynthesisService
from bridgerag.online.schemas import SynthesisDecision

class TestSynthesisService(unittest.TestCase):

    def setUp(self):
        """
        初始化 SynthesisService 及其模拟依赖。
        """
        self.mock_llm_client = MagicMock()
        self.synthesis_service = SynthesisService(llm_client=self.mock_llm_client)

    @patch('bridgerag.online.services.synthesis_service.parse_llm_json_output')
    def test_generate_answer_success_final_answer(self, mock_parser):
        """
        测试 generate_answer 成功生成最终答案的路径。
        """
        mock_response_dict = {
            "decision": "ANSWER",
            "content": "This is the final answer.",
            "summary": "Final summary of facts."
        }
        mock_parser.return_value = mock_response_dict

        decision = self.synthesis_service.generate_answer("question", ["evidence1"])
        
        self.mock_llm_client.generate.assert_called_once()
        self.assertIsInstance(decision, SynthesisDecision)
        self.assertEqual(decision.decision, "ANSWER")
        self.assertEqual(decision.content, "This is the final answer.")

    @patch('bridgerag.online.services.synthesis_service.parse_llm_json_output')
    def test_generate_answer_json_error(self, mock_parser):
        """
        测试 generate_answer 在解析失败时的容错。
        """
        mock_parser.return_value = {} # 模拟解析失败

        decision = self.synthesis_service.generate_answer("question", ["evidence"])
        
        # 验证是否回退到默认的 SUB_QUESTION 决策
        self.assertEqual(decision.decision, "SUB_QUESTION")
        self.assertEqual(decision.content, "question") # content 应该是原始问题
        self.assertIn("Failed to extract JSON", decision.summary)

    @patch('bridgerag.online.services.synthesis_service.parse_llm_json_output')
    def test_generate_answer_validation_error(self, mock_parser):
        """
        测试 generate_answer 在 JSON 有效但不符合 Pydantic 模型时的容错。
        """
        # 缺少 'decision' 字段
        mock_response_dict = {
            "content": "Some content without a decision.",
            "summary": "A summary."
        }
        mock_parser.return_value = mock_response_dict
        
        # Pydantic 在解包 **mock_response_dict 时会引发 ValidationError
        # 这里我们通过让 anside effect 来模拟这个行为
        with patch('bridgerag.online.services.synthesis_service.SynthesisDecision', side_effect=ValidationError([], MagicMock())):
            decision = self.synthesis_service.generate_answer("question", ["evidence"])
        
            # 验证是否回退
            self.assertEqual(decision.decision, "SUB_QUESTION")
            self.assertEqual(decision.content, "question")
            self.assertIn("Failed to validate JSON", decision.summary)

    def test_format_synthesis_prompt(self):
        """
        测试 _format_synthesis_prompt 方法是否正确地将变量映射到模板。
        """
        question = "Q1"
        history = "History"
        evidence = "Evidence"
        
        with patch('bridgerag.online.services.synthesis_service.PROMPTS') as mock_prompts:
            mock_template = MagicMock()
            mock_prompts.__getitem__.return_value = mock_template
            
            # 由于 _format_synthesis_prompt 的逻辑被移入了 generate_answer
            # 我们在这里直接测试 render 的调用
            self.synthesis_service._format_synthesis_prompt(question, history, evidence)
            
            # 验证 render 方法是否被用正确的变量名调用
            mock_template.render.assert_called_once_with(
                user_question=question, 
                history=history, 
                evidence_list=evidence
            )


if __name__ == '__main__':
    unittest.main()
