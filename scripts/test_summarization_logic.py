import sys
from pathlib import Path
from unittest.mock import MagicMock

# 将项目根目录添加到 sys.path
# This ensures that we can import from 'bridgerag'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now we can import the necessary modules
try:
    from bridgerag.core.llm_client import LLMClient
    # It's an internal function, but we can import it for testing purposes
    from bridgerag.offline.steps.build_partitions import _summarize_entity_context
    from bridgerag.config import settings
    # The tokenizer is loaded inside LLMClient, so we don't need a separate import
except ImportError as e:
    print(f"导入模块失败，请确保您在项目根目录下运行此脚本: {e}")
    sys.exit(1)

def run_tests():
    """
    一个独立的诊断脚本，专门用于测试 `_summarize_entity_context` 函数的
    token 阈值判断逻辑是否按预期工作。
    """
    print("=" * 60)
    print("=  启动 `_summarize_entity_context` 逻辑测试脚本  =")
    print("=" * 60)

    # 1. 初始化测试环境
    print("\n[步骤 1/4] 初始化 LLMClient 和 Tokenizer...")
    try:
        # 我们需要一个真实的 LLMClient 来获取真实的 Tokenizer
        # 但在测试中，我们会避免实际调用它的 .generate() 方法
        llm_client = LLMClient()
        tokenizer = llm_client.tokenizer
        print("✅  LLMClient 和 Tokenizer 初始化成功。")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("请确保您的 vLLM 服务配置正确，并且模型文件可访问。")
        return

    # --- 测试用例 ---

    # Case 1: 实体信息完全为空
    print("\n" + "-" * 50)
    print("[步骤 2/4] 测试用例 1: 实体信息完全为空")
    print("-" * 50)
    print("▶️  预期行为: 不调用 LLM，直接返回拼接后的空文本。")
    result1 = _summarize_entity_context(
        doc_id="test_doc_1",
        entity_name="empty_entity",
        descriptions=[],
        relations=[],
        llm_client=llm_client,
        tokenizer=tokenizer
    )
    print(f"    实际返回: '{result1}'")
    if result1 == "":
        print("✅  测试通过: 函数返回了一个空字符串，符合预期。")
    else:
        print(f"❌ 测试失败: 函数应返回空字符串，但返回了其他内容。")

    # Case 2: 实体信息很短 (远小于 300 token)
    print("\n" + "-" * 50)
    print("[步骤 3/4] 测试用例 2: 实体信息很短")
    print("-" * 50)
    print("▶️  预期行为: 不调用 LLM，直接返回拼接后的原始文本。")
    short_descriptions = ["Mick Jagger is the lead singer of the Rolling Stones."]
    short_relations = ["Mick Jagger is a member of The Rolling Stones"]
    # 最终修正：使用三重引号定义一个真正的、包含换行符的多行字符串
    expected_text = """Mick Jagger is the lead singer of the Rolling Stones.

关系:
Mick Jagger is a member of The Rolling Stones"""

    result2 = _summarize_entity_context(
        doc_id="test_doc_2",
        entity_name="short_entity",
        descriptions=short_descriptions,
        relations=short_relations,
        llm_client=llm_client,
        tokenizer=tokenizer
    )
    print(f"    实际返回:\\n---\\n{result2}\\n---")
    # 修正：现在直接进行字符串比较
    if result2.strip() == expected_text.strip():
        print("✅  测试通过: 函数返回了拼接后的原始文本。")
    else:
        print(f"❌ 测试失败: 未返回预期的拼接文本。")
        print(f"    预期 (共 {len(expected_text.strip())} 字符):\\n---\\n{expected_text.strip()}\\n---")
        print(f"    实际 (共 {len(result2.strip())} 字符):\\n---\\n{result2.strip()}\\n---")

    # Case 3: 实体信息很长 (大于 300 token)
    print("\n" + "-" * 50)
    print("[步骤 4/4] 测试用例 3: 实体信息很长 (使用 Mock)")
    print("-" * 50)
    print("▶️  预期行为: 触发 token 阈值，调用 LLM 的 .generate() 方法。")
    
    long_text = "This is a very long text designed to exceed the token threshold of 300 tokens. We will repeat this sentence many times to make sure the length is sufficient. " * 20
    long_descriptions = [long_text]
    long_relations = ["This entity is related to another entity with a very long description to ensure we cross the threshold."]

    # 使用 mock 来“假装”调用 LLM，以验证其是否被调用
    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.generate.return_value = "LLM_WAS_CALLED_SUCCESSFULLY"

    result3 = _summarize_entity_context(
        doc_id="test_doc_3",
        entity_name="long_entity",
        descriptions=long_descriptions,
        relations=long_relations,
        llm_client=mock_llm_client,
        tokenizer=tokenizer
    )
    print(f"    函数返回: '{result3}'")

    if mock_llm_client.generate.called:
        print("✅  测试通过: LLMClient.generate() 方法被成功调用。")
    else:
        print("❌ 测试失败: 实体信息很长，但 LLMClient.generate() 方法未被调用。")

    print("\n" + "=" * 60)
    print("=  所有测试已完成。请检查上面的输出。  =")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
