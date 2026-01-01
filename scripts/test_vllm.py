import openai

# 配置 OpenAI 客户端以连接到本地 vLLM 服务
# 确保这里的端口号（8001）与你启动 vLLM 服务时指定的端口号一致
client = openai.OpenAI(
    base_url="http://localhost:8988/v1",
    api_key="not-needed"  # vLLM 默认不需要 API 密钥
)

# 定义你要使用的模型名称
# 这个名称应该与你启动 vLLM 服务时加载的模型名称或路径相匹配
MODEL_NAME = "Qwen3-8B"

def test_chat_completions():
    """测试 /v1/chat/completions 接口"""
    print("--- 正在测试 /v1/chat/completions 接口 ---")
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "你好，请介绍一下你自己。"}
            ],
            temperature=0.1,
            extra_body={"enable_thinking": False}
        )
        print("✅ [成功] Chat Completions API 响应:")
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ [失败] Chat Completions API 调用失败: {e}")

def test_legacy_completions():
    """测试 /v1/completions 接口"""
    print("\n--- 正在测试 /v1/completions 接口 ---")
    try:
        completion = client.completions.create(
            model=MODEL_NAME,
            prompt="你好，请介绍一下你自己。",
            max_tokens=100,
            temperature=0.1,
            extra_body={"enable_thinking": False}
        )
        print("✅ [成功] Legacy Completions API 响应:")
        print(completion.choices[0].text)
    except Exception as e:
        print(f"❌ [失败] Legacy Completions API 调用失败: {e}")

def test_extraction_with_strict_prompt():
    """使用带有严格指令的Prompt测试实体提取，以验证是否能抑制'thinking'模式"""
    print("\n--- 正在测试带有严格JSON指令的提取Prompt ---")
    
    # 1. 定义一个简短的示例文本
    sample_text = "Edward D. Wood Jr. was an American filmmaker, actor, and author. In his thirties, Wood made a number of low-budget films in the science fiction, comedy and horror genres."

    # 2. 构建一个带有强化指令的Prompt
    # (这是 EXTRACTION_PROMPT_TEMPLATE 的简化版本，移除了强化指令以单独测试 extra_body 的效果)
    strict_prompt = f"""
You are a professional Knowledge Graph builder. Your task is to extract structured entity information from the provided text.
You must strictly return the results in the JSON format defined below.

- `entity_name`: (String) The unique name of the entity.
- `entity_description`: (String) A comprehensive description of the entity.
- `is_named_entity`: (Boolean) Set to `true` if the entity is a specific, proper noun.

### Text to Process
```
{sample_text}
```

Output:
"""
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": strict_prompt}
            ],
            temperature=0.0,  # 使用0温度以获得最确定的输出
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        print("✅ [成功] 严格Prompt的响应:")
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ [失败] 严格Prompt调用失败: {e}")


if __name__ == "__main__":
    print("正在诊断本地 vLLM 服务的 OpenAI 兼容接口...")
    test_chat_completions()
    # test_legacy_completions()
    #test_extraction_with_strict_prompt()
    print("\n诊断完成。")
