import openai

# 指向本地的 vLLM 服务
# vLLM 默认提供的 API 地址格式是 http://<host>:<port>/v1
# API Key 在默认情况下不是必需的，但 openai 库要求提供，所以我们填入一个"EMPTY"
client = openai.OpenAI(
    base_url="http://localhost:8989/v1",
    api_key="EMPTY"
)

print("--- 正在测试 vLLM API 服务 ---")
print(f"模型名称: Qwen2.5-14B")
print("-" * 30)

try:
    # 发起聊天请求
    completion = client.chat.completions.create(
      model="Qwen2.5-14B",  # 这个名称要和 vLLM 启动时 --served-model-name 参数一致
      messages=[
        {"role": "user", "content": "你好，请介绍一下你自己。"}
      ],
      max_tokens=150,  # 限制最大生成长度
      temperature=0.7,   # 控制生成文本的随机性
      extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
      }
    )

    print("✅ vLLM API 调用成功！")
    print("-" * 30)
    print("模型返回内容:")
    # 打印出模型返回的消息内容
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"❌ vLLM API 调用失败，请检查 vLLM 服务是否仍在运行，以及端口号是否正确。")
    print(f"错误详情: {e}")
