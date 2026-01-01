import os
import requests
import json

# --- 配置 ---
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
# 从环境变量读取API密钥，如果不存在，则使用您提供的默认值
# 推荐的做法是将密钥存储在环境变量中，而不是直接硬编码
API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-kevnxgxmsbrzhvqihierepcvnaucpvlxuktfchotrojchuyg")

def test_siliconflow_api():
    """
    发送一个测试请求到硅基流动API，并打印响应。
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "Qwen/Qwen2-7B-Instruct",  # 我们使用一个常用的模型进行测试
        "messages": [
            {
                "role": "user",
                "content": "请用一句话介绍一下硅基流动是什么。"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    print("--- 正在向硅基流动API发送请求 ---")
    print(f"URL: {API_URL}")
    print(f"Headers: {{'Authorization': 'Bearer sk-...'}}") # 不打印完整密钥
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print("------------------------------------")

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        print(f"\n--- API响应 ---")
        print(f"状态码: {response.status_code}")
        
        # 尝试以JSON格式解析并打印响应内容
        try:
            response_json = response.json()
            print("响应内容 (JSON):")
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("响应内容 (Raw Text):")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"\n--- 请求失败 ---")
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_siliconflow_api()
