import openai
from openai import OpenAI
import logging
from bridgerag.config import settings
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class LLMClient:
    """
    一个灵活的LLM客户端，可以根据配置连接到不同的API后端
    （例如本地vLLM服务或硅基流动云端API）。
    """
    def __init__(self):
        """
        初始化LLM客户端。
        它会根据全局`settings`自动配置目标API。
        """
        self.api_type = settings.llm_api_type
        logger.info(f"正在初始化LLMClient，API类型: {self.api_type}")

        if self.api_type == "siliconflow":
            if not settings.siliconflow_api_base or not settings.siliconflow_api_key:
                raise ValueError("硅基流动API配置（URL或密钥）缺失。")
            
            self.model_name = settings.siliconflow_generation_model_name
            self.client = OpenAI(
                base_url=settings.siliconflow_api_base,
                api_key=settings.siliconflow_api_key,
            )
            logger.info(f"LLMClient已配置为使用硅基流动API，模型: {self.model_name}")

        elif self.api_type == "vllm":
            if not settings.vllm_host or not settings.vllm_port:
                raise ValueError("vLLM服务配置（主机或端口）缺失。")

            base_url = f"http://{settings.vllm_host}:{settings.vllm_port}/v1"
            self.model_name = settings.vllm_generation_model_name
            self.client = OpenAI(
                base_url=base_url,
                api_key="not-needed"  # vLLM 默认不需要 API 密钥
            )
            # 根据官方文档，我们需要 tokenizer 来正确禁用思考模式
            try:
                # 优先使用指定的本地路径，如果未提供，则尝试从hub下载
                tokenizer_location = settings.vllm_tokenizer_path or self.model_name
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_location)
                logger.info(f"成功从 '{tokenizer_location}' 加载了 Tokenizer。")
            except Exception as e:
                logger.error(f"为模型 '{self.model_name}' 加载 Tokenizer 失败: {e}")
                raise ValueError(f"无法为 vLLM 模型加载 Tokenizer: {self.model_name}")

            logger.info(f"LLMClient已配置为使用vLLM服务，URL: {base_url}, 模型: {self.model_name}")
            
        else:
            raise ValueError(f"不支持的LLM API类型: '{self.api_type}'")

    def generate(self, prompt: str, max_tokens: int = 8192, temperature: float = 0.1) -> str:
        """
        为给定的提示 (prompt) 生成文本补全。
        使用与OpenAI兼容的chat completions接口。

        参数:
            prompt (str): 发送给语言模型的提示。
            max_tokens (int): 要生成的最大 token 数量。
            temperature (float): 采样温度。值越低，输出越确定。

        返回:
            str: 模型生成的文本。
            
        抛出:
            openai.APIError: 如果 API 调用出现问题。
        """
        logger.debug(f"正在向模型 '{self.model_name}' 发送生成请求, 提示: {prompt[:100]}...")
        try:
            # 根据用户找到并验证成功的方案，正确地在 extra_body 中传递禁用思考模式的参数
            if self.api_type == "vllm":
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else: # 保持对 siliconflow 和其他后端的兼容性
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

            response_text = completion.choices[0].message.content.strip()
            logger.debug(f"收到响应: {response_text[:100]}...")
            return response_text
        except openai.APIError as e:
            logger.error(f"生成文本时发生 API 错误: {e}")
            raise
        except Exception as e:
            logger.error(f"发生意外错误: {e}")
            raise 