import openai
from openai import OpenAI
from typing import List
import logging
from bridgerag.config import settings

logger = logging.getLogger(__name__)

class EmbeddingClient:
    """
    一个灵活的嵌入模型客户端，可以根据配置连接到不同的后端：
    - 'api': 连接到一个兼容OpenAI的远程API服务（如vLLM或硅基流动）。
    - 'local': 在本地CPU或GPU上直接加载并运行SentenceTransformer模型。
    """
    def __init__(self):
        """
        初始化 Embedding 客户端。
        它会根据全局`settings`自动配置目标后端。
        """
        self.api_type = settings.embedding_api_type
        self.local_model = None
        self.client = None
        self.model_name = None

        logger.info(f"正在初始化EmbeddingClient，API类型: {self.api_type}")

        if self.api_type == "local":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("本地嵌入模式需要 `sentence-transformers` 库。请运行 `pip install sentence-transformers`。")
            
            if not settings.local_embedding_model_path:
                raise ValueError("本地嵌入模式需要配置 `local_embedding_model_path`。")
            
            self.model_name = settings.local_embedding_model_path.split('/')[-1]
            try:
                # 显式指定设备为 CPU，以避免与 LLM 争抢 GPU 资源
                self.local_model = SentenceTransformer(
                    self.model_name,
                    device='cpu'
                )
                self.tokenizer = self.local_model.tokenizer
                logger.info(f"EmbeddingClient已配置为本地模式，成功加载 模型: {self.model_name} 到 CPU。")
            except Exception as e:
                logger.error(f"本地加载Embedding模型 '{self.model_name}' 失败: {e}")
                raise

        elif self.api_type == "api":
            target_api = settings.llm_api_type
            logger.info(f"嵌入API模式已启用，将使用 '{target_api}' 的配置。")
            
            if target_api == "siliconflow":
                if not settings.siliconflow_api_base or not settings.siliconflow_api_key:
                    raise ValueError("硅基流动API配置（URL或密钥）缺失。")
                
                self.model_name = settings.siliconflow_embedding_model_name
                self.client = OpenAI(
                    base_url=settings.siliconflow_api_base,
                    api_key=settings.siliconflow_api_key,
                )
                logger.info(f"EmbeddingClient已配置为使用硅基流动API，模型: {self.model_name}")

            elif target_api == "vllm":
                if not settings.vllm_embedding_host or not settings.vllm_embedding_port:
                    raise ValueError("vLLM 嵌入服务配置（主机或端口）缺失。")

                base_url = f"http://{settings.vllm_embedding_host}:{settings.vllm_embedding_port}/v1"
                self.model_name = settings.vllm_embedding_model_name
                self.client = OpenAI(
                    base_url=base_url,
                    api_key="not-needed"
                )
                logger.info(f"EmbeddingClient已配置为使用vLLM服务，URL: {base_url}, 模型: {self.model_name}")
            else:
                raise ValueError(f"不支持的LLM API类型 '{target_api}' 用于嵌入API模式。")
        
        else:
            raise ValueError(f"不支持的嵌入API类型: '{self.api_type}'")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        为一系列文本生成嵌入向量。
        """
        if not texts:
            logger.warning("get_embeddings 被调用时传入了一个空列表。")
            return []
            
        logger.debug(f"正在从模型 '{self.model_name}' 请求 {len(texts)} 条文本的嵌入...")
        
        try:
            if self.api_type == "local":
                # 使用本地模型进行编码
                embeddings = self.local_model.encode(texts, show_progress_bar=False).tolist()
            elif self.api_type == "api":
                # 使用远程API服务
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                embeddings = [item.embedding for item in response.data]
            else:
                # 理论上不会执行到这里，因为__init__已经做了检查
                raise ValueError(f"内部错误：无效的api_type '{self.api_type}'")

            logger.debug(f"成功收到 {len(embeddings)} 个嵌入。")
            return embeddings
        except openai.APIError as e:
            logger.error(f"生成嵌入时发生 API 错误: {e}")
            raise
        except Exception as e:
            logger.error(f"生成嵌入时发生意外错误: {e}")
            raise 