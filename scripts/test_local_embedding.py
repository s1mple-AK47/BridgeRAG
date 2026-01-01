import logging
import sys
from pathlib import Path
import numpy as np

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.core.embedding_client import EmbeddingClient
from bridgerag.config import settings

def setup_logging():
    """配置日志记录，使其输出到控制台。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )

def main():
    """
    测试 EmbeddingClient 在 'local' (CPU) 模式下是否能正常工作。
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("--- 开始测试 EmbeddingClient (local/CPU模式) ---")

    if settings.embedding_api_type != "local":
        logger.error(
            "测试失败: 请确保在 'configs/config.yaml' 文件中, "
            f"'embedding_api_type' 的值被设置为 'local'。 "
            f"当前值为: '{settings.embedding_api_type}'"
        )
        return

    try:
        # 1. 初始化客户端
        # 在 'local' 模式下，这将在CPU上加载 SentenceTransformer 模型
        logger.info("正在初始化 EmbeddingClient...")
        client = EmbeddingClient()
        logger.info("EmbeddingClient 初始化成功。")

        # 2. 准备示例文本
        sample_texts = [
            "你好，世界！",
            "这是一个在CPU上运行的嵌入模型测试。",
            "BridgeRAG项目旨在构建一个强大的知识图谱增强生成系统。"
        ]
        logger.info(f"准备对以下 {len(sample_texts)} 条文本进行编码:\n" + "\n".join(f"- '{text}'" for text in sample_texts))

        # 3. 生成嵌入
        logger.info("正在调用 get_embeddings 方法...")
        embeddings = client.get_embeddings(sample_texts)
        logger.info("get_embeddings 方法执行完毕。")

        # 4. 验证结果
        if embeddings and isinstance(embeddings, list) and len(embeddings) == len(sample_texts):
            logger.info("成功获取嵌入结果！")
            embedding_array = np.array(embeddings)
            logger.info(f"嵌入结果的维度 (shape): {embedding_array.shape}")
            logger.info(f"第一条文本的嵌入向量 (前5维): {embedding_array[0, :5]}...")
            logger.info("--- 测试成功 ---")
        else:
            logger.error(f"测试失败: 未能获取有效的嵌入结果。收到的结果: {embeddings}")

    except Exception as e:
        logger.error(f"在测试过程中发生严重错误: {e}", exc_info=True)
        logger.info("--- 测试失败 ---")

if __name__ == "__main__":
    main()
