from pymilvus import connections, utility
import logging

logger = logging.getLogger(__name__)

class VectorDBConnection:
    """
    管理与 Milvus 向量数据库的连接。
    由于 pymilvus 库使用一个全局的连接管理器，这个类作为一个结构化的包装器，
    用于建立和管理该连接。
    """
    def __init__(self, host: str, port: str, alias: str = "default"):
        """
        初始化并建立到 Milvus 的连接。

        参数:
            host (str): Milvus 服务器的主机名或 IP 地址。
            port (str): Milvus 服务器的端口号。
            alias (str): 连接的别名。
        """
        self.host = host
        self.port = str(port)
        self.alias = alias
        self._connect()

    def _connect(self):
        """建立到 Milvus 的连接。"""
        try:
            logger.info(f"正在以别名 '{self.alias}' 连接到 Milvus: {self.host}:{self.port}...")
            # 如果使用此别名的连接已存在，则先断开。
            if self.alias in connections.list_connections():
                logger.warning(f"别名为 '{self.alias}' 的连接已存在。正在重新连接。")
                connections.disconnect(self.alias)
            
            connections.connect(alias=self.alias, host=self.host, port=self.port)
            logger.info("成功连接到 Milvus。")
        except Exception as e:
            logger.error(f"连接到 Milvus 失败: {e}")
            raise

    def close(self):
        """关闭（断开）与 Milvus 的连接。"""
        self.disconnect()

    def disconnect(self):
        """断开与 Milvus 的连接。"""
        try:
            if self.alias in connections.list_connections():
                connections.disconnect(self.alias)
                logger.info(f"成功从 Milvus 断开连接 (别名: '{self.alias}')。")
        except Exception as e:
            logger.error(f"从 Milvus 断开连接时发生错误: {e}")

    def get_connection_info(self) -> dict:
        """获取当前连接的信息。"""
        try:
            return connections.get_connection_addr(self.alias)
        except Exception as e:
            logger.error(f"为别名 '{self.alias}' 获取 Milvus 连接信息失败: {e}")
            raise

    def list_collections(self) -> list:
        """列出 Milvus 数据库中的所有集合 (collections)。"""
        try:
            return utility.list_collections(using=self.alias)
        except Exception as e:
            logger.error(f"列出 Milvus 集合失败: {e}")
            raise 