from minio import Minio
from minio.error import S3Error
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ObjectStorageConnection:
    """管理与 MinIO 对象存储的连接。"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        """
        初始化 MinIO 客户端。

        参数:
            endpoint (str): MinIO 服务器的 URL (例如, 'localhost:9000')。
            access_key (str): 用于认证的访问密钥。
            secret_key (str): 用于认证的秘密密钥。
            secure (bool): 是否使用 HTTPS。默认为 False。
        """
        self.endpoint = endpoint
        self.client = self._connect(access_key, secret_key, secure)

    def _connect(self, access_key: str, secret_key: str, secure: bool) -> Minio:
        """创建并返回一个 MinIO 客户端实例，并验证连接。"""
        try:
            logger.info(f"正在为端点 '{self.endpoint}' 初始化 MinIO 客户端。")

            # Minio 客户端期望的是 'host:port' 格式，而不是完整的 URL
            # 我们在这里解析它以确保兼容性
            parsed_endpoint = urlparse(self.endpoint)
            minio_host = parsed_endpoint.netloc or parsed_endpoint.path

            client = Minio(
                minio_host,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )
            # 通过列出存储桶来执行简单的连接验证。
            client.list_buckets()
            logger.info(f"成功连接到 MinIO: '{minio_host}'。")
            return client
        except S3Error as e:
            # 处理例如无效凭证之类的错误
            logger.error(f"MinIO 连接验证期间发生 S3 错误: {e}")
            raise
        except Exception as e:
            # 处理例如网络问题之类的其他错误
            logger.error(f"连接到 MinIO '{self.endpoint}' 失败: {e}")
            raise

    def get_client(self) -> Minio:
        """
        返回激活的 MinIO 客户端实例。
        
        返回:
            Minio: 已初始化的 MinIO 客户端。
        """
        return self.client

    def make_bucket_if_not_exists(self, bucket_name: str) -> bool:
        """
        如果存储桶 (bucket) 不存在，则创建它。

        参数:
            bucket_name (str): 要创建的存储桶的名称。

        返回:
            bool: 如果存储桶是新创建的，返回 True；如果已存在，返回 False。
        
        抛出:
            S3Error: 如果发生任何 API 错误。
        """
        try:
            found = self.client.bucket_exists(bucket_name)
            if not found:
                self.client.make_bucket(bucket_name)
                logger.info(f"存储桶 '{bucket_name}' 已创建。")
                return True
            else:
                logger.info(f"存储桶 '{bucket_name}' 已存在。")
                return False
        except S3Error as e:
            logger.error(f"检查或创建存储桶 '{bucket_name}' 失败: {e}")
            raise 