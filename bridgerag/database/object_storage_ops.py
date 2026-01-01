from minio import Minio
from minio.error import S3Error
import logging
from io import BytesIO
from typing import Optional, List

logger = logging.getLogger(__name__)

def upload_text_as_object(
    client: Minio,
    bucket_name: str,
    object_name: str,
    content: str,
    content_type: str = "text/plain; charset=utf-8",
) -> bool:
    """
    将一个字符串内容作为对象上传到 MinIO。

    该函数会将字符串编码为 UTF-8，然后作为字节流上传。
    它会先检查存储桶是否存在，如果不存在则会尝试创建它。

    参数:
        client (Minio): 已初始化的 MinIO 客户端实例。
        bucket_name (str): 目标存储桶的名称。
        object_name (str): 要创建的对象的名称。
        content (str): 要上传的字符串内容。
        content_type (str): 上传对象的 MIME 类型。

    返回:
        bool: 如果上传成功则返回 True，否则返回 False。
    """
    logger.info(f"准备将对象 '{object_name}' 上传到存储桶 '{bucket_name}'...")
    try:
        # 确保存储桶存在
        found = client.bucket_exists(bucket_name)
        if not found:
            logger.warning(f"存储桶 '{bucket_name}' 不存在，将尝试创建它。")
            client.make_bucket(bucket_name)
            logger.info(f"存储桶 '{bucket_name}' 创建成功。")

        # 将字符串内容转换为字节流
        content_bytes = content.encode('utf-8')
        content_stream = BytesIO(content_bytes)
        content_length = len(content_bytes)

        # 上传对象
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=content_stream,
            length=content_length,
            content_type=content_type,
        )
        logger.info(f"成功将对象 '{object_name}' 上传到 MinIO。")
        return True
    except S3Error as e:
        logger.error(f"上传对象 '{object_name}' 到 MinIO 时发生 S3 错误: {e}")
        return False
    except Exception as e:
        logger.error(f"上传对象时发生未知错误: {e}")
        return False

def object_exists(client: Minio, bucket_name: str, object_name: str) -> bool:
    """
    检查指定的对象是否存在于 MinIO 存储桶中。

    参数:
        client (Minio): 已初始化的 MinIO 客户端实例。
        bucket_name (str): 存储桶的名称。
        object_name (str): 要检查的对象的名称。

    返回:
        bool: 如果对象存在则返回 True，否则返回 False。
    """
    try:
        client.stat_object(bucket_name, object_name)
        logger.debug(f"对象 '{object_name}' 在存储桶 '{bucket_name}' 中存在。")
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            logger.debug(f"对象 '{object_name}' 在存储桶 '{bucket_name}' 中不存在。")
            return False
        logger.error(f"检查对象 '{object_name}' 状态时发生 S3 错误: {e}")
        return False
    except Exception as e:
        logger.error(f"检查对象存在性时发生未知错误: {e}")
        return False

def download_object_as_text(client: Minio, bucket_name: str, object_name: str) -> Optional[str]:
    """
    从 MinIO 下载一个对象并将其内容作为 UTF-8 字符串返回。

    参数:
        client (Minio): 已初始化的 MinIO 客户端实例。
        bucket_name (str): 存储桶的名称。
        object_name (str): 要下载的对象的名称。

    返回:
        Optional[str]: 对象的文本内容；如果对象不存在或发生错误，则返回 None。
    """
    logger.info(f"准备从存储桶 '{bucket_name}' 下载对象 '{object_name}'...")
    try:
        response = client.get_object(bucket_name, object_name)
        content_bytes = response.read()
        content_text = content_bytes.decode('utf-8')
        logger.info(f"成功下载并解码对象 '{object_name}'。")
        return content_text
    except S3Error as e:
        if e.code == "NoSuchKey":
            logger.warning(f"尝试下载的对象 '{object_name}' 在存储桶 '{bucket_name}' 中不存在。")
        else:
            logger.error(f"下载对象 '{object_name}' 时发生 S3 错误: {e}")
        return None
    except Exception as e:
        logger.error(f"下载对象时发生未知错误: {e}")
        return None
    finally:
        if 'response' in locals():
            response.close()
            response.release_conn()

def list_objects(client: Minio, bucket_name: str, prefix: str = "") -> List[str]:
    """
    列出 MinIO 存储桶中的对象。

    参数:
        client (Minio): 已初始化的 MinIO 客户端实例。
        bucket_name (str): 存储桶的名称。
        prefix (str, optional): 只列出具有指定前缀的对象。默认为 ""，列出所有对象。

    返回:
        List[str]: 对象名称的列表。
    """
    logger.info(f"准备列出存储桶 '{bucket_name}' 中前缀为 '{prefix}' 的对象...")
    try:
        if not client.bucket_exists(bucket_name):
            logger.warning(f"存储桶 '{bucket_name}' 不存在。")
            return []
            
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
        object_names = [obj.object_name for obj in objects]
        logger.info(f"成功列出 {len(object_names)} 个对象。")
        return object_names
    except S3Error as e:
        logger.error(f"列出对象时发生 S3 错误: {e}")
        return []
    except Exception as e:
        logger.error(f"列出对象时发生未知错误: {e}")
        return []

def delete_object(client: Minio, bucket_name: str, object_name: str) -> bool:
    """
    从 MinIO 存储桶中删除一个对象。

    参数:
        client (Minio): 已初始化的 MinIO 客户端实例。
        bucket_name (str): 存储桶的名称。
        object_name (str): 要删除的对象的名称。

    返回:
        bool: 如果删除成功或对象原本就不存在，则返回 True；如果发生错误，则返回 False。
    """
    logger.info(f"准备从存储桶 '{bucket_name}' 删除对象 '{object_name}'...")
    try:
        client.remove_object(bucket_name, object_name)
        logger.info(f"成功删除对象 '{object_name}'（或该对象本不存在）。")
        return True
    except S3Error as e:
        logger.error(f"删除对象 '{object_name}' 时发生 S3 错误: {e}")
        return False
    except Exception as e:
        logger.error(f"删除对象时发生未知错误: {e}")
        return False 