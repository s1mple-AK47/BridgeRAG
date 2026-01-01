import logging
import os
import sys
from urllib.parse import urlparse

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bridgerag.config import settings
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.vector_db import VectorDBConnection
from bridgerag.core.llm_client import LLMClient
from bridgerag.offline.steps.link_entities import run_entity_linking
from bridgerag.utils.logging_config import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """实体链接脚本的主函数"""
    logger.info("--- 开始执行实体链接脚本 ---")
    
    config = settings
    gdb_conn = None
    
    try:
        # 1. 初始化客户端和数据库连接
        logger.info("正在初始化 LLM 客户端和数据库连接...")
        llm_client = LLMClient()
        gdb_conn = GraphDBConnection(
            uri=config.neo4j_uri, 
            user=config.neo4j_user, 
            password=config.neo4j_password
        )
        neo4j_driver = gdb_conn._driver

        # 初始化 Milvus 连接
        parsed_milvus_uri = urlparse(f"//{config.milvus_uri}" if "://" not in config.milvus_uri else config.milvus_uri)
        milvus_host = parsed_milvus_uri.hostname
        milvus_port = parsed_milvus_uri.port
        VectorDBConnection(host=milvus_host, port=milvus_port)
        
        logger.info("初始化完成。")

        # 定义详细结果的输出文件路径
        details_output_file = os.path.join(project_root, "logs", "entity_linking_details.jsonl")

        # 2. 调用核心实体链接逻辑
        run_entity_linking(
            driver=neo4j_driver,
            llm_client=llm_client,
            milvus_collection_name=config.entity_collection_name,
            max_workers=config.max_workers,
            details_output_path=details_output_file
        )

    except Exception as e:
        logger.error(f"实体链接过程中发生未处理的异常: {e}", exc_info=True)
    finally:
        # 3. 关闭数据库连接
        if gdb_conn:
            gdb_conn.close()
            
    logger.info("--- 实体链接脚本执行完毕 ---")


if __name__ == "__main__":
    main()
