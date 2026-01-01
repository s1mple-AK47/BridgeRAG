import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import Settings
from bridgerag.database.vector_db import VectorDBConnection
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database import vector_ops
from bridgerag.utils.logging_config import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


def initialize_neo4j_indexes():
    """
    连接到 Neo4j 并创建必要的索引以优化查询和写入性能。
    这是一个幂等操作，使用 IF NOT EXISTS 确保不会重复创建。
    """
    logger.info("--- 开始初始化 Neo4j 索引 ---")
    
    settings = Settings()
    graph_db_conn = None
    
    try:
        # 连接到 Neo4j
        logger.info(f"正在连接到 Neo4j at {settings.neo4j_uri}...")
        graph_db_conn = GraphDBConnection(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        logger.info("成功连接到 Neo4j。")
        
        # 定义需要创建的索引
        indexes = [
            ("entity_id_index", "Entity", "entity_id"),
            ("entity_name_index", "Entity", "name"),
            ("chunk_id_index", "Chunk", "chunk_id"),
            ("doc_id_index", "Document", "doc_id"),
        ]
        
        with graph_db_conn._driver.session() as session:
            for index_name, label, property_name in indexes:
                query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON (n.{property_name})"
                logger.info(f"创建索引: {index_name} on {label}.{property_name}")
                session.run(query)
        
        logger.info("--- Neo4j 索引初始化成功 ---")
        
    except Exception as e:
        logger.error("--- Neo4j 索引初始化失败 ---", exc_info=True)
        sys.exit(1)
    finally:
        if graph_db_conn:
            graph_db_conn.close()
            logger.info("Neo4j 连接已关闭。")


def initialize_milvus_collections():
    """
    连接到 Milvus 并确保所有需要的集合都已创建并带有索引。
    这是一个幂等操作，如果集合已存在，它不会重复创建。
    """
    logger.info("--- 开始初始化 Milvus 数据库 ---")
    
    settings = Settings()
    vector_db_conn = None
    
    try:
        # --- 1. 连接到 Milvus ---
        logger.info(f"正在连接到 Milvus at {settings.milvus_uri}...")
        parsed_milvus_uri = urlparse(settings.milvus_uri)
        vector_db_conn = VectorDBConnection(
            host=parsed_milvus_uri.hostname, 
            port=parsed_milvus_uri.port, 
            alias="default"
        )
        logger.info("成功连接到 Milvus。")

        # --- 2. 获取配置参数 ---
        chunk_collection_name = settings.chunk_collection_name
        entity_collection_name = settings.entity_collection_name
        summary_collection_name = settings.summary_collection_name
        embedding_dim = settings.embedding_dim
        
        # 从新的、结构化的配置中加载各自的最大长度
        chunk_len = settings.chunk_max_length
        entity_len = settings.entity_summary_max_length
        summary_len = settings.summary_max_length

        if not all([chunk_collection_name, entity_collection_name, summary_collection_name, embedding_dim, chunk_len, entity_len, summary_len]):
            raise ValueError("一个或多个 Milvus 配置参数缺失。请检查 config.yaml 文件。")

        # --- 3. 创建集合和索引 ---
        logger.info(f"正在检查并创建块集合: '{chunk_collection_name}'...")
        vector_ops.create_chunk_collection(
            collection_name=chunk_collection_name,
            dense_dim=embedding_dim,
            text_max_length=chunk_len
        )

        logger.info(f"正在检查并创建实体集合: '{entity_collection_name}'...")
        vector_ops.create_entity_collection(
            collection_name=entity_collection_name,
            dense_dim=embedding_dim,
            text_max_length=entity_len
        )

        logger.info(f"正在检查并创建摘要集合: '{summary_collection_name}'...")
        vector_ops.create_summary_collection(
            collection_name=summary_collection_name,
            dense_dim=embedding_dim,
            text_max_length=summary_len
        )

        logger.info("--- Milvus 数据库初始化成功 ---")

    except Exception as e:
        logger.error("--- Milvus 数据库初始化失败 ---", exc_info=True)
        sys.exit(1)
    finally:
        # --- 4. 清理连接 ---
        if vector_db_conn:
            vector_db_conn.close()
            logger.info("Milvus 连接已关闭。")

if __name__ == "__main__":
    initialize_neo4j_indexes()
    initialize_milvus_collections()
