import logging
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import Settings
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.object_storage import ObjectStorageConnection
from bridgerag.database.vector_db import VectorDBConnection
from bridgerag.utils.logging_config import setup_logging
from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType
from minio.deleteobjects import DeleteObject

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def clear_neo4j(graph_db_conn: GraphDBConnection):
    """Deletes all nodes and relationships from the Neo4j database."""
    logger.info("--- Clearing Neo4j Database ---")
    try:
        with graph_db_conn.get_driver().session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Successfully deleted all nodes and relationships from Neo4j.")
    except Exception as e:
        logger.error(f"Failed to clear Neo4j database: {e}", exc_info=True)
        raise

def clear_milvus_collections(settings: Settings):
    """仅删除所有指定的 Milvus 集合，不再重建它们。"""
    logger.info("--- Clearing Milvus Collections ---")
    
    collection_names = [
        settings.chunk_collection_name,
        settings.entity_collection_name,
        settings.summary_collection_name
    ]

    try:
        for name in collection_names:
            if utility.has_collection(name, using='default'):
                logger.info(f"Dropping existing Milvus collection: {name}")
                utility.drop_collection(name, using='default')
        logger.info("All specified Milvus collections have been dropped.")

    except Exception as e:
        logger.error(f"Failed to drop Milvus collections: {e}", exc_info=True)
        raise


def clear_minio_bucket(object_storage_conn: ObjectStorageConnection, bucket_name: str):
    """
    Deletes all objects in the specified MinIO bucket by deleting and recreating the bucket.
    """
    logger.info(f"--- Clearing MinIO Bucket: {bucket_name} ---")
    client = object_storage_conn.client
    try:
        if client.bucket_exists(bucket_name):
            # 修正：需要从 list_objects 的结果中提取对象名称，并创建 DeleteObject 列表
            object_names = [obj.object_name for obj in client.list_objects(bucket_name, recursive=True)]
            if object_names:
                objects_to_delete = [DeleteObject(name) for name in object_names]
                errors = client.remove_objects(bucket_name, objects_to_delete)
                for error in errors:
                    logger.error(f"Error occurred when deleting object: {error}")
                logger.info(f"Successfully deleted all objects from MinIO bucket '{bucket_name}'.")
            else:
                logger.info(f"MinIO bucket '{bucket_name}' is already empty.")
        else:
            logger.warning(f"MinIO bucket '{bucket_name}' does not exist. Nothing to clear.")
    except Exception as e:
        logger.error(f"Failed to clear MinIO bucket '{bucket_name}': {e}", exc_info=True)
        raise


def main():
    """Main function to clear all databases."""
    logger.info("=== Starting Database Clearing Process ===")
    
    settings = Settings()
    graph_db_conn = None
    vector_db_conn = None

    try:
        # --- Initialize Connections ---
        graph_db_conn = GraphDBConnection(
            uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password
        )
        vector_db_conn = VectorDBConnection(
            host=settings.milvus_uri.split('//')[1].split(':')[0], 
            port=settings.milvus_uri.split(':')[-1]
        )
        object_storage_conn = ObjectStorageConnection(
            endpoint=settings.minio_endpoint, access_key=settings.minio_access_key, secret_key=settings.minio_secret_key
        )

        # --- Execute Clearing Operations ---
        clear_neo4j(graph_db_conn)
        clear_milvus_collections(settings)
        clear_minio_bucket(object_storage_conn, settings.minio_bucket_name)

        logger.info("=== Database Clearing Process Completed Successfully ===")

    except Exception as e:
        logger.error("An error occurred during the database clearing process.", exc_info=True)
        sys.exit(1)
    finally:
        # --- Close Connections ---
        if graph_db_conn:
            graph_db_conn.close()
        if vector_db_conn:
            vector_db_conn.close()

if __name__ == "__main__":
    main()
