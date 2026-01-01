import logging
import sys
from pathlib import Path
import uuid
from urllib.parse import urlparse

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridgerag.config import Settings
from bridgerag.core.embedding_client import EmbeddingClient
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.object_storage import ObjectStorageConnection
from bridgerag.database.vector_db import VectorDBConnection
from bridgerag.database.vector_ops import (
    create_chunk_collection,
    create_entity_collection,
    create_summary_collection,
)
from bridgerag.offline.steps import save_knowledge_graph
from bridgerag.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class MockEmbeddingClient:
    """A mock embedding client that returns fake embeddings for testing."""
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        logger.info(f"MockEmbeddingClient initialized with embedding dimension: {self.embedding_dim}")

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Returns a list of fake embedding vectors."""
        num_texts = len(texts)
        logger.info(f"MockEmbeddingClient generating {num_texts} fake embeddings.")
        # Create a deterministic, but fake, embedding for each text
        return [[float(i % 10) * 0.1] * self.embedding_dim for i in range(num_texts)]


def create_mock_data(doc_id: str) -> dict:
    """Creates a dictionary of mock data simulating the output of build_partitions."""
    logger.info(f"Generating mock data for doc_id: {doc_id}")
    
    entity1_id = str(uuid.uuid4())
    entity2_id = str(uuid.uuid4())

    return {
        "doc_id": doc_id,
        "document_summary": f"This is a comprehensive summary for the document '{doc_id}'. It synthesizes all key information.",
        "chunks": [
            {
                "chunk_id": f"{doc_id}_0",
                "content": "This is the first chunk of text from the document. It mentions the Galactic Empire.",
            },
            {
                "chunk_id": f"{doc_id}_1",
                "content": "This is the second chunk. It talks about the Rebel Alliance and their struggle.",
            }
        ],
        "entities": [
            {
                "entity_id": entity1_id,
                "entity_name": "Galactic Empire",
                "summary": "The Galactic Empire was the galactic government that replaced the Galactic Republic in the aftermath of the Clone Wars.",
                "is_named_entity": True,
            },
            {
                "entity_id": entity2_id,
                "entity_name": "Rebel Alliance",
                "summary": "The Alliance to Restore the Republic, was a resistance movement formed to challenge the rule of the Galactic Empire.",
                "is_named_entity": True,
            }
        ],
        "relations": [
            {
                "source_entity_id": entity1_id,
                "target_entity_id": entity2_id,
                "description": "The Galactic Empire is the primary antagonist of the Rebel Alliance.",
                "strength": 9,
                "keywords": ["conflict", "antagonist"]
            }
        ]
    }


def main():
    """
    Main function to initialize clients, create mock data, and run the save_partition function.
    """
    TEST_DOC_ID = "test_doc_save_partition"
    
    logger.info("--- Starting Storage Logic Test ---")
    
    clients = {}
    try:
        # --- 1. Initialize all necessary clients ---
        logger.info("Initializing clients...")
        settings = Settings()
        # Use the MockEmbeddingClient instead of the real one for testing
        clients["embedding_client"] = MockEmbeddingClient(settings.embedding_dim)
        clients["graph_db_conn"] = GraphDBConnection(
            uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password
        )
        clients["object_storage_conn"] = ObjectStorageConnection(
            endpoint=settings.minio_endpoint, access_key=settings.minio_access_key, secret_key=settings.minio_secret_key
        )
        # 实例化 VectorDBConnection 以创建和注册 Milvus 连接
        parsed_milvus_uri = urlparse(settings.milvus_uri)
        clients["vector_db_conn"] = VectorDBConnection(
            host=parsed_milvus_uri.hostname, port=parsed_milvus_uri.port, alias="default"
        )

        logger.info("All clients initialized successfully.")

        # --- 2. 确保 Milvus 集合和索引存在 ---
        logger.info("Ensuring Milvus collections and indexes exist...")
        create_chunk_collection(
            collection_name=settings.chunk_collection_name,
            dense_dim=settings.embedding_dim,
            text_max_length=settings.text_max_length,
        )
        create_entity_collection(
            collection_name=settings.entity_collection_name,
            dense_dim=settings.embedding_dim,
            text_max_length=settings.text_max_length,
        )
        create_summary_collection(
            collection_name=settings.summary_collection_name,
            dense_dim=settings.embedding_dim,
            text_max_length=settings.text_max_length,
        )
        logger.info("Milvus collections and indexes are ready.")

        # --- 3. Create mock data ---
        mock_data = create_mock_data(TEST_DOC_ID)

        # --- 4. Call the save_partition function ---
        logger.info(f"Calling save_partition for doc_id '{TEST_DOC_ID}'...")
        
        save_knowledge_graph.save_partition(
            **mock_data,
            # Pass clients
            neo4j_driver=clients["graph_db_conn"]._driver,
            minio_client=clients["object_storage_conn"].client,
            embedding_client=clients["embedding_client"],
            milvus_alias="default",
            # Pass configs
            minio_configs={"bucket_name": settings.minio_bucket_name},
            embedding_configs={"embedding_model": settings.vllm_embedding_model_name},
            milvus_collection_names={
                "chunk_collection_name": settings.chunk_collection_name,
                "entity_collection_name": settings.entity_collection_name,
                "summary_collection_name": settings.summary_collection_name,
            },
            force_rewrite=True  # Use True to ensure a clean state for each test run
        )

        logger.info("--- Storage Logic Test Completed Successfully ---")
        logger.info(f"Please verify that the data for doc_id '{TEST_DOC_ID}' exists in Neo4j, Milvus, and MinIO.")

    except Exception as e:
        logger.error(f"--- Storage Logic Test Failed ---", exc_info=True)
        sys.exit(1)
    finally:
        # --- 4. Clean up connections ---
        if "graph_db_conn" in clients and clients["graph_db_conn"]:
            clients["graph_db_conn"].close()
        if "vector_db_conn" in clients and clients["vector_db_conn"]:
            clients["vector_db_conn"].close()
            logger.info("Milvus connection closed.")

if __name__ == "__main__":
    main()
