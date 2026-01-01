import json
import os
import re
from typing import Set

# Assuming the bridgerag module and its dependencies are in PYTHONPATH
# You might need to adjust the imports based on your project structure
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.vector_db import VectorDBConnection
# from bridgerag.database.object_storage import ObjectStorageConnection # No longer needed
from bridgerag.database import graph_ops, vector_ops #, object_storage_ops # No longer needed

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# These should match your docker-compose.yml and application configs
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

MILVUS_ALIAS = "default"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19531"

# MINIO_ENDPOINT = "localhost:9002"
# MINIO_ACCESS_KEY = "minioadmin"
# MINIO_SECRET_KEY = "minioadmin"

# Define collection and bucket names used by your application
# You might need to verify these names from your application's configuration
CHUNK_COLLECTION_NAME = "bridgerag_chunks"
ENTITY_COLLECTION_NAME = "bridgerag_entities"
SUMMARY_COLLECTION_NAME = "bridgerag_summaries"
VECTOR_COLLECTIONS = [CHUNK_COLLECTION_NAME, ENTITY_COLLECTION_NAME, SUMMARY_COLLECTION_NAME]

# This assumes you have a bucket for original documents.
# If not, you can leave it empty or comment out the Minio part.
# DOCUMENTS_BUCKET = "documents" 

# --- File Paths ---
FINAL_QUESTIONS_PATH = 'final_filtered_questions.jsonl'
SUCCESSFUL_LOGS_PATH = '/home/pangu/gxa_main/BridgeRAG/logs/successful_documents.log'


def get_docs_to_keep(filepath: str) -> Set[str]:
    """Reads the final_filtered_questions.jsonl and returns a set of document IDs to keep."""
    docs_to_keep = set()
    if not os.path.exists(filepath):
        logging.warning(f"File not found: {filepath}. No documents will be marked to keep.")
        return docs_to_keep

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                if "ids" in item and isinstance(item["ids"], list):
                    for doc_id in item["ids"]:
                        docs_to_keep.add(doc_id)
            except json.JSONDecodeError:
                logging.warning(f"Could not parse line in {filepath}: {line.strip()}")
    logging.info(f"Identified {len(docs_to_keep)} unique documents to keep from {filepath}.")
    return docs_to_keep


def get_all_processed_docs(filepath: str) -> Set[str]:
    """Reads the successful_documents.log to get a list of all processed docs."""
    all_docs = set()
    if not os.path.exists(filepath):
        logging.error(f"FATAL: Successful documents log not found at {filepath}. Cannot determine which documents to delete.")
        return all_docs
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                all_docs.add(stripped_line)
    logging.info(f"Found {len(all_docs)} total processed documents in {filepath}.")
    return all_docs


def delete_document_data(doc_id: str, graph_db: GraphDBConnection, vector_db: VectorDBConnection):
    """
    Deletes all data associated with a single document ID from all databases.
    """
    logging.info(f"--- Starting deletion for document: {doc_id} ---")

    # 1. Delete from Graph Database (Neo4j)
    try:
        with graph_db.get_driver().session() as session:
            # Using a transaction to ensure atomicity for the graph deletion part
            session.write_transaction(graph_ops.delete_document_graph, doc_id)
        logging.info(f"[Neo4j] Successfully deleted graph data for doc_id: {doc_id}")
    except Exception as e:
        logging.error(f"[Neo4j] Failed to delete graph data for doc_id: {doc_id}. Reason: {e}")

    # 2. Delete from Vector Database (Milvus)
    for collection in VECTOR_COLLECTIONS:
        try:
            # Note: Milvus filter expressions require double quotes around string values
            filter_expr = f'doc_id == "{doc_id}"'
            vector_ops.delete_vectors_by_filter(collection, filter_expr, alias=MILVUS_ALIAS)
            logging.info(f"[Milvus] Successfully deleted vectors from collection '{collection}' for doc_id: {doc_id}")
        except Exception as e:
            # It's possible a collection doesn't exist, which is not a critical error
            logging.warning(f"[Milvus] Could not delete vectors from collection '{collection}' for doc_id: {doc_id}. Reason: {e}")

    # 3. Delete from Object Storage (Minio) - SKIPPED
    # if DOCUMENTS_BUCKET:
    #     try:
    #         deleted = object_storage_ops.delete_object(object_storage.get_client(), DOCUMENTS_BUCKET, doc_id)
    #         if deleted:
    #             logging.info(f"[Minio] Successfully deleted object '{doc_id}' from bucket '{DOCUMENTS_BUCKET}'.")
    #     except Exception as e:
    #         logging.error(f"[Minio] Failed to delete object '{doc_id}' from bucket '{DOCUMENTS_BUCKET}'. Reason: {e}")
            
    logging.info(f"--- Finished deletion for document: {doc_id} ---")


def main():
    """Main function to orchestrate the database cleanup."""
    logging.info("Starting database cleanup script.")

    # 1. Determine which documents to delete
    docs_to_keep = get_docs_to_keep(FINAL_QUESTIONS_PATH)
    all_docs = get_all_processed_docs(SUCCESSFUL_LOGS_PATH)
    
    if not all_docs:
        logging.error("Could not retrieve the list of all processed documents. Aborting.")
        return

    docs_to_delete = all_docs - docs_to_keep
    
    if not docs_to_delete:
        logging.info("No documents to delete. All processed documents are referenced in the final questions file. Exiting.")
        return
        
    logging.info(f"Found {len(docs_to_delete)} documents to delete.")
    # Log first 5 for preview
    logging.info(f"Deletion preview (first 5): {list(docs_to_delete)[:5]}")

    # 2. Get user confirmation
    confirm = input("Are you sure you want to permanently delete all data for these documents? (yes/no): ")
    if confirm.lower() != 'yes':
        logging.info("User aborted the operation. Exiting.")
        return

    # 3. Initialize database connections
    logging.info("Initializing database connections...")
    try:
        graph_db = GraphDBConnection(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
        vector_db = VectorDBConnection(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
        # object_storage = ObjectStorageConnection(
        #     endpoint=MINIO_ENDPOINT,
        #     access_key=MINIO_ACCESS_KEY,
        #     secret_key=MINIO_SECRET_KEY,
        #     secure=False
        # )
    except Exception as e:
        logging.error(f"Failed to connect to databases. Aborting. Reason: {e}")
        return

    # 4. Execute deletion for each document
    for i, doc_id in enumerate(docs_to_delete):
        logging.info(f"Processing document {i+1}/{len(docs_to_delete)}...")
        delete_document_data(doc_id, graph_db, vector_db)

    # 5. Close connections
    logging.info("Cleanup complete. Closing database connections.")
    graph_db.close()
    vector_db.close()
    
    logging.info("Script finished.")


if __name__ == "__main__":
    main()
