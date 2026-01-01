import logging
import os
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bridgerag.config import settings
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.database.graph_ops import create_same_as_links
from bridgerag.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Configuration ---
# These constants should match the ones in link_entities.py for consistency
# You can easily tweak this threshold to experiment with different linking strategies
SIMILARITY_THRESHOLD = 7.0
LINK_BATCH_SIZE = 50
DETAILS_FILE_PATH = Path(project_root) / "logs" / "entity_linking_details.jsonl"
# Default weights, can be overridden by function arguments
DEFAULT_LLM_SCORE_WEIGHT = 0.7
DEFAULT_EMBEDDING_SCORE_WEIGHT = 0.3


def delete_existing_links(driver):
    """Deletes all existing :SAME_AS relationships."""
    logger.info("Deleting all existing :SAME_AS relationships...")
    query = "MATCH ()-[r:SAME_AS]->() DELETE r"
    with driver.session() as session:
        result = session.run(query)
        summary = result.consume()
        rels_deleted = summary.counters.relationships_deleted
        logger.info(f"Successfully deleted {rels_deleted} relationships.")

def relink_entities_from_details(
    llm_weight: float = DEFAULT_LLM_SCORE_WEIGHT, 
    embedding_weight: float = DEFAULT_EMBEDDING_SCORE_WEIGHT,
    similarity_threshold: float = SIMILARITY_THRESHOLD
):
    """
    Reads detailed entity linking results from a file and creates :SAME_AS
    links in Neo4j based on a similarity threshold and provided weights.
    """
    logger.info("--- Starting script to relink entities from details file ---")
    logger.info(f"Using weights: LLM_WEIGHT={llm_weight}, EMBEDDING_WEIGHT={embedding_weight}")
    logger.info(f"Using Similarity Threshold: {similarity_threshold}")
    
    config = settings
    gdb_conn = None

    if not DETAILS_FILE_PATH.exists():
        logger.error(f"FATAL: Details file not found at '{DETAILS_FILE_PATH}'.")
        logger.error("Please run the `run_entity_linking.py` script first to generate the details file.")
        return

    try:
        # 1. Initialize database connection
        logger.info("Initializing Neo4j database connection...")
        gdb_conn = GraphDBConnection(
            uri=config.neo4j_uri, 
            user=config.neo4j_user, 
            password=config.neo4j_password
        )
        driver = gdb_conn.get_driver()
        logger.info("Database connection successful.")

        # 2. Automatically delete existing links
        delete_existing_links(driver)

        # 3. Read the details file and create links
        links_to_create = []
        total_links_created = 0
        total_records_read = 0

        with open(DETAILS_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                total_records_read += 1
                try:
                    data = json.loads(line)
                    
                    # Dynamically recalculate the final score based on the weights defined in this script
                    llm_score = data.get("llm_score", 0)
                    embedding_score = data.get("embedding_score", 0.0)
                    
                    # The same calculation logic as in link_entities.py
                    final_score = (llm_score * llm_weight) + (
                        (embedding_score * 10) * embedding_weight
                    )

                    # Apply the threshold logic on the recalculated score
                    if final_score >= similarity_threshold:
                        link_data = {
                            "entity_1_id": data["entity_1_id"],
                            "entity_2_id": data["entity_2_id"],
                            "score": round(final_score, 2), # Use the recalculated score
                            "reasoning": data.get("llm_reasoning", ""),
                        }
                        links_to_create.append(link_data)

                        # Write to DB in batches
                        if len(links_to_create) >= LINK_BATCH_SIZE:
                            create_same_as_links(driver, links_to_create)
                            total_links_created += len(links_to_create)
                            logger.info(f"Wrote a batch of {len(links_to_create)} links to the database.")
                            links_to_create.clear()

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed line in details file: {line.strip()} | Error: {e}")
        
        # Write the final batch
        if links_to_create:
            create_same_as_links(driver, links_to_create)
            total_links_created += len(links_to_create)
            logger.info(f"Wrote the final batch of {len(links_to_create)} links.")

        # 4. Report the result
        print("\n" + "="*50)
        print("      Relinking Summary")
        print("="*50)
        print(f"Total records read from file: {total_records_read}")
        print(f"Similarity Threshold applied: >= {similarity_threshold}")
        print(f"Total :SAME_AS relationships created: {total_links_created}")
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"An error occurred during the relinking process: {e}", exc_info=True)
    finally:
        # 5. Close the database connection
        if gdb_conn:
            gdb_conn.close()
            
    logger.info("--- Script finished ---")


if __name__ == "__main__":
    # This allows running the script standalone with default weights
    relink_entities_from_details()
