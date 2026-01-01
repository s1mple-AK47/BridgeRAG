import logging
import os
import sys
from itertools import combinations

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bridgerag.config import settings
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Constants from link_entities.py to define the criteria
MIN_DOCS = 2
MAX_DOCS = 2

def count_candidate_entity_pairs():
    """
    Connects to Neo4j, queries for candidate entities, and calculates the total
    number of pairs that can be formed from same-named entities.
    """
    logger.info("--- Starting script to count candidate entity pairs ---")
    
    config = settings
    gdb_conn = None
    total_pairs = 0

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

        # 2. Query for candidate entities and count pairs
        query = f"""
        MATCH (e:Entity)
        WHERE e.is_named_entity = true
        WITH e.name as name, count(e) as entity_count
        WHERE entity_count >= {MIN_DOCS} AND entity_count <= {MAX_DOCS}
        RETURN name, entity_count
        """
        
        logger.info("Executing Cypher query to find candidate entity groups...")
        with driver.session() as session:
            results = session.run(query)
            
            group_count = 0
            for record in results:
                group_count += 1
                entity_name = record["name"]
                count_in_group = record["entity_count"]
                
                # Calculate pairs for the current group using combinations formula: C(n, 2)
                pairs_in_group = count_in_group * (count_in_group - 1) // 2
                total_pairs += pairs_in_group
                
                logger.debug(
                    f"Found group '{entity_name}' with {count_in_group} entities, "
                    f"forming {pairs_in_group} pairs."
                )
        
        logger.info(f"Found {group_count} groups of same-named entities matching the criteria.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # 3. Close the database connection
        if gdb_conn:
            gdb_conn.close()
            
    logger.info("--- Script finished ---")
    
    # 4. Print the final result
    print("\n" + "="*50)
    print(f"Total number of candidate entity pairs: {total_pairs}")
    print("="*50 + "\n")

if __name__ == "__main__":
    count_candidate_entity_pairs()
