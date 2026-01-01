import logging
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bridgerag.config import settings
from bridgerag.database.graph_db import GraphDBConnection
from bridgerag.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def delete_all_same_as_links(confirm_delete=True):
    """
    Connects to Neo4j and deletes all :SAME_AS relationships between entities.
    :param confirm_delete: If True, will prompt the user for confirmation.
    """
    logger.info("--- Starting script to delete all :SAME_AS links ---")
    
    config = settings
    gdb_conn = None

    if confirm_delete:
        # Safety confirmation prompt
        confirm = input(
            "WARNING: This will permanently delete ALL :SAME_AS relationships from the database.\n"
            "This action cannot be undone.\n"
            "Are you sure you want to continue? (yes/no): "
        )
        if confirm.lower() != 'yes':
            logger.info("User aborted the operation. Exiting.")
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

        # 2. Define and execute the deletion query
        query = """
        MATCH ()-[r:SAME_AS]->()
        DELETE r
        """
        
        logger.info("Executing Cypher query to delete all :SAME_AS links...")
        with driver.session() as session:
            result = session.run(query)
            summary = result.consume()
            relationships_deleted = summary.counters.relationships_deleted
            
            logger.info("Query execution complete.")
            
            # 3. Report the result
            print("\n" + "="*50)
            print("      Deletion Summary")
            print("="*50)
            print(f"Total :SAME_AS relationships deleted: {relationships_deleted}")
            print("="*50 + "\n")

    except Exception as e:
        logger.error(f"An error occurred while deleting links: {e}", exc_info=True)
    finally:
        # 4. Close the database connection
        if gdb_conn:
            gdb_conn.close()
            
    logger.info("--- Script finished ---")

if __name__ == "__main__":
    delete_all_same_as_links()
