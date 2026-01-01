import sys
import os

# Add the project root directory to the Python path to resolve the ModuleNotFoundError
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from bridgerag.config import settings
from bridgerag.utils.logging_config import setup_logging
from bridgerag.core.llm_client import LLMClient
from bridgerag.offline.steps.build_partitions import _extract_entities_and_relations

# Basic setup
setup_logging()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# You can change this text to test different scenarios.
# This text is a sample from the Wikipedia article about "Star Wars".
SAMPLE_TEXT = """
Star Wars is an American epic space opera multimedia franchise created by George Lucas, which began with the eponymous 1977 film and quickly became a worldwide pop-culture phenomenon. The franchise has been expanded into various films and other media, including television series, video games, novels, comic books, theme park attractions, and themed areas, comprising an all-encompassing fictional universe.

The story of the original trilogy focuses on Luke Skywalker's quest to become a Jedi, his struggle with the evil Galactic Empire, and his conflict with his father, the Sith Lord Darth Vader. The prequel trilogy, released later, chronicles the backstory of Anakin Skywalker's transformation into Darth Vader. The sequel trilogy follows the adventures of Rey, a scavenger who discovers her connection to the Force, as she battles the First Order.
"""

def main():
    """
    Main function to run the entity and relation extraction test.
    """
    logger.info("--- Starting LLM Extraction Test ---")

    # --- 1. Initialize LLM Client ---
    # The script will use the LLM configuration from your settings.
    try:
        llm_client = LLMClient()
        logger.info("LLMClient initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
        return

    # --- 2. Define a mock chunk_id for context ---
    chunk_id = "test_chunk_01"
    logger.info(f"Processing sample text for mock chunk_id: '{chunk_id}'")

    # --- 3. Call the core extraction function ---
    # We are directly testing the function responsible for the LLM calls.
    try:
        logger.info("Calling _extract_entities_and_relations...")
        entities, relations = _extract_entities_and_relations(
            chunk_id=chunk_id,
            chunk_content=SAMPLE_TEXT,
            llm_client=llm_client
        )
        logger.info("Function call completed.")
    except Exception as e:
        logger.error(f"An exception occurred during extraction: {e}", exc_info=True)
        return

    # --- 4. Print the results for analysis ---
    logger.info("--- EXTRACTION RESULTS ---")
    logger.info(f"Number of entities extracted: {len(entities)}")
    logger.info(f"Number of relations extracted: {len(relations)}")

    print("\\n--- Entities ---")
    if entities:
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity}")
    else:
        print("No entities were extracted.")

    print("\\n--- Relations ---")
    if relations:
        for i, relation in enumerate(relations, 1):
            print(f"{i}. {relation}")
    else:
        print("No relations were extracted.")

    logger.info("--- LLM Extraction Test Finished ---")


if __name__ == "__main__":
    main()
