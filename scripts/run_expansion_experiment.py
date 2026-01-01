import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bridgerag.utils.logging_config import setup_logging
from scripts.delete_same_as_links import delete_all_same_as_links
from scripts.relink_from_details import relink_entities_from_details
from scripts.run_online_query_benchmark import main as run_benchmark
from bridgerag.config import settings # Import the settings object

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Experiment Configuration ---
# List of SAME_AS_ENTITY_EXPANSION_LIMIT values to test.
EXPANSION_LIMITS_TO_TEST = [2, 4, 5, 6]

# --- Fixed Parameters for Consistency ---
# These parameters are based on previous optimal findings and remain constant.
OPTIMAL_THRESHOLD = 7.0
OPTIMAL_LLM_WEIGHT = 0.7
OPTIMAL_EMBEDDING_WEIGHT = 0.3


def setup_database_links():
    """
    Performs a one-time setup to ensure the graph database has a consistent
    state for the experiment series. It deletes all old links and re-creates
    them using the defined optimal parameters.
    """
    logger.info("\n" + "#"*80)
    logger.info("--- Performing One-Time Database Link Setup ---")
    logger.info(f"Using fixed parameters: Threshold={OPTIMAL_THRESHOLD}, LLM Weight={OPTIMAL_LLM_WEIGHT}")
    
    try:
        # Step 1: Delete all existing :SAME_AS links.
        logger.info("\n[Setup 1/2] Deleting existing :SAME_AS links...")
        delete_all_same_as_links(confirm_delete=False)
        logger.info("[Setup 1/2] Deletion complete.")

        # Step 2: Relink entities with the optimal parameters.
        logger.info(f"\n[Setup 2/2] Relinking entities with optimal parameters...")
        relink_entities_from_details(
            llm_weight=OPTIMAL_LLM_WEIGHT,
            embedding_weight=OPTIMAL_EMBEDDING_WEIGHT,
            similarity_threshold=OPTIMAL_THRESHOLD
        )
        logger.info("[Setup 2/2] Relinking complete.")
        logger.info("--- Database setup finished successfully. ---")
        logger.info("#"*80 + "\n")
        return True
    except Exception as e:
        logger.error(f"!!! Database setup failed: {e}", exc_info=True)
        return False


def run_experiment():
    """
    Orchestrates the entity expansion limit tuning experiment.
    """
    logger.info("--- Starting Entity Expansion Limit Tuning Experiment ---")

    # The one-time database setup.
    if not setup_database_links():
        logger.error("Aborting experiment due to database setup failure.")
        return

    # Determine the total number of questions for resume logic.
    questions_file = Path(project_root) / "final_filtered_questions.jsonl"
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            total_questions = sum(1 for _ in f)
        logger.info(f"Found {total_questions} questions to be processed in each run.")
    except FileNotFoundError:
        logger.error(f"Critical error: Source questions file not found at {questions_file}")
        return

    for limit in EXPANSION_LIMITS_TO_TEST:
        # Resume logic: Check if results for this limit are already complete.
        output_filename_check = f"online_query_results_expansion_{limit}.jsonl"
        output_path_check = Path(project_root) / "logs" / output_filename_check
        
        is_complete = False
        if output_path_check.exists() and output_path_check.stat().st_size > 0:
            lines_in_file = sum(1 for _ in open(output_path_check))
            if lines_in_file >= total_questions:
                is_complete = True

        if is_complete:
            logger.info(f"--- Results for EXPANSION_LIMIT={limit} are complete. Skipping. ---")
            continue

        logger.info("\n" + "="*80)
        logger.info(f"--- Running experiment for SAME_AS_ENTITY_EXPANSION_LIMIT={limit} ---")
        logger.info("="*80)

        # Temporarily modify the setting in memory
        original_limit = settings.SAME_AS_ENTITY_EXPANSION_LIMIT
        logger.info(f"Temporarily setting SAME_AS_ENTITY_EXPANSION_LIMIT from {original_limit} to {limit}")
        settings.SAME_AS_ENTITY_EXPANSION_LIMIT = limit
        
        try:
            # Run the online query benchmark with the modified setting.
            output_filename = f"online_query_results_expansion_{limit}.jsonl"
            output_path = Path(project_root) / "logs" / output_filename
            
            logger.info(f"Running online query benchmark... Results will be saved to: {output_path}")
            run_benchmark(output_file_path=str(output_path))
            logger.info(f"Benchmark for limit={limit} complete.")

        except Exception as e:
            logger.error(f"!!! Experiment run for LIMIT={limit} failed: {e}", exc_info=True)
            logger.error("!!! Skipping to the next limit.")
            continue
        finally:
            # CRITICAL: Always restore the original setting
            settings.SAME_AS_ENTITY_EXPANSION_LIMIT = original_limit
            logger.info(f"Restored SAME_AS_ENTITY_EXPANSION_LIMIT to {original_limit}")
            
    logger.info("\n" + "="*80)
    logger.info("--- Entity Expansion Limit Tuning Experiment Finished ---")
    logger.info("="*80)


if __name__ == "__main__":
    run_experiment()
