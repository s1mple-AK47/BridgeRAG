import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bridgerag.utils.logging_config import setup_logging
from scripts.run_online_query_benchmark import main as run_benchmark
from bridgerag.config import settings # Import the settings object

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Experiment Configuration ---
# List of CHUNK_RETRIEVAL_LIMIT values to test.
CHUNK_LIMITS_TO_TEST = [1, 2, 3, 4]


def run_experiment():
    """
    Orchestrates the chunk retrieval limit tuning experiment.
    
    This script assumes the database :SAME_AS links are already set up correctly
    from a previous experiment (e.g., run_expansion_experiment.py).
    It iterates through a list of chunk retrieval limits, temporarily modifying
    the configuration for each, and runs the benchmark.
    """
    logger.info("--- Starting Chunk Retrieval Limit Tuning Experiment ---")
    logger.warning("This script ASSUMES the database links are correctly configured.")
    
    # Determine the total number of questions for resume logic.
    questions_file = Path(project_root) / "final_filtered_questions.jsonl"
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            total_questions = sum(1 for _ in f)
        logger.info(f"Found {total_questions} questions to be processed in each run.")
    except FileNotFoundError:
        logger.error(f"Critical error: Source questions file not found at {questions_file}")
        return

    for limit in CHUNK_LIMITS_TO_TEST:
        # Resume logic: Check if results for this limit are already complete.
        output_filename_check = f"online_query_results_chunk_{limit}.jsonl"
        output_path_check = Path(project_root) / "logs" / output_filename_check
        
        is_complete = False
        if output_path_check.exists() and output_path_check.stat().st_size > 0:
            lines_in_file = sum(1 for _ in open(output_path_check))
            if lines_in_file >= total_questions:
                is_complete = True

        if is_complete:
            logger.info(f"--- Results for CHUNK_LIMIT={limit} are complete. Skipping. ---")
            continue

        logger.info("\n" + "="*80)
        logger.info(f"--- Running experiment for CHUNK_RETRIEVAL_LIMIT={limit} ---")
        logger.info("="*80)

        # Temporarily modify the setting in memory
        original_limit = settings.CHUNK_RETRIEVAL_LIMIT
        logger.info(f"Temporarily setting CHUNK_RETRIEVAL_LIMIT from {original_limit} to {limit}")
        settings.CHUNK_RETRIEVAL_LIMIT = limit
        
        try:
            # Run the online query benchmark with the modified setting.
            output_filename = f"online_query_results_chunk_{limit}.jsonl"
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
            settings.CHUNK_RETRIEVAL_LIMIT = original_limit
            logger.info(f"Restored CHUNK_RETRIEVAL_LIMIT to {original_limit}")
            
    logger.info("\n" + "="*80)
    logger.info("--- Chunk Retrieval Limit Tuning Experiment Finished ---")
    logger.info("="*80)


if __name__ == "__main__":
    run_experiment()
