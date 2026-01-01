import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bridgerag.utils.logging_config import setup_logging
# The following scripts will be modified to make their main functions importable
from scripts.delete_same_as_links import delete_all_same_as_links
from scripts.relink_from_details import relink_entities_from_details
from scripts.run_online_query_benchmark import main as run_benchmark

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Experiment Configuration ---
# List of (llm_weight, embedding_weight) pairs to test.
# (0.6, 0.4) is skipped as per the user's request.
WEIGHT_PAIRS = [
    (0.4, 0.6),
    (0.5, 0.5),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
    (1.0, 0.0),
]

def run_experiment():
    """
    Orchestrates the hyperparameter tuning experiment.
    
    This script iterates through a predefined list of weights, and for each pair:
    1. Deletes all existing :SAME_AS relationships in the graph.
    2. Re-creates the :SAME_AS links using the current weights.
    3. Runs the online query benchmark and saves the results to a unique file.
    """
    logger.info("--- Starting Hyperparameter Tuning Experiment ---")
    
    # First, determine the total number of questions to expect in a complete run.
    questions_file = Path(project_root) / "final_filtered_questions.jsonl"
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            total_questions = sum(1 for _ in f)
        logger.info(f"Found a total of {total_questions} questions to be processed in each run.")
    except FileNotFoundError:
        logger.error(f"Critical error: The source questions file was not found at {questions_file}")
        return

    for llm_weight, embedding_weight in WEIGHT_PAIRS:
        # Round to handle potential float inaccuracies from generation
        llm_weight = round(llm_weight, 1)
        embedding_weight = round(embedding_weight, 1)

        # Check if the final output file for this weight combination is already complete.
        output_filename_check = f"online_query_results_llm_{llm_weight}.jsonl"
        output_path_check = Path(project_root) / "logs" / output_filename_check
        
        is_complete = False
        if output_path_check.exists() and output_path_check.stat().st_size > 0:
            try:
                with open(output_path_check, 'r', encoding='utf-8') as f:
                    lines_in_file = sum(1 for _ in f)
                if lines_in_file >= total_questions:
                    is_complete = True
            except Exception as e:
                logger.warning(f"Could not read existing results file {output_path_check}: {e}")

        if is_complete:
            logger.info(f"--- Results file for LLM_WEIGHT={llm_weight} is already complete ({lines_in_file}/{total_questions} questions). Skipping. ---")
            continue

        logger.info("\n" + "="*80)
        logger.info(f"--- Running experiment for LLM_WEIGHT={llm_weight}, EMBEDDING_WEIGHT={embedding_weight} ---")
        logger.info("="*80)

        try:
            # Step 1: Delete all existing :SAME_AS links.
            # We will modify the function to accept a flag to bypass the user prompt.
            logger.info("\n[Step 1/3] Deleting existing :SAME_AS links...")
            delete_all_same_as_links(confirm_delete=False)
            logger.info("[Step 1/3] Deletion complete.")

            # Step 2: Relink entities with the current set of weights.
            # We will modify the function to accept weights as arguments.
            logger.info(f"\n[Step 2/3] Relinking entities with weights: LLM={llm_weight}, Embedding={embedding_weight}...")
            relink_entities_from_details(
                llm_weight=llm_weight, 
                embedding_weight=embedding_weight
            )
            logger.info("[Step 2/3] Relinking complete.")

            # Step 3: Run the online query benchmark.
            # We will modify the benchmark script to accept an output path.
            output_filename = f"online_query_results_llm_{llm_weight}.jsonl"
            output_path = Path(project_root) / "logs" / output_filename
            
            logger.info(f"\n[Step 3/3] Running online query benchmark...")
            logger.info(f"Results will be saved to: {output_path}")
            run_benchmark(output_file_path=str(output_path))
            logger.info("[Step 3/3] Benchmark complete.")

        except Exception as e:
            logger.error(f"!!! Experiment run for LLM_WEIGHT={llm_weight} failed: {e}", exc_info=True)
            logger.error("!!! Skipping to the next parameter set.")
            continue
            
    logger.info("\n" + "="*80)
    logger.info("--- Hyperparameter Tuning Experiment Finished ---")
    logger.info("--- You can find the results in the 'logs' directory. ---")
    logger.info("="*80)


if __name__ == "__main__":
    run_experiment()
