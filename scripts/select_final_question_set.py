import os
import json
import logging
from collections import Counter

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
TARGET_QUESTION_COUNT = 150

def select_questions_by_doc_count(input_file: str, output_file: str, target_count: int):
    """
    Selects a subset of questions from a JSONL file, prioritizing those that
    require the fewest documents.

    Args:
        input_file (str): Path to the source JSONL file of filtered questions.
        output_file (str): Path to write the final selected questions.
        target_count (int): The desired number of questions in the final set.
    """
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    # --- Step 1: Read all questions into memory ---
    logging.info(f"Loading questions from '{input_file}'...")
    questions = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line))
    except Exception as e:
        logging.error(f"Failed to read questions file: {e}")
        return
    
    logging.info(f"Loaded {len(questions)} questions.")

    # --- Step 2: Sort questions by the number of required documents ---
    # The key for sorting is the length of the 'ids' list.
    # This places questions with fewer required docs at the beginning of the list.
    questions.sort(key=lambda q: len(q.get('ids', [])))
    logging.info("Successfully sorted questions by document count (ascending).")

    # --- Step 3: Select the top N questions ---
    if len(questions) < target_count:
        logging.warning(
            f"The number of available questions ({len(questions)}) is less than the target ({target_count}). "
            f"All available questions will be used."
        )
        final_questions = questions
    else:
        final_questions = questions[:target_count]
    
    logging.info(f"Selected the top {len(final_questions)} questions.")

    # --- Step 4: Write the selected questions to the output file ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for question in final_questions:
                f.write(json.dumps(question) + '\n')
    except Exception as e:
        logging.error(f"Failed to write to output file: {e}")
        return

    # --- Step 5: Analyze and report on the final dataset ---
    doc_counts = Counter(len(q.get('ids', [])) for q in final_questions)
    
    print("\n" + "="*35)
    print("      Final Question Set Analysis")
    print("="*35)
    logging.info(f"Total questions selected: {len(final_questions)}")
    logging.info("Distribution of documents per question:")
    for count, num_questions in sorted(doc_counts.items()):
        print(f"  - Questions with {count} documents: {num_questions}")
    print("="*35)
    logging.info(f"Final question set saved to '{output_file}'")


if __name__ == "__main__":
    project_root = os.getcwd()
    
    # Input file from the previous filtering step
    source_file = os.path.join(project_root, "2wikimultihop_1000_questions_filtered.jsonl")
    
    # The final, small, curated output file
    final_output_file = os.path.join(project_root, "2wikimultihop_150_questions_final.jsonl")

    select_questions_by_doc_count(source_file, final_output_file, TARGET_QUESTION_COUNT)
