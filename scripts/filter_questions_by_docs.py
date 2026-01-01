import os
import json
import logging

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm library not found. Progress bar will not be shown.")
    print("Install it with: pip install tqdm")
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_questions(questions_file: str, valid_docs_file: str, output_file: str):
    """
    Filters a JSONL file of questions, keeping only those whose required documents
    are all present in a "whitelist" of valid documents.

    Args:
        questions_file (str): Path to the JSONL file containing questions.
        valid_docs_file (str): Path to the filtered JSONL file of valid documents.
        output_file (str): Path to write the final filtered questions.
    """
    # --- Step 1: Create a set of all valid document IDs ---
    logging.info(f"Loading valid document IDs from '{valid_docs_file}'...")
    if not os.path.exists(valid_docs_file):
        logging.error(f"Valid documents file not found: {valid_docs_file}")
        return
    
    valid_doc_ids = set()
    try:
        with open(valid_docs_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    valid_doc_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    continue # Skip malformed lines
    except Exception as e:
        logging.error(f"Failed to read valid documents file: {e}")
        return
        
    logging.info(f"Successfully loaded {len(valid_doc_ids)} valid document IDs.")

    # --- Step 2: Filter the questions file ---
    if not os.path.exists(questions_file):
        logging.error(f"Questions file not found: {questions_file}")
        return

    # Counters for the final report
    total_processed = 0
    kept_count = 0

    logging.info(f"Filtering questions from '{questions_file}'...")
    
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
        
        with open(questions_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, total=total_lines, desc="Filtering Questions"):
                total_processed += 1
                try:
                    data = json.loads(line)
                    required_doc_ids = data.get("ids", [])

                    # Check if the set of required IDs is a subset of the valid IDs
                    if set(required_doc_ids).issubset(valid_doc_ids):
                        outfile.write(line)
                        kept_count += 1

                except (json.JSONDecodeError, KeyError):
                    logging.warning(f"Skipping malformed question line #{total_processed}")
                    continue
    except Exception as e:
        logging.error(f"An error occurred during question filtering: {e}")
        return

    # --- Step 3: Final Report ---
    removed_count = total_processed - kept_count
    print("\n" + "="*30)
    print("      Filtering Complete")
    print("="*30)
    logging.info(f"Total questions processed: {total_processed}")
    logging.info(f"Questions kept: {kept_count}")
    logging.info(f"Questions removed: {removed_count} (due to missing documents)")
    print("="*30)
    logging.info(f"Filtered questions have been saved to '{output_file}'")

if __name__ == "__main__":
    project_root = os.getcwd()
    
    # Input files
    source_questions_file = os.path.join(project_root, "2wikimultihop_1000_questions.jsonl")
    valid_documents_file = os.path.join(project_root, "downloaded_articles", "2wikimultihop_docs_filtered.jsonl")
    
    # Output file
    filtered_questions_output_file = os.path.join(project_root, "2wikimultihop_1000_questions_filtered.jsonl")

    filter_questions(source_questions_file, valid_documents_file, filtered_questions_output_file)
