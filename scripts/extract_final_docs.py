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

def extract_required_docs(questions_file: str, all_docs_file: str, output_file: str):
    """
    Extracts a subset of documents from a large JSONL file based on the
    document IDs required by a given set of questions.

    Args:
        questions_file (str): Path to the final JSONL file of questions.
        all_docs_file (str): Path to the large JSONL file containing all filtered documents.
        output_file (str): Path to write the final, minimal set of required documents.
    """
    # --- Step 1: Collect all unique document IDs required by the questions ---
    logging.info(f"Loading required document IDs from '{questions_file}'...")
    if not os.path.exists(questions_file):
        logging.error(f"Questions file not found: {questions_file}")
        return
        
    required_doc_ids = set()
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    required_doc_ids.update(json.loads(line)['ids'])
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        logging.error(f"Failed to read questions file: {e}")
        return
        
    logging.info(f"The {sum(1 for _ in open(questions_file))} questions require {len(required_doc_ids)} unique documents.")

    # --- Step 2: Filter the main document file ---
    logging.info(f"Extracting these {len(required_doc_ids)} documents from '{all_docs_file}'...")
    if not os.path.exists(all_docs_file):
        logging.error(f"All documents file not found: {all_docs_file}")
        return

    found_docs_count = 0
    
    try:
        with open(all_docs_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)

        with open(all_docs_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, total=total_lines, desc="Extracting Docs"):
                try:
                    data = json.loads(line)
                    doc_id = data.get("id")
                    
                    # If the document's ID is in our required set, write it to the output
                    if doc_id in required_doc_ids:
                        outfile.write(line)
                        found_docs_count += 1
                        
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        logging.error(f"An error occurred during document extraction: {e}")
        return

    # --- Step 3: Final Report ---
    print("\n" + "="*35)
    print("      Document Extraction Complete")
    print("="*35)
    logging.info(f"Total unique documents required: {len(required_doc_ids)}")
    logging.info(f"Documents found and saved: {found_docs_count}")
    if found_docs_count < len(required_doc_ids):
        logging.warning(
            f"Warning: {len(required_doc_ids) - found_docs_count} required documents were not found in the source file."
        )
    print("="*35)
    logging.info(f"Final document set saved to '{output_file}'")


if __name__ == "__main__":
    project_root = os.getcwd()
    
    # Input files
    final_questions_file = os.path.join(project_root, "2wikimultihop_150_questions_final.jsonl")
    all_filtered_docs_file = os.path.join(project_root, "downloaded_articles", "2wikimultihop_docs_filtered.jsonl")
    
    # Output file
    final_docs_output_file = os.path.join(project_root, "downloaded_articles", "2wikimultihop_docs_final.jsonl")

    extract_required_docs(final_questions_file, all_filtered_docs_file, final_docs_output_file)
