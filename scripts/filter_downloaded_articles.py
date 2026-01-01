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

# Define filtering constants
MAX_WORD_COUNT = 2000

def filter_articles(input_file: str, output_file: str):
    """
    Filters a JSONL file of articles based on two criteria:
    1. Removes articles where the 'id' (title) contains quotes.
    2. Removes articles where the 'text' (content) exceeds a word count limit.

    Args:
        input_file (str): Path to the source JSONL file.
        output_file (str): Path to write the filtered JSONL file.
    """
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    # Counters for the final report
    total_processed = 0
    kept_count = 0
    removed_for_quote = 0
    removed_for_length = 0

    logging.info(f"Starting to filter '{input_file}'...")
    logging.info(f"Criteria: No quotes in title, and content <= {MAX_WORD_COUNT} words.")

    # To get a progress bar with tqdm, we need to know the total number of lines first.
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
    except Exception as e:
        logging.error(f"Could not read the input file to determine its size: {e}")
        return
        
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, total=total_lines, desc="Filtering Articles"):
            total_processed += 1
            try:
                data = json.loads(line)
                doc_id = data.get("id", "")
                doc_text = data.get("text", "")

                # Criterion 1: Check for quotes in the document ID (title)
                if "'" in doc_id or '"' in doc_id:
                    removed_for_quote += 1
                    continue

                # Criterion 2: Check for word count in the document text
                word_count = len(doc_text.split())
                if word_count > MAX_WORD_COUNT or word_count < 100:
                    removed_for_length += 1
                    continue

                # If both checks pass, write the line to the output file
                outfile.write(line)
                kept_count += 1

            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Skipping malformed line #{total_processed}: {e}")
                continue
    
    # --- Final Report ---
    print("\n" + "="*30)
    print("      Filtering Complete")
    print("="*30)
    logging.info(f"Total articles processed: {total_processed}")
    logging.info(f"Articles kept: {kept_count}")
    logging.info(f"Total articles removed: {total_processed - kept_count}")
    logging.info(f"  - Removed due to quotes in title: {removed_for_quote}")
    logging.info(f"  - Removed due to exceeding {MAX_WORD_COUNT} words: {removed_for_length}")
    print("="*30)
    logging.info(f"Filtered data has been saved to '{output_file}'")


if __name__ == "__main__":
    project_root = os.getcwd()
    
    source_file = os.path.join(project_root, "downloaded_articles", "2wikimultihop_docs.jsonl")
    filtered_output_file = os.path.join(project_root, "downloaded_articles", "2wikimultihop_docs_filtered.jsonl")

    filter_articles(source_file, filtered_output_file)
