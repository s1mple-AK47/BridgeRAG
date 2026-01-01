import os
import json
import logging
import sys
import time

# --- Path Setup ---
# Add the project root to the Python path.
# This allows us to import modules from the 'scripts' directory, like fetch_wiki_doc.
# This makes the script runnable from any location, not just the project root.
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

# --- Imports from other scripts ---
# Now we can safely import the functions we need from our other script.
from scripts.fetch_wiki_doc import fetch_wikipedia_article

# --- Optional Imports for User Experience ---
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm library not found. Progress bar will not be shown.")
    print("Install it with: pip install tqdm")
    # Define a fallback function if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configure basic logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Suppress Verbose Logging from Imported Modules ---
# Set the logging level for other modules to WARNING to avoid cluttering the console.
# This will hide the detailed INFO messages from the wikipedia library and our fetch script.
logging.getLogger("wikipediaapi").setLevel(logging.WARNING)
logging.getLogger("scripts.fetch_wiki_doc").setLevel(logging.WARNING)


def batch_download_to_jsonl(input_file: str, output_file: str):
    """
    Reads a JSONL file, extracts all unique article titles from the 'ids' field,
    and downloads each article to a single JSONL file.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path for the output JSONL file.
    """
    # 1. Collect all unique article titles
    logging.info(f"Reading article titles from '{input_file}'...")
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return
        
    all_titles = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'ids' in data and isinstance(data['ids'], list):
                    all_titles.update(data['ids'])
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON line: {line.strip()}")
    
    unique_titles = sorted(list(all_titles))
    
    # 2. Implement resumability by checking for already downloaded articles
    processed_titles = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        processed_titles.add(json.loads(line)['id'])
                    except (json.JSONDecodeError, KeyError):
                        continue # Skip malformed lines
            logging.info(f"Found {len(processed_titles)} already downloaded articles in '{output_file}'. Resuming download.")
        except IOError as e:
            logging.error(f"Could not read existing output file at '{output_file}': {e}")
            return

    titles_to_download = [t for t in unique_titles if t not in processed_titles]

    if not titles_to_download:
        logging.info(f"All {len(unique_titles)} required articles are already present in '{output_file}'. Nothing to do.")
        return
    
    logging.info(f"Found {len(unique_titles)} unique articles in total. Attempting to download {len(titles_to_download)} new articles.")

    # 3. Download each article and append to the JSONL file
    completed_count = 0
    # Open the output file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        for title in tqdm(titles_to_download, desc="Downloading Articles"):
            result = fetch_wikipedia_article(title)
            
            if result:
                final_title, content = result
                record = {"id": final_title, "text": content}
                f.write(json.dumps(record) + '\n')
                completed_count += 1
            
            # Be polite to Wikipedia's API by waiting a moment between requests
            time.sleep(0.1)
        
    logging.info(f"Download process finished. Newly downloaded articles: {completed_count}.")


if __name__ == "__main__":
    # Assuming this script is run from the project root directory
    project_root = os.getcwd()
    
    # Input file with the list of questions and article IDs
    questions_file = os.path.join(project_root, "2wikimultihop_1000_questions.jsonl")
    
    # The single output file where all downloaded articles will be stored in JSONL format
    articles_output_file = os.path.join(project_root, "downloaded_articles", "2wikimultihop_docs.jsonl")

    # Ensure the parent directory for the output file exists
    os.makedirs(os.path.dirname(articles_output_file), exist_ok=True)

    batch_download_to_jsonl(questions_file, articles_output_file)
