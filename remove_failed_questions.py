import json
import os
import re

def remove_questions_with_failed_docs(
    input_file, failed_chunks_dir, output_file
):
    """
    Removes questions from a JSONL file if they are associated with failed documents.

    Args:
        input_file (str): Path to the input JSONL file (e.g., filtered_questions.jsonl).
        failed_chunks_dir (str): Path to the directory containing logs for failed chunks.
        output_file (str): Path to the final output JSONL file.
    """
    # 1. Identify all failed documents from the filenames in the failed_chunks directory
    failed_docs = set()
    try:
        for filename in os.listdir(failed_chunks_dir):
            # Extract document name from filename like "failed_chunks_Alex Ferguson.json"
            match = re.match(r"failed_chunks_(.*)\.json", filename)
            if match:
                doc_name = match.group(1).replace("_", " ")
                failed_docs.add(doc_name)
    except FileNotFoundError:
        print(f"Error: The directory '{failed_chunks_dir}' was not found.")
        return

    if not failed_docs:
        print("No failed documents found. The output file will be a copy of the input file.")
    else:
        print(f"Found {len(failed_docs)} failed documents: {list(failed_docs)[:5]}...")


    # 2. Read the input file, filter out questions with failed docs, and write to the output file
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        total_lines = 0
        written_lines = 0
        removed_lines = 0

        for line in f_in:
            total_lines += 1
            try:
                item = json.loads(line)
                
                # 3. Check if any of the document IDs for the question are in the failed set
                doc_ids = item.get("ids", [])
                
                # The intersection of the two sets is non-empty if there is at least one common element
                if not set(doc_ids).intersection(failed_docs):
                    # 4. If no associated docs failed, write the item to the output file
                    f_out.write(json.dumps(item) + '\n')
                    written_lines += 1
                else:
                    removed_lines += 1

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")
            except KeyError as e:
                print(f"Warning: Missing key {e} in line: {line.strip()}")

    print("\nProcessing complete.")
    print(f"Total questions processed: {total_lines}")
    print(f"Questions removed due to failed documents: {removed_lines}")
    print(f"Final questions written: {written_lines}")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    FILTERED_QUESTIONS_PATH = 'filtered_questions.jsonl'
    FAILED_CHUNKS_DIR_PATH = '/home/pangu/gxa_main/BridgeRAG/logs/failed_chunks'
    FINAL_OUTPUT_PATH = 'final_filtered_questions.jsonl'
    
    # First, ensure the input file exists
    if not os.path.exists(FILTERED_QUESTIONS_PATH):
        print(f"Error: Input file '{FILTERED_QUESTIONS_PATH}' not found.")
        print("Please run the previous script 'filter_successful_questions.py' first.")
    else:
        remove_questions_with_failed_docs(
            FILTERED_QUESTIONS_PATH, FAILED_CHUNKS_DIR_PATH, FINAL_OUTPUT_PATH
        )
