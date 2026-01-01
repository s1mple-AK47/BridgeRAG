import json

def filter_questions(data_file, successful_docs_file, output_file):
    """
    Filters questions from a JSONL dataset based on successfully processed documents.

    Args:
        data_file (str): Path to the input JSONL data file.
        successful_docs_file (str): Path to the file containing names of successful documents.
        output_file (str): Path to the output JSONL file.
    """
    # 1. Read successful document names into a set for efficient lookup
    with open(successful_docs_file, 'r', encoding='utf-8') as f:
        successful_docs = {line.strip() for line in f if line.strip()}

    print(f"Loaded {len(successful_docs)} successful document names.")

    # 2. Process the data file and write to the output file
    with open(data_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        total_lines = 0
        written_lines = 0

        for line in f_in:
            total_lines += 1
            try:
                item = json.loads(line)
                
                # 3. Check if all documents for the question are in the successful set
                paragraph_ids = [p['id'] for p in item.get('paragraphs', [])]
                
                if not paragraph_ids:
                    continue

                if all(doc_id in successful_docs for doc_id in paragraph_ids):
                    # 4. Create the new JSON object in the desired format
                    new_item = {
                        "id": item.get("id"),
                        "question": item.get("question"),
                        "answer": item.get("answer"),
                        "ids": paragraph_ids
                    }
                    # 5. Write the new item to the output file
                    f_out.write(json.dumps(new_item) + '\n')
                    written_lines += 1

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")
            except KeyError as e:
                print(f"Warning: Missing key {e} in line: {line.strip()}")

    print(f"Processing complete.")
    print(f"Total questions processed: {total_lines}")
    print(f"Questions with all documents successfully processed: {written_lines}")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    DATA_JSONL_PATH = '/home/pangu/gxa_main/BridgeRAG/data.jsonl'
    SUCCESSFUL_LOGS_PATH = '/home/pangu/gxa_main/BridgeRAG/logs/successful_documents.log'
    OUTPUT_JSONL_PATH = 'filtered_questions.jsonl'
    
    filter_questions(DATA_JSONL_PATH, SUCCESSFUL_LOGS_PATH, OUTPUT_JSONL_PATH)
