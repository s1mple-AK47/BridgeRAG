import os
import json
from datasets import load_dataset
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_and_format_data(dataset_path: str, output_file: str, num_samples: int):
    """
    Loads the 2WikiMultihopQA dataset, extracts a specified number of samples,
    and saves them to a JSONL file in the desired format.

    Args:
        dataset_path (str): The path to the local dataset directory.
        output_file (str): The path for the output JSONL file.
        num_samples (int): The number of samples to extract.
    """
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset path '{dataset_path}' not found.")
        return

    logging.info(f"Loading dataset from '{dataset_path}'...")
    try:
        # Load the 'train' split of the dataset
        dataset = load_dataset(dataset_path, split='train')
        logging.info("Dataset loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        logging.error("Please ensure you have 'datasets' and 'pyarrow' installed: pip install datasets pyarrow")
        return

    if len(dataset) < num_samples:
        logging.warning(f"The dataset contains only {len(dataset)} samples, which is less than the requested {num_samples}.")
        num_samples = len(dataset)

    logging.info(f"Extracting the first {num_samples} samples and writing to '{output_file}'...")
    
    count = 0
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Use .select() to get a subset of the dataset efficiently
            for example in dataset.select(range(num_samples)):
                # The 'ids' are the list of Wikipedia page titles from 'supporting_facts'
                ids = example['supporting_facts']['title']
                
                record = {
                    "question": example['question'],
                    "answer": example['answer'],
                    "ids": ids
                }
                
                f.write(json.dumps(record) + '\n')
                count += 1
        
        logging.info(f"Successfully processed and saved {count} records to '{output_file}'.")

    except Exception as e:
        logging.error(f"An error occurred during file writing: {e}")

if __name__ == "__main__":
    project_root = os.getcwd()
    dataset_directory = os.path.join(project_root, "downloaded_articles", "2WikiMultihopQA")
    output_jsonl_file = os.path.join(project_root, "2wikimultihop_1000_questions.jsonl")
    
    # Define the number of questions to extract
    NUMBER_OF_QUESTIONS = 1000
    
    extract_and_format_data(dataset_directory, output_jsonl_file, NUMBER_OF_QUESTIONS)
