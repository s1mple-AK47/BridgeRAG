import os
from datasets import load_dataset
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def explore_dataset(dataset_path: str):
    """
    Loads and explores the 2WikiMultihopQA dataset from a local directory.
    """
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset path '{dataset_path}' not found.")
        logging.error("Please make sure the path is correct.")
        return

    logging.info(f"Loading dataset from '{dataset_path}'...")
    
    # The `load_dataset` function can directly load from a directory
    # containing the data files and a README.md with dataset metadata.
    # It will automatically find the parquet files for the 'train' split.
    try:
        # We specify the 'train' split since that's what's available.
        # The library will read the README and find the files matching 'data/train-*'.
        dataset = load_dataset(dataset_path, split='train')
        logging.info("Dataset loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        logging.error("Please ensure you have 'datasets' and 'pyarrow' installed: pip install datasets pyarrow")
        return

    print("\n" + "="*30)
    print("      Dataset Information")
    print("="*30)
    print(dataset)

    print("\nLet's look at the first example from the training set:")
    example = dataset[0]

    print("\n------------------------- Example -------------------------")
    print(f"ID: {example['id']}")
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer']}")
    print(f"Type: {example['type']}")

    print("\nSupporting Facts (Evidence needed to answer):")
    supporting_titles = example['supporting_facts']['title']
    print(f" - Titles: {supporting_titles}")
    
    print("\nContext (Paragraphs containing the evidence):")
    context_titles = example['context']['title']
    context_sentences = example['context']['sentences']
    
    # We'll just show the titles and the first sentence from the first 5 contexts for brevity
    print(" - First 5 context snippets:")
    for i in range(min(5, len(context_titles))):
        title = context_titles[i]
        first_sentence = context_sentences[i][0] if context_sentences[i] else "[No sentences]"
        print(f"   - From '{title}': \"{first_sentence}...\"")
    print("-----------------------------------------------------------\n")

if __name__ == "__main__":
    # Assuming the script is run from the project root directory /home/pangu/gxa_main/BridgeRAG
    project_root = os.getcwd()
    dataset_directory = os.path.join(project_root, "downloaded_articles", "2WikiMultihopQA")
    explore_dataset(dataset_directory)
