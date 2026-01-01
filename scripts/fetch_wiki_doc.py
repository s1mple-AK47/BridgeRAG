import wikipediaapi
import argparse
import logging
import re
import os
import sys
from typing import Optional, Tuple

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bridgerag.utils.logging_config import setup_logging

# Setup logging
setup_logging()
# Configure basic logging
# Note: The logger name is derived from the module's file name: 'scripts.fetch_wiki_doc'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sanitize_filename(title: str) -> str:
    """Sanitizes a string to be used as a valid filename."""
    # Replace spaces with underscores, which is more common for filenames
    s = title.replace(" ", "_")
    # Remove characters that are invalid in most filesystems
    s = re.sub(r'[\\/*?:"<>|]', '', s)
    # Replace any remaining invalid characters with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', s)


def fetch_wikipedia_article(title: str) -> Optional[Tuple[str, str]]:
    """
    Fetches a Wikipedia article by its title.

    Args:
        title (str): The exact title of the Wikipedia article.

    Returns:
        Optional[Tuple[str, str]]: A tuple containing the final page title (after redirects)
                                    and the page content. Returns None if page not found or on error.
    """
    try:
        logger.debug(f"Attempting to fetch Wikipedia article: '{title}'")
        
        # Using the Wikipedia-API library
        wiki_wiki = wikipediaapi.Wikipedia(
            user_agent="BridgeRAG-Experiment/1.0 (https://github.com/your-repo)",
            language='en'
        )
        
        page = wiki_wiki.page(title)

        if not page.exists():
            logger.warning(f"Warning: The page with title '{title}' could not be found on Wikipedia.")
            return None

        content = page.text
        # Use the actual page title after potential redirects
        return (page.title, content)

    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching '{title}': {e}", exc_info=False)
        return None


def save_article_to_file(title: str, content: str, output_dir: str):
    """Saves article content to a text file."""
    filename = sanitize_filename(title)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    logger.info(f"Successfully saved article to '{output_path}'")


def main():
    """
    Main function to run the script from the command line for testing a single article.
    It will print the article content to the console.
    """
    parser = argparse.ArgumentParser(
        description="Fetch a Wikipedia article and print its content.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "title",
        type=str,
        help="The exact title of the Wikipedia article to fetch."
    )
    
    args = parser.parse_args()
    
    result = fetch_wikipedia_article(args.title)
    if result:
        final_title, content = result
        save_article_to_file(final_title, content, args.output_dir)

if __name__ == "__main__":
    # To keep the command-line interface clean, we only show INFO and above by default.
    # The batch script can override this for its own purposes.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("wikipediaapi").setLevel(logging.WARNING)
    
    parser = argparse.ArgumentParser(
        description="Fetch a Wikipedia article and print its content.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "title",
        type=str,
        help="The exact title of the Wikipedia article to fetch."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="downloaded_articles",
        help="The directory to save the downloaded article file. Defaults to 'downloaded_articles'."
    )
    
    args = parser.parse_args()
    
    result = fetch_wikipedia_article(args.title)
    if result:
        final_title, content = result
        save_article_to_file(final_title, content, args.output_dir)
