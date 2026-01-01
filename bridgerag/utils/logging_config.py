import yaml
import logging.config
from pathlib import Path

def setup_logging():
    """
    Loads logging configuration from the YAML file and sets it up.
    """
    config_file = Path(__file__).resolve().parent.parent / "configs" / "logging.yaml"
    if config_file.exists():
        with open(config_file, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging configuration file not found at {config_file}. Using basic config.")
