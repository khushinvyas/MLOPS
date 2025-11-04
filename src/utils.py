# src/utils.py

import yaml
import logging
from typing import Dict, Any

# Configure logging for consistent output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params(params_path: str = 'params.yaml') -> Dict[str, Any]:
    """
    Loads parameters from a YAML file.

    Args:
        params_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the parameters.
    """
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logging.info(f"Parameters loaded successfully from {params_path}")
        return params
    except FileNotFoundError:
        logging.error(f"Error: The parameter file at {params_path} was not found.")
        exit(1)  # Exit if configuration is missing
    except yaml.YAMLError as e:
        logging.error(f"Error parsing the YAML file {params_path}: {e}")
        exit(1)  # Exit if configuration is malformed