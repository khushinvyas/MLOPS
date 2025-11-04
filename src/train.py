# src/train.py

import pandas as pd
import joblib
import os
import logging
import argparse
from typing import Dict, Any

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from utils import load_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_instance(model_name: str, params: Dict[str, Any]):
    """Initializes a model instance based on its name and parameters."""
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    elif model_name == "XGBoostRegressor":
        return XGBRegressor(**params)
    elif model_name == "LightGBMRegressor":
        return LGBMRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

# --- FIX is here: We now accept 'model_name' as a direct argument ---
def train_model(model_name: str, model_config: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Loads data, trains a single model using params, and saves it."""
    processed_data_dir = 'data/processed'
    model_output_dir = 'models'
    
    # We get the params and filename from the model_config dictionary
    model_params = model_config['params']
    model_filename = model_config['file_name']

    logging.info(f"--- Starting training for: {model_name} ---")
    
    X_train = pd.read_csv(os.path.join(processed_data_dir, 'X_train.csv'), index_col='datetime', parse_dates=True)
    y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train.csv'), index_col='datetime', parse_dates=True).iloc[:, 0]

    model = get_model_instance(model_name, model_params)
    
    logging.info(f"Training {model_name}...")
    model.fit(X_train, y_train)

    logging.info(f"Training for {model_name} complete.")

    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, model_filename)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a specific model from params.yaml")
    parser.add_argument("--model-name", required=True, help="The name of the model to train")
    args = parser.parse_args()

    all_params = load_params()

    model_name = args.model_name
    if model_name in all_params['models']:
        model_config = all_params['models'][model_name]
        # --- FIX is here: Pass the model_name to the function ---
        train_model(model_name, model_config, all_params)
    else:
        logging.error(f"Model '{model_name}' not found in params.yaml")
        exit(1)