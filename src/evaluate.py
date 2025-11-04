# src/evaluate.py
import pandas as pd
import joblib
import os
import json
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import load_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model_config: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Loads a model and evaluates its performance on the test set."""
    model_name = model_config['name']
    model_filename = model_config['file_name']
    
    model_path = os.path.join('models', model_filename)
    processed_data_dir = 'data/processed'
    metrics_output_dir = 'metrics'
    plots_dir = params['validation']['plots_dir']

    logging.info(f"--- Evaluating model: {model_name} ---")
    model = joblib.load(model_path)

    X_test = pd.read_csv(os.path.join(processed_data_dir, 'X_test.csv'), index_col='datetime', parse_dates=True)
    y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv'), index_col='datetime', parse_dates=True).iloc[:, 0]
    
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    metrics = {'mae': mae, 'rmse': rmse, 'r2_score': r2}

    os.makedirs(metrics_output_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics for {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    os.makedirs(plots_dir, exist_ok=True)
    residuals = y_test - predictions

    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, residuals, marker='.', linestyle='None', alpha=0.6)
    plt.title(f'{model_name} - Residuals Over Time')
    plt.savefig(os.path.join(plots_dir, f'{model_name}_residuals_over_time.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f'{model_name} - Distribution of Residuals')
    plt.savefig(os.path.join(plots_dir, f'{model_name}_residuals_histogram.png'))
    plt.close()
    
    logging.info(f"Validation plots saved for {model_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a specific model")
    parser.add_argument("--model-name", required=True, help="The name of the model to evaluate")
    args = parser.parse_args()

    all_params = load_params()

    # --- SIMPLIFIED LOGIC ---
    # Directly access the model config from the dictionary.
    model_name = args.model_name
    if model_name in all_params['models']:
        # We need to pass the model name along with the config
        model_config_data = all_params['models'][model_name]
        model_config_data['name'] = model_name # Add the name to the dictionary
        evaluate_model(model_config_data, all_params)
    else:
        logging.error(f"Model '{model_name}' not found in params.yaml")
        exit(1)