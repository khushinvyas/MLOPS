# src/preprocess.py

import pandas as pd
import os
import logging
from typing import Dict, Any

from utils import load_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_lagged_features(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Shifts the dataframe to create lagged features for time series forecasting."""
    df_lagged = df.shift(lag)
    df_lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
    return df_lagged

def preprocess_data(input_path: str, output_dir: str, params: Dict[str, Any]) -> None:
    """Loads, processes, and splits data into training and testing sets."""
    logging.info(f"Starting data preprocessing from {input_path}")
    df = pd.read_csv(input_path, index_col='datetime', parse_dates=True)

    # Clean data: ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.ffill(inplace=True)

    # Create time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year

    target_column = params['preprocess']['target_column']
    features_to_use = params['train']['features']
    
    # Separate target (y) from features (X) BEFORE lagging
    y = df[[target_column]]
    X = df[features_to_use]

    # Create lagged features to use past data for future predictions
    logging.info("Creating lagged features to prevent data leakage...")
    X_lagged = create_lagged_features(X, lag=1)

    # Combine target with lagged features and drop rows with NaN values
    full_df = pd.concat([y, X_lagged], axis=1)
    full_df.dropna(inplace=True)
    
    y = full_df[[target_column]]
    X = full_df.drop(columns=[target_column])

    # Chronological data split
    test_split_ratio = params['preprocess']['test_split_ratio']
    test_size = int(len(full_df) * test_split_ratio)
    
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'))
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'))
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'))
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'))

    logging.info(f"Data preprocessing complete. Output saved to '{output_dir}'")
    logging.info(f"Shapes after lagging -> X_train: {X_train.shape}, y_train: {y_train.shape}")

def load_raw_file(input_path: str) -> pd.DataFrame:
    """Attempts to load the raw dataset in a robust way.

    Supports the original .csv layout as well as the semicolon-delimited
    `household_power_consumption.txt` which contains separate Date and Time
    columns. Returns a DataFrame indexed by a datetime index named 'datetime'.
    """
    # Try common CSV pattern first
    try:
        if input_path.endswith('.txt'):
            # The original dataset uses ';' as separator and has Date + Time
            df = pd.read_csv(input_path, sep=';', low_memory=False)
            if 'Date' in df.columns and 'Time' in df.columns:
                # Date format in this dataset is usually day/month/year
                df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), dayfirst=True, errors='coerce')
                df.set_index('datetime', inplace=True)
                df.drop(['Date', 'Time'], axis=1, inplace=True, errors='ignore')
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                df.set_index('datetime', inplace=True)
            else:
                # Fallback: attempt to coerce the first column or index to datetime
                try:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                except Exception:
                    pass
        else:
            df = pd.read_csv(input_path, index_col='datetime', parse_dates=True)
    except Exception:
        # Last-resort attempt: try automatic delimiter detection
        df = pd.read_csv(input_path, sep=None, engine='python', low_memory=False)
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), dayfirst=True, errors='coerce')
            df.set_index('datetime', inplace=True)
            df.drop(['Date', 'Time'], axis=1, inplace=True, errors='ignore')

    # Ensure index is datetime-like
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
        except Exception:
            pass

    # Rename index to 'datetime' if not already
    df.index.name = 'datetime'
    return df


if __name__ == "__main__":
    params = load_params()
    # Use the actual raw filename present in the repo. This loader is robust
    # and will handle both the .txt semicolon-delimited original dataset and
    # CSV variants if present.
    input_file = 'data/raw/household_power_consumption.txt'
    output_directory = 'data/processed'
    # Load raw data robustly
    raw_df = load_raw_file(input_file)

    # Write the raw dataframe to a temporary CSV so the existing pipeline
    # and downstream code that expects a CSV-like structure will continue to work.
    # We pass this DataFrame through the same preprocessing function by
    # saving it to a small temp file path and letting preprocess_data read it.
    tmp_raw_path = os.path.join('data', 'raw', 'household_power_consumption_prepared.csv')
    os.makedirs(os.path.dirname(tmp_raw_path), exist_ok=True)
    raw_df.to_csv(tmp_raw_path)

    preprocess_data(tmp_raw_path, output_directory, params)