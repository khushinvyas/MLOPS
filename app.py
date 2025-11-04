# app.py

import joblib
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime
import logging
import os

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:  # boto3 optional for local-only runs
    boto3 = None
    BotoCoreError = ClientError = Exception

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask application
app = Flask(__name__)

# --- 1. Define models and their paths ---
# This dictionary maps the user-friendly name (for the dropdown) to the model file.
MODELS = {
    'XGBoost Regressor': 'models/xgb_model.pkl',
    'Random Forest Regressor': 'models/rf_model.pkl',
    'LightGBM Regressor': 'models/lgbm_model.pkl'
}

# --- 1b. Optionally map S3 keys from environment ---
S3_BUCKET = os.getenv('MODEL_S3_BUCKET')
S3_KEYS = {
    'XGBoost Regressor': os.getenv('XGB_MODEL_KEY'),
    'Random Forest Regressor': os.getenv('RF_MODEL_KEY'),
    'LightGBM Regressor': os.getenv('LGBM_MODEL_KEY'),
}

def download_from_s3_if_needed(local_path: str, s3_key: str) -> None:
    if not S3_BUCKET or not s3_key or os.path.exists(local_path):
        return
    if boto3 is None:
        logging.warning("boto3 not available; cannot download models from S3.")
        return
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logging.info(f"Downloading model from s3://{S3_BUCKET}/{s3_key} -> {local_path}")
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, s3_key, local_path)
    except (BotoCoreError, ClientError) as e:
        logging.warning(f"Failed to download {s3_key} from S3: {e}")

# --- 2. Load all models at startup ---
LOADED_MODELS = {}
for name, path in MODELS.items():
    try:
        # Try to fetch from S3 first if missing
        download_from_s3_if_needed(path, S3_KEYS.get(name))
        LOADED_MODELS[name] = joblib.load(path)
        logging.info(f"Successfully loaded model: {name} from {path}")
    except FileNotFoundError:
        logging.warning(f"Warning: Model file not found at {path}. This model will be unavailable.")
    except Exception as e:
        logging.warning(f"Warning: Failed to load model '{name}' from {path}: {e}. Skipping.")

# This is the order of features the models were trained on (from params.yaml)
FEATURE_ORDER = [
    'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1',
    'Sub_metering_2', 'Sub_metering_3', 'hour_of_day', 'day_of_week', 'month', 'year'
]

# --- 3. Define the web routes ---

@app.route('/')
def home():
    """Renders the main page with the form and model dropdown."""
    # Pass the list of available (successfully loaded) model names to the template
    available_models = list(LOADED_MODELS.keys())
    return render_template('index.html', model_names=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, makes a prediction, and returns the result."""
    available_models = list(LOADED_MODELS.keys())
    try:
        # --- Get user input from the form ---
        form_values = request.form.to_dict()
        
        # --- Get the selected model from the dropdown ---
        selected_model_name = form_values.get('model_choice')
        if not selected_model_name or selected_model_name not in LOADED_MODELS:
            return render_template('index.html', model_names=available_models,
                                   error="Please select a valid, available model.")

        # Retrieve the loaded model object
        model = LOADED_MODELS[selected_model_name]

        # --- Feature Engineering: Create time-based features ---
        datetime_str = form_values.get('datetime')
        if not datetime_str:
            return render_template('index.html', model_names=available_models,
                                   error="Please select a date and time.", selected_model=selected_model_name)
        
        dt_object = datetime.fromisoformat(datetime_str)
        
        features = {
            'hour_of_day': dt_object.hour,
            'day_of_week': dt_object.weekday(),
            'month': dt_object.month,
            'year': dt_object.year
        }

        # --- Assemble the full feature set ---
        for key in ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
            features[key] = float(form_values[key])

        # --- Prepare data for the model ---
        input_df = pd.DataFrame([features], columns=FEATURE_ORDER)

        # --- Make a prediction ---
        prediction = model.predict(input_df)[0]

        # --- Display the result ---
        prediction_text = f"Predicted Energy Consumption: {prediction:.4f} kW (using {selected_model_name})"
        
        # Pass the selected model back to the template to keep the dropdown state
        return render_template('index.html', prediction_text=prediction_text, 
                               model_names=available_models, selected_model=selected_model_name)

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return render_template('index.html', model_names=available_models, 
                               error=f"An error occurred: {e}", selected_model=request.form.get('model_choice'))
# --- Run the application ---
if __name__ == '__main__':
    if not LOADED_MODELS:
        logging.warning("No models loaded; UI will show an error until models are added.")
    app.run(host='0.0.0.0', port=5000, debug=True)