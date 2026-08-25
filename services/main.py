import os
import joblib
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from google.cloud import storage

app = Flask(__name__)

# GCS Bucket Configuration
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "ainy-fin-models")
MODEL_FILENAME = "xgboost_bhs_model.joblib"
LOCAL_MODEL_PATH = f"/tmp/{MODEL_FILENAME}"

loaded_model = None

def download_model_from_gcs():
    """Downloads the latest model file from GCS to local temp storage."""
    global loaded_model
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILENAME)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        loaded_model = joblib.load(LOCAL_MODEL_PATH)
        print(f"Successfully loaded {MODEL_FILENAME} from GCS bucket '{BUCKET_NAME}'.")
        return True
    except Exception as e:
        print(f"Error loading model from GCS: {e}")
        return False

# Attempt initial model download on container boot
download_model_from_gcs()

FEATURES = [
    'Close', 'Volume', 'targetMedianPrice', 'beta', 
    'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'shortRatio', 
    'epsForward', 'forwardPE', 'pegRatio', 'revenueGrowth',
    'dividendYield', 'fiftyDayAverage', 'averageAnalystRating_float'
]

BHS_DESCS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "model_loaded": loaded_model is not None
    }), 200

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """
    Call this webhook from your weekly training job after uploading a new model to GCS.
    """
    success = download_model_from_gcs()
    if success:
        return jsonify({"status": "success", "message": "Model reloaded successfully."}), 200
    return jsonify({"status": "error", "message": "Failed to reload model from GCS."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts tickers to evaluate live via Yahoo Finance feature extraction.
    Payload format: {"tickers": ["AAPL", "AMZN", "MU"]}
    """
    if loaded_model is None:
        return jsonify({"error": "No model loaded into memory."}), 503

    data = request.get_json() or {}
    orgs = data.get("tickers", ["AAPL", "AMZN", "MU"])

    try:
        # Extract features
        tickers = yf.Tickers(orgs)
        metrics_df_list = []
        for org in orgs:
            info = tickers.tickers[org].info
            metrics_df_list.append(pd.DataFrame([info]).fillna(0))

        current_metrics_df = pd.concat(metrics_df_list, ignore_index=True)
        current_metrics_df.rename(columns={"previousClose": "Close", "volume": "Volume"}, inplace=True)
        
        # Format ratings
        current_metrics_df['averageAnalystRating_float'] = pd.to_numeric(
            current_metrics_df['averageAnalystRating'].astype(str).str.split('-').str[0], 
            errors='coerce'
        ).fillna(0.0)

        # Align input features
        input_df = current_metrics_df[FEATURES].fillna(0)

        # Predict
        out_pred = loaded_model.predict(input_df)
        
        results = []
        for ticker, score in zip(orgs, out_pred):
            # Class index mapping safely (0-indexed to descriptions)
            idx = max(0, min(int(score), len(BHS_DESCS) - 1))
            results.append({
                "symbol": ticker,
                "score_class": int(score),
                "recommendation": BHS_DESCS[idx]
            })

        return jsonify({"status": "success", "predictions": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
