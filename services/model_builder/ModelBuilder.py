import os
from datetime import datetime

import joblib
import pandas as pd
import requests
from google.cloud import storage
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Environment variables configured in Cloud Run
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "ainy-fin-models")
APP_ENGINE_URL = os.environ.get("APP_ENGINE_URL", "https://your-app-id.appspot.com")
MODEL_FILENAME = "xgboost_bhs_model.joblib"

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class ModelBuilder:
    def __init__(self, builder: str, n_estimators: int = 100):
        self.builder = builder
        self.n_estimators = n_estimators
        self.gb_model = None

    def train(self, feature_df):
        target_column = "bhsScore"
        self.training_columns = [
            "bhsScore",
            "Close",
            "Volume",
            "targetMedianPrice",
            "beta",
            "fiftyTwoWeekLow",
            "fiftyTwoWeekHigh",
            "shortRatio",
            "epsForward",
            "forwardPE",
            "pegRatio",
            "revenueGrowth",
            "dividendYield",
            "fiftyDayAverage",
            "averageAnalystRating_float",
        ]
        df = feature_df[self.training_columns]
        # print(target_column,":", feature_df[target_column])
        # print("feature_df:", df.shape, feature_df.columns)

        # 7. Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 8. Split data for training
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("final_df y_train:", feature_df[target_column].unique())

        # 9. Initialize Gradient Boosting Classifier
        # Multi-class uses 'multinomial' deviance loss automatically
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42
        )

        # 10. Train the model
        self.gb_model.fit(X_train, y_train)
        print("\ngb_model classes:", self.gb_model.n_classes_, self.gb_model.classes_)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        self.xg_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            objective="multi:softprob",
        )

        self.xg_model.fit(X_train, y_train)
        print("\nxg_model classes:", self.xg_model.n_classes_, self.xg_model.classes_)

        return True

    def test(self):
        # 1. Input
        input = {
            "Company": ["ICE", "MANH", "AAPL", "AMZN"],
            "Close": [],
            "Volume": [],
            "targetMedianPrice": [],
            "beta": [1.086, 1.444, 1.229],
            "fiftyTwoWeekLow": [18.5, 22.1, 15.2],
            "fiftyTwoWeekHigh": [2.5, 3.1, 1.8],
            "shortRatio": [2.8, 3.3, 2.0],
            "epsForward": [16.1, 19.5, 14.0],
            "forwardPE": [4.2, 3.8, 4.5],
            "pegRatio": [2.40, 1.83, 0.81],
            "revenueGrowth": [0.166, 0.166, 0.331],
            "dividendYield": [0.37, 0.00, 0.37],
            "fiftyDayAverage": [290.1468, 256.997, 619.2546],
            "averageAnalystRating_float": [2.0, 1.3, 1.3],
        }

        est_tz = timezone(timedelta(hours=-5))
        current_time = datetime.now(est_tz)
        # Create a target time object for 4:00 PM today in EST
        target_time = datetime(
            current_time.year,
            current_time.month,
            current_time.day,
            16,
            0,
            0,
            tzinfo=est_tz,
        )

        # Check if current time is greater than 4 PM EST
        if current_time > target_time:
            test_date = date.today() - timedelta(days=1)
        else:
            test_date = date.today() - timedelta(days=2)

        print("test_date", test_date)
        orgs = ["AAPL", "AMZN", "MU"]
        tickers = yf.Tickers(orgs)
        metrics_df_list = []
        # metrics_data = tickers.history(period=None,start=test_date, end=date.today())
        # print("metrics_data:",metrics_data.columns,metrics_data)
        for org in orgs:
            metrics_df_list.append(pd.DataFrame([tickers.tickers[org].info]).fillna(0))

        current_metrics_df = pd.concat(metrics_df_list, ignore_index=True)
        current_metrics_df.rename(
            columns={"previousClose": "Close", "volume": "Volume"}, inplace=True
        )
        current_metrics_df["averageAnalystRating_float"] = pd.to_numeric(
            current_metrics_df["averageAnalystRating"].str.split("-").str[0],
            errors="coerce",
        ).astype(float)
        if "bhsScore" in self.training_columns:
            self.training_columns.remove("bhsScore")
        input_df = current_metrics_df[self.training_columns]

        # 2. Load the data into a Pandas DataFrame
        # input_df = pd.DataFrame(input)
        # input_df = input_df.drop(columns=["Company"])
        input_df.fillna(0, inplace=True)

        # 3. Make Test predictions
        y_pred = self.gb_model.predict(self.X_test)
        predictions = [round(value) for value in y_pred]
        # print(f"Test Input: {self.X_test}")
        # print(f"Test y_test: {self.y_test}\n")
        # print(f"Test Predictions: {predictions}\n")

        accuracy = accuracy_score(self.y_test, predictions)
        print("gb_model Accuracy: %.2f%%\n" % (accuracy * 100.0))

        bhs_descs = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]

        # 4. Make Output predictions
        # print(f"Input: {input_df}\n")
        out_pred = self.gb_model.predict(input_df)
        print(f"BHS out_pred: {out_pred}")
        bhs_desc = [bhs_descs[value - 1] for value in out_pred]
        print(f"BHS: {bhs_desc}")

        # 5. Make Test predictions
        y_pred = self.xg_model.predict(self.X_test)
        predictions = [round(value) for value in y_pred]
        # print(f"Test Predictions: {predictions}\n")

        le = LabelEncoder()
        y_test_encoded = le.fit_transform(self.y_test)
        accuracy = accuracy_score(y_test_encoded, predictions)
        print("xg_model Accuracy: %.2f%%\n" % (accuracy * 100.0))

        # 6. Make Output predictions
        out_pred = self.xg_model.predict(input_df) + 1
        print(f"BHS out_pred: {out_pred}")
        bhs_desc = [bhs_descs[value] for value in out_pred]
        print(f"BHS: {bhs_desc}")


class ModelTrainer:
    def __init__(self):
        self.training_columns = [
            "bhsScore",
            "Close",
            "Volume",
            "targetMedianPrice",
            "beta",
            "fiftyTwoWeekLow",
            "fiftyTwoWeekHigh",
            "shortRatio",
            "epsForward",
            "forwardPE",
            "pegRatio",
            "revenueGrowth",
            "dividendYield",
            "fiftyDayAverage",
            "averageAnalystRating_float",
        ]

    def load_historical_data(self) -> pd.DataFrame:
        """
        Fetch your historical training data from BigQuery, Cloud Storage, or APIs.
        """
        print("Fetching historical training feature dataset...")
        # Placeholder: Load from your data source
        # feature_df = pd.read_parquet("gs://ainy-fin-data/historical_features.parquet")
        return pd.DataFrame()

    def train_and_export(self, feature_df: pd.DataFrame):
        df = feature_df[self.training_columns].dropna()
        X = df.drop(columns=["bhsScore"])
        y = df["bhsScore"] - 1  # Shift 1-5 rating to 0-4 for XGBoost

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training XGBoost Model on {len(X_train)} samples...")
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            objective="multi:softprob",
        )
        model.fit(X_train, y_train)

        # 1. Save locally to ephemeral container storage
        local_path = f"/tmp/{MODEL_FILENAME}"
        joblib.dump(model, local_path)
        print("Model saved locally.")

        # 2. Upload to Google Cloud Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILENAME)
        blob.upload_from_filename(local_path)
        print(f"Successfully uploaded model to gs://{BUCKET_NAME}/{MODEL_FILENAME}")

        # 3. Notify App Engine to reload the model from GCS
        try:
            reload_endpoint = f"{APP_ENGINE_URL}/reload-model"
            resp = requests.post(reload_endpoint, timeout=10)
            print(f"App Engine notification status: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Failed to notify App Engine service: {e}")


def main():
    print(f"=== Starting Weekly Training Job Execution: {datetime.utcnow()} ===")
    trainer = ModelTrainer()
    feature_df = trainer.load_historical_data()

    if not feature_df.empty:
        trainer.train_and_export(feature_df)
        print("=== Training Job Completed Successfully ===")
    else:
        print("Warning: Feature DataFrame was empty. Aborting training.")


if __name__ == "__main__":
    main()
