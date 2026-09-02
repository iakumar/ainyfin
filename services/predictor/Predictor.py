import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import requests
import yfinance as yf
from google.cloud import storage
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from ..model_builder.ModelBuilder import ModelBuilder


BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "anypug.appspot.com")
DIRECTORY_NAME = "ainyfin/models"
APP_ENGINE_URL = os.environ.get(
    "APP_ENGINE_URL", "https://ainyfin.appspot.com"
)
GBMODEL_FILENAME = "gboost_bhs_model.joblib"
XGBMODEL_FILENAME = "xgboost_bhs_model.joblib"

class Predictor:
    DATA_DIR = "/Users/rithuhegde/ainyfin/services/data"

    def __init__(self, builder: str, n_estimators: int = 100):
        self.target_column = "bhsScore"
        self.bhs_descs = ["Sell", "Hold", "Buy"]

    def init_runtime(self):
        local_path = f"/tmp/{XGBMODEL_FILENAME}"
        self.xg_model = joblib.load(local_path)
        print("Model running:",  self.xg_model)

        # 1. Get raw normalized importance scores
        importances = self.xg_model.feature_importances_

        # 2. Map scores to feature names in a clean DataFrame
        feature_imp_df = pd.DataFrame({
            'Feature': self.xg_model.feature_names_in_,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print(feature_imp_df)



    def create_input(self, tickers):
        price_df = yf.download(tickers, period="1d").stack(level=1).reset_index()
        price_df["Date"] = pd.to_datetime(price_df["Date"]).astype("datetime64[ns]")
        print(price_df)

        #need to folter by tickers
        metrics_df = pd.read_csv(self.DATA_DIR+'/financial_data.csv')
        metrics_df["Date"] = pd.to_datetime(metrics_df["Date"]).astype("datetime64[ns]")
        print("metrics_df:\n", metrics_df)

        price_df = price_df.sort_values("Date").reset_index(drop=True)
        metrics_df = metrics_df.sort_values("Date").reset_index(drop=True)

        # 6. Merge historical data and fundamental info together
        merged_df = pd.merge_asof(price_df, metrics_df, left_on='Date',  right_on='Date',
                                  by='Ticker',    direction='backward')

        #print("metrics_df after price:\n",  merged_df)

        zero_fill_cols = [
            "ResearchAndDevelopment",
            "InterestExpense",
            "InterestIncome",
            "TotalUnusualItems",
            "Goodwill",
            "Receivables",
            "Inventory",
            "NetPPE",
            "StockBasedCompensation",
            "CommonStockDividendPaid",
            "RepurchaseOfCapitalStock",
        ]
        for col in zero_fill_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)
            else:
                merged_df[col] = 0

        modelBuilder = ModelBuilder("xg",100)
        merged_df = modelBuilder.compute_financial_snapshot(merged_df)
        #print("metrics_df after compute_financial_snapshot:\n",  merged_df)

        merged_df.fillna(0, inplace=True)
        print("create_input final-df:\n",  merged_df)
        input_columns = [col for col in self.training_columns if col != "bhsScore"]
        merged_df[input_columns + ['Ticker']].to_csv(ModelBuilder.DATA_DIR+'/testdata.csv')
        return merged_df[input_columns]


    def predict(self, tickerlist, input_df):
        out_pred = self.xg_model.predict(input_df)
        bhs_desc = [self.bhs_descs[value] for value in out_pred]
        # Map ticker to description into a dict
        ticker_bhs_map = dict(zip(tickerlist, bhs_desc))
        print(f"BHS Predictions: {ticker_bhs_map}")

        # Or print line-by-line
        #for ticker, desc in zip(tickerlist, bhs_desc):
        #  print(f"{ticker}: {desc}")

def main(args:list):
    if len(args) < 1:
        print("Ticker list needed")
        return

    predictor = Predictor("xg",100)
    predictor.init_runtime()
    tickerlist = [ticker.strip() for ticker in args[0].split(",")]
    input_df = predictor.create_input(tickerlist)
    predictor.predict(tickerlist, input_df)


if __name__ == "__main__":
    main(sys.argv[1:])
