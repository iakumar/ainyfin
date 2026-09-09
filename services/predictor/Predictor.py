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

from services.model_builder.ModelBuilder import ModelBuilder


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
        #1 get current price
        price_df = yf.download(tickers, period="1d").stack(level=1).reset_index()
        price_df["Date"] = pd.to_datetime(price_df["Date"]).astype("datetime64[ns]")
        print(price_df)

        #need to filter by tickers
        financial_df = pd.read_csv(self.DATA_DIR+'/financial_data.csv')
        financial_df["Date"] = pd.to_datetime(financial_df["Date"]).astype("datetime64[ns]")
        print("financial_df:\n", financial_df)

        price_df = price_df.sort_values("Date").reset_index(drop=True)
        financial_df = financial_df.sort_values("Date").reset_index(drop=True)

        # 2. Merge historical data and fundamental info together
        merged_df = pd.merge_asof(price_df, financial_df, left_on='Date',  right_on='Date',
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

        existing_cols = [c for c in zero_fill_cols if c in merged_df.columns]
        if existing_cols:
            merged_df[existing_cols] = merged_df[existing_cols].fillna(0)

        missing_cols = [c for c in zero_fill_cols if c not in merged_df.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=merged_df.index, columns=missing_cols, dtype=float)
            merged_df = pd.concat([merged_df, missing_df], axis=1)

        modelBuilder = ModelBuilder("xg",100)
        snapshot_df = modelBuilder.compute_financial_snapshot(merged_df)
        snapshot_df = snapshot_df.sort_values(['Date']).reset_index(drop=True)
        #print("metrics_df after compute_financial_snapshot:\n",  snapshot_df)

        financial_trends_df = modelBuilder.compute_financial_trends(financial_df)
        financial_trends_df = financial_trends_df.sort_values(['Date']).reset_index(drop=True)

        merged_df = pd.merge_asof(snapshot_df, financial_trends_df, left_on='Date', right_on='Date',
                                  by='Ticker', direction='backward', suffixes=('', '_Trends'))

        merged_df.fillna(0, inplace=True)
        print("create_input final-df:\n",  merged_df)
        input_columns = [col for col in modelBuilder.training_columns if col != "bhsScore"]
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
