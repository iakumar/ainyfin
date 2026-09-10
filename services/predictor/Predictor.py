import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import yfinance as yf
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from services.consts.AinySchema import AinySchema
from services.model_builder.ModelBuilder import ModelBuilder

BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "anypug.appspot.com")
DIRECTORY_NAME = "ainyfin/models"
APP_ENGINE_URL = os.environ.get(
    "APP_ENGINE_URL", "https://ainyfin.appspot.com"
)
GBMODEL_FILENAME = "gboost_bhs_model.joblib"
XGBMODEL_FILENAME = "xgboost_bhs_model.joblib"

class Predictor:

    def __init__(self, builder: str, n_estimators: int = 100):
        self.target_column = "bhsScore"

    def init_runtime(self):
        local_path = f"/tmp/{XGBMODEL_FILENAME}"
        print("Loading Model:",  local_path)
        self.xg_model = joblib.load(local_path)
        print("Model running:",  self.xg_model)

        # 1. Get raw normalized importance scores
        importances = self.xg_model.feature_importances_

        # 2. Map scores to feature names in a clean DataFrame
        feature_imp_df = pd.DataFrame({
            'Feature': self.xg_model.feature_names_in_,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        #print(feature_imp_df)



    def create_input(self, tickers):
        #1 get current price
        price_df = yf.download(tickers, period="1d", progress=False)
        if not price_df.empty:
            # Drop columns where all values are NaN (failed tickers)
            price_df = price_df.dropna(how="all", axis=1)
            price_df = price_df.stack(level=1, future_stack=True).reset_index()
        else:
            print("No price data found for tickers:", tickers)
            return pd.DataFrame()  # Return an empty DataFrame if no data is found

        print("price_df tickers:", price_df['Ticker'].unique())

        price_df["Date"] = pd.to_datetime(price_df["Date"]).astype("datetime64[ns]")
        #print("price_df:\n", price_df)

        #need to filter by tickers
        financial_df = pd.read_csv(AinySchema.DATA_DIR+'/financial_data.csv')
        financial_df["Date"] = pd.to_datetime(financial_df["Date"]).astype("datetime64[ns]")
        #print("financial_df:\n", financial_df)

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
        input_columns = [col for col in modelBuilder.training_columns if col != "bhsScore"]
        merged_df[input_columns + ['Ticker']].to_csv(AinySchema.DATA_DIR+'/testdata.csv')
        print("create_input final-df:\n",  input_columns + ['Ticker'])
        print("create_input final-df:\n",  merged_df[input_columns + ['Ticker']])
        return merged_df[input_columns + ['Ticker']]


    def predict(self, tickerlist, input_df):
        out_pred = self.xg_model.predict(input_df)
        out_pred_desc = [AinySchema.BHS_DESCS[value] for value in out_pred]
        # Map ticker to description into a dict
        ticker_bhs_map = dict(zip(tickerlist, out_pred_desc))
        print(f"BHS Predictions: {ticker_bhs_map}")

        # Or print line-by-line
        #for ticker, desc in zip(tickerlist, bhs_desc):
        #  print(f"{ticker}: {desc}")
        return ticker_bhs_map

    def predict_with_explanations(self, tickerlist:list[str], df: pd.DataFrame, top_k_reasons: int = 4) -> dict:
        """Predicts BHS signals and extracts the top fundamental drivers for each signal.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame processed with features matching the trained model.
        top_k_reasons : int
            Number of top contributing features to output per ticker.

        Returns:
        --------
        dict
            Dictionary mapping Tickers to their signal and primary driving reasons.
        """
        # 1. Prepare features for prediction
        #
        modelBuilder = ModelBuilder("xg",100)
        input_columns = [col for col in modelBuilder.training_columns if col != "bhsScore"]
        X = df[input_columns].copy()
        tickers = tickerlist

        # 2. Get class probabilities and predicted indices
        preds_proba = self.xg_model.predict_proba(X)
        preds = self.xg_model.predict(X)

        # Map target encoding back to string labels (e.g., {0: 'Buy', 1: 'Hold', 2: 'Sell'})
        label_map = {0: "Buy", 1: "Hold", 2: "Sell"}
        signals = [label_map[p] for p in preds]

        # 3. Compute SHAP values for feature attribution
        explainer = shap.TreeExplainer(self.xg_model)
        shap_values = explainer.shap_values(X)

        results = {}
        for i, ticker in enumerate(tickers):
            pred_class_idx = preds[i]
            signal = signals[i]

            # Extract exact probability for the predicted class
            confidence = preds_proba[i][pred_class_idx]

            # Extract full class probabilities dictionary
            prob_distribution = {
                label_map[j]: round(float(preds_proba[i][j]), 4)
                for j in range(len(label_map))
            }

            # Handle SHAP multi-class output (list of arrays or 3D array)
            if isinstance(shap_values, list):
                sample_shap = shap_values[pred_class_idx][i]
            else:  # shape: (n_samples, n_features, n_classes)
                sample_shap = shap_values[i, :, pred_class_idx]

            # Rank features by absolute SHAP impact for this prediction
            top_indices = np.argsort(np.abs(sample_shap))[::-1][:top_k_reasons]

            p_reasons = []
            n_reasons = []
            for idx in top_indices:
                feat_name = input_columns[idx]
                feat_val = X.iloc[i, idx]
                shap_impact = sample_shap[idx]

                if shap_impact > 0:
                    p_reasons.append(f"{feat_name} ({feat_val:.4f})")
                else:
                    n_reasons.append(f"{feat_name} ({feat_val:.4f})")

            #results[ticker] = {"Signal": signal, "Primary_Reasons": reasons}
            results[ticker] = {"Signal": signal,
                               "Confidence": f"{confidence:.1%}",
                               "Probabilities": prob_distribution,
                               "P_Reasons": p_reasons,
                               "N_Reasons": n_reasons
                               }

        return results


def main(args:list):
    if len(args) < 1:
        print("Ticker list needed")
        return

    predictor = Predictor("xg",100)
    tickerlist:list[str] = [ticker.strip() for ticker in args[0].split(",")]
    input_df = predictor.create_input(tickerlist)
    if input_df.empty == False:
        predictor.init_runtime()
        valid_tickers = input_df['Ticker'].tolist()
        #results = predictor.predict(valid_tickers, input_df)
        results = predictor.predict_with_explanations(valid_tickers, input_df)
        for ticker, info in results.items():
            p_reasons_str = ", ".join(info["P_Reasons"])
            n_reasons_str = ", ".join(info["N_Reasons"])
            print(f"{ticker}: {info['Signal']} | {info['Confidence']} | {info['Probabilities']} | Supported by: {p_reasons_str} | Weakened by: {n_reasons_str}")
    else:
        print("No input data available for tickers.")

if __name__ == "__main__":
    main(sys.argv[1:])
