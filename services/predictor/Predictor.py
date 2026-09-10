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

        print("price_df tickers:", price_df['Ticker'].unique().tolist())

        price_df["Date"] = pd.to_datetime(price_df["Date"]).astype("datetime64[ns]")
        #print("price_df:\n", price_df)

        #need to filter by tickers
        financial_df = pd.read_csv(AinySchema.DATA_DIR+'/financial_data.csv')
        financial_df = financial_df[financial_df['Ticker'].isin(price_df['Ticker'])]
        if financial_df.empty:
            print("No financial data found for tickers:", price_df['Ticker'].unique().tolist())
            return pd.DataFrame()  # Return an empty DataFrame if no data is found

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


    def predict_with_explanations(
        self,
        tickerlist: list[str],
        df: pd.DataFrame,
        top_k_reasons: int = 5,
        min_confidence: float = 0.40,
        min_margin: float = 0.05,
    ) -> dict:
        """Predicts BHS signals, applies conviction filtering, and extracts SHAP drivers.

        Parameters:
        -----------
        tickerlist : list[str]
            List of ticker symbols matching the rows of df.
        df : pd.DataFrame
            Processed dataframe containing feature columns.
        top_k_reasons : int
            Number of primary feature drivers to extract per ticker.
        min_confidence : float
            Minimum probability required for the top prediction (default 0.40).
        min_margin : float
            Minimum gap required between top and 2nd class probability (default 0.05).

        Returns:
        --------
        dict
            Structured output containing filtered actions, confidence, margin, and SHAP drivers.
        """
        # 1. Align features
        modelBuilder = ModelBuilder("xg", 100)
        input_columns = [
            col for col in modelBuilder.training_columns if col != "bhsScore"
        ]
        X = df[input_columns].copy()

        # 2. Inference
        preds_proba = self.xg_model.predict_proba(X)
        preds = self.xg_model.predict(X)
        label_map = {0: "Buy", 1: "Hold", 2: "Sell"}

        # 3. Compute SHAP values
        explainer = shap.TreeExplainer(self.xg_model)
        shap_output = explainer(X) if hasattr(explainer, "__call__") else explainer.shap_values(X)

        results = {}
        for i, ticker in enumerate(tickerlist):
            raw_pred_idx = preds[i]
            raw_signal = label_map[raw_pred_idx]

            # Extract full probability distribution
            prob_distribution = {
                label_map[j]: round(float(preds_proba[i][j]), 4)
                for j in range(len(label_map))
            }

            # Calculate signal margin
            sorted_probs = sorted(
                prob_distribution.values(), reverse=True
            )
            top_prob = sorted_probs[0]
            runner_up_prob = sorted_probs[1]
            margin = top_prob - runner_up_prob

            # Apply conviction threshold logic
            passes_confidence = top_prob >= min_confidence
            passes_margin = margin >= min_margin

            if passes_confidence and passes_margin:
                actionable_signal = raw_signal
                status = "ACTIONABLE"
            else:
                actionable_signal = "NO_TRADE (Hold Cash)"
                status = "FILTERED (Low Conviction)"

            # Extract SHAP array for raw predicted class
            if isinstance(shap_output, shap.Explanation):
                # shap 0.40+ Explanation object
                sample_shap = shap_output.values[i, :, raw_pred_idx]
            elif isinstance(shap_output, list):
                # Classic list of 2D arrays [n_samples, n_features] per class
                sample_shap = shap_output[raw_pred_idx][i]
            else:
                # 3D array [n_samples, n_features, n_classes]
                sample_shap = shap_output[i, :, raw_pred_idx]

            # Rank features by absolute impact
            top_indices = np.argsort(np.abs(sample_shap))[::-1][:top_k_reasons]

            p_reasons = []
            n_reasons = []
            for idx in top_indices:
                feat_name = input_columns[idx]
                feat_val = X.iloc[i, idx]
                shap_impact = sample_shap[idx]

                formatted_reason = f"{feat_name} ({feat_val:.4f})"
                if shap_impact > 0:
                    p_reasons.append(formatted_reason)
                else:
                    n_reasons.append(formatted_reason)

            # Store output payload
            results[ticker] = {
                "Action": actionable_signal,
                "Raw_Signal": raw_signal,
                "Status": status,
                "Confidence": f"{top_prob:.1%}",
                "Margin": f"{margin:.1%}",
                "Probabilities": prob_distribution,
                "Supported_By": p_reasons,
                "Weakened_By": n_reasons,
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
            supp_str = ", ".join(info["Supported_By"]) if info["Supported_By"] else ""
            weak_str = ", ".join(info["Weakened_By"]) if info["Weakened_By"] else ""

            print(
                f"{ticker}: {info['Action']} | {info['Confidence']} | "
                f"{info['Probabilities']} | Supported by: {supp_str} | Weakened by: {weak_str}"
            )
    else:
        print("No input data available for tickers.")

if __name__ == "__main__":
    main(sys.argv[1:])
