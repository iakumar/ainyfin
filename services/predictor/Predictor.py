import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import yfinance as yf

from services.consts.AinySchema import AinySchema
from services.model_builder.ModelBuilder_CL import ModelBuilder

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

    def init_runtime(self, modelBuilder:ModelBuilder):
        """
        local_path = f"/tmp/{XGBMODEL_FILENAME}"
        print("Loading Model:",  local_path)
        self.xg_model = joblib.load(local_path)
        """

        self.xg_model = modelBuilder.load_model()
        print("Model Running:",  self.xg_model)

        # 1. Get raw normalized importance scores
        importances = self.xg_model.feature_importances_

        # 2. Map scores to feature names in a clean DataFrame
        feature_imp_df = pd.DataFrame({
            'Feature': self.xg_model.feature_names_in_,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print(feature_imp_df)


    def create_input(self, modelBuilder:ModelBuilder, tickers):
        # 1. Fetch Current Live Market Prices
        new_price_df = yf.download(tickers, period="3d", progress=False)
        if not new_price_df.empty:
            # Drop columns where all values are NaN (failed tickers)
            new_price_df = new_price_df.dropna(how="all", axis=1)
            new_price_df = new_price_df.stack(level=1, future_stack=True).reset_index()
        else:
            print("No price data found for tickers:", tickers)
            return pd.DataFrame()

        ticker_sic_df = pd.read_csv(AinySchema.DATA_DIR + "ticker_sic.csv")
        new_price_df = new_price_df.merge(ticker_sic_df[["Ticker", "SIC"]], on="Ticker", how="left")

        old_price_df = pd.read_csv(AinySchema.DATA_DIR + "ainyfin_data.csv")
        old_price_df = old_price_df.merge(ticker_sic_df[["Ticker", "SIC"]], on="Ticker", how="left")
        sic_list = old_price_df["SIC"].unique().tolist()
        #load old prices within the sectors (SIC)
        old_price_df = old_price_df[old_price_df["SIC"].isin(sic_list)].copy()
        old_price_df['Date'] = pd.to_datetime(old_price_df['Date'])
        latest_date = old_price_df['Date'].max()
        latest_year_df = old_price_df[old_price_df['Date'] >= (latest_date - pd.Timedelta(days=365))]
        price_df = pd.concat([new_price_df,latest_year_df])

        #print("price_df tickers:", price_df['Ticker'].unique().tolist())
        price_df["Date"] = pd.to_datetime(price_df["Date"]).astype("datetime64[ns]")
        price_df = price_df.sort_values("Date").reset_index(drop=True)

        # 2. Load Full Historical Financial Dataset
        full_financial_df = pd.read_csv(AinySchema.DATA_DIR + "financial_data.csv")
        full_financial_df["Date"] = pd.to_datetime(full_financial_df["Date"]).astype("datetime64[ns]")

        # Filter for target tickers
        if full_financial_df["Ticker"].isin(price_df["Ticker"]).empty:
            print("No financial data found for tickers:", price_df["Ticker"].tolist())
            return pd.DataFrame()

        merged_df = modelBuilder.build_features(price_df, financial_data=full_financial_df)

        if merged_df.empty or merged_df["Close"].isna().all():
            print("No matching financial data found for target tickers.")
            return pd.DataFrame()

        #print("merged_df:", merged_df.columns.tolist())
        expected_features =  self.xg_model.get_booster().feature_names
        print("expected_features:", expected_features)
        missing_columns = [col for col in expected_features if col not in merged_df.columns]
        print("Missing columns:", missing_columns)
        if len(missing_columns) > 0:
            print("Warning: Some expected training columns are missing in the final merged DataFrame.")
            return pd.DataFrame()  # Return an empty DataFrame if critical columns are missing

        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        largest_date_rows:pd.DataFrame = merged_df[merged_df['Date'] == merged_df['Date'].max()]
        largest_date_rows = largest_date_rows[largest_date_rows["Ticker"].isin(tickers)].copy()

        # Save artifact for debugging
        final_df = largest_date_rows[expected_features + ["Ticker"]].reset_index(drop=True)
        final_df.to_csv(AinySchema.DATA_DIR + "testdata.csv", index=False)
        print("Final input data generated successfully:\n", final_df)
        return final_df


    def predict(self, tickerlist, input_df):
        out_pred = self.xg_model.predict(input_df)
        out_pred_desc = [AinySchema.BHS_DESCS[value] for value in out_pred]
        # Map ticker to description into a dict
        ticker_bhs_map = dict(zip(tickerlist, out_pred_desc))
        print(f"BHS Predictions: {ticker_bhs_map}")

        #Or print line-by-line
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
        """
        Predicts BHS signals, applies conviction filtering, and extracts SHAP drivers.

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
        input_columns =  self.xg_model.get_booster().feature_names
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
                actionable_signal = "NO_TRADE(Hold)"
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
    modelBuilder = ModelBuilder("xg")
    predictor.init_runtime(modelBuilder)

    tickerlist:list[str] = [ticker.strip() for ticker in args[0].split(",")]
    input_df = predictor.create_input(modelBuilder, tickerlist)
    if input_df.empty == False:
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
