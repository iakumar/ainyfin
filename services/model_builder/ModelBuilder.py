import sys
import os
import typing
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from google.cloud import storage
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "anypug.appspot.com")
DIRECTORY_NAME = "ainyfin/models"
APP_ENGINE_URL = os.environ.get(
    "APP_ENGINE_URL", "https://ainyfin.appspot.com"
)
GBMODEL_FILENAME = "gboost_bhs_model.joblib"
XGBMODEL_FILENAME = "xgboost_bhs_model.joblib"

class ModelBuilder:
    DATA_DIR = "/Users/rithuhegde/ainyfin/services/data"

    def __init__(self, builder: str, n_estimators: int = 100):
        self.target_column = "bhsScore"
        self.bhs_descs = ["Sell", "Hold", "Buy"]
        # Explicit list of generated ratio columns
        self.ratio_columns = [
            "Price_To_Earnings",
            "Price_To_FreeCashFlow",
            "Price_To_Book",
            "Price_To_Sales",
            "EV_To_EBITDA",
            "EV_To_EBIT",
            "Gross_Margin",
            "Operating_Margin",
            "EBITDA_Margin",
            "Net_Margin",
            "FCF_Margin",
            "Return_On_Equity",
            "Return_On_Assets",
            "Debt_To_Equity",
            "Debt_To_Assets",
            "Current_Ratio",
            "Quick_Ratio",
            "Interest_Coverage",
            "Cash_To_Debt",
            "Asset_Turnover",
            "Working_Capital_Turnover",
            "CFO_To_NetIncome",
            "SBC_To_Revenue",
            "CapEx_To_CFO",
            "CapEx_To_Revenue",
            "R_And_D_To_Revenue",
        ]

        self.training_columns = self.ratio_columns +  ["Close", "bhsScore"]
        print("training_columns:", self.training_columns)
        self.builder = builder
        self.n_estimators = n_estimators
        self.xg_model = None


    def load_train_data(self) -> pd.DataFrame:
        """
        Fetch your historical training feature dataset...
        """
        print("Fetching historical training feature dataset...")
        raw_df = pd.read_csv(ModelBuilder.DATA_DIR+'/ainyfina-input.csv')
        feature_df = self.compute_financial_ratios(raw_df).dropna()
        feature_df[['Ticker', "Close", "bhsScore"]] = raw_df[['Ticker', "Close", "bhsScore"]].replace([np.inf, -np.inf], np.nan)
        return feature_df


    def compute_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes standardized valuation, leverage, profitability, and quality ratios
        from raw SEC fundamental and market price data.

        Parameters:
        -----------
        df : pd.DataFrame
            Must contain raw fundamental columns and 'Close' market price.

        Returns:
        --------
        pd.DataFrame
            DataFrame augmented with engineered ratio features.
        """

        # Define helper to prevent Division-by-Zero and np.inf errors
        def safe_divide(
            numerator: pd.Series, denominator: pd.Series
        ) -> pd.Series:
            return np.where(
                (denominator == 0) | (denominator.isna()),
                np.nan,
                numerator / denominator,
            )

        # Create a copy to prevent mutating the input DataFrame
        data = df.copy()

        # 1. Valuation Ratios
        data["Price_To_Earnings"] = safe_divide(data["Close"], data["DilutedEPS"])
        data["Price_To_FreeCashFlow"] = safe_divide(
            data["Close"] * data["DilutedAverageShares"], data["FreeCashFlow"]
        )
        data["Price_To_Book"] = safe_divide(
            data["Close"] * data["DilutedAverageShares"], data["StockholdersEquity"]
        )
        data["Price_To_Sales"] = safe_divide(
            data["Close"] * data["DilutedAverageShares"], data["TotalRevenue"]
        )
        data["EV_To_EBITDA"] = safe_divide(
            (data["Close"] * data["DilutedAverageShares"])
            + data["TotalDebt"]
            - data["CashCashEquivalentsAndShortTermInvestments"],
            data["NormalizedEBITDA"],
        )
        data["EV_To_EBIT"] = safe_divide(
            (data["Close"] * data["DilutedAverageShares"])
            + data["TotalDebt"]
            - data["CashCashEquivalentsAndShortTermInvestments"],
            data["EBIT"],
        )

        # 2. Profitability & Margins
        data["Gross_Margin"] = safe_divide(
            data["GrossProfit"], data["OperatingRevenue"]
        )
        data["Operating_Margin"] = safe_divide(
            data["OperatingIncome"], data["TotalRevenue"]
        )
        data["EBITDA_Margin"] = safe_divide(
            data["NormalizedEBITDA"], data["TotalRevenue"]
        )
        data["Net_Margin"] = safe_divide(
            data["NetIncome"], data["TotalRevenue"]
        )
        data["FCF_Margin"] = safe_divide(
            data["FreeCashFlow"], data["TotalRevenue"]
        )
        data["Return_On_Equity"] = safe_divide(
            data["NetIncomeCommonStockholders"], data["CommonStockEquity"]
        )
        data["Return_On_Assets"] = safe_divide(
            data["NetIncome"], data["TotalAssets"]
        )

        # 3. Leverage, Solvency & Liquidity
        data["Debt_To_Equity"] = safe_divide(
            data["TotalDebt"], data["StockholdersEquity"]
        )
        data["Debt_To_Assets"] = safe_divide(
            data["TotalDebt"], data["TotalAssets"]
        )
        data["Current_Ratio"] = safe_divide(
            data["CurrentAssets"], data["CurrentLiabilities"]
        )
        data["Quick_Ratio"] = safe_divide(
            data["CashCashEquivalentsAndShortTermInvestments"] + data["AccountsReceivable"],
            data["CurrentLiabilities"],
        )
        data["Interest_Coverage"] = safe_divide(
            data["EBIT"], data["InterestExpense"].abs()
        )
        data["Cash_To_Debt"] = safe_divide(
            data["CashCashEquivalentsAndShortTermInvestments"], data["TotalDebt"]
        )

        # 4. Operational Efficiency
        data["Asset_Turnover"] = safe_divide(
            data["TotalRevenue"], data["TotalAssets"]
        )
        data["Working_Capital_Turnover"] = safe_divide(
            data["TotalRevenue"], data["WorkingCapital"]
        )

        # 5. Earnings Quality & Capital Allocation
        data["CFO_To_NetIncome"] = safe_divide(
            data["OperatingCashFlow"], data["NetIncome"]
        )
        data["SBC_To_Revenue"] = safe_divide(
            data["StockBasedCompensation"], data["TotalRevenue"]
        )
        data["CapEx_To_CFO"] = safe_divide(
            data["CapitalExpenditure"].abs(), data["OperatingCashFlow"]
        )
        data["CapEx_To_Revenue"] = safe_divide(
            data["CapitalExpenditure"].abs(), data["TotalRevenue"]
        )
        data["R_And_D_To_Revenue"] = safe_divide(
            data["ResearchAndDevelopment"], data["TotalRevenue"]
        )

        # Clean extreme inf / -inf values on calculated ratio columns only
        data[self.ratio_columns] = data[self.ratio_columns].replace([np.inf, -np.inf], np.nan)

        return data


    def transformFeatureXY(self, X: pd.DataFrame, org: str) -> pd.DataFrame:
        df_transformed = X.T
        # 1. Convert date headers from the index into a dedicated 'Date' column
        df_transformed = df_transformed.reset_index().rename(columns={'index': 'Date'})
        # 2. Ensure Date is datetime type
        df_transformed['Date'] = pd.to_datetime(df_transformed['Date'])
        df_transformed['Ticker'] = org
        return df_transformed


    def load_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        feature_df = self.load_train_data()[self.training_columns].dropna()
        # print(self.target_column,":", feature_df[self.target_column])
        # print("feature_df:", df.shape, feature_df.columns)

        # 7. Separate features and target
        X = feature_df.drop(columns=[self.target_column])
        y = feature_df[self.target_column] - 1  # Shift 1-5 rating to 0-4 for XGBoost

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        return X_test, y_test


    def train(self, feature_df):
        df = feature_df[self.training_columns].dropna()

        # 1. Separate features and target
        X = df.drop(columns=[self.target_column]).dropna()
        y = df[self.target_column] - 1  # Shift 1-3 rating to 0-3 for XGBoost

        n_splits: int = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_predictions = np.zeros(len(df))
        cv_accuracies = []
        cv_f1_scores = []

        print(f"\n=== Starting {n_splits}-Fold Stratified Cross-Validation ===")

        print("final_df y_train:", feature_df[self.target_column].unique())

        # 3. Iterate through folds
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # 9. Initialize Gradient Boosting Classifier
            # Multi-class uses 'multinomial' deviance loss automatically
            """
            fold_model = XGBClassifier(
                n_estimators=self.n_estimators,
                        learning_rate=0.05,
                        max_depth=5,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
            )
            """

            fold_model = XGBClassifier(
                    n_estimators=50,
                    max_depth=3,  # Reduce depth from 5 to 3
                    learning_rate=0.03,
                    subsample=0.8,  # Randomly sample 80% of rows per tree
                    colsample_bytree=0.8,  # Randomly sample 80% of features per tree
                    reg_alpha=1.0,  # L1 regularization
                    reg_lambda=1.0,  # L2 regularization
            )

            # 10. Train the model
            fold_model.fit(X_train_fold, y_train_fold)
            print("fold_model classes:", fold_model.n_classes_, fold_model.classes_)

            # Predict on validation fold
            val_preds = fold_model.predict(X_val_fold)
            oof_predictions[val_idx] = val_preds
            bhs_desc = [self.bhs_descs[value] for value in val_preds]
            print(f"BHS: {bhs_desc}")

            # Metric evaluation
            fold_acc = accuracy_score(y_val_fold, val_preds)
            fold_f1 = f1_score(y_val_fold, val_preds, average="weighted")

            cv_accuracies.append(fold_acc)
            cv_f1_scores.append(fold_f1)

            print(
                f"Fold {fold}/{n_splits} - Accuracy: {fold_acc * 100:.2f}% | Weighted"
                f" F1: {fold_f1:.4f}"
            )

            # 4. Out-of-fold aggregate summary
            mean_acc = np.mean(cv_accuracies)
            std_acc = np.std(cv_accuracies)
            mean_f1 = np.mean(cv_f1_scores)

            print("-" * 50)
            print(f"CV Mean Accuracy: {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")
            print(f"CV Mean Weighted F1: {mean_f1:.4f}")
            print("-" * 50)

        # 5. Retrain final model on 100% of the dataset for production deployment
        print("Retraining final production model on entire dataset...")

        """
        final_xg_model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_depth=5,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        )
        """

        final_xg_model = XGBClassifier(
            n_estimators=50,
            max_depth=3,  # Reduce depth from 5 to 3
            learning_rate=0.03,
            subsample=0.8,  # Randomly sample 80% of rows per tree
            colsample_bytree=0.8,  # Randomly sample 80% of features per tree
            reg_alpha=1.0,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
        )

        final_xg_model.fit(X, y)

        # 6. Save final production model locally
        local_path = f"/tmp/{XGBMODEL_FILENAME}"
        joblib.dump(final_xg_model, local_path)
        print(f"Model successfully saved to {local_path}")

         # 2. Upload to Google Cloud Storage
        #storage_client = storage.Client()
        #bucket = storage_client.bucket(BUCKET_NAME)
        #blob_path = f"{DIRECTORY_NAME}/{MODEL_FILENAME}"
        #blob = bucket.blob(blob_path)
        #blob.upload_from_filename(local_path)
        #print(f"Successfully uploaded model to gs://{blob_path}")

        # 3. Notify App Engine to reload the model from GCS
        try:
            reload_endpoint = f"{APP_ENGINE_URL}/reload-model"
            resp = requests.post(reload_endpoint, timeout=10)
            print(f"App Engine notification status: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Failed to notify App Engine service: {e}")

        return True


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


    def create_input(self, orgs):
        # Check if current time is greater than 4 PM EST
        est_tz = timezone(timedelta(hours=-5))
        current_time = datetime.now(est_tz)
        target_time = datetime(
            current_time.year,
            current_time.month,
            current_time.day,
            16,
            0,
            0,
            tzinfo=est_tz,
        )

        if current_time > target_time:
            predict_date = target_time
        else:
            predict_date = target_time - timedelta(days=1)
        if predict_date.weekday() in {5, 6, 0}:
            weekendoffset =  4 - predict_date.weekday()
            predict_date = target_time + timedelta(days=weekendoffset)

        print("predict_date:", predict_date)
        price_df = yf.download(orgs, start=predict_date, end=predict_date, auto_adjust=True)
        price_df = price_df.stack(level=1).reset_index()
        price_df = price_df.sort_values('Date').reset_index(drop=True)
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        print("price_df\n:",  price_df)

        metrics_df_list = []
        for org in orgs:
            org = org.strip()
            ticker_obj = yf.Ticker(org)
            #2 Financials
            finacials_df = ticker_obj.get_financials(freq='quarterly')
            finacials_df_x = self.transformFeatureXY(finacials_df, org)

            balancesheet = ticker_obj.get_balancesheet(freq='quarterly')
            balancesheet_x = self.transformFeatureXY(balancesheet, org)
            metrics_df = pd.merge(finacials_df_x, balancesheet_x, on=['Ticker','Date'], how='inner')

            cashflow:DataFrame = ticker_obj.get_cashflow(freq='quarterly')
            cashflow_x = self.transformFeatureXY(cashflow, org)
            metrics_df = pd.merge(metrics_df, cashflow_x, on=['Ticker','Date'], how='inner')

            metrics_df = metrics_df.sort_values("Date").reset_index(drop=True)

            actions = ticker_obj.get_actions(period="max").reset_index().rename(columns={'index': 'Date'})
            actions['Ticker'] = org
            actions['Date'] = pd.to_datetime(actions['Date']).dt.date
            actions["Date"] = pd.to_datetime(actions["Date"])
            actions = actions.sort_values('Date').reset_index(drop=True)

            metrics_df = pd.merge_asof(metrics_df, actions, left_on='Date', right_on='Date', by='Ticker',
                                       direction='backward')

            price_targets = pd.DataFrame([ticker_obj.get_analyst_price_targets()])
            price_targets['Ticker'] = org
            price_targets['Date'] = datetime.now(timezone.utc).date()
            price_targets["Date"] = pd.to_datetime(price_targets["Date"])
            price_targets = price_targets.sort_values('Date')
            metrics_df = pd.merge_asof(metrics_df, price_targets, left_on='Date', right_on='Date', by='Ticker',
                                       direction='backward')
            metrics_df_list.append(metrics_df)

        metrics_df = pd.concat(metrics_df_list, ignore_index=True, sort=False)
        #metrics_df['Date'] = pd.to_datetime(metrics_df['Date'])
        metrics_df = metrics_df.sort_values('Date')

        merged_df = pd.merge_asof(price_df, metrics_df, left_on='Date', right_on='Date',
                                  by='Ticker', direction='backward')
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

        merged_df.fillna(0, inplace=True)
        feature_df = self.compute_financial_ratios(merged_df)
        feature_df[['Ticker', "Close"]] = merged_df[['Ticker', "Close"]].replace([np.inf, -np.inf], np.nan)
        print("create_input final-df:\n",  feature_df)
        input_columns = [col for col in self.training_columns if col != "bhsScore"]
        #feature_df[input_columns + ['Ticker']].to_csv(ModelBuilder.DATA_DIR+'/testdata.csv')
        return feature_df[input_columns]


    def test_model(self):
        # 1. Input
        input_df = pd.read_csv(ModelBuilder.DATA_DIR+'/testdata.csv')

        X_test, y_test = self.load_test_data()
        #print(f"Test data:\n", X_test, y_test )

        # 3. Make Test predictions
        y_pred = self.xg_model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        # print(f"Test Predictions: {predictions}\n")

        le = LabelEncoder()
        y_test_encoded = le.fit_transform(y_test)
        accuracy = accuracy_score(y_test_encoded, predictions)
        print("xg_model Accuracy: %.2f%%\n" % (accuracy * 100.0))


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
    tickerlist = args[0].split(",")
    print(f"=== Starting Weekly Training Job Execution: {datetime.now(ZoneInfo('America/New_York')).date()} ===")
    modelBuilder = ModelBuilder("xg",100)
    #feature_df = modelBuilder.load_train_data()
    #success = modelBuilder.train(feature_df)
    success = True
    if success == True:
        print("=== Training Job Completed Successfully ===")
        modelBuilder.init_runtime()
        #modelBuilder.test_model()
        input_df = modelBuilder.create_input(tickerlist)
        modelBuilder.predict(tickerlist, input_df)
    else:
        print("Model Building failed")


if __name__ == "__main__":
    main(sys.argv[1:])
