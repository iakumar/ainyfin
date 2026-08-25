from datetime import datetime, timedelta, timezone
from typing import final
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


@final
class FeatureExtractor:

    DATA_DIR="/Users/rithuhegde/ainyfin/services/data"
    def __init__(self, usecase: str):
        self.usecase = usecase
        self.orgs = ["ICE"]
        self.status = ''

    def download(self):
        # 2. Load historical price data
        #orgs = ["ICE", "MANH", 'MSFT', 'NVDA']
        #orgs = [
        #    "ICE", "MANH", 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'LLY', 'AVGO',
        #    'JPM', 'TSLA', 'WMT', 'XOM', 'UNH', 'MA', 'PG', 'JNJ', 'COST', 'MRK',
        #    'HD', 'ABBV', 'CVX', 'NFLX', 'CRM', 'BAC', 'PEP', 'AMD', 'LIN', 'ACN',
        #    'ORCL', 'TMO', 'IBM', 'CSCO', 'DIS', 'QCOM', 'CAT', 'TMUS', 'DHR', 'INTU',
        #    'VZ', 'UBER', 'WFC', 'GE', 'AMGN', 'PM', 'COP', 'UNP', 'LOW', 'ISRG', "MU"
        #]
        #
        #
        # 1. data for start and end
        training_window = 4 * 365
        look_ahead = 30
        today = datetime.now(timezone.utc).date()
        begin_start_date = today - timedelta(days=(training_window+look_ahead))
        begin_end_date = today - timedelta(days= look_ahead)

        """
        #1 Price
        price_df = yf.download(orgs, start=begin_start_date, end=begin_end_date, auto_adjust=True)
        price_df = price_df.stack(level=1).reset_index()
        price_df = price_df.sort_values('Date').reset_index(drop=True)
        price_df["TargetDate"] = price_df["Date"] + pd.to_timedelta(look_ahead+1, unit='D')
        price_df.to_csv(FeatureExtractor.DATA_DIR+'/price.csv')
        """

        for org in self.orgs:
            ticker_obj = yf.Ticker(org)
            print("Downloading metrics data for:",org)

            """
            #2 Financials
            finacials_df = ticker_obj.get_financials(freq='quarterly')
            finacials_df_x = self.transformFeatureXY(finacials_df, org)
            finacials_df_x.to_csv(FeatureExtractor.DATA_DIR+'/finacials.csv')
            """

            actions = ticker_obj.get_actions(period="max")
            actions['Ticker'] = org
            actions.to_csv(FeatureExtractor.DATA_DIR+'/actions.csv')

            """
            price_targets = pd.DataFrame([ticker_obj.get_analyst_price_targets()])
            price_targets['Ticker'] = org
            price_targets['Date'] = datetime.now(timezone.utc).date()
            price_targets.to_csv(FeatureExtractor.DATA_DIR+'/price_targets.csv')

            balancesheet = ticker_obj.get_balancesheet(freq='quarterly')
            balancesheet_x = self.transformFeatureXY(balancesheet, org)
            balancesheet_x.to_csv(FeatureExtractor.DATA_DIR+'/balancesheet.csv')

            cashflow = ticker_obj.get_cashflow(freq='quarterly')
            cashflow_x = self.transformFeatureXY(cashflow, org)
            cashflow_x.to_csv(FeatureExtractor.DATA_DIR+'/cashflow.csv')
            """

            print("get_info\n",ticker_obj.get_info())

            self.status = 'Done'


    def transformFeatureXY(self, X, org):
        df_transformed = X.T
        # 1. Convert date headers from the index into a dedicated 'Date' column
        df_transformed = df_transformed.reset_index().rename(columns={'index': 'Date'})
        # 2. Ensure Date is datetime type
        df_transformed['Date'] = pd.to_datetime(df_transformed['Date'])
        df_transformed['Ticker'] = org
        return df_transformed


    def load(self):
        price_df = pd.read_csv(FeatureExtractor.DATA_DIR+'/price.csv')
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df["TargetDate"] = pd.to_datetime(price_df["TargetDate"])
        #print("price_df:\n",price_df)

        # Self-merge example
        price_target_df = pd.merge_asof(
            price_df.sort_values('TargetDate'),
            price_df[['Ticker', 'Date', 'Close']].sort_values('Date'),
            by='Ticker',
            left_on='TargetDate',
            right_on='Date',
            direction='nearest',
            suffixes=('', '_Target'))

        #print("price_target_df:\n",price_target_df.head(100))

        price_target_df['Pct_Diff'] = ((price_target_df['Close_Target'] - price_target_df['Close']) / price_target_df['Close']) * 100
        #print("price_target_df:",price_target_df['Pct_Diff'])
        price_target_df.dropna(subset=['Pct_Diff'],inplace=True)
        price_target_df['bhsScore'] = pd.qcut(price_target_df['Pct_Diff'].round(0).astype(int), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

        # 4. Clean up the temporary column if needed
        price_target_df.drop(columns=["Close_Target"],inplace=True)
        print("price_target_df:\n",price_target_df)

        # 5. View the structure of current P/E and other metrics
        metrics_df_list = []
        for org in self.orgs:
            print("loading metrics data for:",org)
            finacials_df = pd.read_csv(FeatureExtractor.DATA_DIR+'/finacials.csv')
            metrics_df_list.append(finacials_df)
            balancesheet = pd.read_csv(FeatureExtractor.DATA_DIR+'/balancesheet.csv')
            metrics_df = pd.merge(finacials_df, balancesheet, on=['Ticker','Date'], how='inner')
            cashflow = pd.read_csv(FeatureExtractor.DATA_DIR+'/cashflow.csv')
            metrics_df = pd.merge(metrics_df, cashflow, on=['Ticker','Date'], how='inner')
            metrics_df_list.append(metrics_df)

        metrics_df = pd.concat(metrics_df_list, ignore_index=True, sort=False)
        print("metrics_df:\n", metrics_df)

        # 6. Merge historical data and fundamental info together
        self.feature_df = pd.merge(price_target_df, metrics_df, on="symbol", how="left")

        print(self.feature_df['averageAnalystRating'].unique())

        self.feature_df['averageAnalystRating_float'] = pd.to_numeric(self.feature_df['averageAnalystRating'].str.split('-').str[0], errors='coerce').astype(float)
        self.feature_df.drop(columns=['averageAnalystRating'], inplace=True)
        self.feature_df.fillna(0, inplace=True)
        #print(self.feature_df[['Date','symbol','averageAnalystRating_float']])
        print(self.feature_df[self.feature_df['symbol'] == 'AAPL'][['Date','symbol','targetMedianPrice','averageAnalystRating_float']])
        self.status = 'Done'


def main():
    print(f"=== Starting Feature Extraction : {datetime.now(ZoneInfo('America/New_York')).date()} ===")
    featureExtractor = FeatureExtractor("bhs_inv")
    featureExtractor.download();
    #featureExtractor.load();
    if featureExtractor.status == 'Done':
        print("=== Feature Extraction Completed Successfully ===")
    else:
        print("Warning: Feature DataFrame was empty. Aborting Extraction.")


if __name__ == "__main__":
    main()
