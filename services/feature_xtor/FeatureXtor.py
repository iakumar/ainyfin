from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from services.consts.AinySchema import AinySchema

from .EdgarXDI import EdgarXDI


class FeatureExtractor:

    def __init__(self, usecase: str):
        self.usecase = usecase
        self.symbols:list[str] = ['ICE','MANH','AMD','GOOG','MSFT','HOOD','TT','NET','ETN','PSX','AAPL','NVDA',
                        'EOG','GRMN','PANW','XOM','SNOW','TSLA','MRK','IQV','WFC','CBRE',
                        'TMO','NFLX','INTU','MDLZ','Z','XYZ','AMZN','LRCX','GE','GS','PLTR']
        #self.symbols = [
        #    "ICE", "MANH", 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'LLY', 'AVGO',
        #    'JPM', 'TSLA', 'WMT', 'XOM', 'UNH', 'MA', 'PG', 'JNJ', 'COST', 'MRK',
        #    'HD', 'ABBV', 'CVX', 'NFLX', 'CRM', 'BAC', 'PEP', 'AMD', 'LIN', 'ACN',
        #    'ORCL', 'TMO', 'IBM', 'CSCO', 'DIS', 'QCOM', 'CAT', 'TMUS', 'DHR', 'INTU',
        #    'VZ', 'UBER', 'WFC', 'GE', 'AMGN', 'PM', 'COP', 'UNP', 'LOW', 'ISRG', "MU"
        #
        # HOOD, TT, NET, ETN, PSX, EOG, GRMN, PANW, AMLP, IWN, XOM, SNOW, TSLA, MRK, IQV,
        # WFC, CBRE, TMO, NFLX, INTU, MDLZ, Z, XYZ, GE, GS, AMZN, LRCX, GE, GS, PLTR,
        #]
        #

        self.funds = ['CCMAZ', 'FBGRX', 'FELV','AMLP', 'IWN' ]
        self.status = ''


    def download(self):
        # 1. setup params for start and end
        training_window = 4 * 365
        look_ahead = 30
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=(training_window+look_ahead))
        end_date = today - timedelta(days= look_ahead)


        #2 Price
        price_df = yf.download(self.symbols, start=start_date, end=end_date, auto_adjust=True)
        price_df = price_df.stack(level=1).reset_index()
        price_df = price_df.sort_values('Date').reset_index(drop=True)
        price_df["TargetDate"] = price_df["Date"] + pd.to_timedelta(look_ahead+1, unit='D')
        price_df.to_csv(AinySchema.DATA_DIR+'/price.csv')

        """
        #2 Financials
        self.downloadFinancials(start_date=start_date, end_date=end_date)

        #3 balancesheet
        self.downloadBalancesheet(start_date=start_date, end_date=end_date)

        #4 cashflow
        self.downloadCashflow(start_date=start_date, end_date=end_date)

        #5 misc
        self.downloadMisc(start_date=start_date, end_date=end_date)
        """

        edgarXDI = EdgarXDI("BHS")
        edgarXDI.downloadFinancialData(self.symbols)

        self.status = 'Done'


    def downloadFinancials(self, start_date, end_date):
        raw_dfs = []
        for symbol in self.symbols:
            print("Downloading financials for:",symbol)
            ticker_obj = yf.Ticker(symbol)

            # Fetch financial statements (columns are dates, index are line items)
            fin = ticker_obj.get_financials(freq="quarterly")
            if fin is None or fin.empty:
                continue

            # Transpose so rows are dates and columns are financial line items
            df = fin.T.reset_index().rename(columns={"index": "Date"})
            df["Ticker"] = symbol
            df["Date"] = pd.to_datetime(df["Date"])
            # Filter: start <= Date <= end
            df = df[
                (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
            ]

            raw_dfs.append(df)

        if not raw_dfs:
            return

        # -------------------------------------------------------------
        # Phase 1: Outer Union (Combines all unique columns across stocks)
        # -------------------------------------------------------------
        unified_df = pd.concat(raw_dfs, axis=0, ignore_index=True, join="outer")

        # Sort deterministically
        unified_df = unified_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # -------------------------------------------------------------
        # Phase 2: Standardize & Impute Missing Financial Features
        # -------------------------------------------------------------

        # Core standardized line items shared across most companies
        core_features = [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "OperatingRevenue",
            "CostOfGoodsAndServicesSold",
            "GrossProfit",
            "OperatingExpense",
            "ResearchAndDevelopment",
            "SellingGeneralAndAdministration",
            "SellingAndMarketingExpense",
            "GeneralAndAdministrativeExpense",
            "OperatingIncomeLoss",
            "EBITDA",
            "NormalizedEBITDA",
            "EBIT",
            "NetInterestIncome",
            "InterestExpense",
            "InterestIncome",
            "PretaxIncome",
            "TaxProvision",
            "TaxRateForCalcs",
            "NetIncome",
            "NetIncomeLoss",
            "NetIncomeContinuousOperations",
            "DilutedNIAvailtoComStockholders",
            "BasicEPS",
            "EarningsPerShareDiluted",
            "BasicAverageShares",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
        ]

        # Ensure all core columns exist (add as NaN if completely missing from batch)
        for col in core_features:
            if col not in unified_df.columns:
                unified_df[col] = np.nan

        # Fill missing granular line items with 0 (e.g., R&D is 0 for non-tech companies)
        zero_fill_cols = [
            "ResearchAndDevelopment",
            "InterestExpense",
            "InterestIncome",
            "TotalUnusualItems",
        ]
        for col in zero_fill_cols:
            if col in unified_df.columns:
                unified_df[col] = unified_df[col].fillna(0)

        # Calculate resilient fallback metrics if primary metrics are missing
        if "GrossProfit" in unified_df.columns and "RevenueFromContractWithCustomerExcludingAssessedTax" in unified_df.columns:
            unified_df["GrossProfit"] = unified_df["GrossProfit"].fillna(
                unified_df["RevenueFromContractWithCustomerExcludingAssessedTax"] - unified_df.get("CostOfGoodsAndServicesSold", 0)
            )

        # Keep metadata + core features (or retain all columns if preferred)
        final_cols = ["Date", "Ticker"] + [
            c for c in core_features if c in unified_df.columns
        ]
        normalized_df = unified_df[final_cols]
        normalized_df.to_csv(AinySchema.DATA_DIR+'/financials.csv', index=False, header=True)


    def downloadBalancesheet(self, start_date, end_date):
        raw_dfs = []
        for symbol in self.symbols:
            print("Downloading balancesheet for:",symbol)
            ticker_obj = yf.Ticker(symbol)

            # Fetch financial statements (columns are dates, index are line items)
            fin = ticker_obj.get_balancesheet(freq="quarterly")
            if fin is None or fin.empty:
                continue

            # Transpose so rows are dates and columns are financial line items
            df = fin.T.reset_index().rename(columns={"index": "Date"})
            df["Ticker"] = symbol
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[
                (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
            ]
            raw_dfs.append(df)

        if not raw_dfs:
            return

        # -------------------------------------------------------------
        # Phase 1: Outer Union (Combines all unique columns across stocks)
        # -------------------------------------------------------------
        unified_df = pd.concat(raw_dfs, axis=0, ignore_index=True, join="outer")

        # Sort deterministically
        unified_df = unified_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # Core standardized line items shared across most companies
        core_features = [
            "TotalAssets",
            "CurrentAssets",
            "CashCashEquivalentsAndShortTermInvestments",
            "CashAndCashEquivalents",
            "OtherShortTermInvestments",
            "Receivables",
            "AccountsReceivable",
            "Inventory",
            "NetPPE",  # Property, Plant & Equipment
            "GrossPPE",
            "GoodwillAndOtherIntangibleAssets",
            "Goodwill",
            "OtherIntangibleAssets",
            "TotalLiabilitiesNetMinorityInterest",
            "CurrentLiabilities",
            "AccountsPayable",
            "CurrentDebt",
            "LongTermDebt",
            "TotalDebt",  # CurrentDebt + LongTermDebt
            "StockholdersEquity",
            "StockholdersEquity",
            "RetainedEarnings",
            "WorkingCapital",
        ]

        # Ensure all core columns exist (add as NaN if completely missing from batch)
        for col in core_features:
            if col not in unified_df.columns:
                unified_df[col] = np.nan

        # Fill missing granular line items with 0
        zero_fill_cols = [
            "Goodwill",
            "Receivables",
            "Inventory",
            "NetPPE",
        ]
        for col in zero_fill_cols:
            if col in unified_df.columns:
                unified_df[col] = unified_df[col].fillna(0)

        # Calculate resilient fallback metrics if primary metrics are missing
        if "TotalDebt" in unified_df.columns:
            unified_df["TotalDebt"] = unified_df["TotalDebt"].fillna(
                unified_df.get("CurrentDebt", 0.0) + unified_df.get("LongTermDebt", 0.0)
            )

        # Keep metadata + core features (or retain all columns if preferred)
        final_cols = ["Date", "Ticker"] + [
            c for c in core_features if c in unified_df.columns
        ]
        normalized_df = unified_df[final_cols].copy()
        normalized_df.to_csv(AinySchema.DATA_DIR+'/balancesheet.csv', index=False, header=True)


    def downloadCashflow(self, start_date, end_date):
        raw_dfs = []
        for symbol in self.symbols:
            print("Downloading cashflow for:",symbol)
            ticker_obj = yf.Ticker(symbol)

            # Fetch financial statements (columns are dates, index are line items)
            fin = ticker_obj.get_cashflow(freq="quarterly")
            if fin is None or fin.empty:
                continue

            # Transpose so rows are dates and columns are financial line items
            df = fin.T.reset_index().rename(columns={"index": "Date"})
            df["Ticker"] = symbol
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[
                (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
            ]
            raw_dfs.append(df)

        if not raw_dfs:
            return

        # -------------------------------------------------------------
        # Phase 1: Outer Union (Combines all unique columns across stocks)
        # -------------------------------------------------------------
        unified_df = pd.concat(raw_dfs, axis=0, ignore_index=True, join="outer")

        # Sort deterministically
        unified_df = unified_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # Core standardized line items shared across most companies
        core_features = [
            "NetCashProvidedByUsedInOperatingActivities",
            "InvestingCashFlow",
            "FinancingCashFlow",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "FreeCashFlow",  # NetCashProvidedByUsedInOperatingActivities - PaymentsToAcquirePropertyPlantAndEquipment
            "StockBasedCompensation",
            "DepreciationAndAmortization",
            "RepurchaseOfCapitalStock",
            "CommonStockDividendPaid",
            "ChangeInWorkingCapital",
            "NetIssuancePaymentsOfDebt",
        ]

        # Ensure all core columns exist (add as NaN if completely missing from batch)
        for col in core_features:
            if col not in unified_df.columns:
                unified_df[col] = np.nan

        # Fill missing granular line items with 0
        zero_fill_cols = [
            "StockBasedCompensation",
            "CommonStockDividendPaid",
            "RepurchaseOfCapitalStock",
        ]
        for col in zero_fill_cols:
            if col in unified_df.columns:
                unified_df[col] = unified_df[col].fillna(0)

        # Calculate resilient fallback metrics if primary metrics are missing
        if "FreeCashFlow" in unified_df.columns:
            unified_df["FreeCashFlow"] = unified_df["FreeCashFlow"].fillna(
                unified_df.get("NetCashProvidedByUsedInOperatingActivities", 0.0)
                - np.abs(unified_df.get("PaymentsToAcquirePropertyPlantAndEquipment", 0.0))
            )

        # Keep metadata + core features (or retain all columns if preferred)
        final_cols = ["Date", "Ticker"] + [
            c for c in core_features if c in unified_df.columns
        ]
        normalized_df = unified_df[final_cols].copy()
        normalized_df.to_csv(AinySchema.DATA_DIR+'/cashflow.csv', index=False, header=True)


    def downloadMisc(self, start_date, end_date):
        write_header = True
        mode = "w"
        for symbol in self.symbols:
            ticker_obj = yf.Ticker(symbol)
            print("Downloading metrics data for:",symbol)

            actions = ticker_obj.get_actions(period="max").reset_index().rename(columns={'index': 'Date'})
            actions['Ticker'] = symbol
            actions['Date'] = pd.to_datetime(actions['Date']).dt.date
            # Filter: start <= Date <= end
            actions = actions[
                (actions["Date"] >= start_date) & (actions["Date"] <= end_date)
            ]
            actions.to_csv(AinySchema.DATA_DIR+'/actions.csv', mode=mode, index=False, header=write_header)

            price_targets = pd.DataFrame([ticker_obj.get_analyst_price_targets()])
            price_targets['Ticker'] = symbol
            price_targets['Date'] = datetime.now(timezone.utc).date()
            price_targets.to_csv(AinySchema.DATA_DIR+'/price_targets.csv',mode=mode, index=False, header=write_header)

            write_header = False
            mode = "a"


    def transformFeatureXY(self, X: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df_transformed = X.T
        # 1. Convert date headers from the index into a dedicated 'Date' column
        df_transformed = df_transformed.reset_index().rename(columns={'index': 'Date'})
        # 2. Ensure Date is datetime type
        df_transformed['Date'] = pd.to_datetime(df_transformed['Date'])
        df_transformed['Ticker'] = symbol
        return df_transformed


    def load(self):
        # 1. Load the price data
        price_df = pd.read_csv(AinySchema.DATA_DIR+'/price.csv')
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df["TargetDate"] = pd.to_datetime(price_df["TargetDate"])
        #print("price_df:\n",price_df)

        #2 Self-merge to find the nearest price on the target date for each stock
        price_target_df = pd.merge_asof(
            price_df.sort_values('TargetDate'),
            price_df[['Ticker', 'Date', 'Close']].sort_values('Date'),
            by='Ticker',
            left_on='TargetDate',
            right_on='Date',
            direction='nearest',
            suffixes=('', '_Target'))

        price_target_df['Pct_Diff'] = ((price_target_df['Close_Target'] - price_target_df['Close']) / price_target_df['Close']) * 100
        #print("price_target_df:",price_target_df['Pct_Diff'])
        price_target_df.dropna(subset=['Pct_Diff'],inplace=True)
        price_target_df['bhsScore'] = pd.qcut(price_target_df['Pct_Diff'].round(0).astype(int), q=3, labels=[1, 2, 3]).astype(int)

        # 3. Clean up the temporary column if needed
        price_target_df.drop(columns=['Unnamed: 0'],inplace=True)
        print("price_target_df:\n",price_target_df)

        # 4. current P/E and other metrics
        """
        financials_df = pd.read_csv(AinySchema.DATA_DIR+'/financials.csv')
        balancesheet = pd.read_csv(AinySchema.DATA_DIR+'/balancesheet.csv')
        metrics_df = pd.merge(financials_df, balancesheet, on=['Ticker','Date'], how='inner')
        cashflow = pd.read_csv(AinySchema.DATA_DIR+'/cashflow.csv')
        metrics_df = pd.merge(metrics_df, cashflow, on=['Ticker','Date'], how='inner')
        """

        # 5. Both DataFrames MUST be sorted chronologically by date
        metrics_df = pd.read_csv(AinySchema.DATA_DIR+'/financial_data.csv')
        metrics_df['Date'] = pd.to_datetime(metrics_df['Date'])
        metrics_df = metrics_df.sort_values('Date').reset_index(drop=True)
        print("metrics_df:\n", metrics_df)

        price_target_df['Date'] = pd.to_datetime(price_target_df['Date'])
        price_target_df = price_target_df.sort_values('Date').reset_index(drop=True)

        # 6. Merge historical data and fundamental info together
        merged_df = pd.merge_asof(price_target_df, metrics_df, left_on='Date',  right_on='Date',
                                  by='Ticker',    direction='backward')    # Uses last available fundamental data (no look-ahead leakage)

        #merged_df.fillna(0, inplace=True)
        #print(merged_df[merged_df['Ticker'] == 'ICE'][['Ticker', 'Date', 'Close', 'TargetDate', 'Close_Target', 'TaxRateForCalcs']].head(25))
        merged_df.to_csv(AinySchema.DATA_DIR+'/ainyfin_data.csv')

        self.status = 'Done'


def main():
    print(f"=== Starting Feature Extraction : {datetime.now(ZoneInfo('America/New_York')).date()} ===")
    featureExtractor = FeatureExtractor("BHS")
    featureExtractor.download();
    if featureExtractor.status == 'Done':
        print("=== Feature Download Completed Successfully ===")
        featureExtractor.load();
        if featureExtractor.status == 'Done':
            print("=== Feature Load Completed Successfully ===")
    else:
        print("Warning: Feature DataFrame was empty. Aborting Extraction.")


if __name__ == "__main__":
    main()
