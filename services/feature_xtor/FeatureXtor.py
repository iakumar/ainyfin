import tempfile
import time
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
        self.look_ahead_days = 30

        self.symbols: list[str] = [
            'AAPL', 'AMD', 'AMZN', 'ABNB', 'CBRE', 'CMCSA', 'CRWD', 'DAL', 'DUK', 'EOG',
            'ETN', 'GE', 'GEV', 'GOOG', 'GOOGL', 'GRMN', 'GS', 'HOOD', 'ICE', 'INTU',
            'IQV', 'IR', 'JNJ', 'JPM', 'KO', 'LRCX', 'LYB', 'MANH', 'MDLZ', 'META',
            'MRK', 'MSFT', 'NET', 'NFLX', 'NVDA', 'PANW', 'PG', 'PLTR', 'PSX', 'RH',
            'SNOW', 'SO', 'T', 'TMO', 'TSLA', 'TT', 'TROW', 'V', 'WFC', 'WMT',
            'XOM', 'XYZ', 'Z']

        self.symbols2: list[str] = [
            # --- Original List ---
            'AAPL', 'AMD', 'AMZN', 'ABNB', 'CBRE', 'CMCSA', 'CRWD', 'DAL', 'DUK', 'EOG',
            'ETN', 'GE', 'GEV', 'GOOG', 'GOOGL', 'GRMN', 'GS', 'HOOD', 'ICE', 'INTU',
            'IQV', 'IR', 'JNJ', 'JPM', 'KO', 'LRCX', 'LYB', 'MANH', 'MDLZ', 'META',
            'MRK', 'MSFT', 'NET', 'NFLX', 'NVDA', 'PANW', 'PG', 'PLTR', 'PSX', 'RH',
            'SNOW', 'SO', 'T', 'TMO', 'TSLA', 'TT', 'TROW', 'V', 'WFC', 'WMT',
            'XOM', 'XYZ', 'Z',

            # --- Appended S&P 500 Components ---
            'A', 'AAL', 'ABBV', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM',
            'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM',
            'ALB', 'ALGN', 'ALL', 'ALLE', 'ALNT', 'AMCR', 'AME', 'AMGN', 'AMP', 'AMT',
            'ANET', 'AON', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO',
            'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'BA', 'BAC', 'BALL', 'BAX',
            'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BIO', 'BKNG', 'BKR',
            'BLK', 'BLDR', 'BMY', 'BR', 'BRO', 'BSX', 'BWA', 'BXP', 'C',
            'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CCI', 'CCK', 'CCL', 'CDNS',
            'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF',
            'CL', 'CLX', 'CME', 'CMG', 'CMI', 'CMS', 'CNC',
            'CNP', 'COF', 'COO', 'COP', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL',
            'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CTVA', 'CVS', 'CVX',
            'CZR', 'D', 'DE', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS',
            'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DVN', 'DXCM',
            'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EME', 'EMN',
            'EMR', 'ENPH', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETR', 'ETSY',
            'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX',
            'FDS', 'FDX', 'FE', 'FICO', 'FIS', 'FITB', 'FMC', 'FOX',
            'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GEN', 'GILD', 'GIS', 'GL',
            'GLW', 'GM', 'GNRC', 'GPC', 'GPN', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA',
            'HD', 'HIG', 'HII', 'HLT', 'HON', 'HPE', 'HPQ', 'HRL',
            'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY',
            'INTC', 'INVH', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI',
            'JKHY', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB',
            'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN',
            'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LULU', 'LUV', 'LVS', 'LW', 'LYV',
            'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDT', 'MET',
            'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMM', 'MNST', 'MO', 'MOH',
            'MOS', 'MPC', 'MPWR', 'MRNA', 'MS', 'MSCI', 'MSI', 'MSM', 'MTB', 'MTCH',
            'MTD', 'MU', "NCLH", 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NI', 'NKE', 'NOC',
            'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVR', 'NWS', 'NWSA', 'NXPI',
            'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PARA',
            'PAYC', 'PAYX', 'PCAR', 'PCG', 'PKG', 'PM', 'PNC', 'PNR', 'PNW',
            'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PTC', 'PWR', 'PYPL', 'QCOM',
            'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK',
            'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW',
            'SJM', 'SLB', 'SNA', 'SNPS', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX',
            'STZ', 'SWK', 'SWKS', 'SYK', 'SYY', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL',
            'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TRV',
            'TSCO', 'TSN', 'TTWO', 'TXN', 'TXT', 'TYL', 'UA', 'UAA', 'UAL', 'UBER',
            'UHS', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'VFC', 'VICI', 'VLO', 'VMC',
            'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBD',
            'WDC', 'WEC', 'WELL', 'WFRD', 'WHR', 'WM', 'WMB', 'WRB', 'WST', 'WTW',
            'WY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS'
        ]

        self.funds:list[str] = [
            '857480610', '857480628', 'AMLP', 'IWN', 'CCMAZ', 'FBCGX', 'FGKFX', 'FBGRX', 'FFSFX', 'FLKSX', 'FXAIX', 'FELV', 'GLD',
            'WFPRX'
        ]
        self.status:str = 'Done'


    def download(self):
        self.status = 'Started'

        # 1. setup params for start and end
        training_window = 4 * 365
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=training_window)
        end_date = today


        #2 Price
        price_df = self.download_prices_in_chunks(self.symbols, start_date, end_date)
        if not price_df.empty:
            # Drop columns where all values are NaN (failed tickers)
            price_df = price_df.dropna(how="all", axis=1)
            price_df = price_df.loc[:, ~price_df.columns.duplicated()]
            if isinstance(price_df.columns, pd.MultiIndex):
                # Stack by 'Ticker' level specifically (usually level 1 or level='Ticker')
                level_to_stack = "Ticker" if "Ticker" in price_df.columns.names else 1
                price_df = price_df.stack(level=level_to_stack, future_stack=True).reset_index()
            else:
                price_df = price_df.reset_index()
        else:
            print("No price data found for tickers:", self.symbols)
            return pd.DataFrame()

        price_df = price_df.sort_values('Date').reset_index(drop=True)
        price_df.to_csv(AinySchema.DATA_DIR+'/price.csv')

        edgarXDI = EdgarXDI("BHS")
        edgarXDI.downloadFinancialData(self.symbols)
        self.status = 'Done'


    def download_prices_in_chunks(self, symbols, start_date, end_date, chunk_size=50, delay_seconds=5):
        all_dfs = []
        # Split list into smaller chunks
        symbol_chunks = [
            symbols[i : i + chunk_size] for i in range(0, len(symbols), chunk_size)
        ]

        for idx, chunk in enumerate(symbol_chunks, 1):
            print(
                f"Downloading chunk {idx}/{len(symbol_chunks)} ({len(chunk)} tickers)..."
            )
            success = False
            max_retries = 3

            temp_dir = tempfile.gettempdir()
            yf.set_tz_cache_location(temp_dir)
            for attempt in range(1, max_retries + 1):
                try:
                    # Set a timeout on the request via threads/session parameters if needed
                    df = yf.download(
                        chunk,
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                    )
                    if not df.empty:
                        all_dfs.append(df)
                        success = True
                        break
                except Exception as e:
                    print(
                        f"Attempt {attempt} failed for chunk {idx}: {e}. Retrying..."
                    )
                    time.sleep(delay_seconds * attempt)

            if not success:
                print(f"Failed to download chunk {idx} after {max_retries} retries.")

            time.sleep(delay_seconds)  # Cooldown between batches

        if not all_dfs:
            return pd.DataFrame()

        # Combine chunk DataFrames along columns
        combined_df = pd.concat(all_dfs, axis=1)
        return combined_df


    def load(self):
        # 1. Load the price data
        price_df = pd.read_csv(AinySchema.DATA_DIR+'/price.csv')
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df["TargetDate"] = price_df["Date"] + pd.to_timedelta(self.look_ahead_days, unit='D')
        #print("price_df:\n",price_df)

        #2 Self-merge to find the nearest price on the target date for each stock
        price_target_df = pd.merge_asof(
            price_df.sort_values(["TargetDate"]),
            price_df[["Ticker", "Date", "Close"]].sort_values(["Date"]),
            by="Ticker",
            left_on="TargetDate",
            right_on="Date",
            direction="forward",
            tolerance=pd.Timedelta(days=3),
            suffixes=("", "_Target")
        )

        #Filter out records where TargetDate is in the future
        today = pd.Timestamp.today().normalize()
        price_target_df = price_target_df[price_target_df["Date_Target"] <= today].copy()

        price_target_df['Pct_Diff'] = (price_target_df['Close_Target']/price_target_df['Close'] - 1) * 100
        #print("price_target_df:",price_target_df['Pct_Diff'])
        price_target_df.dropna(subset=['Pct_Diff'],inplace=True)

        hold_threshold = 3.0
        bins = [-np.inf, -hold_threshold, hold_threshold, np.inf]
        # 1:Sell, 2: Hold, 3: Buy
        labels = [1, 2, 3]
        price_target_df['bhsScore'] = pd.cut(price_target_df['Pct_Diff'], bins=bins, labels=labels, include_lowest=True).astype(int)

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
    featureExtractor.download()
    if featureExtractor.status == 'Done':
        print("=== Feature Download Completed Successfully ===")
        featureExtractor.load();
        if featureExtractor.status == 'Done':
            print("=== Feature Load Completed Successfully ===")
    else:
        print("Warning: Feature DataFrame was empty. Aborting Extraction.")


if __name__ == "__main__":
    main()
