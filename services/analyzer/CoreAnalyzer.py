"""
AinyFin Core + Swing Trading Strategy Engine
============================================
Combines Long-Term Core Equity Evaluation (Moat, Growth, Institutional Backing)
with Tactical Swing Trading (Intraday Volatility Bands, Range Analysis, Friction Buffer).
"""

import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from services.consts.AinySchema import AinySchema
from services.feature_xtor.FeatureXtor import FeatureExtractor


class AinyFinSwingEngine:
    def __init__(
        self,
        core_conviction_threshold: float = 65.0,
        swing_buy_buffer: float = 0.015,   # 1.5% below entry band for swing buy
        swing_sell_buffer: float = 0.020,  # 2.0% above exit band for swing trim
        min_swing_volatility_pct: float = 2.0  # Min intraday range % for swing candidate
    ):
        self.core_threshold = core_conviction_threshold
        self.swing_buy_buffer = swing_buy_buffer
        self.swing_sell_buffer = swing_sell_buffer
        self.min_swing_vol = min_swing_volatility_pct

    def EvaluateCoreEligibility(self, row: pd.Series) -> Dict[str, Any]:
        """
        Evaluates fundamental quality, moat proxies, growth, and liquidity
        to determine if an asset qualifies for a Core Long-Term Position.
        """
        score = 0.0
        max_score = 100.0
        reasons = []

        # 1. Liquidity & Institutional Scale (20 pts)
        dollar_vol = row.get('Dollar_Volume_Log', 0.0)
        if dollar_vol >= 22.0:
            score += 20.0
            reasons.append("High Liquidity/Scale (Dollar_Volume_Log >= 22)")
        elif dollar_vol >= 20.0:
            score += 12.0

        # 2. Earnings & Revenue Expansion (20 pts)
        net_inc_growth = row.get('NetIncome_YoY_Growth', 0.0)
        rev_growth = row.get('Revenue_YoY_Growth', 0.0)
        if net_inc_growth > 0.15 or rev_growth > 0.15:
            score += 20.0
            reasons.append("Strong Fundamental Growth (>15% YoY Expansion)")
        elif net_inc_growth > 0.05 or rev_growth > 0.05:
            score += 10.0

        # 3. Moat & Reinvestment / ROIC (20 pts)
        roic = row.get('Return_On_Invested_Capital', 0.0)
        rd_to_rev = row.get('R_And_D_To_Revenue', 0.0)
        gross_margin = row.get('Gross_Margin', 0.0)
        if roic > 0.12 or rd_to_rev > 0.08 or gross_margin > 0.40:
            score += 20.0
            reasons.append("Moat Proxies Present (High ROIC, R&D, or Gross Margin)")

        # 4. Cash Flow & Capital Allocation (20 pts)
        fcf_margin = row.get('FCF_Margin', 0.0)
        buyback_yield = row.get('Buyback_Yield', 0.0)
        if fcf_margin > 0.15 or buyback_yield > 0.01:
            score += 20.0
            reasons.append("Robust Cash Flow / Shareholder Returns")

        # 5. Volatility & Balance Sheet Risk Discipline (20 pts)
        realized_vol = row.get('Realized_Vol_60D', 0.5)
        current_ratio = row.get('Current_Ratio', 1.0)
        if realized_vol < 0.60 and current_ratio >= 1.0:
            score += 20.0
            reasons.append("Stable Balance Sheet & Controlled Volatility")

        qualifies = score >= self.core_threshold
        return {
            "Ticker": row.get('Ticker', 'UNKNOWN'),
            "Core_Score": score,
            "Core_Eligible": qualifies,
            "Drivers": reasons
        }

    def CalculateVolatilityStats(self, ticker:str, df_intraday: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates price fluctuation statistics:
        - Average Daily High-Low Range
        - Days with High-Low Diff >= 2%
        - Days with UP / DOWN swings >= 2% from Prev Close
        """
        if df_intraday.empty:
            return {}

        df_intraday['High_Low_Diff'] = df_intraday['High'] - df_intraday['Low']
        df_intraday['High_Low_Pct'] = (df_intraday['High_Low_Diff'] / df_intraday['Close']) * 100.0
        df_intraday['Up_Diff_Pct'] = ((df_intraday['High'] - df_intraday['Prev_Close']) / df_intraday['Prev_Close']) * 100.0
        df_intraday['Down_Diff_Pct'] = ((df_intraday['Prev_Close'] - df_intraday['Low']) / df_intraday['Prev_Close']) * 100.0

        total_days = len(df_intraday)
        avg_price = df_intraday['Close'].mean()
        avg_range = df_intraday['High_Low_Diff'].mean()
        avg_range_pct = df_intraday['High_Low_Pct'].mean()

        days_range_gte_2pct = (df_intraday['High_Low_Pct'] >= 2.0).sum()
        up_days_gte_2pct = (df_intraday['Up_Diff_Pct'] >= 2.0).sum()
        down_days_gte_2pct = (df_intraday['Down_Diff_Pct'] >= 2.0).sum()

        return {
            "Ticker": ticker,
            "Total_Trading_Days": total_days,
            "Avg_Stock_Price": avg_price,
            "Avg_Daily_Range": avg_range,
            "Avg_Daily_Range_Pct": avg_range_pct,
            "Days_Range_GTE_2Pct": days_range_gte_2pct,
            "Up_Days_GTE_2Pct": up_days_gte_2pct,
            "Down_Days_GTE_2Pct": down_days_gte_2pct,
            "Swing_Candidate_Score": (days_range_gte_2pct / total_days) * 100.0 if total_days > 0 else 0.0
        }

    def GenerateSwingSignals(
        self,
        ticker,
        feature_row: pd.Series,
        current_price: float,
        prob_buy: float,
        vol_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generates tactical Core + Swing allocation signals based on conviction and price bands.
        """
        core_eval = self.EvaluateCoreEligibility(feature_row)
        avg_pct_range = vol_stats.get('Avg_Daily_Range_Pct', 3.0) / 100.0

        # Define dynamic entry/exit bounds based on asset's statistical daily range
        lower_swing_band = current_price * (1.0 - (avg_pct_range * 0.75))
        upper_swing_band = current_price * (1.0 + (avg_pct_range * 0.75))

        action = "HOLD_CORE"
        target_lot = "0%"
        rationale = []

        # Decision Logic: Model Conviction + Core Eligibility + Price Range
        if prob_buy >= 0.65:
            if core_eval["Core_Eligible"]:
                action = "BUY_CORE_AND_SWING"
                target_lot = "Full Allocation (70% Core / 30% Swing)"
                rationale.append("High model conviction + strong fundamental core score.")
            else:
                action = "BUY_TACTICAL_SWING_ONLY"
                target_lot = "30% Tactical Swing Lot"
                rationale.append("High model conviction, but failed Core eligibility criteria.")

        elif prob_buy <= 0.35:
            if core_eval["Core_Eligible"]:
                action = "TRIM_SWING_HOLD_CORE"
                target_lot = "Sell Swing Portion / Keep Core"
                rationale.append("Low short-term conviction; take profit/cut swing lot while holding core.")
            else:
                action = "SELL_ALL"
                target_lot = "100% Exit"
                rationale.append("Low conviction and lacks core fundamental support.")

        else:  # Neutral Model Zone (0.35 < prob_buy < 0.65)
            if avg_pct_range >= (self.min_swing_vol / 100.0):
                action = "RANGE_SWING_ACCUMULATE"
                target_lot = "15% Scale-In Lot"
                rationale.append(f"High volatility asset ({avg_pct_range*100:.2f}% avg range) in neutral trend; trade boundaries.")
            else:
                action = "HOLD_NO_TRADE"
                target_lot = "0%"
                rationale.append("Neutral conviction and low intraday range.")

        return {
            "Ticker": ticker,
            "Current_Price": current_price,
            "Model_Prob_Buy": prob_buy,
            "Core_Score": core_eval["Core_Score"],
            "Core_Eligible": core_eval["Core_Eligible"],
            "Recommended_Action": action,
            "Target_Lot_Size": target_lot,
            "Swing_Buy_Band": round(lower_swing_band, 2),
            "Swing_Sell_Band": round(upper_swing_band, 2),
            "Rationale": " | ".join(rationale)
        }


    def create_input(self, tickerlist:list[str]):
        # 1. Load training data
        training_file: str = Path(AinySchema.DATA_DIR) / "training_data.csv"
        training_data = pd.read_csv(training_file)
        column_list:list[str] = ['Ticker', 'Date', 'Close', 'High', 'Low', 'Dollar_Volume_Log', 'NetIncome_YoY_Growth', 'Revenue_YoY_Growth',
                 'Return_On_Invested_Capital', 'R_And_D_To_Revenue', 'Gross_Margin', 'FCF_Margin', 'Buyback_Yield', 'Realized_Vol_60D', 'Current_Ratio']
        input_df:pd.DataFrame = training_data[column_list]
        input_df = input_df[input_df["Ticker"].isin(tickerlist)].copy()
        input_df["Date"] = pd.to_datetime(input_df["Date"])
        today = datetime.now(timezone.utc).date()
        cutoff_date = pd.Timestamp(today - timedelta(days=365))
        input_df = input_df[input_df["Date"] >= cutoff_date].copy()
        input_df = input_df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
        input_df['Prev_Close'] = input_df.groupby('Ticker')['Close'].shift(1)
        return input_df

    def get_current_price(self, tickerlist:list[str]):
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=3)
        end_date = today + timedelta(days=1)
        featureExtractor = FeatureExtractor("BHS")
        new_price_df = featureExtractor.download_prices_in_chunks(tickerlist, start_date, end_date)
        if not new_price_df.empty:
            new_price_df = new_price_df.dropna(how="all", axis=1)
            new_price_df = new_price_df.loc[:, ~new_price_df.columns.duplicated()]
            if isinstance(new_price_df.columns, pd.MultiIndex):
                level_to_stack = "Ticker" if "Ticker" in new_price_df.columns.names else 1
                new_price_df = new_price_df.stack(level=level_to_stack, future_stack=True).reset_index()
            else:
                new_price_df = new_price_df.reset_index()
        else:
            print("No price data found for tickers:", tickerlist)
            return pd.DataFrame()
        new_price_df.dropna(subset=["Close"], inplace=True)
        latest_price_df:pd.DataFrame = new_price_df[new_price_df['Date'] == new_price_df['Date'].max()]
        return latest_price_df


def main(args:list[str]):
    tickerlist:list[str] = []

    if len(args) < 1:
        featureExtractor = FeatureExtractor("BHS")
        tickerlist = featureExtractor.symbols
    else:
        tickerlist = [ticker.strip() for ticker in args[0].split(",")]

    if len(tickerlist) < 1:
        print("Ticker list needed")
        return

    # 1. Instantiate Engine & Execute Analysis
    engine = AinyFinSwingEngine(core_conviction_threshold=65.0)
    input_df = engine.create_input(tickerlist)
    latest_price_df =  engine.get_current_price(tickerlist)

    for ticker in tickerlist:
        input_ticker_df = input_df[input_df["Ticker"] == ticker].copy()
        input_ticker_df['Date'] = pd.to_datetime(input_ticker_df['Date'])
        latest_row:pd.DataFrame = input_ticker_df[input_ticker_df['Date'] == input_ticker_df['Date'].max()]
        feature_series:pd.Series = latest_row.iloc[0]

        """
        print("closes:",input_ticker_df['Close'])
        print("highs:",input_ticker_df['High'])
        print("lows:",input_ticker_df['Low'])
        print("prev_closes:",input_ticker_df['Prev_Close'])
        """

        # 2. Compute High-Low Daily Volatility
        volatility_df:pd.DataFrame = pd.DataFrame({
            'Close': input_ticker_df['Close'],
            'High': input_ticker_df['High'],
            'Low': input_ticker_df['Low'],
            'Prev_Close': input_ticker_df['Prev_Close']
        })

        print("=== Ticker:",ticker,"===\n")

        # 2. Compute Volatility Statistics
        vol_stats = engine.CalculateVolatilityStats(ticker,
                                                    volatility_df)
        print("=== Volatility & Fluctuation Profile ===")
        for k, v in vol_stats.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

        latest_price_ticker_df = latest_price_df[latest_price_df["Ticker"] == ticker]
        #print("latest_price_ticker_df:\n",latest_price_ticker_df)
        current_price = latest_price_ticker_df.iloc[0].get('Close')
        # 3. Generate Strategy Trade Decision
        trade_signal = engine.GenerateSwingSignals(
            ticker,
            feature_row=feature_series,
            current_price=current_price,
            prob_buy=0.612,  # 61.2% model probability
            vol_stats=vol_stats
        )

        print("\n=== Strategic Allocation Signal ===")
        for k, v in trade_signal.items():
            print(f"  {k}: {v}")

        print("===" * 20)

if __name__ == "__main__":
    main(sys.argv[1:])
