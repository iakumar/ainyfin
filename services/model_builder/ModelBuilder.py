import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from services.consts.AinySchema import AinySchema

BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "anypug.appspot.com")
DIRECTORY_NAME = "ainyfin/models"
APP_ENGINE_URL = os.environ.get(
    "APP_ENGINE_URL", "https://ainyfin.appspot.com"
)
GBMODEL_FILENAME = "gboost_bhs_model.joblib"
XGBMODEL_FILENAME = "xgboost_bhs_model.joblib"

class ModelBuilder:
    def __init__(self, builder: str, n_estimators: int = 100):
        self.target_column:str = "bhsScore"
        self.bhs_descs:list[str] = ["Sell", "Hold", "Buy"]
        # Explicit list of generated ratio columns
        self.ratio_columns:list[str] = [
            "Price_To_Earnings",
            "Price_To_FreeCashFlow",
            "Price_To_Book",
            "Price_To_Sales",
            "EV_To_EBITDA",
            "EV_To_EBIT",
            "Gross_Margin",
            "Net_Margin",
            "FCF_Margin",
            "Operating_Margin",
            "EBITDA_Margin",
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
            "Accrual_Ratio",
            "Payout_To_FCF",
            "NonOperating_Income_Reliance",
            "Goodwill_To_Assets",
            "Had_Goodwill_Impairment",
            "Effective_Tax_Rate"
        ]

        # Collect trend feature columns for inf handling
        self.trend_columns:list[str] = [
            "Revenue_YoY_Growth",
            "Revenue_QoQ_Growth",
            "EBITDA_YoY_Growth",
            "EPS_YoY_Growth",
            "FCF_YoY_Growth",
            "Operating_CashFlow_YoY_Growth",
            "Revenue_Acceleration",
            "Gross_Margin_YoY_Delta",
            "EBITDA_Margin_YoY_Delta",
            "FCF_Margin_YoY_Delta",
            "Operating_Margin_QoQ_Delta",
            "Debt_YoY_Growth",
            "Shares_Outstanding_YoY_Change",
            "CapEx_YoY_Growth",
            "ROIC_YoY_Delta",
            "Working_Capital_YoY_Change",
            "ROE_Volatility_8Q",
            "Margin_Volatility_8Q"
        ]

        self.training_columns:list[str] = self.ratio_columns + self.trend_columns + ["bhsScore"]
        #print("training_columns:", self.training_columns)
        self.builder:str = builder
        self.n_estimators:int = n_estimators
        self.xg_model:any = None


    def getMostRecentMarketDate(self):
        # Check if current time is greater than 4 PM EST
        est_tz = timezone(timedelta(hours=-4))
        current_time = datetime.now().astimezone()
        target_time = datetime(
            current_time.year,
            current_time.month,
            current_time.day,
            17,
            0,
            0,
            tzinfo=est_tz,
        )

        us_market_bday = pd.tseries.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
        print("us_market_bday:",us_market_bday)
        # Convert to pandas Timestamps to make calendar math work perfectly
        current_ts = pd.Timestamp(current_time)
        target_ts = pd.Timestamp(target_time)

        # Rule 1: Weekends & Holidays (Sat/Sun/Holidays always roll backward to prior trading day)
        if not us_market_bday.is_on_offset(target_ts):
            predict_date = target_ts - us_market_bday

        # Rule 2: It is Monday (or the first trading day of the week) and we haven't reached target_time yet
        elif target_ts.weekday() == 0 and current_ts < target_ts:
            predict_date = target_ts - us_market_bday # Rolls back 2 trading days (e.g., past Friday to Thursday)
            print("predict_date 2:", predict_date, current_ts, target_ts)

        # Rule 3: target_time is in the past, and it is a valid weekday/trading day
        elif current_ts > target_ts:
            predict_date = target_ts
            print("predict_date 3:", predict_date)

        # Rule 4: Normal weekdays (Tue-Fri) where current_time hasn't reached target_time yet
        else:
            predict_date = target_ts - us_market_bday
            print("predict_date 4:", predict_date)

        predict_date_end = predict_date + timedelta(days=1)
        return predict_date, predict_date_end


    def load_train_data(self) -> pd.DataFrame:
        """
        Fetch your historical training feature dataset...
        """
        print("Loading ainyfin_data training feature dataset...")
        raw_df = pd.read_csv(AinySchema.DATA_DIR+'/ainyfin_data.csv')
        raw_df = raw_df.sort_values(['Ticker','Date'])
        snapshot_df = self.compute_financial_snapshot(raw_df)
        snapshot_df.to_csv(AinySchema.DATA_DIR+'/snapshot.csv')
        snapshot_df[['Ticker', "bhsScore"]] = raw_df[['Ticker', "bhsScore"]].replace([np.inf, -np.inf], np.nan)
        snapshot_df['Date'] = pd.to_datetime(snapshot_df['Date'])
        snapshot_df = snapshot_df.sort_values(['Date']).reset_index(drop=True)

        financial_df = pd.read_csv(AinySchema.DATA_DIR+'/financial_data.csv')
        financial_df['Date'] = pd.to_datetime(financial_df['Date'])
        financial_df = financial_df.sort_values(['Ticker','Date']).reset_index(drop=True)
        financial_trends_df = self.compute_financial_trends(financial_df)
        financial_trends_df = financial_trends_df.sort_values(['Date']).reset_index(drop=True)
        financial_trends_df.to_csv(AinySchema.DATA_DIR+'/trends.csv')

        merged_df = pd.merge_asof(snapshot_df, financial_trends_df, left_on='Date', right_on='Date',
                                  by='Ticker', direction='backward', suffixes=('', '_Trends'))
        return merged_df


    def compute_financial_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes standardized valuation, leverage, profitability, and quality ratios

        from raw SEC fundamental and market price data.

        Parameters:
        -----------
        df : pd.DataFrame
            Must contain raw fundamental columns and market price.

        Returns:
        --------
        pd.DataFrame
            DataFrame augmented with engineered ratio features.
        """

        def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
            """
                Performs element-wise division, returning np.nan for invalid or zero denominator
                rather than coercing to 0.0 or inf.
            """
            # Replace zeros in denominator with NaN before division to propagate NaN correctly
            denom_clean = denominator.replace(0, np.nan)
            result = numerator / denom_clean

            # Replace inf / -inf results (e.g., non-zero numerator divided by ~0) with NaN
            result = result.replace([np.inf, -np.inf], np.nan)
            return result

        def get_col(col_name: str) -> pd.Series:
                if col_name in data.columns:
                    return data[col_name].astype(float)
                return pd.Series(np.nan, index=data.index)

        # Create a copy to prevent mutating the input DataFrame
        data = df.copy()

        # =========================================================================
        # PRE-COMPUTATION & TAG MAPPINGS
        # =========================================================================

        # 1. Total Revenue / Operating Revenue
        revenue = get_col("RevenueFromContractWithCustomerExcludingAssessedTax").where(
            lambda x: x != 0, get_col("Revenues")
        )

        # 2. Net Income
        net_income = get_col("NetIncomeLoss").where(
            lambda x: x != 0, get_col("ProfitLoss")
        )

        # 3. Balance Sheet Aggregates
        total_assets = get_col("Assets")
        current_assets = get_col("AssetsCurrent")
        current_liabilities = get_col("LiabilitiesCurrent")
        stockholders_equity = get_col("StockholdersEquity")

        # 4. Total Debt & Liquidity
        # Use TotalDebt column if present, otherwise compute sum of current + non-current debt
        total_debt = get_col("TotalDebt").where(
            lambda x: x != 0,
            get_col("LongTermDebtCurrent") + get_col("LongTermDebtNoncurrent"),
        )

        cash_and_equivalents = get_col(
            "CashCashEquivalentsAndShortTermInvestments"
        ).where(
            lambda x: x != 0,
            get_col("CashCashEquivalentsAtCarryingValue")
            + get_col("ShortTermInvestments"),
        )

        accounts_receivable = get_col("AccountsReceivableNetCurrent").where(
            lambda x: x != 0, get_col("AccountsReceivableNet")
        )

        # 5. Earnings Metrics (EBIT & Normalized EBITDA)
        operating_income = get_col("OperatingIncomeLoss")
        interest_expense = get_col("InterestExpense").where(
            lambda x: x != 0,
            get_col(
                "InterestExpenseNonoperating", get_col("InterestExpenseOperating")
            ),
        )

        # EBIT (Operating Income + Non-operating items if required, defaulting to Operating Income)
        ebit = operating_income.where(lambda x: x != 0, net_income + interest_expense)

        # Depreciation & Amortization
        dna = get_col("DepreciationDepletionAndAmortization").where(
            lambda x: x != 0,
            get_col("DepreciationAndAmortization").where(
                lambda x: x != 0,
                get_col("Depreciation")
                + get_col("AmortizationOfIntangibleAssets"),
            ),
        )

        # Adjustments for Normalized EBITDA
        sbc = get_col("ShareBasedCompensation").where(
            lambda x: x != 0,
            get_col(
                "SharebasedCompensationArrangementBySharebasedPaymentAwardCompensationCost1",
                get_col("AllocatedShareBasedCompensationExpense"),
            ),
        )

        restructuring = get_col("RestructuringCosts").where(
            lambda x: x != 0,
            get_col("RestructuringAndRelatedCostIncurredCost"),
        )

        impairments = (
            get_col("GoodwillImpairmentLoss")
            + get_col("ImpairmentOfIntangibleAssetsExcludingGoodwill")
            + get_col("InventoryWriteDown")
        )

        gains_losses = (
            get_col("GainLossOnSaleOfPropertyPlantEquipment")
            + get_col("GainLossOnSaleOfBusiness")
            + get_col("EquitySecuritiesFvNiGainLoss")
        )

        normalized_ebitda = (
            operating_income + dna + sbc + restructuring + impairments
        ) - gains_losses

        # 6. Expenses & Cash Flows
        gross_profit = get_col("GrossProfit").where(
            lambda x: x != 0, get_col("GrossProfit_Calculated")
        )

        cfo = get_col("NetCashProvidedByUsedInOperatingActivities")
        capex = get_col("PaymentsToAcquirePropertyPlantAndEquipment").abs()
        fcf = get_col("FreeCashFlow").where(lambda x: x != 0, cfo - capex)
        rd_expense = get_col("ResearchAndDevelopmentExpense")

        # Share Count and Market Cap
        shares_diluted = get_col(
            "WeightedAverageNumberOfDilutedSharesOutstanding"
        ).where(
            lambda x: x != 0,
            get_col("WeightedAverageNumberOfSharesOutstandingBasic"),
        )
        market_cap = get_col("Close") * shares_diluted
        enterprise_value = market_cap + total_debt - cash_and_equivalents

        # =========================================================================
        # COMPUTED FINANCIAL RATIOS
        # =========================================================================

        # 1. Valuation Ratios
        data["Price_To_Earnings"] = safe_divide(
            data["Close"], get_col("EarningsPerShareDiluted")
        )
        data["Price_To_FreeCashFlow"] = safe_divide(market_cap, fcf)
        data["Price_To_Book"] = safe_divide(market_cap, stockholders_equity)
        data["Price_To_Sales"] = safe_divide(market_cap, revenue)
        data["EV_To_EBITDA"] = safe_divide(enterprise_value, normalized_ebitda)
        data["EV_To_EBIT"] = safe_divide(enterprise_value, ebit)

        # 2. Profitability & Margins
        data["Gross_Margin"] = safe_divide(gross_profit, revenue)
        data["Operating_Margin"] = safe_divide(operating_income, revenue)
        data["EBITDA_Margin"] = safe_divide(normalized_ebitda, revenue)
        data["Net_Margin"] = safe_divide(net_income, revenue)
        data["FCF_Margin"] = safe_divide(fcf, revenue)
        data["Return_On_Equity"] = safe_divide(net_income, stockholders_equity)
        data["Return_On_Assets"] = safe_divide(net_income, total_assets)

        # 3. Leverage, Solvency & Liquidity
        data["Debt_To_Equity"] = safe_divide(total_debt, stockholders_equity)
        data["Debt_To_Assets"] = safe_divide(total_debt, total_assets)
        data["Current_Ratio"] = safe_divide(current_assets, current_liabilities)
        data["Quick_Ratio"] = safe_divide(
            cash_and_equivalents + accounts_receivable, current_liabilities
        )
        data["Interest_Coverage"] = safe_divide(ebit, interest_expense.abs())
        data["Cash_To_Debt"] = safe_divide(cash_and_equivalents, total_debt)

        # 4. Operational Efficiency
        data["Asset_Turnover"] = safe_divide(revenue, total_assets)
        data["Working_Capital_Turnover"] = safe_divide(
            revenue, get_col("WorkingCapital")
        )

        # 5. Earnings Quality & Capital Allocation
        data["CFO_To_NetIncome"] = safe_divide(cfo, net_income)
        data["SBC_To_Revenue"] = safe_divide(sbc, revenue)
        data["CapEx_To_CFO"] = safe_divide(capex, cfo)
        data["CapEx_To_Revenue"] = safe_divide(capex, revenue)
        data["R_And_D_To_Revenue"] = safe_divide(rd_expense, revenue)

        #Literature-Validated Earnings Quality Signals
        data["Accrual_Ratio"] = safe_divide(net_income - cfo, total_assets)
        data["Payout_To_FCF"] = safe_divide(
            get_col("PaymentsOfDividendsCommonStock").abs()
            + get_col("PaymentsForRepurchaseOfCommonStock").abs(),
            fcf,
        ).clip(-5.0, 5.0)

        data["NonOperating_Income_Reliance"] = safe_divide(
            get_col("OtherNonoperatingIncomeExpense").abs(), net_income.abs()
        )

        data["Goodwill_To_Assets"] = safe_divide(get_col("Goodwill"), total_assets)

        data["Had_Goodwill_Impairment"] = (get_col("GoodwillImpairmentLoss") > 0).astype(int)

        tax_base = get_col(
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
        )
        data["Effective_Tax_Rate"] = safe_divide(get_col("IncomeTaxExpenseBenefit"), tax_base).clip(0.0,0.50)

        # Clean extreme inf / -inf values on calculated ratio columns only
        if hasattr(self, "ratio_columns") and self.ratio_columns:
            data[self.ratio_columns] = data[self.ratio_columns].replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            # Fallback if ratio_columns attribute is not defined on class instance
            ratio_cols = [
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
                "Accrual_Ratio",
                "Payout_To_FCF",
                "NonOperating_Income_Reliance",
                "Goodwill_To_Assets",
                "Had_Goodwill_Impairment",
                "Effective_Tax_Rate"
            ]

            for col in ratio_cols:
               if col in data.columns:
                  data[col] = data[col].replace([np.inf, -np.inf], np.nan)

        return data


    def compute_financial_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes quarterly YoY momentum, acceleration, and margin delta signals

        from raw fundamental line items and calculated ratios.

        Parameters:
        -----------
        df : pd.DataFrame
            Must contain 'Ticker', 'Date', and normalized fundamental features/ratios
            sorted chronologically by Ticker and Date.

        Returns:
        --------
        pd.DataFrame
            DataFrame augmented with fundamental trend features.
        """

        data = df.copy()
        # Ensure dataset is sorted strictly by Ticker and Date
        data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # -------------------------------------------------------------
        # HELPER FUNCTIONS
        # -------------------------------------------------------------
        def get_col(col_name: str, fallback_val=0) -> pd.Series:
            """Safely fetch a column or return a default series if missing."""
            if col_name in data.columns:
                return data[col_name].fillna(fallback_val)
            return pd.Series(fallback_val, index=data.index)

        def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
            """Vectorized safe percentage change helper supporting sign-flips & zero division."""
            prev = series.groupby(data["Ticker"]).shift(periods)
            denom = prev.abs()
            return (series - prev).divide(denom).where(
                (denom != 0) & denom.notna(), np.nan
            )

        def safe_delta(series: pd.Series, periods: int) -> pd.Series:
            """Vectorized safe difference (delta) helper for margins/ratios."""
            prev = series.groupby(data["Ticker"]).shift(periods)
            return series - prev

        def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
            """Helper to prevent Division-by-Zero and np.inf errors."""
            return np.where(
                (denominator == 0) | (denominator.isna()),
                np.nan,
                numerator / denominator,
            )

        # -------------------------------------------------------------
        # BASE VARIABLE MAPPINGS & FALLBACK DERIVATIONS
        # -------------------------------------------------------------
        # Revenue (Primary SEC tag with fallback to Revenues)
        revenue = get_col("RevenueFromContractWithCustomerExcludingAssessedTax").where(
            lambda x: x != 0, get_col("Revenues")
        )

        # Net Income
        net_income = get_col("NetIncomeLoss").where(
            lambda x: x != 0, get_col("ProfitLoss")
        )

        # Depreciation & Amortization
        dna = get_col("DepreciationDepletionAndAmortization").where(
            lambda x: x != 0,
            get_col("DepreciationAndAmortization").where(
                lambda x: x != 0,
                get_col("Depreciation")
                + get_col("AmortizationOfIntangibleAssets"),
            ),
        )

        # Operating Cash Flow & CapEx
        cfo = get_col("NetCashProvidedByUsedInOperatingActivities")
        capex = get_col("PaymentsToAcquirePropertyPlantAndEquipment").abs()

        # Derived Free Cash Flow (use EDGAR tag if present, fallback to CFO - CapEx)
        fcf = get_col("FreeCashFlow").where(lambda x: x != 0, cfo - capex)

        # Operating Income & Derived EBITDA
        operating_income = get_col("OperatingIncomeLoss")
        sbc = get_col("ShareBasedCompensation").where(
            lambda x: x != 0,
            get_col(
                "SharebasedCompensationArrangementBySharebasedPaymentAwardCompensationCost1",
                get_col("AllocatedShareBasedCompensationExpense"),
            ),
        )

        restructuring = get_col("RestructuringCosts").where(
            lambda x: x != 0,
            get_col("RestructuringAndRelatedCostIncurredCost"),
        )

        impairments = (
            get_col("GoodwillImpairmentLoss")
            + get_col("ImpairmentOfIntangibleAssetsExcludingGoodwill")
            + get_col("InventoryWriteDown")
        )

        gains_losses = (
            get_col("GainLossOnSaleOfPropertyPlantEquipment")
            + get_col("GainLossOnSaleOfBusiness")
            + get_col("EquitySecuritiesFvNiGainLoss")
        )

        # EBITDA / Normalized EBITDA
        normalized_ebitda = get_col("NormalizedEBITDA").where(
            lambda x: x != 0,
            (operating_income + dna + sbc + restructuring + impairments)
            - gains_losses,
        )

        ebitda = operating_income + dna

        # Gross Profit & Cost of Goods Sold
        cogs = get_col("CostOfGoodsAndServicesSold").where(
            lambda x: x != 0, get_col("CostOfRevenue")
        )
        gross_profit = get_col("GrossProfit").where(
            lambda x: x != 0,
            get_col("GrossProfit_Calculated").where(
                lambda x: x != 0, revenue - cogs
            ),
        )

        # Diluted Shares
        shares_diluted = get_col(
            "WeightedAverageNumberOfDilutedSharesOutstanding"
        ).where(
            lambda x: x != 0,
            get_col("WeightedAverageNumberOfSharesOutstandingBasic"),
        )

        # Total Debt & Working Capital
        total_debt = get_col("TotalDebt").where(
            lambda x: x != 0,
            get_col("LongTermDebtCurrent") + get_col("LongTermDebtNoncurrent"),
        )

        working_capital = get_col("WorkingCapital").where(
            lambda x: x != 0,
            get_col("AssetsCurrent") - get_col("LiabilitiesCurrent"),
        )

        stockholders_equity = get_col("StockholdersEquity")

        # -------------------------------------------------------------
        # 1. Growth Velocity (YoY: 4 Quarters, QoQ: 1 Quarter)
        # -------------------------------------------------------------
        data["Revenue_QoQ_Growth"] = safe_pct_change(revenue, 1)
        data["EBITDA_YoY_Growth"] = safe_pct_change(normalized_ebitda, 4)
        data["EPS_YoY_Growth"] = safe_pct_change(
            get_col("EarningsPerShareDiluted"), 4
        )
        data["FCF_YoY_Growth"] = safe_pct_change(fcf, 4)
        data["Operating_CashFlow_YoY_Growth"] = safe_pct_change(cfo, 4)

        # -------------------------------------------------------------
        # 2. Fundamental Acceleration / Deceleration
        # -------------------------------------------------------------
        data["Revenue_YoY_Growth"] = safe_pct_change(revenue, 4)
        data["Revenue_Acceleration"] = (
            data.groupby("Ticker")["Revenue_YoY_Growth"]
            .diff(1)
            .clip(-1.0,1.0)
        )

        # -------------------------------------------------------------
        # 3. Margin Expansion / Contraction (Basis Point Deltas)
        # -------------------------------------------------------------
        data["Gross_Margin"] = safe_divide(gross_profit, revenue)
        data["Gross_Margin_YoY_Delta"] = safe_delta(data["Gross_Margin"], 4)

        data["EBITDA_Margin"] = safe_divide(ebitda, revenue)
        data["EBITDA_Margin_YoY_Delta"] = safe_delta(data["EBITDA_Margin"], 4)

        data["FCF_Margin"] = safe_divide(fcf, revenue)
        data["FCF_Margin_YoY_Delta"] = safe_delta(data["FCF_Margin"], 4)

        data["Operating_Margin"] = safe_divide(operating_income, revenue)
        data["Operating_Margin_QoQ_Delta"] = safe_delta(
            data["Operating_Margin"], 1
        )

        # -------------------------------------------------------------
        # 4. Capital Structure & Reinvestment Velocity
        # -------------------------------------------------------------
        data["Debt_YoY_Growth"] = safe_pct_change(total_debt, 4)
        data["Shares_Outstanding_YoY_Change"] = safe_pct_change(shares_diluted, 4)
        data["CapEx_YoY_Growth"] = safe_pct_change(capex, 4)

        # -------------------------------------------------------------
        # 5. Financial Health Trajectory Signals
        # -------------------------------------------------------------
        data["Return_On_Equity"] = safe_divide(net_income, stockholders_equity)
        data["ROIC_YoY_Delta"] = safe_delta(data["Return_On_Equity"], 4)
        data["Working_Capital_YoY_Change"] = safe_pct_change(working_capital, 4)

        # -------------------------------------------------------------
        # 6. Fundamental Stability & Volatility (8-Quarter Rolling Window)
        # -------------------------------------------------------------
        # Pre-compute base metrics if not already present in the incoming DataFrame
        if "Return_On_Equity" not in data.columns:
            data["Return_On_Equity"] = safe_divide(net_income, stockholders_equity)
        if "Net_Margin" not in data.columns:
            data["Net_Margin"] = safe_divide(net_income, revenue)

        data["ROE_Volatility_8Q"] = (
            data.groupby("Ticker")["Return_On_Equity"]
            .transform(lambda s: s.rolling(8, min_periods=4).std())
        )

        data["Margin_Volatility_8Q"] = (
            data.groupby("Ticker")["Net_Margin"]
            .transform(lambda s: s.rolling(8, min_periods=4).std())
        )

        # Clean extreme inf / -inf values generated by zero division
        if hasattr(self, "trend_columns") and self.trend_columns:
            data[self.trend_columns] = data[self.trend_columns].replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            # Fallback if self.trend_columns is not explicitly assigned
            trend_cols = [
                "Revenue_QoQ_Growth",
                "EBITDA_YoY_Growth",
                "EPS_YoY_Growth",
                "FCF_YoY_Growth",
                "Operating_CashFlow_YoY_Growth",
                "Revenue_YoY_Growth",
                "Revenue_Acceleration",
                "Gross_Margin_YoY_Delta",
                "EBITDA_Margin_YoY_Delta",
                "FCF_Margin_YoY_Delta",
                "Operating_Margin_QoQ_Delta",
                "Debt_YoY_Growth",
                "Shares_Outstanding_YoY_Change",
                "CapEx_YoY_Growth",
                "ROIC_YoY_Delta",
                "Working_Capital_YoY_Change",
                "ROE_Volatility_8Q",
                "Margin_Volatility_8Q"
            ]
            data[trend_cols] = data[trend_cols].replace([np.inf, -np.inf], np.nan)

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
        feature_df = self.load_train_data()[self.training_columns]
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
        df = feature_df[self.training_columns]

        # 1. Separate features and target
        X = df.drop(columns=[self.target_column])
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
            n_estimators=100,
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


    def test_model(self):
        # 1. Input
        X_test, y_test = self.load_test_data()
        #print(f"Test data:\n", X_test, y_test )

        # 3. Make Test predictions
        y_pred = self.xg_model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        print(f"Test Predictions: {predictions}\n")

        le = LabelEncoder()
        y_test_encoded = le.fit_transform(y_test)
        accuracy = accuracy_score(y_test_encoded, predictions)
        print("xg_model Accuracy: %.2f%%\n" % (accuracy * 100.0))


    def predict(self, tickerlist, input_df):
        out_pred = self.xg_model.predict(input_df)
        bhs_desc = [self.bhs_descs[value] for value in out_pred]
        # Map ticker to description into a dict
        ticker_bhs_map = dict(zip(tickerlist, bhs_desc))
        #print(f"BHS Predictions: {ticker_bhs_map}")

        # Or print line-by-line
        for ticker, desc in zip(tickerlist, bhs_desc):
          print(f"{ticker}: {desc}")


def main(args:list):
    print(f"=== Starting Weekly Training Job Execution: {datetime.now(ZoneInfo('America/New_York')).date()} ===")
    modelBuilder = ModelBuilder("xg",100)
    feature_df = modelBuilder.load_train_data()
    success = modelBuilder.train(feature_df)
    success = True
    if success == True:
        print("=== Training Job Completed Successfully ===")
        modelBuilder.init_runtime()
        modelBuilder.test_model()
    else:
        print("Model Building failed")


if __name__ == "__main__":
    main(sys.argv[1:])
