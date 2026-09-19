import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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
        self.symbols:list[str] = []
        # Explicit list of generated ratio columns
        self.snapshot_columns:list[str] = [
            # Margins & Profitability
            "Net_Margin",
            "FCF_Margin",
            "Operating_Margin",
            "EBITDA_Margin",
            "Return_On_Equity",
            "Return_On_Assets",
            # Balance Sheet & Leverage
            "Debt_To_Equity",
            "Debt_To_Assets",
            "Current_Ratio",
            "Quick_Ratio",
            "Interest_Coverage",
            # Efficiency & Cash Flow Quality
            "Asset_Turnover",
            "Working_Capital_Turnover",
            "CFO_To_NetIncome",
            "SBC_To_Revenue",
            "CapEx_To_CFO",
            "CapEx_To_Revenue",
            "R_And_D_To_Revenue",
            "Accrual_Ratio",
            # Accounting Quality & Risk
            "Payout_To_FCF",
            "NonOperating_Income_Reliance",
            "Goodwill_To_Assets",
            "Effective_Tax_Rate",
        ]

        # Collect trend feature columns for inf handling
        self.trend_columns:list[str] = [
            # -------------------------------------
            # 1. Growth & Momentum Trends
            # -------------------------------------
            "Revenue_YoY_Growth",
            "Revenue_QoQ_Growth",
            "EBITDA_YoY_Growth",
            "EPS_YoY_Growth",
            "FCF_YoY_Growth",
            "Operating_CashFlow_YoY_Growth",
            "Revenue_Acceleration",
            # -------------------------------------
            # 2. Trajectory Deltas & Volatility
            # -------------------------------------
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

        self.training_columns:list[str] = self.snapshot_columns + self.trend_columns + ["bhsScore"]
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
        raw_df = pd.read_csv(f"{AinySchema.DATA_DIR}/ainyfin_data.csv")
        raw_df = raw_df.sort_values(["Ticker", "Date"])
        snapshot_df = self.compute_financial_snapshot(raw_df)

        snapshot_df.to_csv(f"{AinySchema.DATA_DIR}/snapshot.csv", index=False)
        snapshot_df[["Ticker", "bhsScore"]] = raw_df[["Ticker", "bhsScore"]].replace([np.inf, -np.inf], np.nan)
        snapshot_df["Date"] = pd.to_datetime(snapshot_df["Date"])
        snapshot_df = snapshot_df.sort_values(["Date"]).reset_index(drop=True)

        financial_df = pd.read_csv(f"{AinySchema.DATA_DIR}/financial_data.csv")
        financial_df["Date"] = pd.to_datetime(financial_df["Date"])
        financial_df = financial_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        financial_trends_df = self.compute_financial_trends(financial_df)

        financial_trends_df = financial_trends_df.sort_values(["Date"]).reset_index(drop=True)
        financial_trends_df.to_csv(f"{AinySchema.DATA_DIR}/trends.csv", index=False)

        merged_df = pd.merge_asof(
            snapshot_df,
            financial_trends_df,
            left_on="Date",
            right_on="Date",
            by="Ticker",
            direction="backward",
            suffixes=("", "_Trends"),
        )
        self.symbols = merged_df["Ticker"].unique().tolist()
        return merged_df


    def compute_financial_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes standardized valuation, leverage, profitability, and quality ratios
        from raw SEC fundamental and market price data with robust outlier handling.
        """

        def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
            """Performs element-wise division, returning np.nan for invalid, zero, or inf values."""
            denom_clean = denominator.replace(0, np.nan)
            result = numerator / denom_clean
            return result.replace([np.inf, -np.inf], np.nan)

        def get_col(col_name: str, fallback_val=np.nan) -> pd.Series:
            """Safely fetch a column or return a default series if missing."""
            if col_name in data.columns:
                return pd.to_numeric(data[col_name], errors="coerce")
            return pd.Series(fallback_val, index=data.index)

        # Preserve index alignment upfront
        data = df.copy()

        # =========================================================================
        # PRE-COMPUTATION & TAG MAPPINGS
        # =========================================================================

        revenue = (
            get_col("RevenueFromContractWithCustomerExcludingAssessedTax")
            .where(lambda x: x.notna() & (x != 0), get_col("Revenues"))
            .where(lambda x: x.notna() & (x != 0), get_col("SalesRevenueNet"))
        )

        net_income = get_col("NetIncomeLoss").where(
            lambda x: x.notna() & (x != 0), get_col("ProfitLoss")
        )

        total_assets = get_col("Assets")
        current_assets = get_col("AssetsCurrent")
        current_liabilities = get_col("LiabilitiesCurrent")
        stockholders_equity = get_col("StockholdersEquity")

        working_capital = get_col("WorkingCapital").where(
            lambda x: x.notna() & (x != 0), current_assets - current_liabilities
        )

        total_debt = get_col("TotalDebt").where(
            lambda x: x.notna() & (x != 0),
            get_col("LongTermDebtCurrent").fillna(0)
            + get_col("LongTermDebtNoncurrent").fillna(0),
        )

        cash_and_equivalents = get_col(
            "CashCashEquivalentsAndShortTermInvestments"
        ).where(
            lambda x: x.notna() & (x != 0),
            get_col("CashCashEquivalentsAtCarryingValue").fillna(0)
            + get_col("ShortTermInvestments").fillna(0),
        )

        accounts_receivable = get_col("AccountsReceivableNetCurrent").where(
            lambda x: x.notna() & (x != 0), get_col("AccountsReceivableNet")
        )

        operating_income = get_col("OperatingIncomeLoss").where(
            lambda x: x.notna() & (x != 0),
            get_col("OperatingIncome").where(
                lambda x: x.notna() & (x != 0),
                revenue - get_col("OperatingExpenses"),
            ),
        )

        interest_expense = get_col("InterestExpense").where(
            lambda x: x.notna() & (x != 0),
            get_col("InterestExpenseNonoperating").where(
                lambda x: x.notna() & (x != 0),
                get_col("InterestExpenseOperating"),
            ),
        )

        ebit = operating_income.where(
            lambda x: x.notna() & (x != 0),
            net_income.fillna(0) + interest_expense.fillna(0),
        )

        dna = get_col("DepreciationDepletionAndAmortization").where(
            lambda x: x.notna() & (x != 0),
            get_col("DepreciationAndAmortization").where(
                lambda x: x.notna() & (x != 0),
                get_col("DepreciationAmortizationAndAccretion").where(
                    lambda x: x.notna() & (x != 0),
                    get_col("Depreciation").fillna(0)
                    + get_col("AmortizationOfIntangibleAssets").fillna(0),
                ),
            ),
        )

        sbc = get_col("ShareBasedCompensation").where(
            lambda x: x.notna() & (x != 0),
            get_col(
                "SharebasedCompensationArrangementBySharebasedPaymentAwardCompensationCost1"
            ).where(
                lambda x: x.notna() & (x != 0),
                get_col("AllocatedShareBasedCompensationExpense"),
            ),
        )

        restructuring = get_col("RestructuringCosts").where(
            lambda x: x.notna() & (x != 0),
            get_col("RestructuringAndRelatedCostIncurredCost"),
        )

        impairments = (
            get_col("GoodwillImpairmentLoss").fillna(0)
            + get_col("ImpairmentOfIntangibleAssetsExcludingGoodwill").fillna(0)
            + get_col("InventoryWriteDown").fillna(0)
        )

        gains_losses = (
            get_col("GainLossOnSaleOfPropertyPlantEquipment").fillna(0)
            + get_col("GainLossOnSaleOfBusiness").fillna(0)
            + get_col("EquitySecuritiesFvNiGainLoss").fillna(0)
        )

        ebitda_base = operating_income.fillna(0) + dna.fillna(0)
        normalized_ebitda = (
            ebitda_base
            + sbc.fillna(0)
            + restructuring.fillna(0)
            + impairments
            - gains_losses
        ).where(lambda x: x != 0, ebitda_base)

        gross_profit = get_col("GrossProfit").where(
            lambda x: x.notna() & (x != 0), get_col("GrossProfit_Calculated")
        )

        cfo = get_col("NetCashProvidedByUsedInOperatingActivities")
        capex = get_col("PaymentsToAcquirePropertyPlantAndEquipment").abs()
        fcf = get_col("FreeCashFlow").where(
            lambda x: x.notna() & (x != 0), cfo - capex
        )
        rd_expense = get_col("ResearchAndDevelopmentExpense")

        shares_diluted = get_col(
            "WeightedAverageNumberOfDilutedSharesOutstanding"
        ).where(
            lambda x: x.notna() & (x != 0),
            get_col("WeightedAverageNumberOfSharesOutstandingBasic"),
        )

        close = get_col("Close")
        eps = get_col("EarningsPerShareDiluted")

        # Base column population
        data["Revenue"] = revenue
        data["Net_Income"] = net_income
        data["Gross_Profit"] = gross_profit
        data["EBITDA"] = normalized_ebitda
        data["Free_Cash_Flow"] = fcf

        # Valuation setup
        pe_primary = safe_divide(close, eps)
        market_cap = close * shares_diluted
        pe_fallback = safe_divide(market_cap, net_income)
        pe_ratio = pe_primary.combine_first(pe_fallback)
        data["Price_To_Earnings"] = pe_ratio.where(pe_ratio > 0, np.nan).clip(0.0, 500.0)

        raw_de = safe_divide(total_debt, stockholders_equity)
        valid_de = raw_de.where(stockholders_equity > 0, np.nan)
        data["Debt_To_Equity"] = valid_de.clip(lower=0.0, upper=10.0)

        total_capital = total_debt + stockholders_equity
        debt_to_capital = safe_divide(total_debt, total_capital)
        data["Debt_To_Capital"] = debt_to_capital.where(
            total_capital > 0, np.nan
        ).clip(lower=0.0, upper=1.0)

        enterprise_value = (
            market_cap + total_debt.fillna(0) - cash_and_equivalents.fillna(0)
        )

        # =========================================================================
        # COMPUTED FINANCIAL RATIOS & OUTLIER MITIGATION
        # =========================================================================

        # 1. Valuation Ratios
        data["Price_To_FreeCashFlow"] = safe_divide(market_cap, fcf).where(
            fcf > 0, np.nan
        ).clip(0.0, 500.0)
        data["Price_To_Book"] = safe_divide(
            market_cap, stockholders_equity
        ).where(stockholders_equity > 0, np.nan).clip(0.0, 100.0)
        data["Price_To_Sales"] = safe_divide(market_cap, revenue).where(
            revenue > 0, np.nan
        ).clip(0.0, 100.0)
        data["EV_To_EBITDA"] = safe_divide(
            enterprise_value, normalized_ebitda
        ).where(normalized_ebitda > 0, np.nan).clip(0.0, 200.0)
        data["EV_To_EBIT"] = safe_divide(enterprise_value, ebit).where(
            ebit > 0, np.nan
        ).clip(0.0, 200.0)

        # 2. Profitability & Margins
        data["Return_On_Assets"] = safe_divide(net_income, total_assets).clip(-2.0, 2.0)
        data["Gross_Margin"] = safe_divide(gross_profit, revenue).clip(-1.0, 1.0)
        data["Operating_Margin"] = safe_divide(operating_income, revenue).clip(-2.0, 1.0)
        data["Net_Margin"] = safe_divide(net_income, revenue).clip(-2.0, 1.0)

        # 3. Leverage, Solvency & Liquidity
        data["Debt_To_Assets"] = safe_divide(total_debt, total_assets).clip(0.0, 5.0)
        data["Current_Ratio"] = safe_divide(current_assets, current_liabilities).clip(0.0, 20.0)
        data["Quick_Ratio"] = safe_divide(
            cash_and_equivalents + accounts_receivable, current_liabilities
        ).clip(0.0, 20.0)
        data["Interest_Coverage"] = safe_divide(
            ebit, interest_expense.abs()
        ).clip(-50.0, 100.0)
        data["Cash_To_Debt"] = safe_divide(cash_and_equivalents, total_debt).clip(0.0, 20.0)

        # 4. Operational Efficiency
        data["Asset_Turnover"] = safe_divide(revenue, total_assets).clip(0.0, 10.0)
        data["Working_Capital_Turnover"] = safe_divide(
            revenue, working_capital
        ).clip(-20.0, 20.0)

        # 5. Earnings Quality & Capital Allocation
        data["CFO_To_NetIncome"] = safe_divide(cfo, net_income).clip(-10.0, 10.0)
        data["SBC_To_Revenue"] = safe_divide(sbc, revenue).clip(0.0, 1.0)
        data["CapEx_To_CFO"] = safe_divide(capex, cfo).clip(0.0, 10.0)
        data["CapEx_To_Revenue"] = safe_divide(capex, revenue).clip(0.0, 1.0)
        data["R_And_D_To_Revenue"] = safe_divide(rd_expense, revenue).clip(0.0, 1.0)

        data["Accrual_Ratio"] = safe_divide(net_income - cfo, total_assets).clip(-2.0, 2.0)
        data["Payout_To_FCF"] = safe_divide(
            get_col("PaymentsOfDividendsCommonStock").abs().fillna(0)
            + get_col("PaymentsForRepurchaseOfCommonStock").abs().fillna(0),
            fcf,
        ).clip(-5.0, 5.0)

        data["NonOperating_Income_Reliance"] = safe_divide(
            get_col("OtherNonoperatingIncomeExpense").abs(), net_income.abs()
        ).clip(0.0, 5.0)

        data["Goodwill_To_Assets"] = safe_divide(get_col("Goodwill"), total_assets).clip(0.0, 1.0)

        impairment_col = "GoodwillImpairmentLoss"
        if impairment_col in data.columns:
            data["Had_Goodwill_Impairment"] = (
                data[impairment_col].fillna(0) > 0
            ).astype(int)
        else:
            data["Had_Goodwill_Impairment"] = np.nan

        tax_base = get_col(
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
        )
        data["Effective_Tax_Rate"] = safe_divide(get_col("IncomeTaxExpenseBenefit"), tax_base).clip(0.0, 0.50)

        # Downstream caller safeguard
        data = self.normalize_financial_snapshot(data, date_col="Date", sector_col="SIC")
        return data


    def normalize_financial_snapshot(self, df: pd.DataFrame, date_col:str ="Date", sector_col: str = "SIC") -> pd.DataFrame:
        """
        Applies sector normalization and outlier clipping to computed financial snapshot ratios.
        """

        data = df.copy()

        # 1. Define feature groups
        sector_norm_cols = [
            "EV_To_EBITDA",
            "EV_To_EBIT",
            "Price_To_Earnings",
            "Price_To_FreeCashFlow",
            "Price_To_Book",
            "Price_To_Sales",
            "CapEx_To_Revenue",
            "R_And_D_To_Revenue",
            "SBC_To_Revenue",
            "Asset_Turnover",
            "Working_Capital_Turnover",
            "Debt_To_Equity",
            "Debt_To_Assets",
        ]

        unbounded_ratio_cols = [
            "CapEx_To_CFO",
            "CFO_To_NetIncome",
            "Interest_Coverage",
            "Accrual_Ratio",
            "NonOperating_Income_Reliance",
        ]

        # 2. Apply Sector Z-Scores
        if sector_col in data.columns:
            for col in sector_norm_cols:
                if col in data.columns:
                    mean = data.groupby([date_col, sector_col])[col].transform("mean")
                    std = data.groupby([date_col, sector_col])[col].transform("std")

                    # Avoid division by zero or NaN std
                    std = std.replace(0, np.nan).fillna(1.0)
                    mean = mean.fillna(0.0)

                    # Compute Sector Z-score
                    data[f"{col}_SectorZ"] = (data[col] - mean) / std
                    data[f"{col}_SectorZ"] = data[f"{col}_SectorZ"].clip(-4.0, 4.0)

        # 3. Clip extreme ratio outliers (Winsorization)
        for col in unbounded_ratio_cols:
            if col in data.columns:
                lower = data[col].quantile(0.01)
                upper = data[col].quantile(0.99)
                data[col] = data[col].clip(lower=lower, upper=upper)

        return data


    def compute_financial_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        # Sort and preserve index integrity
        data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        def _get_col(name: str, fallback=0.0) -> pd.Series:
            if name in data.columns:
                return pd.to_numeric(data[name], errors="coerce").fillna(fallback)
            return pd.Series(fallback, index=data.index)

        def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
            res = np.where((den == 0) | den.isna(), np.nan, num / den)
            return pd.Series(res, index=data.index)

        # Re-grouping dynamically on the sorted DataFrame
        grouped = data.groupby("Ticker")

        def _pct_change(series: pd.Series, periods: int, max_clip: float = 5.0) -> pd.Series:
            # Group directly using the sorted index alignment
            prev = series.groupby(data["Ticker"]).shift(periods)
            denom = prev.abs()
            res = (series - prev).divide(denom).where((denom != 0) & denom.notna(), np.nan)
            return pd.Series(res, index=data.index).clip(-max_clip, max_clip)

        def _delta(series: pd.Series, periods: int) -> pd.Series:
            prev = series.groupby(data["Ticker"]).shift(periods)
            return pd.Series(series - prev, index=data.index)

        # --- BASE VARIABLE MAPPINGS ---
        rev_contract = _get_col("RevenueFromContractWithCustomerExcludingAssessedTax")
        rev_gen = _get_col("Revenues")
        revenue = pd.Series(np.where(rev_contract != 0, rev_contract, rev_gen), index=data.index)

        net_inc_raw = _get_col("NetIncomeLoss")
        profit_loss = _get_col("ProfitLoss")
        net_income = pd.Series(np.where(net_inc_raw != 0, net_inc_raw, profit_loss), index=data.index)

        dna = _get_col("DepreciationDepletionAndAmortization")
        dna_alt = _get_col("DepreciationAndAmortization")
        dna_sum = _get_col("Depreciation") + _get_col("AmortizationOfIntangibleAssets")
        dna_final = pd.Series(np.where(dna != 0, dna, np.where(dna_alt != 0, dna_alt, dna_sum)), index=data.index)

        cfo = _get_col("NetCashProvidedByUsedInOperatingActivities")
        capex = _get_col("PaymentsToAcquirePropertyPlantAndEquipment").abs()
        fcf_tag = _get_col("FreeCashFlow")
        fcf = pd.Series(np.where(fcf_tag != 0, fcf_tag, cfo - capex), index=data.index)

        op_inc = _get_col("OperatingIncomeLoss")
        sbc_arr = _get_col("SharebasedCompensationArrangementBySharebasedPaymentAwardCompensationCost1")
        sbc_alloc = _get_col("AllocatedShareBasedCompensationExpense")
        sbc = _get_col("ShareBasedCompensation").where(lambda x: x != 0, np.where(sbc_arr != 0, sbc_arr, sbc_alloc))

        restruct = _get_col("RestructuringCosts").where(
            lambda x: x != 0, _get_col("RestructuringAndRelatedCostIncurredCost")
        )
        impairments = (
            _get_col("GoodwillImpairmentLoss")
            + _get_col("ImpairmentOfIntangibleAssetsExcludingGoodwill")
            + _get_col("InventoryWriteDown")
        )
        gains_losses = (
            _get_col("GainLossOnSaleOfPropertyPlantEquipment")
            + _get_col("GainLossOnSaleOfBusiness")
            + _get_col("EquitySecuritiesFvNiGainLoss")
        )

        norm_ebitda = pd.Series((op_inc + dna_final + sbc + restruct + impairments) - gains_losses, index=data.index)
        ebitda = pd.Series(op_inc + dna_final, index=data.index)

        cogs = _get_col("CostOfGoodsAndServicesSold").where(lambda x: x != 0, _get_col("CostOfRevenue"))
        gp_calc = _get_col("GrossProfit_Calculated").where(lambda x: x != 0, revenue - cogs)
        gross_profit = pd.Series(_get_col("GrossProfit").where(lambda x: x != 0, gp_calc), index=data.index)

        shares_diluted = pd.Series(
            _get_col("WeightedAverageNumberOfDilutedSharesOutstanding").where(
                lambda x: x != 0, _get_col("WeightedAverageNumberOfSharesOutstandingBasic")
            ),
            index=data.index,
        )

        total_debt = pd.Series(
            _get_col("TotalDebt").where(
                lambda x: x != 0, _get_col("LongTermDebtCurrent") + _get_col("LongTermDebtNoncurrent")
            ),
            index=data.index,
        )

        working_cap = pd.Series(
            _get_col("WorkingCapital").where(
                lambda x: x != 0, _get_col("AssetsCurrent") - _get_col("LiabilitiesCurrent")
            ),
            index=data.index,
        )

        equity = _get_col("StockholdersEquity")
        eps_diluted = _get_col("EarningsPerShareDiluted")

        # --- FEATURE COMPUTATION ---
        data["Revenue_QoQ_Growth"] = _pct_change(revenue, 1, max_clip=2.0)
        data["Revenue_YoY_Growth"] = _pct_change(revenue, 4, max_clip=5.0)
        data["EBITDA_YoY_Growth"] = _pct_change(norm_ebitda, 4, max_clip=5.0)
        data["EPS_YoY_Growth"] = _pct_change(eps_diluted, 4, max_clip=5.0)
        data["FCF_YoY_Growth"] = _pct_change(fcf, 4, max_clip=5.0)
        data["Operating_CashFlow_YoY_Growth"] = _pct_change(cfo, 4, max_clip=5.0)

        data["Revenue_Acceleration"] = data.groupby("Ticker")["Revenue_YoY_Growth"].diff(1).clip(-1.0, 1.0)

        data["Gross_Margin"] = _safe_divide(gross_profit, revenue).clip(-1.0, 1.0)
        data["Gross_Margin_YoY_Delta"] = _delta(data["Gross_Margin"], 4).clip(-0.5, 0.5)

        data["EBITDA_Margin"] = _safe_divide(ebitda, revenue).clip(-2.0, 1.0)
        data["EBITDA_Margin_YoY_Delta"] = _delta(data["EBITDA_Margin"], 4).clip(-0.5, 0.5)

        data["FCF_Margin"] = _safe_divide(fcf, revenue).clip(-2.0, 1.0)
        data["FCF_Margin_YoY_Delta"] = _delta(data["FCF_Margin"], 4).clip(-0.5, 0.5)

        data["Operating_Margin"] = _safe_divide(op_inc, revenue).clip(-2.0, 1.0)
        data["Operating_Margin_QoQ_Delta"] = _delta(data["Operating_Margin"], 1).clip(-0.5, 0.5)

        data["Debt_YoY_Growth"] = _pct_change(total_debt, 4, max_clip=5.0)
        data["Shares_Outstanding_YoY_Change"] = _pct_change(shares_diluted, 4, max_clip=2.0)
        data["CapEx_YoY_Growth"] = _pct_change(capex, 4, max_clip=5.0)

        data["Return_On_Equity"] = _safe_divide(net_income, equity).clip(-5.0, 5.0)
        data["ROIC_YoY_Delta"] = _delta(data["Return_On_Equity"], 4).clip(-1.0, 1.0)
        data["Working_Capital_YoY_Change"] = _pct_change(working_cap, 4, max_clip=5.0)

        data["Net_Margin"] = _safe_divide(net_income, revenue).clip(-2.0, 1.0)

        data["ROE_Volatility_8Q"] = (
            data.groupby("Ticker")["Return_On_Equity"]
            .transform(lambda s: s.rolling(8, min_periods=4).std())
            .clip(0.0, 2.0)
        )

        data["Margin_Volatility_8Q"] = (
            data.groupby("Ticker")["Net_Margin"]
            .transform(lambda s: s.rolling(8, min_periods=4).std())
            .clip(0.0, 2.0)
        )

        # Execute downstream transformations safely
        data = self.normalize_financial_trends(data, date_col="Date", sector_col="SIC")
        data = self.build_composite_valuation_features(data)

        return data


    def normalize_financial_trends(self, df: pd.DataFrame, date_col:str ='Date', sector_col: str = "SIC") -> pd.DataFrame:
        """
        Applies cross-sectional sector Z-scoring to computed trend metrics
        to eliminate sectoral bias in growth rates and margin volatility.
        """
        data = df.copy()

        trend_cols_to_normalize = [
            "Revenue_YoY_Growth",
            "EBITDA_YoY_Growth",
            "EPS_YoY_Growth",
            "FCF_YoY_Growth",
            "Gross_Margin_YoY_Delta",
            "EBITDA_Margin_YoY_Delta",
            "Operating_Margin_QoQ_Delta",
            "CapEx_YoY_Growth",
            "ROIC_YoY_Delta",
            "ROE_Volatility_8Q",
            "Margin_Volatility_8Q",
        ]

        if sector_col in data.columns:
            for col in trend_cols_to_normalize:
                if col in data.columns:
                    mean = data.groupby([date_col, sector_col])[col].transform("mean")
                    std = data.groupby([date_col, sector_col])[col].transform("std")

                    # Avoid division by zero or NaN std
                    std = std.replace(0, np.nan).fillna(1.0)
                    mean = mean.fillna(0.0)

                    # Compute Sector Z-score
                    data[f"{col}_SectorZ"] = (data[col] - mean) / std
                    data[f"{col}_SectorZ"] = data[f"{col}_SectorZ"].clip(-4.0, 4.0)

        return data

    def build_composite_valuation_features(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines individual sector-normalized valuation multiples into non-redundant
        composite rank features to reduce multi-collinearity in tree models.
        """
        data = df.copy()

        val_z_cols = [
            c
            for c in [
                "EV_To_EBITDA_SectorZ",
                "Price_To_Earnings_SectorZ",
                "Price_To_FreeCashFlow_SectorZ",
                "Price_To_Book_SectorZ",
            ]
            if c in data.columns
        ]

        if val_z_cols:
            # Lower Z-score = Cheaper stock relative to sector
            data["Composite_Valuation_SectorZ"] = data[val_z_cols].mean(axis=1)

        # 2. Earnings & Cash Yields (Inverses of P/E and P/FCF are naturally linear and non-explosive)
        if "Price_To_Earnings" in data.columns:
            data["Earnings_Yield"] = (
                (1.0 / data["Price_To_Earnings"]).replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
            )

        if "Price_To_FreeCashFlow" in data.columns:
            data["FCF_Yield"] = (
                (1.0 / data["Price_To_FreeCashFlow"])
                .replace([np.inf, -np.inf], np.nan)
                .clip(-0.5, 0.5)
            )

        return data



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
            # Predict on validation fold
            val_preds = fold_model.predict(X_val_fold)

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
            n_estimators=150,
            max_depth=3,  # Reduce depth from 5 to 3
            learning_rate=0.03,
            subsample=0.8,  # Row sampling to improve generalization
            colsample_bytree=0.8,  # Feature sampling per tree
            reg_alpha=1.0,  # L1 regularization (sparsity)
            reg_lambda=2.0,  # Increased L2 regularization to stabilize collinear features
            objective="multi:softprob",  # Explicitly set for 3-class target (Buy/Hold/Sell)
            num_class=3,  # Number of target categories
            random_state=42,
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
        #print(f"Test Predictions: {predictions}\n")
        out_pred_desc = [AinySchema.BHS_DESCS[value] for value in y_pred]
        for ticker, desc in zip(self.symbols, out_pred_desc):
          print(f"{ticker}: {desc}")

        le = LabelEncoder()
        y_test_encoded = le.fit_transform(y_test)
        accuracy = accuracy_score(y_test_encoded, predictions)
        print("xg_model Accuracy: %.2f%%\n" % (accuracy * 100.0))
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))


    def predict(self, tickerlist, input_df):
        out_pred = self.xg_model.predict(input_df)
        out_pred_desc = [AinySchema.BHS_DESCS[value] for value in out_pred]
        # Map ticker to description into a dict
        ticker_bhs_map = dict(zip(tickerlist, out_pred_desc))
        #print(f"BHS Predictions: {ticker_bhs_map}")

        # Or print line-by-line
        for ticker, desc in zip(tickerlist, out_pred_desc):
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
