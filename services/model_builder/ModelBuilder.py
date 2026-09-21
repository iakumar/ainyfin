"""
AinyFin ModelBuilder
--------------------

Purpose:
    Train and evaluate the AinyFin stock classification model.

Architecture:
    Fundamentals + Fundamental Trends + Market State
                         |
                         v
                    XGBoost
                         |
                         v
                 Buy / Hold / Sell
                         |
                         v
            Probability + Confidence

Important:
    This class is deliberately designed for time-series financial data.

    It does NOT use random train/test splitting for evaluation.
    Primary validation is chronological / walk-forward.

    Expected target:
        bhsScore
        1 = Sell
        2 = Hold
        3 = Buy

    If your target mapping differs, change TARGET_LABELS below.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from xgboost import XGBClassifier
from services.consts.AinySchema import AinySchema

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class ModelConfig:

    # -----------------------------------------------------------------
    # Files
    # -----------------------------------------------------------------

    # ainyfin_data.csv is the daily price/target source.
    training_file: str = f"{AinySchema.DATA_DIR}/ainyfin_data.csv"

    # financial_data.csv is the authoritative SEC fundamental source.
    financial_file: str = f"{AinySchema.DATA_DIR}financial_data.csv"

    model_file: str = "models/ainyfin_xgb_model.json"

    # -----------------------------------------------------------------
    # Target
    # -----------------------------------------------------------------

    target_column: str = "bhsScore"

    # Your current mapping:
    # 1 = Sell
    # 2 = Hold
    # 3 = Buy
    #
    # XGBoost requires zero-based class labels.
    #
    # Therefore:
    #   Sell = 0
    #   Hold = 1
    #   Buy  = 2
    target_labels: dict[int, str] = None

    # -----------------------------------------------------------------
    # Date / ticker
    # -----------------------------------------------------------------

    ticker_column: str = "Ticker"
    date_column: str = "Date"

    # -----------------------------------------------------------------
    # Walk-forward validation
    # -----------------------------------------------------------------

    n_walk_forward_folds: int = 5

    # Minimum training observations.
    min_train_rows: int = 50000

    # -----------------------------------------------------------------
    # Chronological test
    # -----------------------------------------------------------------

    chronological_test_fraction: float = 0.20

    # -----------------------------------------------------------------
    # XGBoost
    # -----------------------------------------------------------------

    n_estimators: int = 600

    max_depth: int = 6

    learning_rate: float = 0.04

    subsample: float = 0.80

    colsample_bytree: float = 0.80

    min_child_weight: int = 5

    reg_alpha: float = 0.05

    reg_lambda: float = 1.0

    objective: str = "multi:softprob"

    eval_metric: str = "mlogloss"

    random_state: int = 42

    n_jobs: int = -1

    # -----------------------------------------------------------------
    # Feature cleaning
    # -----------------------------------------------------------------

    clip_feature_min: float = -1e6

    clip_feature_max: float = 1e6

    # -----------------------------------------------------------------
    # Signal
    # -----------------------------------------------------------------

    # These are deliberately NOT used as model-training thresholds.
    # They are only reporting defaults.
    no_trade_confidence: float = 0.50


# ---------------------------------------------------------------------
# Main ModelBuilder
# ---------------------------------------------------------------------

class ModelBuilder:
    """
    AinyFin model training / validation / prediction engine.
    """

    def __init__(self, builder: str):
        self.builder = builder
        self.config = ModelConfig()

        if self.config.target_labels is None:
            self.config.target_labels = {
                0: "SELL",
                1: "HOLD",
                2: "BUY",
            }

        # -------------------------------------------------------------
        # Model features
        # -------------------------------------------------------------

        self.snapshot_columns = [

            # ---------------------------------------------------------
            # Valuation
            # ---------------------------------------------------------

            "Price_To_Earnings",
            "Price_To_FreeCashFlow",
            "Price_To_Book",
            "Price_To_Sales",

            "EV_To_EBITDA",
            "EV_To_EBIT",

            # ---------------------------------------------------------
            # Profitability
            # ---------------------------------------------------------

            "Gross_Margin",
            "Net_Margin",
            "FCF_Margin",
            "Operating_Margin",
            "EBITDA_Margin",

            "Return_On_Equity",
            "Return_On_Assets",
            "Return_On_Invested_Capital",

            # ---------------------------------------------------------
            # Leverage / liquidity
            # ---------------------------------------------------------

            "Debt_To_Equity",
            "Debt_To_Assets",

            "Current_Ratio",
            "Quick_Ratio",

            "Interest_Coverage",
            "Cash_To_Debt",

            "Net_Debt_To_EBITDA",

            # ---------------------------------------------------------
            # Efficiency
            # ---------------------------------------------------------

            "Asset_Turnover",
            "Working_Capital_Turnover",

            # ---------------------------------------------------------
            # Cash flow quality
            # ---------------------------------------------------------

            "CFO_To_NetIncome",
            "CapEx_To_CFO",
            "CapEx_To_Revenue",

            # ---------------------------------------------------------
            # Accounting / quality
            # ---------------------------------------------------------

            "SBC_To_Revenue",
            "Accrual_Ratio",

            "Goodwill_To_Assets",
            "Intangibles_Plus_Goodwill_To_Assets",

            "Effective_Tax_Rate",
            "NonOperating_Income_Reliance",

            # ---------------------------------------------------------
            # Capital allocation
            # ---------------------------------------------------------

            "Payout_To_FCF",
            "Buyback_To_FCF",
            "Reinvestment_Rate",

            # ---------------------------------------------------------
            # Other fundamental indicators
            # ---------------------------------------------------------

            "R_And_D_To_Revenue",
            "SGA_Intensity",

            "Retained_Earnings_To_Assets",
            "AOCI_To_Equity",

            "Operating_Lease_To_Assets",

            "Deferred_Tax_To_NetIncome",
            "Cash_Vs_Book_Tax_Gap",

            "Dividend_Yield",
            "Dividend_Per_Share_YoY_Growth",
            "Buyback_Yield",

            "Had_Goodwill_Impairment",
        ]

        # -------------------------------------------------------------
        # Fundamental trend features
        # -------------------------------------------------------------

        self.trend_columns = [

            "Revenue_YoY_Growth",
            "Revenue_QoQ_Growth",
            "Revenue_Acceleration",

            "NetIncome_YoY_Growth",
            "EPS_YoY_Growth",
            "EBITDA_YoY_Growth",
            "FCF_YoY_Growth",
            "Operating_CashFlow_YoY_Growth",

            "Gross_Margin_YoY_Delta",
            "EBITDA_Margin_YoY_Delta",
            "FCF_Margin_YoY_Delta",
            "Operating_Margin_QoQ_Delta",

            "ROA_YoY_Delta",
            "ROE_YoY_Delta",

            # Correctly calculated ROIC trend.
            "ROIC_YoY_Delta",

            "Debt_YoY_Growth",

            "Shares_Outstanding_YoY_Change",

            "CapEx_YoY_Growth",

            "Working_Capital_YoY_Change",

            "ROE_Volatility_8Q",
            "Margin_Volatility_8Q",

            # Working capital / operating efficiency
            "DSO",
            "DIO",
            "DPO",

            "AR_Growth_vs_Revenue_Growth",
            "Inventory_Growth_vs_Revenue_Growth",

            "Cash_Conversion_Cycle",

            "Billings_Proxy_YoY",

            "Deferred_Revenue_To_Revenue",
            "Deferred_Revenue_YoY_Growth",

            "Acquisition_Intensity",

            "Net_Debt_Issuance_To_Assets",

            "Realized_Vol_60D",

            "Daily_Range_Pct",

            "Dollar_Volume_Log",
            "Volume_Vs_60D_Avg",

            "Momentum_1M",
            "Momentum_3M",
            "Momentum_12M_1M",

            "Price_Vs_52W_High",

        ]

        # -------------------------------------------------------------
        # Sector normalized features
        # -------------------------------------------------------------
        #
        # These are retained because they are already part of your
        # current architecture.
        #
        # IMPORTANT:
        # Sector normalization should be performed using only
        # information available at that date.
        #
        # The method below normalizes cross-sectionally by date/sector
        # when Sector is available.
        # -------------------------------------------------------------

        self.sector_base_columns = [

            "Revenue_YoY_Growth",
            "EPS_YoY_Growth",
            "EBITDA_YoY_Growth",
            "FCF_YoY_Growth",

            "Gross_Margin",
            "Net_Margin",
            "Operating_Margin",
            "FCF_Margin",

            "Gross_Margin_YoY_Delta",
            "EBITDA_Margin_YoY_Delta",
            "FCF_Margin_YoY_Delta",
            "Operating_Margin_QoQ_Delta",

            "ROA_YoY_Delta",
            "ROE_YoY_Delta",

            "Debt_To_Assets",
            "Debt_To_Equity",

            "Price_To_Earnings",
            "Price_To_Sales",
            "Price_To_Book",
            "Price_To_FreeCashFlow",

            "CapEx_YoY_Growth",

            "Working_Capital_Turnover",
            "Working_Capital_YoY_Change",

            "Margin_Volatility_8Q",
            "ROE_Volatility_8Q",

            "Cash_To_Debt",

            "Net_Debt_To_EBITDA",

            "EV_To_EBITDA",
            "EV_To_EBIT",

            "FCF_Margin",

            "Asset_Turnover",

            "Cash_Conversion_Cycle",

            "DSO",
            "DIO",
            "DPO",

            "R_And_D_To_Revenue",

            "Billings_Proxy_YoY",

            "Reinvestment_Rate",

            "Buyback_Yield",

            "Dividend_Yield",

            "Intangibles_Plus_Goodwill_To_Assets",

            "Deferred_Revenue_To_Revenue",

        ]

        # Remove accidental duplicates while preserving order.
        self.available_training_columns = list(dict.fromkeys(self.snapshot_columns + self.trend_columns))

        self.model:XGBClassifier = None

        self.feature_importance:pd.DataFrame = None

        self.feature_columns: list[str] = []
        print(
            f"ModelBuilder initialized with "
            f"{len(self.available_training_columns)} model features."
        )

    # =================================================================
    # Utility methods
    # =================================================================

    @staticmethod
    def _safe_divide(
        numerator,
        denominator,
        default=np.nan,
    ):
        """
        Safe element-wise division.
        """

        numerator = pd.to_numeric(
            numerator,
            errors="coerce",
        )

        denominator = pd.to_numeric(
            denominator,
            errors="coerce",
        )

        result = numerator / denominator.replace(
            0,
            np.nan,
        )

        if default is not np.nan:
            result = result.fillna(default)

        return result

    @staticmethod
    def _winsorize_series(
        series: pd.Series,
        lower: float = 0.01,
        upper: float = 0.99,
    ) -> pd.Series:

        if series.dropna().empty:
            return series

        low = series.quantile(lower)
        high = series.quantile(upper)

        return series.clip(
            lower=low,
            upper=high,
        )

    @staticmethod
    def _yoy(series: pd.Series) -> pd.Series:

        return series.pct_change(
            periods=4
        )

    @staticmethod
    def _qoq(series: pd.Series) -> pd.Series:

        return series.pct_change(
            periods=1
        )

    @staticmethod
    def _delta(
        series: pd.Series,
        periods: int = 4,
    ) -> pd.Series:

        return series - series.shift(periods)

    @staticmethod
    def _rolling_volatility(
        series: pd.Series,
        window: int = 8,
    ) -> pd.Series:

        return series.rolling(
            window=window,
            min_periods=max(3, window // 2),
        ).std()

    # =================================================================
    # Correct ROIC
    # =================================================================

    def compute_roic(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate actual ROIC.

        ROIC ≈ NOPAT / Invested Capital

        NOPAT:
            EBIT * (1 - effective tax rate)

        Invested capital:
            Debt + Equity - Cash

        This intentionally does NOT use ROE.

        Previous versions had ROIC_YoY_Delta derived from ROE,
        which made the feature name misleading.
        """

        ebit = self._first_existing(
            data,
            [
                "OperatingIncome",
                "Operating_Income",
                "EBIT",
                "OperatingIncomeLoss",
            ],
        )

        debt = self._first_existing(
            data,
            [
                "TotalDebt",
                "Debt",
                "LongTermDebt",
                "LongTermDebtNoncurrent",
                "LongTermDebtCurrent",
            ],
        )

        equity = self._first_existing(
            data,
            [
                "StockholdersEquity",
                "Stockholders_Equity",
                "StockholdersEquityAbstract",
                "TotalEquity",
                "Equity",
            ],
        )

        cash = self._first_existing(
            data,
            [
                "CashAndCashEquivalents",
                "CashAndShortTermInvestments",
                "CashCashEquivalentsAndShortTermInvestments",
                "CashAndCashEquivalentsAtCarryingValue",
            ],
        )

        tax_rate = self._first_existing(
            data,
            [
                "Effective_Tax_Rate",
                "Cash_Tax_Rate",
            ],
        )

        if ebit is None:
            return pd.Series(
                np.nan,
                index=data.index,
                name="Return_On_Invested_Capital",
            )

        if debt is None:
            debt = pd.Series(
                0.0,
                index=data.index,
            )

        if equity is None:
            return pd.Series(
                np.nan,
                index=data.index,
                name="Return_On_Invested_Capital",
            )

        if cash is None:
            cash = pd.Series(
                0.0,
                index=data.index,
            )

        if tax_rate is None:
            tax_rate = pd.Series(
                0.21,
                index=data.index,
            )

        tax_rate = pd.to_numeric(
            tax_rate,
            errors="coerce",
        )

        # Prevent pathological tax rates from destroying ROIC.
        tax_rate = tax_rate.clip(
            lower=0.0,
            upper=0.50,
        )

        nopat = (
            pd.to_numeric(
                ebit,
                errors="coerce",
            )
            * (1.0 - tax_rate)
        )

        invested_capital = (
            pd.to_numeric(
                debt,
                errors="coerce",
            )
            +
            pd.to_numeric(
                equity,
                errors="coerce",
            )
            -
            pd.to_numeric(
                cash,
                errors="coerce",
            )
        )

        invested_capital = invested_capital.where(
            invested_capital > 0
        )

        roic = self._safe_divide(
            nopat,
            invested_capital,
        )

        return roic.clip(
            lower=-5.0,
            upper=5.0,
        )

    # =================================================================
    # Column helpers
    # =================================================================

    @staticmethod
    def _first_existing(
        data: pd.DataFrame,
        candidates: list[str],
    ):
        for column in candidates:
            if column in data.columns:
                series = data[column]

                # Prefer the first candidate that actually contains data
                if series.notna().any():
                    return series

        return None

    # =================================================================
    # Fundamental snapshot
    # =================================================================

    def compute_financial_snapshot(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        data = data.copy()

        # -------------------------------------------------------------
        # Basic numeric conversion
        # -------------------------------------------------------------

        for column in data.columns:
            if column in [
                self.config.ticker_column,
                self.config.date_column,
                "Sector",
            ]:
                continue

            if (data[column].dtype == "object"):
                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce",
                )

        # -------------------------------------------------------------
        # Profitability
        # -------------------------------------------------------------

        revenue = self._first_existing(
            data,
            [
                "RevenueFromContractWithCustomerExcludingAssessedTax",  # Primary ASC 606 GAAP tag
                "Revenues",                                             # Aggregate GAAP tag
                "SalesRevenueNet",                                      # Standard product/service sales tag
                "TotalRevenue_Consolidated",                            # Consolidated dataset tag
                "RevenueFromContractWithCustomerIncludingAssessedTax", # Alternative ASC 606 tag
            ],
        )

        gross_profit = self._first_existing(
            data,
            [
                "GrossProfit",
                "Gross_Profit",
            ],
        )

        operating_income = self._first_existing(
            data,
            [
                "OperatingIncome",
                "Operating_Income",
                "OperatingIncomeLoss",
            ],
        )

        net_income = self._first_existing(
            data,
            [
                "NetIncome",
                "NetIncomeLoss",
            ],
        )

        ebitda = self._first_existing(
            data,
            [
                "EBITDA",
                "EBITDA_Calculated",
            ],
        )

        fcf = self._first_existing(
            data,
            [
                "FreeCashFlow",
                "FCF",
            ],
        )

        # financial_data.csv does not contain a standard EBITDA column.
        # Derive EBITDA when possible from operating income plus D&A.
        if ebitda is None and operating_income is not None:
            da = self._first_existing(
                data,
                [
                    "DepreciationAndAmortization",
                    "DepreciationDepletionAndAmortization",
                    "DepreciationAmortizationAndAccretionNet",
                    "Depreciation",
                ],
            )
            if da is not None:
                data["EBITDA_Calculated"] = (
                    pd.to_numeric(operating_income, errors="coerce")
                    + pd.to_numeric(da, errors="coerce").abs()
                )
                ebitda = data["EBITDA_Calculated"]

        if revenue is not None:

            data["Gross_Margin"] = (
                self._safe_divide(
                    gross_profit,
                    revenue,
                )
                if gross_profit is not None
                else np.nan
            )

            data["Operating_Margin"] = (
                self._safe_divide(
                    operating_income,
                    revenue,
                )
                if operating_income is not None
                else np.nan
            )

            data["Net_Margin"] = (
                self._safe_divide(
                    net_income,
                    revenue,
                )
                if net_income is not None
                else np.nan
            )

            data["FCF_Margin"] = (
                self._safe_divide(
                    fcf,
                    revenue,
                )
                if fcf is not None
                else np.nan
            )

            data["EBITDA_Margin"] = (
                self._safe_divide(
                    ebitda,
                    revenue,
                )
                if ebitda is not None
                else np.nan
            )

        # -------------------------------------------------------------
        # Balance sheet
        # -------------------------------------------------------------

        equity = self._first_existing(
            data,
            [
                "StockholdersEquity",
                "Stockholders_Equity",
                "StockholdersEquityAbstract",
                "TotalEquity",
                "Equity",
            ],
        )

        assets = self._first_existing(
            data,
            [
                "Assets",
                "TotalAssets",
            ],
        )

        current_assets = self._first_existing(
            data,
            [
                "AssetsCurrent",
                "CurrentAssets",
            ],
        )

        current_liabilities = self._first_existing(
            data,
            [
                "LiabilitiesCurrent",
                "CurrentLiabilities",
            ],
        )

        total_debt = self._first_existing(
            data,
            [
                "TotalDebt",
                "Debt",
            ],
        )

        cash = self._first_existing(
            data,
            [
                "CashAndCashEquivalents",
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsAndShortTermInvestments",
                "Cash",
                "CashAndShortTermInvestments",
            ],
        )

        if equity is not None:

            data["Return_On_Equity"] = (
                self._safe_divide(
                    net_income,
                    equity,
                )
                if net_income is not None
                else np.nan
            )

        if assets is not None:

            data["Return_On_Assets"] = (
                self._safe_divide(
                    net_income,
                    assets,
                )
                if net_income is not None
                else np.nan
            )

        if total_debt is not None and equity is not None:

            data["Debt_To_Equity"] = self._safe_divide(
                total_debt,
                equity,
            )

        if total_debt is not None and assets is not None:

            data["Debt_To_Assets"] = self._safe_divide(
                total_debt,
                assets,
            )

        if current_assets is not None and current_liabilities is not None:

            data["Current_Ratio"] = self._safe_divide(
                current_assets,
                current_liabilities,
            )

            working_capital = (
                current_assets
                - current_liabilities
            )

            data["_Working_Capital"] = working_capital

        # -------------------------------------------------------------
        # Quick ratio
        # -------------------------------------------------------------

        inventory = self._first_existing(
            data,
            [
                "Inventory",
                "InventoryNet",
            ],
        )

        if (
            current_assets is not None
            and current_liabilities is not None
        ):

            if inventory is not None:

                quick_assets = (
                    current_assets
                    - inventory
                )

            else:

                quick_assets = current_assets

            data["Quick_Ratio"] = self._safe_divide(
                quick_assets,
                current_liabilities,
            )

            interest_expense = self._first_existing(data, ["InterestExpense",
                                                           "InterestExpenseNonoperating",
                                                           "InterestExpenseDebt",
                                                           "FinancingInterestExpense"])
            if interest_expense is not None and ebitda is not None:
                data["Interest_Coverage"] = self._safe_divide(
                    ebitda,
                    pd.to_numeric(interest_expense, errors="coerce").abs()
                ).clip(-50.0, 100.0)

        # -------------------------------------------------------------
        # Cash / debt
        # -------------------------------------------------------------

        if cash is not None and total_debt is not None:

            data["Cash_To_Debt"] = self._safe_divide(
                cash,
                total_debt,
            )

            data["_Net_Debt"] = (
                total_debt - cash
            )

        # -------------------------------------------------------------
        # Working capital turnover
        # -------------------------------------------------------------

        if (
            revenue is not None
            and "_Working_Capital" in data.columns
        ):

            wc = data["_Working_Capital"]

            # Avoid pathological ratios caused by near-zero WC.
            wc = wc.where(
                wc.abs() > 1e-9
            )

            data["Working_Capital_Turnover"] = (
                self._safe_divide(
                    revenue,
                    wc,
                )
            )

        # -------------------------------------------------------------
        # Asset turnover
        # -------------------------------------------------------------

        if revenue is not None and assets is not None:

            data["Asset_Turnover"] = self._safe_divide(
                revenue,
                assets,
            )

        # -------------------------------------------------------------
        # Free cash flow conversion
        # -------------------------------------------------------------

        cfo = self._first_existing(
            data,
            [
                "NetCashProvidedByUsedInOperatingActivities",
                "OperatingCashFlow",
                "CFO",
            ],
        )

        if (
            cfo is not None
            and net_income is not None
        ):

            data["CFO_To_NetIncome"] = (
                self._safe_divide(
                    cfo,
                    net_income,
                )
            )

        capex = self._first_existing(
            data,
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "CapitalExpenditures",
                "CapEx",
            ],
        )

        if cfo is not None and capex is not None:

            # SEC data can represent capex as either positive or
            # negative depending on source. Normalize it to positive
            # expenditure before calculating ratios.
            capex_abs = pd.to_numeric(
                capex,
                errors="coerce",
            ).abs()

            data["CapEx_To_CFO"] = self._safe_divide(
                capex_abs,
                cfo.abs(),
            )

            if revenue is not None:

                data["CapEx_To_Revenue"] = self._safe_divide(
                    capex_abs,
                    revenue,
                )

        # -------------------------------------------------------------
        # Actual ROIC
        # -------------------------------------------------------------

        data["Return_On_Invested_Capital"] = (
            self.compute_roic(data)
        )

        # -------------------------------------------------------------
        # Effective tax rate
        # -------------------------------------------------------------

        tax_expense = self._first_existing(
            data,
            [
                "IncomeTaxExpenseBenefit",
                "IncomeTaxExpense",
            ],
        )

        pretax_income = self._first_existing(
            data,
            [
                "IncomeBeforeTax",
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            ],
        )

        if (
            tax_expense is not None
            and pretax_income is not None
        ):

            data["Effective_Tax_Rate"] = (
                self._safe_divide(
                    tax_expense,
                    pretax_income,
                )
                .clip(
                    lower=-1.0,
                    upper=1.0,
                )
            )

        # -------------------------------------------------------------
        # Goodwill / intangibles
        # -------------------------------------------------------------

        goodwill = self._first_existing(
            data,
            [
                "Goodwill",
                "GoodwillAndIntangibleAssets",
            ],
        )

        intangibles = self._first_existing(
            data,
            [
                "FiniteLivedIntangibleAssetsNet",
                "IntangibleAssetsNetExcludingGoodwill",
                "IntangibleAssets",
            ],
        )

        if assets is not None:

            if goodwill is not None:

                data["Goodwill_To_Assets"] = (
                    self._safe_divide(
                        goodwill,
                        assets,
                    )
                )

            if (
                goodwill is not None
                and intangibles is not None
            ):

                data[
                    "Intangibles_Plus_Goodwill_To_Assets"
                ] = self._safe_divide(
                    goodwill + intangibles,
                    assets,
                )

        # -------------------------------------------------------------
        # SBC
        # -------------------------------------------------------------

        sbc = self._first_existing(
            data,
            [
                "ShareBasedCompensation",
                "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodTotal",
                "StockBasedCompensation",
            ],
        )

        if (
            sbc is not None
            and revenue is not None
        ):

            data["SBC_To_Revenue"] = self._safe_divide(
                sbc.abs(),
                revenue,
            )

        # -------------------------------------------------------------
        # R&D
        # -------------------------------------------------------------

        rnd = self._first_existing(
            data,
            [
                "ResearchAndDevelopmentExpense",
                "ResearchAndDevelopment",
            ],
        )

        if rnd is not None and revenue is not None:

            data["R_And_D_To_Revenue"] = self._safe_divide(
                rnd.abs(),
                revenue,
            )

        # -------------------------------------------------------------
        # SGA
        # -------------------------------------------------------------

        sga = self._first_existing(
            data,
            [
                "SellingGeneralAndAdministrativeExpense",
                "SGA",
            ],
        )

        if sga is not None and revenue is not None:

            data["SGA_Intensity"] = self._safe_divide(
                sga.abs(),
                revenue,
            )

        # -------------------------------------------------------------
        # Net debt / EBITDA
        # -------------------------------------------------------------

        if (
            "_Net_Debt" in data.columns
            and ebitda is not None
        ):

            data["Net_Debt_To_EBITDA"] = (
                self._safe_divide(
                    data["_Net_Debt"],
                    ebitda,
                )
            )

        # -------------------------------------------------------------
        # Non-operating income reliance
        # -------------------------------------------------------------

        non_operating = self._first_existing(
            data,
            [
                "NonOperatingIncome",
                "OtherIncomeExpenseNet",
                "OtherNonoperatingIncomeExpense",
            ],
        )

        if (
            non_operating is not None
            and pretax_income is not None
        ):

            data["NonOperating_Income_Reliance"] = (
                self._safe_divide(
                    non_operating.abs(),
                    pretax_income.abs(),
                )
            )

        # -------------------------------------------------------------
        # Reinvestment
        # -------------------------------------------------------------

        if (
            capex is not None
            and cfo is not None
        ):

            data["Reinvestment_Rate"] = (
                self._safe_divide(
                    capex.abs(),
                    cfo.abs(),
                )
            )

        # -------------------------------------------------------------
        # Accrual ratio
        # -------------------------------------------------------------

        if (
            net_income is not None
            and cfo is not None
            and assets is not None
        ):

            data["Accrual_Ratio"] = (
                self._safe_divide(
                    net_income - cfo,
                    assets,
                )
            )


        dividends = self._first_existing(
            data,
                [
                    "PaymentsOfDividendsCommonStock",  # Primary US-GAAP common cash dividend
                    "PaymentsOfDividends",            # Aggregate dividend payments
                    "PaymentsOfDividendsPreferredStockAndPreferenceStock", # Preferred dividends fallback if analyzing equity cash outflows
                ],
        )

        buybacks = self._first_existing(
            data,
            [
                "PaymentsForRepurchaseOfCommonStock",
                "PaymentsForRepurchaseOfWarrants",
            ],
        )

        data["Payout_To_FCF"] = self._safe_divide(dividends + buybacks, fcf).clip(-5, 5)
        data["Buyback_To_FCF"] = self._safe_divide(buybacks, fcf).clip(-5, 5)

        retained_earnings = self._first_existing(
            data,
            ["RetainedEarningsAccumulatedDeficit"],
        )
        if retained_earnings is not None and assets is not None:
            data["Retained_Earnings_To_Assets"] = self._safe_divide(
                retained_earnings, assets
            )

        aoci = self._first_existing(
            data,
            ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
        )
        if aoci is not None and equity is not None:
            data["AOCI_To_Equity"] = self._safe_divide(
                aoci, equity.abs()
            )

        lease_asset = self._first_existing(
            data,
            ["OperatingLeaseRightOfUseAsset"],
        )
        if lease_asset is not None and assets is not None:
            data["Operating_Lease_To_Assets"] = self._safe_divide(
                lease_asset, assets
            )

        deferred_tax = self._first_existing(
            data,
            ["DeferredIncomeTaxExpenseBenefit"],
        )
        if deferred_tax is not None and net_income is not None:
            data["Deferred_Tax_To_NetIncome"] = self._safe_divide(
                deferred_tax, net_income.abs()
            )

        # Cash tax rate and book-vs-cash tax gap.
        cash_taxes = self._first_existing(
            data,
            ["IncomeTaxesPaidNet"],
        )
        if cash_taxes is not None and pretax_income is not None:
            data["Cash_Tax_Rate"] = self._safe_divide(
                cash_taxes.abs(), pretax_income.abs()
            ).clip(-1.0, 1.0)

        if "Cash_Tax_Rate" in data.columns and "Effective_Tax_Rate" in data.columns:
            data["Cash_Vs_Book_Tax_Gap"] = (
                data["Cash_Tax_Rate"] - data["Effective_Tax_Rate"]
            )

        goodwill_impairment = self._first_existing(
            data,
            ["GoodwillImpairmentLoss"],
        )
        if goodwill_impairment is not None:
            data["Had_Goodwill_Impairment"] = (
                pd.to_numeric(goodwill_impairment, errors="coerce")
                .fillna(0)
                .abs()
                .gt(1e-12)
                .astype(float)
            )

        # Working-capital efficiency. For quarterly income-statement flows,
        # use ~91.25 days per quarter so the ratios remain comparable.
        accounts_receivable = self._first_existing(
            data, ["AccountsReceivableNetCurrent", "AccountsReceivableNet"]
        )
        inventory_wc = self._first_existing(
            data, ["InventoryNet"]
        )
        accounts_payable = self._first_existing(
            data, ["AccountsPayableCurrent"]
        )
        cogs = self._first_existing(
            data,
            ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
        )

        if accounts_receivable is not None and revenue is not None:
            data["DSO"] = self._safe_divide(
                accounts_receivable, revenue
            ) * 91.25

        if inventory_wc is not None and cogs is not None:
            data["DIO"] = self._safe_divide(
                inventory_wc, cogs.abs()
            ) * 91.25

        if accounts_payable is not None and cogs is not None:
            data["DPO"] = self._safe_divide(
                accounts_payable, cogs.abs()
            ) * 91.25

        if {"DSO", "DIO", "DPO"}.issubset(data.columns):
            data["Cash_Conversion_Cycle"] = (
                data["DSO"] + data["DIO"] - data["DPO"]
            )

        # Revenue-quality growth diagnostics.
        if accounts_receivable is not None and revenue is not None:
            ar_growth = accounts_receivable.groupby(data[self.config.ticker_column]).transform(
                lambda x: x.pct_change(4)
            )
            revenue_growth = revenue.groupby(data[self.config.ticker_column]).transform(
                lambda x: x.pct_change(4)
            )
            data["AR_Growth_vs_Revenue_Growth"] = ar_growth - revenue_growth

        if inventory_wc is not None and revenue is not None:
            inv_growth = inventory_wc.groupby(data[self.config.ticker_column]).transform(
                lambda x: x.pct_change(4)
            )
            revenue_growth = revenue.groupby(data[self.config.ticker_column]).transform(
                lambda x: x.pct_change(4)
            )
            data["Inventory_Growth_vs_Revenue_Growth"] = inv_growth - revenue_growth

        deferred_revenue = self._first_existing(
            data, ["DeferredRevenueCurrent", "DeferredRevenueNoncurrent"]
        )
        if deferred_revenue is not None and revenue is not None:
            # Use both current and non-current deferred revenue where both
            # exist. If only one exists, _first_existing supplies it.
            if "DeferredRevenueCurrent" in data.columns and "DeferredRevenueNoncurrent" in data.columns:
                deferred_revenue = (
                    pd.to_numeric(data["DeferredRevenueCurrent"], errors="coerce").fillna(0)
                    + pd.to_numeric(data["DeferredRevenueNoncurrent"], errors="coerce").fillna(0)
                )
            data["Deferred_Revenue_To_Revenue"] = self._safe_divide(
                deferred_revenue, revenue
            )
            data["Deferred_Revenue_YoY_Growth"] = deferred_revenue.groupby(
                data[self.config.ticker_column]
            ).transform(lambda x: x.pct_change(4))

            billings_proxy = revenue + deferred_revenue.groupby(
                data[self.config.ticker_column]
            ).diff(1)
            data["Billings_Proxy_YoY"] = billings_proxy.groupby(
                data[self.config.ticker_column]
            ).transform(lambda x: x.pct_change(4))

        acquisitions = self._first_existing(
            data, ["PaymentsToAcquireBusinessesNetOfCashAcquired"]
        )
        if acquisitions is not None and revenue is not None:
            data["Acquisition_Intensity"] = self._safe_divide(
                acquisitions.abs(), revenue
            )

        debt_issued = self._first_existing(
            data, ["ProceedsFromIssuanceOfLongTermDebt"]
        )
        debt_repaid = self._first_existing(
            data, ["RepaymentsOfLongTermDebt"]
        )
        if debt_issued is not None or debt_repaid is not None:
            issued = (
                pd.to_numeric(debt_issued, errors="coerce")
                if debt_issued is not None
                else pd.Series(0.0, index=data.index)
            )
            repaid = (
                pd.to_numeric(debt_repaid, errors="coerce")
                if debt_repaid is not None
                else pd.Series(0.0, index=data.index)
            )
            if assets is not None:
                data["Net_Debt_Issuance_To_Assets"] = self._safe_divide(
                    issued.fillna(0) - repaid.abs().fillna(0), assets.abs()
                )

        # -------------------------------------------------------------
        # Clean extreme fundamental ratios
        # -------------------------------------------------------------

        ratio_columns = [
            c
            for c in self.snapshot_columns
            if c in data.columns
        ]

        for column in ratio_columns:

            data[column] = (
                pd.to_numeric(
                    data[column],
                    errors="coerce",
                )
            )

            data[column] = (
                data[column]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

            data[column] = (
                self._winsorize_series(
                    data[column]
                )
            )

        return data

    # =================================================================
    # Fundamental trends
    # =================================================================

    def compute_financial_trends(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        data = data.copy()

        if self.config.ticker_column in data.columns:

            data = data.sort_values(
                [
                    self.config.ticker_column,
                    self.config.date_column,
                ]
            )

        grouped = data.groupby(
            self.config.ticker_column,
            group_keys=False,
        )

        # -------------------------------------------------------------
        # Helper
        # -------------------------------------------------------------

        def add_yoy(
            output_name: str,
            source_candidates: list[str],
        ):

            source = self._first_existing(
                data,
                source_candidates,
            )

            if source is not None:

                data[output_name] = grouped[
                    source.name
                ].transform(
                    lambda x: x.pct_change(4)
                )

        def add_qoq(
            output_name: str,
            source_candidates: list[str],
        ):

            source = self._first_existing(
                data,
                source_candidates,
            )

            if source is not None:

                data[output_name] = grouped[
                    source.name
                ].transform(
                    lambda x: x.pct_change(1)
                )

        # -------------------------------------------------------------
        # Growth
        # -------------------------------------------------------------

        add_yoy(
            "Revenue_YoY_Growth",
            [
                "Revenue",
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
                "TotalRevenue_Consolidated",
            ],
        )

        add_qoq(
            "Revenue_QoQ_Growth",
            [
                "Revenue",
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
                "TotalRevenue_Consolidated",
            ],
        )

        add_yoy(
            "NetIncome_YoY_Growth",
            [
                "NetIncome",
                "NetIncomeLoss",
            ],
        )

        add_yoy(
            "EPS_YoY_Growth",
            [
                "EPS",
                "DilutedEPS",
                "EarningsPerShareDiluted",
            ],
        )

        add_yoy(
            "EBITDA_YoY_Growth",
            [
                "EBITDA",
                "EBITDA_Calculated",
            ],
        )

        add_yoy(
            "FCF_YoY_Growth",
            [
                "FreeCashFlow",
                "FCF",
            ],
        )

        add_yoy(
            "Operating_CashFlow_YoY_Growth",
            [
                "NetCashProvidedByUsedInOperatingActivities",
                "OperatingCashFlow",
                "CFO",
            ],
        )

        # -------------------------------------------------------------
        # Revenue acceleration
        # -------------------------------------------------------------

        if "Revenue_YoY_Growth" in data.columns:

            data["Revenue_Acceleration"] = grouped[
                "Revenue_YoY_Growth"
            ].transform(
                lambda x: x.diff()
            )

        # -------------------------------------------------------------
        # Margin trends
        # -------------------------------------------------------------

        for source, output in [
            (
                "Gross_Margin",
                "Gross_Margin_YoY_Delta",
            ),
            (
                "EBITDA_Margin",
                "EBITDA_Margin_YoY_Delta",
            ),
            (
                "FCF_Margin",
                "FCF_Margin_YoY_Delta",
            ),
        ]:

            if source in data.columns:

                data[output] = grouped[
                    source
                ].transform(
                    lambda x: x.diff(4)
                )

        if "Operating_Margin" in data.columns:

            data["Operating_Margin_QoQ_Delta"] = grouped[
                "Operating_Margin"
            ].transform(
                lambda x: x.diff(1)
            )

        # -------------------------------------------------------------
        # ROA / ROE trend
        # -------------------------------------------------------------

        for source, output in [
            (
                "Return_On_Assets",
                "ROA_YoY_Delta",
            ),
            (
                "Return_On_Equity",
                "ROE_YoY_Delta",
            ),
        ]:

            if source in data.columns:

                data[output] = grouped[
                    source
                ].transform(
                    lambda x: x.diff(4)
                )

        # -------------------------------------------------------------
        # ACTUAL ROIC trend
        # -------------------------------------------------------------

        if "Return_On_Invested_Capital" in data.columns:

            data["ROIC_YoY_Delta"] = grouped[
                "Return_On_Invested_Capital"
            ].transform(
                lambda x: x.diff(4)
            )

        # -------------------------------------------------------------
        # Debt
        # -------------------------------------------------------------

        if "TotalDebt" in data.columns:

            data["Debt_YoY_Growth"] = grouped[
                "TotalDebt"
            ].transform(
                lambda x: x.pct_change(4)
            )

        # -------------------------------------------------------------
        # Working capital
        # -------------------------------------------------------------

        if "_Working_Capital" in data.columns:

            data["Working_Capital_YoY_Change"] = grouped[
                "_Working_Capital"
            ].transform(
                lambda x: x.diff(4)
            )

        # -------------------------------------------------------------
        # Shares outstanding
        # -------------------------------------------------------------

        shares = self._first_existing(
            data,
            [
                "CommonStockSharesOutstanding",
                "EntityCommonStockSharesOutstanding",
                "WeightedAverageNumberOfSharesOutstandingBasic",
                "WeightedAverageNumberOfDilutedSharesOutstanding",
            ],
        )

        if shares is not None:

            data["Shares_Outstanding_YoY_Change"] = grouped[
                shares.name
            ].transform(
                lambda x: x.pct_change(4)
            )

        # -------------------------------------------------------------
        # CapEx growth
        # -------------------------------------------------------------

        capex = self._first_existing(
            data,
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "CapitalExpenditures",
                "CapEx",
            ],
        )

        if capex is not None:

            data["CapEx_YoY_Growth"] = grouped[
                capex.name
            ].transform(
                lambda x: x.abs().pct_change(4)
            )

        # -------------------------------------------------------------
        # Volatility
        # -------------------------------------------------------------

        if "Return_On_Equity" in data.columns:

            data["ROE_Volatility_8Q"] = grouped[
                "Return_On_Equity"
            ].transform(
                lambda x: x.rolling(
                    8,
                    min_periods=4,
                ).std()
            )

        if "Operating_Margin" in data.columns:

            data["Margin_Volatility_8Q"] = grouped[
                "Operating_Margin"
            ].transform(
                lambda x: x.rolling(
                    8,
                    min_periods=4,
                ).std()
            )

        dividends = self._first_existing(
            data,
                [
                    "PaymentsOfDividendsCommonStock",  # Primary US-GAAP common cash dividend
                    "PaymentsOfDividends",            # Aggregate dividend payments
                    "PaymentsOfDividendsPreferredStockAndPreferenceStock", # Preferred dividends fallback if analyzing equity cash outflows
                ],
        )

        dps = self._first_existing(
            data,
            [
                "CommonStockDividendsPerShareDeclared",
                "CommonStockDividendsPerShareCashPaid",
            ]
        )

        if dps is None or dps.isna().all():
            dps = (dividends.abs() / shares)

        # Calculate 4-Quarter (YoY) Growth grouped by Ticker
        data["Dividend_Per_Share_YoY_Growth"] = data.groupby("Ticker")[dps.name if hasattr(dps, 'name') else "DPS"].transform(
            lambda s: s.pct_change(periods=4)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


        # -------------------------------------------------------------
        # Normalize extreme values
        # -------------------------------------------------------------

        for column in self.trend_columns:

            if column not in data.columns:
                continue

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            data[column] = data[column].replace(
                [np.inf, -np.inf],
                np.nan,
            )

            data[column] = (
                data[column]
                .clip(
                    lower=-10,
                    upper=10,
                )
            )

        return data

    # =================================================================
    # Sector Z-score
    # =================================================================

    def add_sector_zscores(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add sector-relative versions of selected features.

        Requires:
            Sector

        If Sector is not available, this method simply returns data.

        The normalization is cross-sectional by Date + Sector.
        """

        data = data.copy()

        if "Sector" not in data.columns:
            return data

        if self.config.date_column not in data.columns:
            return data

        for column in self.sector_base_columns:

            if column not in data.columns:
                continue

            output = f"{column}_SectorZ"

            grouped = data.groupby(
                [
                    self.config.date_column,
                    "Sector",
                ],
                observed=True,
            )[column]

            mean = grouped.transform("mean")
            std = grouped.transform("std")

            data[output] = (
                (data[column] - mean)
                /
                std.replace(0, np.nan)
            )

            data[output] = data[output].clip(
                -5,
                5,
            )

        return data

    # =================================================================
    # Market features
    # =================================================================

    def compute_market_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute price-derived market features.

        These are intentionally retained in the combined model because
        the latest AinyFin experiment shows that market-state variables
        may add predictive information.

        Critical:
            These features must be computed strictly from prices
            available on or before the observation date.
        """

        data = data.copy()

        if "Close" not in data.columns:
            return data

        if self.config.ticker_column not in data.columns:
            return data

        data = data.sort_values([self.config.ticker_column,"Date",])

        grouped = data.groupby(
            self.config.ticker_column,
            group_keys=False,
        )

        close = pd.to_numeric(data["Close"],  errors="coerce")

        # -------------------------------------------------------------
        # Momentum 1M
        # -------------------------------------------------------------

        data["Momentum_1M"] = grouped["Close"].transform(lambda x: x.pct_change(21))

        # -------------------------------------------------------------
        # Momentum 3M
        # -------------------------------------------------------------

        data["Momentum_3M"] = grouped["Close"].transform(lambda x: x.pct_change(63))

        # -------------------------------------------------------------
        # Momentum 12M minus 1M
        # -------------------------------------------------------------

        momentum_12m = grouped["Close"].transform(lambda x: x.pct_change(252))

        momentum_1m = data["Momentum_1M"]

        data["Momentum_12M_1M"] = (momentum_12m - momentum_1m)

        # -------------------------------------------------------------
        # 52-week high
        # -------------------------------------------------------------

        rolling_high = grouped["Close"].transform(
            lambda x: x.rolling(
                252,
                min_periods=20,
            ).max()
        )

        data["Price_Vs_52W_High"] = (close / rolling_high)

        # -------------------------------------------------------------
        # Realized volatility
        # -------------------------------------------------------------

        daily_return = grouped[
            "Close"
        ].transform(
            lambda x: x.pct_change()
        )

        data["Realized_Vol_60D"] = grouped[
            "Close"
        ].transform(
            lambda x: x.pct_change()
            .rolling(
                60,
                min_periods=20,
            )
            .std()
            * np.sqrt(252)
        )

        # -------------------------------------------------------------
        # Daily range
        # -------------------------------------------------------------

        if (
            "High" in data.columns
            and "Low" in data.columns
        ):
            data["Daily_Range_Pct"] = (
                (
                    pd.to_numeric(
                        data["High"],
                        errors="coerce",
                    )
                    -
                    pd.to_numeric(
                        data["Low"],
                        errors="coerce",
                    )
                )
                /
                close
            )

        # -------------------------------------------------------------
        # Volume features
        # -------------------------------------------------------------

        if "Volume" in data.columns:
            volume = pd.to_numeric(
                data["Volume"],
                errors="coerce",
            )

            avg_volume = grouped[
                "Volume"
            ].transform(
                lambda x: x.rolling(
                    60,
                    min_periods=20,
                ).mean()
            )

            data["Volume_Vs_60D_Avg"] = (
                volume / avg_volume
            )

            data["Dollar_Volume_Log"] = np.log1p(
                volume * close
            )

        return data

    # =================================================================
    # Build feature dataframe
    # =================================================================

    def _prepare_financial_data(
        self,
        financial_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build quarterly fundamental features from financial_data.csv.

        financial_data.csv is the authoritative source for SEC fundamentals.
        Trends MUST be calculated while the data is still quarterly. Doing
        them after merging to daily prices would create false daily changes.
        """
        data = financial_data.copy()

        required = [
            self.config.ticker_column,
            self.config.date_column,
        ]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(
                "financial_data.csv is missing required columns: "
                f"{missing}"
            )

        data[self.config.ticker_column] = (
            data[self.config.ticker_column]
            .astype(str)
            .str.upper()
            .str.strip()
        )
        data[self.config.date_column] = pd.to_datetime(
            data[self.config.date_column],
            errors="coerce",
        )
        data = data.dropna(
            subset=[
                self.config.ticker_column,
                self.config.date_column,
            ]
        )

        data = data.sort_values(
            [
                self.config.ticker_column,
                self.config.date_column,
            ]
        ).reset_index(drop=True)

        # Fundamental snapshot and trends are calculated BEFORE the
        # quarterly data is merged into daily prices.
        data = self.compute_financial_snapshot(data)
        data = self.compute_financial_trends(data)
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values(["Date"]).reset_index(drop=True)
        return data


    def build_features(
        self,
        data: pd.DataFrame,
        financial_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Assemble the final daily modeling dataset.

        Sources:
            1. ainyfin_data.csv
               Daily prices + target (bhsScore).
            2. financial_data.csv
               Quarterly SEC fundamentals.

        Fundamental trends are calculated on the quarterly financial data
        first, then point-in-time merged into daily observations.
        """
        price_data = data.copy()

        # -------------------------------------------------------------
        # Daily price/target data
        # -------------------------------------------------------------
        price_data[self.config.ticker_column] = (
            price_data[self.config.ticker_column]
            .astype(str)
            .str.upper()
            .str.strip()
        )
        price_data[self.config.date_column] = pd.to_datetime(
            price_data[self.config.date_column],
            errors="coerce",
        )

        price_data = price_data.dropna(
            subset=[
                self.config.ticker_column,
                self.config.date_column,
            ]
        )

        price_data = price_data.sort_values(
            [
                self.config.ticker_column,
                self.config.date_column,
            ]
        ).reset_index(drop=True)

        if financial_data is None:
            raise ValueError(
                "financial_data must be supplied. "
                "The model now uses financial_data.csv as the authoritative "
                "source for fundamental features."
            )

        # -------------------------------------------------------------
        # SEC fundamentals
        # -------------------------------------------------------------
        fundamentals = self._prepare_financial_data(
            financial_data
        )

        # -------------------------------------------------------------
        # Point-in-time merge
        # -------------------------------------------------------------
        #
        # For each daily price observation, use the most recent financial
        # filing that was available on or before that date.
        #
        # This is deliberately NOT a simple merge on quarter-end Date.
        # -------------------------------------------------------------
        left = price_data.sort_values([self.config.date_column])
        right = fundamentals.sort_values([self.config.date_column])

        data = pd.merge_asof(
            left,
            right,
            by=self.config.ticker_column,
            left_on="Date",
            right_on="Date",
            direction="backward",
            allow_exact_matches=True,
        )
        #print("Merged data:\n", data)

        # -------------------------------------------------------------
        # Market features
        # -------------------------------------------------------------
        data = self.compute_market_features(data)
        data.to_csv(f"{AinySchema.DATA_DIR}merged_data_with_market_features.csv", index=False)
        #print("compute_market_features data:\n", data)

        # -------------------------------------------------------------
        # Price-dependent valuation features
        # -------------------------------------------------------------
        data = self.compute_valuation_features(data)
        print("compute_valuation_features data:\n", data)

        # -------------------------------------------------------------
        # Sector-relative features
        # -------------------------------------------------------------
        data = self.add_sector_zscores(data)
        print("add_sector_zscores data:\n", data)

        return data



    def calculate_ebitda(self, df: pd.DataFrame) -> pd.Series:

        # 1. Depreciation & Amortization
        dna = self._first_existing(
            df,
            [
                "DepreciationDepletionAndAmortization",
                "DepreciationAndAmortization",
                "DepreciationAmortizationAndAccretionNet",
                "Depreciation",
            ],
        )

        if dna is None:
            dna = pd.Series(0.0, index=df.index)
        else:
            dna = pd.to_numeric(dna, errors="coerce").fillna(0.0)

        # 2. Operating Income
        operating_income = self._first_existing(
            df,
            [
                "OperatingIncomeLoss",
                "OperatingIncome",
                "Operating_Income",
            ],
        )

        if operating_income is not None:
            operating_income = pd.to_numeric(
                operating_income,
                errors="coerce"
            )

            ebitda_top_down = operating_income + dna
        else:
            ebitda_top_down = pd.Series(np.nan, index=df.index)

        # 3. Bottom-up fallback
        net_income = self._first_existing(
            df,
            [
                "NetIncomeLoss",
                "ProfitLoss",
            ],
        )

        if net_income is not None:

            net_income = pd.to_numeric(
                net_income,
                errors="coerce"
            )

            interest_expense = self._first_existing(
                df,
                [
                    "InterestExpense",
                    "InterestExpenseNonoperating",
                ],
            )

            if interest_expense is None:
                interest_expense = pd.Series(0.0, index=df.index)
            else:
                interest_expense = pd.to_numeric(
                    interest_expense,
                    errors="coerce"
                ).fillna(0.0)

            tax = self._first_existing(
                df,
                [
                    "IncomeTaxExpenseBenefit",
                    "CurrentIncomeTaxExpenseBenefit",
                ],
            )

            if tax is None:
                tax = pd.Series(0.0, index=df.index)
            else:
                tax = pd.to_numeric(
                    tax,
                    errors="coerce"
                ).fillna(0.0)

            ebitda_bottom_up = (
                net_income
                + interest_expense
                + tax
                + dna
            )

        else:
            ebitda_bottom_up = pd.Series(
                np.nan,
                index=df.index
            )

        return ebitda_top_down.fillna(ebitda_bottom_up)


    # =================================================================
    # Price-dependent valuation features
    # =================================================================
    def compute_valuation_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute valuation ratios only after daily Close has been merged
        with the point-in-time financial fundamentals.

        Missing denominator data remains NaN rather than being converted
        to zero.
        """
        data = data.copy()

        close = self._first_existing(
            data,
            ["Close"],
        )

        revenue = self._first_existing(
            data,
            [
                "RevenueFromContractWithCustomerExcludingAssessedTax",  # Primary ASC 606 GAAP tag
                "Revenues",                                             # Aggregate GAAP tag
                "SalesRevenueNet",                                      # Standard product/service sales tag
                "TotalRevenue_Consolidated",                            # Consolidated dataset tag
                "RevenueFromContractWithCustomerIncludingAssessedTax", # Alternative ASC 606 tag
            ],
        )

        net_income = self._first_existing(
            data,
            ["NetIncome", "NetIncomeLoss"],
        )

        fcf = self._first_existing(
            data,
            ["FreeCashFlow", "FCF"],
        )

        equity = self._first_existing(
            data,
            [
                "StockholdersEquity",
                "Stockholders_Equity",
                "StockholdersEquityAbstract",
                "TotalEquity",
                "Equity",
            ],
        )

        assets = self._first_existing(
            data,
            ["Assets", "TotalAssets"],
        )

        ebitda = self.calculate_ebitda(data)

        operating_income = self._first_existing(
            data,
            [
                "OperatingIncome",
                "Operating_Income",
                "OperatingIncomeLoss",
            ],
        )

        shares = self._first_existing(
            data,
            [
                "CommonStockSharesOutstanding",
                "EntityCommonStockSharesOutstanding",
                "WeightedAverageNumberOfSharesOutstandingBasic",
                "WeightedAverageNumberOfDilutedSharesOutstanding",
            ],
        )

        debt = self._first_existing(data, ["TotalDebt", "Debt"])

        cash = self._first_existing(
            data,
            [
                "CashAndCashEquivalents",
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsAndShortTermInvestments",
                "Cash",
                "CashAndShortTermInvestments",
            ],
        )

        if close is None:
            return data

        close = pd.to_numeric(close, errors="coerce")

        # Market capitalization is needed for P/E, P/B, and P/S.
        if shares is not None:
            market_cap = close * pd.to_numeric(
                shares,
                errors="coerce",
            )
            data["_Market_Cap"] = market_cap
        else:
            market_cap = None

        if market_cap is not None:
            if net_income is not None:
                data["Price_To_Earnings"] = self._safe_divide(
                    market_cap,
                    net_income,
                )

            if equity is not None:
                data["Price_To_Book"] = self._safe_divide(
                    market_cap,
                    equity,
                )

            if revenue is not None:
                data["Price_To_Sales"] = self._safe_divide(
                    market_cap,
                    revenue,
                )

            if fcf is not None:
                data["Price_To_FreeCashFlow"] = self._safe_divide(
                    market_cap,
                    fcf,
                )

        print("market_cap:\n", market_cap)
        print("operating_income:\n", operating_income)
        print("net_income:\n", net_income)
        print("ebitda:\n", ebitda)
        print("debt:\n",debt)
        print("revenue:\n",revenue)

        if market_cap is not None and ebitda is not None:
            if debt is not None:
                net_debt = (
                    pd.to_numeric(debt, errors="coerce")
                    - (
                        pd.to_numeric(cash, errors="coerce")
                        if cash is not None
                        else 0.0
                    )
                )
                enterprise_value = market_cap + net_debt
                data["EV_To_EBITDA"] = self._safe_divide(
                    enterprise_value,
                    ebitda,
                )

        if market_cap is not None and operating_income is not None:
            if debt is not None:
                net_debt = (
                    pd.to_numeric(debt, errors="coerce")
                    - (
                        pd.to_numeric(cash, errors="coerce")
                        if cash is not None
                        else 0.0
                    )
                )
                enterprise_value = market_cap + net_debt
                data["EV_To_EBIT"] = self._safe_divide(
                    enterprise_value,
                    operating_income,
                )

        buybacks = self._first_existing(
            data,
            [
                "PaymentsForRepurchaseOfCommonStock",
                "PaymentsForRepurchaseOfWarrants",
            ],
        )

        if buybacks is None:
            buybacks = pd.Series(0.0, index=data.index)
        else:
            buybacks = pd.to_numeric(buybacks, errors="coerce").fillna(0.0)

        if market_cap is None:
            market_cap = pd.Series(0.0, index=data.index)
        else:
            market_cap = pd.to_numeric(market_cap, errors="coerce").fillna(0.0)

        data["Buyback_Yield"] = buybacks.abs() / market_cap.abs()

        dividends = self._first_existing(
            data,
                [
                    "PaymentsOfDividendsCommonStock",  # Primary US-GAAP common cash dividend
                    "PaymentsOfDividends",            # Aggregate dividend payments
                    "PaymentsOfDividendsPreferredStockAndPreferenceStock", # Preferred dividends fallback if analyzing equity cash outflows
                ],
        )

        if dividends is None:
            dividends = pd.Series(0.0, index=data.index)
        else:
            dividends = pd.to_numeric(dividends, errors="coerce").fillna(0.0)

        # Take absolute value as cash flow items are sometimes reported as negative numbers
        data["Dividend_Yield"] = dividends.abs() / market_cap.abs()
        return data

    # =================================================================
    # Source-column audit
    # =================================================================

    def audit_source_columns(
        self,
        financial_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Report which source columns required by ModelBuilder are available.

        This is based on the actual financial_data.csv schema supplied for
        AinyFin. It helps catch schema drift before a training run.
        """
        checks = {
            "Revenue": [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
            ],
            "NetIncome": ["NetIncomeLoss", "ProfitLoss"],
            "OperatingIncome": ["OperatingIncomeLoss"],
            "EBITDA": [
                "EBITDA",
                "OperatingIncomeLoss + DepreciationAndAmortization",
            ],
            "FreeCashFlow": ["FreeCashFlow"],
            "Cash": [
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsAndShortTermInvestments",
            ],
            "Equity": ["StockholdersEquity", "StockholdersEquityAbstract"],
            "Debt": ["TotalDebt", "LongTermDebt", "LongTermDebtNoncurrent"],
            "Shares": [
                "CommonStockSharesOutstanding",
                "WeightedAverageNumberOfSharesOutstandingBasic",
                "WeightedAverageNumberOfDilutedSharesOutstanding",
            ],
            "CapEx": ["PaymentsToAcquirePropertyPlantAndEquipment"],
            "SBC": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
            "R&D": ["ResearchAndDevelopmentExpense"],
            "SG&A": ["SellingGeneralAndAdministrativeExpense"],
            "Goodwill": ["Goodwill"],
            "Intangibles": ["IntangibleAssetsNetExcludingGoodwill"],
            "Sector": ["SIC"],
            "FilingDate": [
                "FilingDate", "filing_date", "FiledDate", "acceptedDate",
                "AcceptedDate", "SEC_Filing_Date", "Date",
            ],
        }

        rows = []
        columns = set(financial_data.columns)
        for logical_name, candidates in checks.items():
            found = [c for c in candidates if c in columns]
            rows.append({
                "Logical_Field": logical_name,
                "Available": bool(found) or logical_name == "EBITDA",
                "Matched_Columns": "; ".join(found),
            })

        return pd.DataFrame(rows)

    # =================================================================
    # Expand model feature schema
    # =================================================================

    def get_available_training_columns(
        self,
        data: pd.DataFrame,
    ) -> list[str]:

        available = [
            column
            for column in self.available_training_columns
            if column in data.columns
        ]

        return available

    # =================================================================
    # Clean model data
    # =================================================================

    def prepare_model_dataframe(
        self,
        data: pd.DataFrame,
        require_target: bool = True,
    ) -> pd.DataFrame:

        data = data.copy()

        missing = [
            column
            for column in self.available_training_columns
            if column not in data.columns
        ]

        if missing:
            print("\nWARNING: Missing model features:")

            for column in missing:
                print(f"{column}")

        available = self.get_available_training_columns(data)

        if len(available) < 10:
            raise ValueError(
                "Too few model features available. "
                f"Found {len(available)}."
            )

        self.feature_columns = available

        # -------------------------------------------------------------
        # Numeric cleanup
        # -------------------------------------------------------------

        for column in available:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            data[column] = data[column].replace(
                [np.inf, -np.inf],
                np.nan,
            )

            data[column] = data[column].clip(
                self.config.clip_feature_min,
                self.config.clip_feature_max,
            )

        # -------------------------------------------------------------
        # Target
        # -------------------------------------------------------------

        if require_target:

            if (
                self.config.target_column
                not in data.columns
            ):

                raise ValueError(
                    f"Missing target column: "
                    f"{self.config.target_column}"
                )

            data[self.config.target_column] = (
                pd.to_numeric(
                    data[
                        self.config.target_column
                    ],
                    errors="coerce",
                )
            )

            data = data.dropna(
                subset=[
                    self.config.target_column
                ]
            )

            # Convert:
            #
            # 1 -> 0
            # 2 -> 1
            # 3 -> 2
            #
            data["_Target"] = (
                data[
                    self.config.target_column
                ].astype(int)
                - 1
            )

        # -------------------------------------------------------------
        # Sort
        # -------------------------------------------------------------

        data = data.sort_values(
            [
                self.config.date_column,
                self.config.ticker_column,
            ]
        )

        return data

    # =================================================================
    # Load training data
    # =================================================================

    def load_train_data(
        self,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load BOTH source files.

        ainyfin_data.csv:
            Daily prices and bhsScore target.

        financial_data.csv:
            SEC quarterly fundamentals.

        The two files are combined inside build_features().
        """
        training_filename = (
            filename or self.config.training_file
        )
        financial_filename = self.config.financial_file

        print("\nLoading AinyFin source datasets...")

        # -------------------------------------------------------------
        # Daily price / target source
        # -------------------------------------------------------------
        if not os.path.exists(training_filename):
            raise FileNotFoundError(
                f"Training file not found: {training_filename}"
            )

        print(
            f"Reading daily/target data: "
            f"{training_filename}"
        )

        price_data = pd.read_csv(
            training_filename
        )

        print(
            f"  Daily source: "
            f"{len(price_data):,} rows / "
            f"{len(price_data.columns)} columns"
        )

        # -------------------------------------------------------------
        # SEC fundamental source
        # -------------------------------------------------------------
        if not os.path.exists(financial_filename):
            raise FileNotFoundError(
                f"Financial file not found: {financial_filename}"
            )

        print(
            f"Reading SEC fundamentals: "
            f"{financial_filename}"
        )

        financial_data = pd.read_csv(
            financial_filename
        )

        print(
            f"  Financial source: "
            f"{len(financial_data):,} rows / "
            f"{len(financial_data.columns)} columns"
        )

        print("\nFinancial source schema audit:")
        print(self.audit_source_columns(financial_data).to_string(index=False))

        # -------------------------------------------------------------
        # Build combined feature dataset
        # -------------------------------------------------------------
        data = self.build_features(
            price_data,
            financial_data=financial_data,
        )

        data = self.prepare_model_dataframe(
            data,
            require_target=True,
        )

        print(
            f"\nFinal modeling dataset: "
            f"{len(data):,} rows / "
            f"{len(self.feature_columns)} model features"
        )

        print(
            f"Date range: "
            f"{data[self.config.date_column].min()} "
            f"to "
            f"{data[self.config.date_column].max()}"
        )

        print("\nTarget distribution:")

        print(
            data[
                self.config.target_column
            ]
            .value_counts(
                normalize=True
            )
            .sort_index()
        )

        # Audit how many daily rows actually received fundamentals.
        if "FinancialDate" in data.columns:
            fundamental_coverage = (
                data["FinancialDate"].notna().mean()
            )
            print(
                "\nFundamental coverage: "
                f"{fundamental_coverage:.2%}"
            )

        return data

    # =================================================================
    # Chronological split
    # =================================================================

    def chronological_split(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        dates = np.sort(
            data[
                self.config.date_column
            ].dropna().unique()
        )

        split_index = int(
            len(dates)
            * (
                1.0
                -
                self.config.chronological_test_fraction
            )
        )

        split_date = dates[
            split_index
        ]

        train = data[
            data[
                self.config.date_column
            ]
            < split_date
        ].copy()

        test = data[
            data[
                self.config.date_column
            ]
            >= split_date
        ].copy()

        return train, test

    # =================================================================
    # Create model
    # =================================================================

    def create_model(
        self,
    ) -> XGBClassifier:

        return XGBClassifier(

            objective=self.config.objective,

            num_class=3,

            n_estimators=self.config.n_estimators,

            max_depth=self.config.max_depth,

            learning_rate=self.config.learning_rate,

            subsample=self.config.subsample,

            colsample_bytree=self.config.colsample_bytree,

            min_child_weight=self.config.min_child_weight,

            reg_alpha=self.config.reg_alpha,

            reg_lambda=self.config.reg_lambda,

            eval_metric=self.config.eval_metric,

            random_state=self.config.random_state,

            n_jobs=self.config.n_jobs,

            tree_method="hist",

        )

    # =================================================================
    # Metrics
    # =================================================================

    def evaluate_predictions(
        self,
        y_true,
        y_pred,
        title: str = "Evaluation",
    ) -> dict[str, float]:

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
        )

        weighted_f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )

        print(
            f"\n=== {title} ==="
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_accuracy:.4f}"
        )

        print(
            f"Macro F1: "
            f"{macro_f1:.4f}"
        )

        print(
            f"Weighted F1: "
            f"{weighted_f1:.4f}"
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(
                y_true,
                y_pred,
                digits=4,
                zero_division=0,
            )
        )

        print(
            "Confusion Matrix:"
        )

        print(
            confusion_matrix(
                y_true,
                y_pred,
            )
        )

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    # =================================================================
    # Majority baseline
    # =================================================================

    def evaluate_baseline(
        self,
        y,
    ):

        values, counts = np.unique(
            y,
            return_counts=True,
        )

        majority_class = values[
            np.argmax(counts)
        ]

        predictions = np.full(
            len(y),
            majority_class,
        )

        print(
            "\n=== Majority-Class Baseline ==="
        )

        self.evaluate_predictions(
            y,
            predictions,
            title="Majority Baseline",
        )

        print(
            f"Majority class: "
            f"{majority_class}"
        )

    # =================================================================
    # Walk-forward validation
    # =================================================================

    def walk_forward_validation(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Expanding-window walk-forward validation.

        Example:

            Fold 1:
                Train: oldest -> T1
                Test:  T1 -> T2

            Fold 2:
                Train: oldest -> T2
                Test:  T2 -> T3

            ...

        This is the primary validation method.
        """

        print(
            "\n"
            + "=" * 70
        )

        print(
            "=== "
            f"{self.config.n_walk_forward_folds}"
            "-Fold Walk-Forward Validation ==="
        )

        print(
            "=" * 70
        )

        data = data.sort_values(
            self.config.date_column
        )

        unique_dates = np.array(
            sorted(
                data[
                    self.config.date_column
                ].dropna().unique()
            )
        )

        n_dates = len(
            unique_dates
        )

        if n_dates < 10:

            raise ValueError(
                "Not enough unique dates "
                "for walk-forward validation."
            )

        # -------------------------------------------------------------
        # Divide the historical period into approximately equal
        # validation windows.
        # -------------------------------------------------------------

        fold_boundaries = np.linspace(
            0,
            n_dates,
            self.config.n_walk_forward_folds + 2,
            dtype=int,
        )

        results = []

        for fold in range(
            self.config.n_walk_forward_folds
        ):

            train_end_index = (
                fold_boundaries[
                    fold + 1
                ]
            )

            validation_start_index = (
                train_end_index
            )

            validation_end_index = (
                fold_boundaries[
                    fold + 2
                ]
            )

            train_end_date = unique_dates[
                train_end_index - 1
            ]

            validation_start_date = (
                unique_dates[
                    validation_start_index
                ]
            )

            validation_end_date = (
                unique_dates[
                    validation_end_index - 1
                ]
            )

            train = data[
                data[
                    self.config.date_column
                ]
                <= train_end_date
            ]

            validation = data[
                (
                    data[
                        self.config.date_column
                    ]
                    >= validation_start_date
                )
                &
                (
                    data[
                        self.config.date_column
                    ]
                    <= validation_end_date
                )
            ]

            if len(train) < self.config.min_train_rows:

                print(
                    f"\nFold {fold + 1}: "
                    f"SKIPPED — only "
                    f"{len(train):,} training rows."
                )

                continue

            X_train = train[
                self.feature_columns
            ]

            y_train = train[
                "_Target"
            ]

            X_validation = validation[
                self.feature_columns
            ]

            y_validation = validation[
                "_Target"
            ]

            # ---------------------------------------------------------
            # Missing-value handling
            #
            # XGBoost handles NaN natively.
            # ---------------------------------------------------------

            model = self.create_model()

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_validation
            )

            metrics = self.evaluate_predictions(
                y_validation,
                predictions,
                title=f"Fold {fold + 1}",
            )

            results.append(
                {
                    "fold": fold + 1,

                    "train_rows": len(train),

                    "validation_rows": len(validation),

                    "train_end": train_end_date,

                    "validation_start":
                        validation_start_date,

                    "validation_end":
                        validation_end_date,

                    **metrics,
                }
            )

        results_df = pd.DataFrame(
            results
        )

        if not results_df.empty:

            print(
                "\n"
                + "-" * 70
            )

            print(
                "Walk-forward Accuracy: "
                f"{results_df['accuracy'].mean():.4f} "
                f"+/- "
                f"{results_df['accuracy'].std():.4f}"
            )

            print(
                "Walk-forward Balanced Accuracy: "
                f"{results_df['balanced_accuracy'].mean():.4f}"
            )

            print(
                "Walk-forward Macro F1: "
                f"{results_df['macro_f1'].mean():.4f}"
            )

            print(
                "Walk-forward Weighted F1: "
                f"{results_df['weighted_f1'].mean():.4f}"
            )

            print(
                "-" * 70
            )

        return results_df

    # =================================================================
    # Chronological test
    # =================================================================

    def chronological_test(
        self,
        data: pd.DataFrame,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "=== Chronological Test ==="
        )

        print(
            "=" * 70
        )

        train, test = (
            self.chronological_split(
                data
            )
        )

        print(
            f"Training rows: "
            f"{len(train):,}"
        )

        print(
            f"Test rows: "
            f"{len(test):,}"
        )

        print(
            f"Training end: "
            f"{train[self.config.date_column].max()}"
        )

        print(
            f"Test start: "
            f"{test[self.config.date_column].min()}"
        )

        # -------------------------------------------------------------
        # Baseline
        # -------------------------------------------------------------

        self.evaluate_baseline(
            test["_Target"].values
        )

        # -------------------------------------------------------------
        # Train
        # -------------------------------------------------------------

        X_train = train[
            self.feature_columns
        ]

        y_train = train[
            "_Target"
        ]

        X_test = test[
            self.feature_columns
        ]

        y_test = test[
            "_Target"
        ]

        model = self.create_model()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )

        metrics = self.evaluate_predictions(
            y_test,
            predictions,
            title="Chronological Test",
        )

        # -------------------------------------------------------------
        # Prediction frame
        # -------------------------------------------------------------

        prediction_df = test[
            [
                self.config.ticker_column,
                self.config.date_column,
            ]
        ].copy()

        prediction_df[
            "Actual_Class"
        ] = y_test.values

        prediction_df[
            "Predicted_Class"
        ] = predictions

        prediction_df[
            "P_Sell"
        ] = probabilities[:, 0]

        prediction_df[
            "P_Hold"
        ] = probabilities[:, 1]

        prediction_df[
            "P_Buy"
        ] = probabilities[:, 2]

        prediction_df[
            "Confidence"
        ] = probabilities.max(
            axis=1
        )

        sorted_probabilities = (
            np.sort(
                probabilities,
                axis=1
            )
        )

        prediction_df[
            "Confidence_Margin"
        ] = (
            sorted_probabilities[:, -1]
            -
            sorted_probabilities[:, -2]
        )

        prediction_df[
            "Predicted_Signal"
        ] = [
            self.config.target_labels.get(
                int(x),
                "UNKNOWN",
            )
            for x in predictions
        ]

        return (
            model,
            prediction_df,
            metrics,
        )

    # =================================================================
    # Feature importance
    # =================================================================

    def compute_feature_importance(
        self,
        model: Optional[XGBClassifier] = None,
    ) -> pd.DataFrame:

        model = model or self.model

        if model is None:
            raise ValueError("No trained model available.")

        importance = (model.feature_importances_)

        result = pd.DataFrame(
            {
                "Feature":
                    self.feature_columns,
                "Importance":
                    importance,
            }
        )

        result = result.sort_values(
            "Importance",
            ascending=False,
        ).reset_index(
            drop=True
        )

        self.feature_importance = result

        print("\n" + "=" * 70)

        print("=== Global Feature Importance ===")

        print(
            result.to_string(
                index=False
            )
        )

        return result

    # =================================================================
    # Train production model
    # =================================================================

    def train(
        self,
        data: pd.DataFrame,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "=== Training Production Model ==="
        )

        print(
            "=" * 70
        )

        X = data[
            self.feature_columns
        ]

        y = data[
            "_Target"
        ]

        print(
            f"Training rows: "
            f"{len(X):,}"
        )

        print(
            f"Features: "
            f"{len(self.feature_columns)}"
        )

        self.model = self.create_model()

        self.model.fit(
            X,
            y,
        )

        self.compute_feature_importance(
            self.model
        )

        return self.model

    # =================================================================
    # Probability prediction
    # =================================================================

    def predict(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict signals with probabilities.

        Returns:

            Ticker
            Date
            Predicted_Signal
            P_Sell
            P_Hold
            P_Buy
            Confidence
            Confidence_Margin
            Signal_Strength
        """

        if self.model is None:

            raise ValueError(
                "Model has not been trained."
            )

        data = data.copy()

        missing = [
            column
            for column in self.feature_columns
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Prediction data is missing "
                f"features: {missing}"
            )

        X = data[
            self.feature_columns
        ]

        probabilities = (
            self.model.predict_proba(
                X
            )
        )

        predictions = (
            probabilities.argmax(
                axis=1
            )
        )

        result = data[
            [
                c
                for c in [
                    self.config.ticker_column,
                    self.config.date_column,
                ]
                if c in data.columns
            ]
        ].copy()

        result[
            "Predicted_Class"
        ] = predictions

        result[
            "P_Sell"
        ] = probabilities[:, 0]

        result[
            "P_Hold"
        ] = probabilities[:, 1]

        result[
            "P_Buy"
        ] = probabilities[:, 2]

        result[
            "Confidence"
        ] = probabilities.max(
            axis=1
        )

        sorted_probabilities = (
            np.sort(
                probabilities,
                axis=1
            )
        )

        result[
            "Confidence_Margin"
        ] = (
            sorted_probabilities[:, -1]
            -
            sorted_probabilities[:, -2]
        )

        result[
            "Predicted_Signal"
        ] = [
            self.config.target_labels.get(
                int(x),
                "UNKNOWN",
            )
            for x in predictions
        ]

        # -------------------------------------------------------------
        # Signal strength
        #
        # This is deliberately descriptive.
        # Do not interpret it as a validated investment threshold yet.
        # -------------------------------------------------------------

        result[
            "Signal_Strength"
        ] = result.apply(
            lambda row:
                self._signal_strength(
                    row["Confidence"],
                    row["Confidence_Margin"],
                ),
            axis=1,
        )

        # -------------------------------------------------------------
        # No-trade layer
        #
        # We retain the model's class prediction but distinguish it
        # from an actionable signal.
        # -------------------------------------------------------------

        result[
            "Action"
        ] = result.apply(
            self._action_from_prediction,
            axis=1,
        )

        return result

    # =================================================================
    # Signal strength
    # =================================================================

    @staticmethod
    def _signal_strength(
        confidence: float,
        margin: float,
    ) -> str:

        if confidence >= 0.65 and margin >= 0.20:
            return "STRONG"

        if confidence >= 0.55 and margin >= 0.10:
            return "MODERATE"

        return "WEAK"

    # =================================================================
    # Action layer
    # =================================================================

    def _action_from_prediction(
        self,
        row,
    ) -> str:

        confidence = float(
            row["Confidence"]
        )

        margin = float(
            row["Confidence_Margin"]
        )

        signal = row[
            "Predicted_Signal"
        ]

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # These are NOT claimed to be optimal thresholds.
        # They are deliberately conservative defaults until confidence
        # calibration and return-bucket analysis have been performed.
        # -------------------------------------------------------------

        if (
            confidence < 0.50
            or margin < 0.05
        ):

            return "NO_TRADE"

        return signal

    # =================================================================
    # Save model
    # =================================================================

    def save_model(
        self,
        filename: Optional[str] = None,
    ):
        if self.model is None:
            raise ValueError("No model to save.")

        filename = (filename or self.config.model_file)
        self.model.save_model(filename)
        print(f"\nModel saved to: " f"{filename}")

        # -------------------------------------------------------------
        # Save feature schema
        # -------------------------------------------------------------

        schema_file = (filename + ".features.txt")

        with open(
            schema_file,
            "w",
            encoding="utf-8",
        ) as f:
            for feature in self.feature_columns:
                f.write(feature + "\n")

        print(f"Feature schema saved to: " f"{schema_file}")

    # =================================================================
    # Load model
    # =================================================================

    def load_model(
        self,
        filename: Optional[str] = None,
    ):
        filename = (
            filename
            or self.config.model_file
        )

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Model file not found: "
                f"{filename}"
            )

        self.model = self.create_model()
        self.model.load_model(filename)

        schema_file = (filename+ ".features.txt")

        if os.path.exists(
            schema_file
        ):
            with open(
                schema_file,
                "r",
                encoding="utf-8",
            ) as f:
                self.feature_columns = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]
        else:
            self.feature_columns = (
                self.available_training_columns
            )

        print(
            f"Model loaded from: "
            f"{filename}"
        )

        print(
            f"Feature schema: "
            f"{len(self.feature_columns)} features"
        )

        return self.model

    # =================================================================
    # End-to-end training
    # =================================================================

    def run_training(self, filename: Optional[str] = None):
        print("\n" + "=" * 70)
        print("=== Starting AinyFin Training ===")
        print("=" * 70)
        print(datetime.now())

        # -------------------------------------------------------------
        # Load
        # -------------------------------------------------------------

        data = self.load_train_data(filename)

        # -------------------------------------------------------------
        # Walk-forward validation
        # -------------------------------------------------------------

        walk_forward_results = (self.walk_forward_validation(data))

        # -------------------------------------------------------------
        # Chronological holdout
        # -------------------------------------------------------------

        (
            chronological_model,
            chronological_predictions,
            chronological_metrics,
        ) = self.chronological_test(data)

        # -------------------------------------------------------------
        # Train production model on ALL historical data
        # -------------------------------------------------------------

        self.train(data)

        # -------------------------------------------------------------
        # Save
        # -------------------------------------------------------------

        self.save_model()
        print("\n"+ "=" * 70)
        print("=== AinyFin Training Complete ===")
        print("=" * 70)

        return {
            "data": data,

            "walk_forward_results":
                walk_forward_results,

            "chronological_predictions":
                chronological_predictions,

            "chronological_metrics":
                chronological_metrics,

            "model":
                self.model,

            "feature_importance":
                self.feature_importance,
        }


# =====================================================================
# Main
# =====================================================================

def main():
    builder = ModelBuilder("xg")
    results = builder.run_training()
    print("\nFinal model features:")
    for i, feature in enumerate(
        builder.feature_columns,
        start=1,
    ):
        print(f"{i:3d}. {feature}")


if __name__ == "__main__":

    main()
