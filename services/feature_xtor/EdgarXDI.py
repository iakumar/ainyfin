import pandas as pd
from edgar import Company, set_identity

from services.consts.AinySchema import AinySchema


class EdgarXDI:

    DATE: str = "Date"
    TICKER: str = "Ticker"

    def __init__(self, usecase: str):
        self.usecase:str = usecase
        self.status:str = ''

    def convert_quarter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts DataFrame columns formatted as 'QX YYYY' to 'MM-DD-YYYY' date strings."""

        # Map quarter indicators to standard calendar quarter-end month & day
        quarter_map = {
            "Q1": "03-31",
            "Q2": "06-30",  # Note: June has 30 days
            "Q3": "09-30",
            "Q4": "12-31",
        }

        new_columns = {}
        for col in df.columns:
            # Process only columns matching the 'QX YYYY' pattern
            if (
                isinstance(col, str)
                and col.startswith("Q")
                and len(col.split()) == 2
            ):
                q_part, year_part = col.split()
                if q_part in quarter_map:
                    new_date_str = f"{quarter_map[q_part]}-{year_part}"
                    new_columns[col] = new_date_str

        # Rename matched columns in-place or return renamed DataFrame
        return df.rename(columns=new_columns)


    def cleanup_financial_featureset(self, symbol:str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downloads financial statements for a given ticker and returns a cleaned DataFrame.

        Args:
            ticker (str): The stock ticker symbol.
        Returns:
            pd.DataFrame: Cleaned DataFrame with financial data.
        """

        df = df.rename(columns={'concept': 'Date'})

        # 2. Identify metadata columns vs. quarterly financial date columns
        metadata_cols = ['Date']
        quarter_cols = [c for c in df.columns if c.startswith("Q")]

        # 3. Re-index the DataFrame to put concept and quarters first
        df_clean = df[metadata_cols + quarter_cols]
        df_clean = self.convert_quarter_columns(df_clean).set_index('Date').T.reset_index()
        df_clean.rename(columns={'index': 'Date'}, inplace=True)
        df_clean['Ticker'] = symbol
        print("df_clean:\n",df_clean)

        return df_clean


    def downloadFinancialData(self, symbols:list[str]):
        set_identity("info@iakumar.com")

        financial_data_list = []
        for symbol in symbols:
            company = Company(symbol)
            df1:pd.DataFrame = company.income_statement(periods=16, period='quarterly', as_dataframe=True).reset_index()
            df1 = self.cleanup_financial_featureset(symbol, df1)
            #print("income_statement df:\n",df1.columns)
            # Concatenate all financial data into a single DataFrame

            df2 = company.balance_sheet(periods=16, period='quarterly', as_dataframe=True).reset_index()
            df2 = self.cleanup_financial_featureset(symbol, df2)
            #print("balance_sheet df:\n",df2.columns)
            bs_dups = ["AdditionalItems"]
            df2 = df2.drop(columns=[c for c in bs_dups if c in df2.columns])

            df3 = company.cash_flow_statement(periods=16, period='quarterly', as_dataframe=True).reset_index()
            df3 = self.cleanup_financial_featureset(symbol, df3)
            cf_dups = [
                "AdditionalItems",
                "AmortizationOfIntangibleAssets",
                "EquitySecuritiesWithoutReadilyDeterminableFairValueImpairmentLossAnnualAmount",
                "Goodwill",
                "IncomeLossFromEquityMethodInvestments",
                "IncomeTaxExpenseBenefit",
                "InterestExpenseNonoperating",
                "NetIncomeLoss",
                "NonoperatingIncomeExpense",
                "OperatingExpenses",
                "OtherNonoperatingIncomeExpense",
                "GeneralAndAdministrativeExpense",
                "IncomeLossFromContinuingOperations",
                "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
                "CashCashEquivalentsAndShortTermInvestments",
                "DebtAndEquitySecuritiesGainLoss",
                "DebtSecuritiesRealizedGainLoss",
                "EquitySecuritiesFvNiRealizedGainLoss",
                "ForeignCurrencyTransactionGainLossBeforeTax",
                "NetIncomeLossAvailableToCommonStockholdersBasic",
                "GainLossOnDerivativeInstrumentsNetPretax",
                "GainLossOnInvestments",
                "GoodwillImpairmentLoss",
                "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
                "IncomeLossFromDiscontinuedOperationsNetOfTax",
                "OtherOperatingIncomeExpenseNet",
                "RestructuringCosts",
                "NetIncomeLossAttributableToNoncontrollingInterest",
                "AccretionExpenseIncludingAssetRetirementObligations",
                "DividendsPayableCurrentAndNoncurrent",
                "OperatingLeaseExpense",
                "DividendsPayableCurrent",
                "CapitalizedComputerSoftwareAmortization1",
                "SharebasedCompensationArrangementBySharebasedPaymentAwardCompensationCost1",
                "InvestmentIncomeInterest",
                "DebtAndEquitySecuritiesRealizedGainLoss",
                "EquipmentExpense",
                "OperatingLeaseLeaseIncome",
                "EnvironmentalRemediationExpense",
                "GainLossOnSalesOfMortgageBackedSecuritiesMBS",
                "GainsLossesOnSalesOfInvestmentRealEstate",
                "CryptoAssetRealizedGainLossNonoperating",
                "InsuranceServicesRevenue",
                "FairValueOptionChangesInFairValueGainLoss1",
                "FinancingReceivableExcludingAccruedInterestCreditLossExpenseReversal",
                "OtherInterestAndDividendIncome",
                "DebtAndEquitySecuritiesUnrealizedGainLoss",
                "DebtSecuritiesTradingGainLoss",
                "LiabilityForUnpaidClaimsAndClaimsAdjustmentExpenseIncurredClaims1",
                "DefinedBenefitPlanRecognizedNetGainLossDueToSettlements1"

            ]
            df3 = df3.drop(columns=[c for c in cf_dups if c in df3.columns])

            #print("cash_flow_statement df:\n",df3.columns)
            for df in [df1, df2, df3]:
                df["Date"] = pd.to_datetime(df["Date"])

            merged_df = df1.merge(df2, on=["Date", "Ticker"], how="outer", suffixes=("", "_bs"))
            merged_df = merged_df.merge(
                df3, on=["Date", "Ticker"], how="outer", suffixes=("", "_cf")
            )
            merged_df = merged_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
            financial_data_list.append(merged_df)

        final_df = pd.concat(financial_data_list, ignore_index=True).copy()
        final_df['TotalDebt'] = final_df['LongTermDebtNoncurrent'] + final_df['LongTermDebtCurrent']
        final_df['WorkingCapital'] = final_df['AssetsCurrent'] - final_df['LiabilitiesCurrent']
        final_df['FreeCashFlow'] = final_df['NetCashProvidedByUsedInOperatingActivities'] - final_df['PaymentsToAcquirePropertyPlantAndEquipment']

        final_df.to_csv(AinySchema.DATA_DIR+"/financial_data.csv", index=False)
        print("Final merged financial data saved to 'financial_data.csv'.")
