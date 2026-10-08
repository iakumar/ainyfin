import pandas as pd
import datetime
from edgar import Company, set_identity
from edgar import Fund

from services.consts.AinySchema import AinySchema


class EdgarXDI:
    DATE: str = "Date"
    TICKER: str = "Ticker"

    def __init__(self, usecase: str):
        self.usecase: str = usecase
        self.status: str = ""

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
            if isinstance(col, str) and col.startswith("Q") and len(col.split()) == 2:
                q_part, year_part = col.split()
                if q_part in quarter_map:
                    new_date_str = f"{quarter_map[q_part]}-{year_part}"
                    new_columns[col] = new_date_str

        # Rename matched columns in-place or return renamed DataFrame
        return df.rename(columns=new_columns)

    def cleanup_financial_featureset(
        self, symbol: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Downloads financial statements for a given ticker and returns a cleaned DataFrame.

        Args:
            ticker (str): The stock ticker symbol.
        Returns:
            pd.DataFrame: Cleaned DataFrame with financial data.
        """

        df = df.rename(columns={"concept": "Date"})

        # 2. Identify metadata columns vs. quarterly financial date columns
        metadata_cols = ["Date"]
        quarter_cols = [c for c in df.columns if c.startswith("Q")]

        # 3. Re-index the DataFrame to put concept and quarters first
        df_clean = df[metadata_cols + quarter_cols]
        df_clean = (
            self.convert_quarter_columns(df_clean).set_index("Date").T.reset_index()
        )
        df_clean.rename(columns={"index": "Date"}, inplace=True)
        df_clean["Ticker"] = symbol
        # print("df_clean:\n",df_clean)

        return df_clean

    def downloadFinancialData(self, symbols: list[str]):
        set_identity("info@iakumar.com")

        financial_data_list = []
        for symbol in symbols:
            print(f"downloadFinancialData: {symbol}\n")
            downloaded:bool = False
            df1:pd.DataFrame = pd.DataFrame()
            df2:pd.DataFrame = pd.DataFrame()
            df3:pd.DataFrame = pd.DataFrame()
            try:
                downloaded = False
                company = Company(symbol)
                df1 = company.income_statement(
                    periods=16, period="quarterly", as_dataframe=True
                ).reset_index()
                df1 = self.cleanup_financial_featureset(symbol, df1)
                df1["SIC"] = company.sic

                # print("income_statement df:\n",df1.columns)
                # Concatenate all financial data into a single DataFrame

                df2 = company.balance_sheet(
                    periods=16, period="quarterly", as_dataframe=True
                ).reset_index()
                df2 = self.cleanup_financial_featureset(symbol, df2)
                # print("balance_sheet df:\n",df2.columns)
                bs_dups = ["AdditionalItems"]
                df2 = df2.drop(columns=[c for c in bs_dups if c in df2.columns])

                df3 = company.cash_flow_statement(
                    periods=16, period="quarterly", as_dataframe=True
                ).reset_index()
                df3 = self.cleanup_financial_featureset(symbol, df3)
                cf_dups = [
                    "AccretionExpenseIncludingAssetRetirementObligations",
                    "AdditionalItems",
                    "AmortizationOfIntangibleAssets",
                    "BankOwnedLifeInsuranceIncome",
                    "CapitalizedComputerSoftwareAmortization1",
                    "CashCashEquivalentsAndShortTermInvestments",
                    "CryptoAssetRealizedGainLossNonoperating",
                    "DebtAndEquitySecuritiesGainLoss",
                    "DebtAndEquitySecuritiesRealizedGainLoss",
                    "DebtAndEquitySecuritiesUnrealizedGainLoss",
                    "DebtSecuritiesAvailableForSaleExcludingAccruedInterestAllowanceForCreditLossNotPreviouslyRecorded",
                    "DebtSecuritiesRealizedGainLoss",
                    "DebtSecuritiesTradingGainLoss",
                    "DeferredPolicyAcquisitionCostAmortizationExpense",
                    "DeferredPolicyAcquisitionCostsAndPresentValueOfFutureProfitsAmortization1",
                    "DefinedBenefitPlanRecognizedNetGainLossDueToSettlements1",
                    "DepreciationAndAmortization",
                    "DiscontinuedOperationIncomeLossFromDiscontinuedOperationDuringPhaseOutPeriodNetOfTax",
                    "DividendsPayableCurrent",
                    "DividendsPayableCurrentAndNoncurrent",
                    "EnvironmentalRemediationExpense",
                    "EquipmentExpense",
                    "EquitySecuritiesFvNiRealizedGainLoss",
                    "EquitySecuritiesWithoutReadilyDeterminableFairValueImpairmentLossAnnualAmount",
                    "FairValueOptionChangesInFairValueGainLoss1",
                    "FinancingInterestExpense",
                    "FinancingReceivableExcludingAccruedInterestCreditLossExpenseReversal",
                    "ForeignCurrencyTransactionGainLossBeforeTax",
                    "ForeignCurrencyTransactionGainLossRealized",
                    "GainLossOnDerivativeInstrumentsNetPretax",
                    "GainLossOnInvestments",
                    "GainLossOnSalesOfMortgageBackedSecuritiesMBS",
                    "GainLossRelatedToLitigationSettlement",
                    "GainsLossesOnSalesOfInvestmentRealEstate",
                    "GeneralAndAdministrativeExpense",
                    "Goodwill",
                    "GoodwillImpairmentLoss",
                    "IncomeLossFromContinuingOperations",
                    "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
                    "IncomeLossFromDiscontinuedOperationsNetOfTax",
                    "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
                    "IncomeLossFromEquityMethodInvestments",
                    "IncomeTaxExpenseBenefit",
                    "InsuranceServicesRevenue",
                    "InterestExpenseNonoperating",
                    "InterestIncomeOperatingPaidInKind",
                    "InvestmentIncomeInterest",
                    "LiabilityForUnpaidClaimsAndClaimsAdjustmentExpenseIncurredClaims1",
                    "LitigationSettlementLoss",
                    "MarketRiskBenefitChangeInFairValueGainLoss",
                    "MarketableSecuritiesRealizedGainLossExcludingOtherThanTemporaryImpairments",
                    "NetIncomeLoss",
                    "NetIncomeLossAttributableToNoncontrollingInterest",
                    "NetIncomeLossAvailableToCommonStockholdersBasic",
                    "NonoperatingIncomeExpense",
                    "OperatingCostsAndExpenses",
                    "OperatingExpenses",
                    "OperatingLeaseExpense",
                    "OperatingLeaseLeaseIncome",
                    "OtherInterestAndDividendIncome",
                    "OtherNonoperatingIncomeExpense",
                    "OtherOperatingIncomeExpenseNet",
                    "PublicUtilitiesAllowanceForFundsUsedDuringConstructionCapitalizedCostOfEquity",
                    "RealEstateTaxExpense",
                    "RealizedInvestmentGainsLosses",
                    "RestrictedCashAndInvestmentsCurrent",
                    "RestructuringCosts",
                    "RevenueFromContractWithCustomerIncludingAssessedTax",
                    "SalesTypeLeaseSellingProfitLoss"
                    "SharebasedCompensationArrangementBySharebasedPaymentAwardCompensationCost1",
                ]
                df3 = df3.drop(columns=[c for c in cf_dups if c in df3.columns])
                downloaded = True
            except Exception as e:
                print(f"Error downloading income statement for {symbol}: {e}")

            if downloaded == False:
                print("Company:", company)
                #print("Companyget_filings:", company.get_filings())
                current_year = datetime.datetime.today().year
                years_list = [current_year - i for i in range(5)]
                #filings = company.get_filings(year=years_list)
                #for filing in filings:
                #    print(f"{filing.form}: {filing.company} ({filing.filing_date})")
                #    print(filing.to_dict())
                #    print(filing.obj())

                filings_list = company.get_filings(form=["NPORT-P",'N-30D'], trigger_full_load=False)
                # Extract portfolio holdings DataFrame from the most recent NPORT-P filing
                for filing in filings_list:
                    if filing == None or filing.obj() == None:
                        print("filing_obj is None for filing:", filing)
                    else:
                        print(f"{filing.form}: {filing.company} ({filing.filing_date})")
                        print(filing.to_dict())
                        filing_obj = filing.obj()
                        print("filing_obj:\n",filing_obj)
                        holdings_df = filing_obj.investment_data()
                        print("Holdings:\n",holdings_df.head())

            if df1.empty or df2.empty or df3.empty:
                print(f"Skipping {symbol} due to empty DataFrames.")
                continue

            for df in [df1, df2, df3]:
                df["Date"] = pd.to_datetime(df["Date"])

            merged_df = df1.merge(
                df2, on=["Date", "Ticker"], how="outer", suffixes=("", "_bs")
            )
            merged_df = merged_df.merge(
                df3, on=["Date", "Ticker"], how="outer", suffixes=("", "_cf")
            )
            merged_df = merged_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
            financial_data_list.append(merged_df)

        final_df = pd.concat(financial_data_list, ignore_index=True).copy()

        abstract_cols = [c for c in final_df.columns if c.endswith("Abstract")]
        final_df = final_df.drop(columns=abstract_cols)
        final_df.to_csv(AinySchema.DATA_DIR + "financial_data.csv", index=False)
        ticker_sic_df = final_df[["Ticker", "SIC"]].dropna().drop_duplicates()
        ticker_sic_df.to_csv(AinySchema.DATA_DIR + "ticker_sic.csv", index=False)

        print("Final merged financial data saved to 'financial_data.csv'.")
