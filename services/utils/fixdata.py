import sys

import pandas as pd

from services.consts.AinySchema import AinySchema


def fix(args:list):
    """
    fdata_df = pd.read_csv(AinySchema.DATA_DIR+"financial_data.csv")
    abstract_cols = [c for c in fdata_df.columns if c.endswith("Abstract")]
    fdata_df = fdata_df.drop(columns=abstract_cols)
    fdata_df.to_csv(AinySchema.DATA_DIR + "financial_data.csv",index=False)
    ticker_sic_df = fdata_df[["Ticker", "SIC"]].dropna().drop_duplicates()
    ticker_sic_df.to_csv(AinySchema.DATA_DIR + "ticker_sic.csv",index=False)
    """

    price_df = pd.read_csv(AinySchema.DATA_DIR+"reco_ainyfin-26-10-19.csv")
    price_df["Date"] = '2022-10-19'
    price_df = price_df[["Ticker", 'Date', "Action", "Recommendation", "Price", "Swing_Buy", "Swing_Sell", "Lot_Size", "Confidence", 'Probabilities', "Rationale", "SupportedBy", "WeakenedBy"]]
    price_df.to_csv(AinySchema.DATA_DIR + "reco",index=False)

    #price_df.sort_values(['Date','Volume']).to_csv(AinySchema.DATA_DIR+'price-sort.csv')

if __name__ == "__main__":
    fix(sys.argv[1:])
