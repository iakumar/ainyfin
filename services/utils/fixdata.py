import sys

import numpy as np
import pandas as pd

from services.consts.AinySchema import AinySchema

def fix(args:list):
    fdata_df = pd.read_csv(AinySchema.DATA_DIR+"financial_data.csv")
    abstract_cols = [c for c in fdata_df.columns if c.endswith("Abstract")]
    fdata_df = fdata_df.drop(columns=abstract_cols)
    fdata_df.to_csv(AinySchema.DATA_DIR + "financial_data.csv",index=False)
    ticker_sic_df = fdata_df[["Ticker", "SIC"]].dropna().drop_duplicates()
    ticker_sic_df.to_csv(AinySchema.DATA_DIR + "ticker_sic.csv",index=False)

if __name__ == "__main__":
    fix(sys.argv[1:])
