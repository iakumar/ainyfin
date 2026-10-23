import sys
import os
from datetime import datetime, timezone

import pandas as pd
from pathlib import Path

from services.consts.AinySchema import AinySchema


def print_recommendations(args:list[str]):

    if len(args) > 0:
        formatted_date = args[0]
        print(f"Printing recommendation data for date: {formatted_date}")
        files = [AinySchema.DATA_DIR + "reco_ainyfin-"+formatted_date+".csv"]
    else:
        directory = Path(AinySchema.DATA_DIR)
        files = [f for f in directory.glob('reco_ainyfin-*.csv') if f.is_file()]

    reco_df_list = []
    for file in files:
        #reco_df = pd.read_csv(AinySchema.DATA_DIR + "reco_ainyfin-"+formatted_date+".csv")
        print(f"Reading recommendation data from file: {file}")
        reco_df = pd.read_csv(file)
        reco_df_list.append(reco_df)

    combined_df = pd.concat(reco_df_list, ignore_index=True)
    #print(combined_df)
    combined_df = combined_df.sort_values(['Ticker',"Date"])

    html_table_styled = combined_df.to_html(
        index=False,
        classes="table table-striped table-hover",
        justify="left"
    )

    with open("data/recommendations.html", "w") as f:
        f.write(html_table_styled)

if __name__ == "__main__":
    print_recommendations(sys.argv[1:])
