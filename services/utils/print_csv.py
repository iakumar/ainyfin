import sys
from datetime import datetime, timezone

import pandas as pd

from services.consts.AinySchema import AinySchema


def print_recommendations(args:list[str]):
    if len(args) > 0:
        formatted_date = args[0]
    else:
        today = datetime.now(timezone.utc).date()
        formatted_date = today.strftime("%m-%d-%y")

    print(f"Printing recommendation data for date: {formatted_date}")

    reco_df = pd.read_csv(AinySchema.DATA_DIR + "reco_ainyfin-"+formatted_date+".csv")
    #print(reco_df.to_markdown(index=False))

    html_table_styled = reco_df.to_html(
        index=False,
        classes="table table-striped table-hover",
        justify="left"
    )

    with open("data/recommendations.html", "w") as f:
        f.write(html_table_styled)

if __name__ == "__main__":
    print_recommendations(sys.argv[1:])
