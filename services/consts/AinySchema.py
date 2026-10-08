import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class AinySchema:
    BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME", "anypug.appspot.com")
    DIRECTORY_NAME: str = "ainyfin/models"
    APP_ENGINE_URL: str = os.environ.get(
        "APP_ENGINE_URL", "https://ainyfin.appspot.com"
    )

    DATA_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "data") + "/"
    DATE: str = "Date"
    TICKER: str = "Ticker"
    REVENUE: str = "RevenueFromContractWithCustomerExcludingAssessedTax"
    OPERATING_INCOME: str = "OperatingIncomeLoss"
    BHS_DESCS: ClassVar[list[str]] = ["Sell", "Hold", "Buy"]


schema = AinySchema()
