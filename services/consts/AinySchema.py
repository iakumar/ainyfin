from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

@dataclass(frozen=True)
class AinySchema:
    DATA_DIR: str = str(Path(__file__).resolve().parent.parent/"data")
    DATE: str = "Date"
    TICKER: str = "Ticker"
    REVENUE: str = "RevenueFromContractWithCustomerExcludingAssessedTax"
    OPERATING_INCOME: str = "OperatingIncomeLoss"
    BHS_DESCS: ClassVar[list[str]] = ["Sell", "Hold", "Buy"]

schema = AinySchema()
