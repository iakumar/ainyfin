from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AinySchema:
    DATA_DIR: str = str(Path(__file__).resolve().parent.parent/"data")
    DATE: str = "Date"
    TICKER: str = "Ticker"
    REVENUE: str = "RevenueFromContractWithCustomerExcludingAssessedTax"
    OPERATING_INCOME: str = "OperatingIncomeLoss"

schema = AinySchema()
