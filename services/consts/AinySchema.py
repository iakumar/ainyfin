from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class AinySchema:
    DATE: str = "Date"
    TICKER: str = "Ticker"
    REVENUE: str = "RevenueFromContractWithCustomerExcludingAssessedTax"
    OPERATING_INCOME: str = "OperatingIncomeLoss"
