"""Data models for historical data fetching."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class BarFrequency(StrEnum):
    """Bar size/frequency options supported by IB API."""

    ONE_SEC = "1 sec"
    FIVE_SEC = "5 sec"
    TEN_SEC = "10 sec"
    FIFTEEN_SEC = "15 sec"
    THIRTY_SEC = "30 sec"
    ONE_MIN = "1 min"
    TWO_MIN = "2 min"
    THREE_MIN = "3 min"
    FIVE_MIN = "5 min"
    FIFTEEN_MIN = "15 min"
    TWENTY_MIN = "20 min"
    THIRTY_MIN = "30 min"
    ONE_HOUR = "1 hour"
    FOUR_HOUR = "4 hours"
    EIGHT_HOUR = "8 hours"
    ONE_DAY = "1 day"
    ONE_WEEK = "1 week"
    ONE_MONTH = "1 month"


class ContractSpec(BaseModel):
    """Contract specification for historical data requests."""

    symbol: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")
    sec_type: str = Field(
        default="STK", description="Security type (e.g., 'STK', 'FUT', 'OPT')"
    )
    exchange: str = Field(
        default="SMART", description="Exchange (e.g., 'SMART', 'NASDAQ', 'NYSE')"
    )
    currency: str = Field(default="USD", description="Currency (e.g., 'USD', 'EUR')")


# Constants
DEFAULT_START_DATE = datetime(2020, 1, 1)
DEFAULT_END_DATE = lambda: datetime.now()  # Use lambda to get current time at call time
