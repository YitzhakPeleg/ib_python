"""Data models for historical data fetching and trading signals."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class SecurityType(StrEnum):
    STOCK = "STK"
    OPTION = "OPT"
    FUTURE = "FUT"


class Exchange(StrEnum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    ARCA = "ARCA"
    SMART = "SMART"  # IB API's SMART is equivalent to the exchange.


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    JPY = "JPY"


class Duration(BaseModel):
    unit: Literal["S", "M", "D", "W"]
    value: PositiveInt


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
    sec_type: SecurityType = Field(
        default=SecurityType.STOCK,
        description="Security type (e.g., 'STK', 'FUT', 'OPT')",
    )
    exchange: Exchange = Field(
        default=Exchange.SMART, description="Exchange (e.g., 'SMART', 'NASDAQ', 'NYSE')"
    )
    currency: Currency = Field(
        default=Currency.USD, description="Currency (e.g., 'USD', 'EUR')"
    )


# ============================================================================
# Trading Signal Models
# ============================================================================


class SignalType(IntEnum):
    """Trading signal types with numeric values for ML models."""

    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class TradeSetup:
    """Complete trade setup with entry, stop-loss, and take-profit levels."""

    date: int  # YYYYMMDD format
    signal: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # Model probability/confidence score
    risk_reward_ratio: float = 2.0  # Default 2:1 ratio

    @property
    def risk_amount(self) -> float:
        """Calculate risk amount (distance from entry to stop-loss)."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_amount(self) -> float:
        """Calculate reward amount (distance from entry to take-profit)."""
        return abs(self.take_profit - self.entry_price)

    def __repr__(self) -> str:
        return (
            f"TradeSetup(date={self.date}, signal={self.signal.name}, "
            f"entry={self.entry_price:.2f}, sl={self.stop_loss:.2f}, "
            f"tp={self.take_profit:.2f}, confidence={self.confidence:.2%})"
        )


@dataclass
class SignalResult:
    """Result of a completed trade for backtesting."""

    setup: TradeSetup
    outcome: Literal["win", "loss", "breakeven", "open"]
    exit_price: float
    pnl: float  # Profit/Loss in dollars
    pnl_percent: float  # Profit/Loss as percentage
    bars_held: int  # Number of bars from entry to exit

    @property
    def r_multiple(self) -> float:
        """Calculate R-multiple (PnL / Risk)."""
        if self.setup.risk_amount == 0:
            return 0.0
        return self.pnl / self.setup.risk_amount
