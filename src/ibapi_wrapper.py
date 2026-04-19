"""Enhanced IBapi wrapper with multi-request support."""

import threading
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Dict, Literal, Union

import polars as pl
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from loguru import logger

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Request:
    """Represents a single historical data request with its state."""

    data: list = field(default_factory=list)
    ready: threading.Event = field(default_factory=threading.Event)

    def export(
        self, format: Literal["polars", "pandas"] = "polars"
    ) -> Union[pl.DataFrame, "pd.DataFrame"]:
        """Export request data as a DataFrame.

        Args:
            format: Output format - "polars" (default) or "pandas"

        Returns:
            Polars or Pandas DataFrame with columns: DateTime, Open, High, Low, Close, Volume

        Raises:
            ValueError: If data is empty
            ImportError: If required library (pandas) is not installed
        """
        if not self.data:
            raise ValueError("No data to export")

        if format.lower() == "polars":
            df = pl.DataFrame(
                self.data,
                schema=["DateTime", "Open", "High", "Low", "Close", "Volume"],
                orient="row",
            )
            return df
        elif format.lower() == "pandas":
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas is required for 'pandas' format. "
                    "Install with: pip install pandas"
                )
            df = pd.DataFrame(
                self.data,
                columns=["DateTime", "Open", "High", "Low", "Close", "Volume"],
            )
            return df
        else:
            raise ValueError(f"Unknown format: {format}. Use 'polars' or 'pandas'")


class IBapi(EWrapper, EClient):
    """Enhanced IBapi wrapper with support for multiple concurrent data requests."""

    def __init__(self):
        EClient.__init__(self, self)
        self.requests: Dict[int, Request] = {}  # Store Request objects per reqId

    def _ensure_request_tracking(self, reqId: int) -> None:
        """Ensure tracking structures are initialized for a given reqId."""
        if reqId not in self.requests:
            self.requests[reqId] = Request()

    def historicalData(self, reqId: int, bar) -> None:
        """Callback for each historical bar received."""
        self._ensure_request_tracking(reqId)
        self.requests[reqId].data.append(
            [bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume]
        )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        """Callback when all historical data has been received."""
        self._ensure_request_tracking(reqId)
        logger.info(
            f"Historical data request {reqId} completed. Received {len(self.requests[reqId].data)} bars."
        )
        self.requests[reqId].ready.set()  # Signal that data is ready
        logger.debug(f"Data for reqId {reqId}: {self.requests[reqId].ready.is_set()=} ")

    def get_data(
        self, reqId: int, format: Literal["polars", "pandas"] = "polars"
    ) -> Union[pl.DataFrame, "pd.DataFrame"]:
        """Get accumulated data for a specific request ID and remove it from tracking.

        Args:
            reqId: Request ID to retrieve data for
            format: Output format - "polars" (default) or "pandas"

        Returns:
            DataFrame with columns: DateTime, Open, High, Low, Close, Volume

        Raises:
            ValueError: If no data exists for this request ID or data is empty
            ImportError: If pandas format requested but not installed
        """
        if reqId not in self.requests:
            raise ValueError(f"No request found with ID {reqId}")

        request = self.requests[reqId]
        df = request.export(format=format)
        self.remove_request(reqId)
        return df

    def wait_for_data(self, reqId: int, timeout: timedelta | None = None) -> bool:
        """Wait for data to be ready for a specific request ID.

        Args:
            reqId: Request ID to wait for
            timeout: Maximum seconds to wait

        Returns:
            True if data arrived, False if timeout
        """
        if timeout:
            self._ensure_request_tracking(reqId)
            logger.debug(
                f"Waiting for data for reqId {reqId} with timeout {timeout}..."
            )
            return self.requests[reqId].ready.wait(timeout=timeout.total_seconds())
        return True  # If no timeout, assume data will arrive eventually

    def remove_request(self, reqId: int) -> None:
        """Remove a request from tracking."""
        self.requests.pop(reqId, None)
