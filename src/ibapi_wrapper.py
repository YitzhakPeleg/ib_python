"""Enhanced IBapi wrapper with multi-request support."""

import threading
from typing import Dict

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from loguru import logger


class IBapi(EWrapper, EClient):
    """Enhanced IBapi wrapper with support for multiple concurrent data requests."""

    def __init__(self):
        EClient.__init__(self, self)
        self.data: Dict[int, list] = {}  # Store data per reqId
        self.data_ready: Dict[int, threading.Event] = {}  # Events per reqId

    def _ensure_request_tracking(self, reqId: int) -> None:
        """Ensure tracking structures are initialized for a given reqId."""
        if reqId not in self.data:
            self.data[reqId] = []
        if reqId not in self.data_ready:
            self.data_ready[reqId] = threading.Event()

    def historicalData(self, reqId: int, bar) -> None:
        """Callback for each historical bar received."""
        self._ensure_request_tracking(reqId)
        self.data[reqId].append(
            [bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume]
        )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        """Callback when all historical data has been received."""
        self._ensure_request_tracking(reqId)
        logger.info(
            f"Historical data request {reqId} completed. Received {len(self.data[reqId])} bars."
        )
        self.data_ready[reqId].set()  # Signal that data is ready for this reqId

    def get_data(self, reqId: int) -> list:
        """Get accumulated data for a specific request ID."""
        return self.data.get(reqId, [])

    def wait_for_data(self, reqId: int, timeout: float = 30.0) -> bool:
        """Wait for data to be ready for a specific request ID.
        
        Args:
            reqId: Request ID to wait for
            timeout: Maximum seconds to wait
            
        Returns:
            True if data arrived, False if timeout
        """
        self._ensure_request_tracking(reqId)
        return self.data_ready[reqId].wait(timeout=timeout)

    def reset_request(self, reqId: int) -> None:
        """Reset tracking for a specific request ID."""
        if reqId in self.data:
            self.data[reqId] = []
        if reqId in self.data_ready:
            self.data_ready[reqId] = threading.Event()
