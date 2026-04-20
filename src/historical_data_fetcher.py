"""Historical data fetcher using IBapi wrapper."""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import polars as pl
from ibapi.client import EClient
from ibapi.contract import Contract
from loguru import logger
from rich.pretty import pretty_repr

from date_converter import add_date_int_column
from ibapi_wrapper import IBapi
from models import BarFrequency, ContractSpec

DEFAULT_START_DATE = datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0)


class HistoricalDataFetcher(IBapi):
    """High-level wrapper for fetching historical market data from Interactive Brokers."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1):
        """Initialize the fetcher.

        Args:
            host: IB Gateway/TWS host
            port: IB Gateway/TWS port
            client_id: Client ID for IB connection
        """
        # Initialize parent IBapi class
        super().__init__()

        self.host = host
        self.port = port
        self.client_id = client_id
        self._api_thread: Optional[threading.Thread] = None
        self._connected = False

        # Progress bar tracking per request ID
        self._connect()

    def _connect(self) -> None:
        """Connect to IB Gateway/TWS and start the event loop."""
        if self._connected:
            logger.warning("Already connected to IB")
            return

        # Call parent EClient.connect() method
        EClient.connect(self, host=self.host, port=self.port, clientId=self.client_id)
        self._api_thread = threading.Thread(target=self.run, daemon=True)
        self._api_thread.start()

        # Wait for connection to establish with a 5-second timeout
        logger.info(f"Connecting to IB at {self.host}:{self.port}...")
        for i in range(10):  # 10 iterations at 500ms intervals = 5s
            time.sleep(0.5)
            if self.isConnected():
                self._connected = True
                logger.info(f"Connected to IB at {self.host}:{self.port}")
                break
        else:
            raise RuntimeError(f"Could not connect to IB at {self.host}:{self.port}")

    def close(self) -> None:
        """Disconnect from IB Gateway/TWS."""
        if self._connected:
            logger.info("Disconnecting from IB")
            self.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    @staticmethod
    def _frequency_to_seconds(frequency: BarFrequency) -> float:
        """Convert BarFrequency enum to seconds.

        Args:
            frequency: BarFrequency enum value

        Returns:
            Number of seconds in one bar period
        """
        freq_str = frequency.value.lower()

        # Parse "N unit" format (e.g., "1 hour", "5 min")
        parts = freq_str.split()
        if len(parts) != 2:
            raise ValueError(f"Unexpected frequency format: {frequency.value}")

        value, unit = int(parts[0]), parts[1]

        unit_to_seconds = {
            "sec": 1,
            "min": 60,
            "hour": 3600,
            "day": 86400,
            "week": 604800,
            "month": 2592000,  # Approximate: 30 days
        }

        if unit not in unit_to_seconds:
            raise ValueError(f"Unknown time unit: {unit}")

        return value * unit_to_seconds[unit]

    def _calculate_expected_bars(
        self, duration: timedelta, frequency: BarFrequency
    ) -> int:
        """Calculate expected number of bars based on duration and frequency.

        Args:
            duration: Time period for historical data
            frequency: Bar size/frequency

        Returns:
            Expected number of bars (rounded up)
        """
        duration_seconds = duration.total_seconds()
        freq_seconds = self._frequency_to_seconds(frequency)

        # Calculate expected bars, rounded up
        expected_bars = int(duration_seconds / freq_seconds) + 1
        return max(1, expected_bars)  # At least 1

    def get_historical_data(
        self,
        *,
        contract: ContractSpec,
        end_date: Optional[datetime] = None,
        duration: timedelta,
        frequency: BarFrequency = BarFrequency.ONE_HOUR,
        regular_trading_hours: bool = True,
        timeout: timedelta | None = None,
        timezone: Optional[str] = None,
    ) -> pl.DataFrame:
        """Fetch historical market data.

        Args:
            contract: Contract specification
            start_date: Start date (None -> 2020-01-01)
            end_date: End date (None -> now)
            frequency: Bar size/frequency
            regular_trading_hours: Only include regular trading hours data

        Returns:
            Polars DataFrame with columns: DateTime, Open, High, Low, Close, Volume

        Raises:
            RuntimeError: If connection fails or request times out
        """
        # Auto-connect if not already connected
        if not self._connected:
            self._connect()

        # Use defaults for dates if not provided
        if end_date is None:
            end_date = datetime.now()

        # Create Contract object from ContractSpec
        ib_contract = Contract()
        ib_contract.symbol = contract.symbol
        ib_contract.secType = contract.sec_type
        ib_contract.exchange = contract.exchange
        ib_contract.currency = contract.currency

        # Format dates for IB API (yyyymmdd HH:mm:ss UTC)
        end_datetime_str = end_date.strftime("%Y%m%d %H:%M:%S UTC")

        # Calculate duration string

        if duration < timedelta(days=1):
            duration_str = f"{int(duration.total_seconds())} S"
        else:
            duration_str = f"{duration // timedelta(days=1)} D"

        # Use next available reqId
        req_id = 1
        while req_id in self.requests:
            req_id += 1

        logger.info(
            f"Requesting historical data:\n"
            f"ticker = {contract.symbol}\n"
            f"duration = {duration}\n"
            f"end_date = {end_datetime_str}\n"
            f"bar frequency = {frequency.value}"
        )

        # Request tracking will be auto-created on first historicalData callback

        logger.debug(
            "Sending historical data request with reqId:\n"
            + pretty_repr(
                dict(
                    reqId=req_id,
                    contract=ib_contract,
                    endDateTime=end_datetime_str,
                    durationStr=duration_str,
                    barSizeSetting=frequency.value,  # Use the BarFrequency enum value
                    whatToShow="TRADES",
                    useRTH=1 if regular_trading_hours else 0,
                    formatDate=2,  # epoch time in seconds
                    keepUpToDate=False,
                    chartOptions=[],
                ),
            )
        )
        # Request historical data
        self.reqHistoricalData(
            reqId=req_id,
            contract=ib_contract,
            endDateTime=end_datetime_str,
            durationStr=duration_str,
            barSizeSetting=frequency.value,  # Use the BarFrequency enum value
            whatToShow="TRADES",
            useRTH=1 if regular_trading_hours else 0,
            formatDate=2,  # epoch time in seconds
            keepUpToDate=False,
            chartOptions=[],
        )
        # Wait for data to be ready with a timeout (e.g., 60 seconds)
        if not self.wait_for_data(req_id, timeout=timeout):
            logger.warning(
                f"Historical data request {req_id} timed out after {timeout}"
            )
            self.remove_request(req_id)  # Clean up request tracking
            raise RuntimeError(f"Historical data request {req_id} timed out")

        # Get accumulated data as DataFrame and remove request
        try:
            df = self.get_data(req_id)
        except ValueError:
            logger.warning(f"No data received for {contract.symbol}")
            return pl.DataFrame(
                schema=["DateTime", "Open", "High", "Low", "Close", "Volume"]
            )

        # Convert DateTime from epoch seconds to datetime
        try:
            df = df.with_columns(
                [
                    (pl.col("DateTime").cast(pl.Int64) * 1000)
                    .cast(pl.Datetime(time_unit="ms", time_zone=timezone or "UTC"))
                    .alias("DateTime")
                ]
            )
            logger.info(f"Retrieved {len(df)} bars for {contract.symbol}")
        except Exception as e:
            logger.warning(f"Could not convert DateTime to datetime: {e}")

        return df

    def __enter__(self):
        """Support context manager entry."""
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager exit."""
        self.close()


if __name__ == "__main__":
    # Example usage
    with HistoricalDataFetcher() as fetcher:
        contract = ContractSpec(symbol="AAPL")
        freq = BarFrequency.ONE_MIN
        # get_historical_data auto-connects if needed
        df = fetcher.get_historical_data(
            contract=contract,
            # end_date=datetime(2024, 4, 16, 15, 0, 0),
            duration=timedelta(days=365),
            frequency=freq,
            timeout=timedelta(minutes=30),
            timezone="US/Eastern",
        )
        df = add_date_int_column(df)
    print(df)
    output_file = f"{contract.symbol}_{freq.value.replace(' ', '_')}.parquet"
    df.write_parquet(output_file)
    logger.info(f"Saved historical data to {output_file}")
