"""Historical data fetcher using IBapi wrapper."""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import polars as pl
from ibapi.contract import Contract
from loguru import logger
from rich.pretty import pretty_repr

from ibapi_wrapper import IBapi
from models import BarFrequency, ContractSpec

DEFAULT_START_DATE = datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0)


class HistoricalDataFetcher:
    """High-level wrapper for fetching historical market data from Interactive Brokers."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1):
        """Initialize the fetcher.

        Args:
            host: IB Gateway/TWS host
            port: IB Gateway/TWS port
            client_id: Client ID for IB connection
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ibapi = IBapi()
        self._api_thread: Optional[threading.Thread] = None
        self._connected = False

    def connect(self) -> None:
        """Connect to IB Gateway/TWS and start the event loop."""
        if self._connected:
            logger.warning("Already connected to IB")
            return

        self.ibapi.connect(host=self.host, port=self.port, clientId=self.client_id)
        self._api_thread = threading.Thread(target=self.ibapi.run, daemon=True)
        self._api_thread.start()

        # Wait for connection to establish with a 5-second timeout
        logger.info(f"Connecting to IB at {self.host}:{self.port}...")
        for i in range(10):  # 10 iterations at 500ms intervals = 5s
            time.sleep(0.5)
            if self.ibapi.isConnected():
                self._connected = True
                logger.info(f"Connected to IB at {self.host}:{self.port}")
                break
        else:
            raise RuntimeError(f"Could not connect to IB at {self.host}:{self.port}")

    def disconnect(self) -> None:
        """Disconnect from IB Gateway/TWS."""
        if self._connected:
            self.ibapi.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    def get_historical_data(
        self,
        *,
        contract: ContractSpec,
        end_date: Optional[datetime] = None,
        duration: timedelta,
        frequency: BarFrequency = BarFrequency.ONE_HOUR,
        regular_trading_hours: bool = True,
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
            RuntimeError: If not connected or request times out
        """
        if not self._connected:
            raise RuntimeError("Not connected to IB. Call connect() first.")

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
        while req_id in self.ibapi.data:
            req_id += 1

        logger.info(
            f"Requesting historical data:\n"
            f"ticker = {contract.symbol}\n"
            f"duration = {duration} = {duration_str[:-2]} seconds\n"
            f"end_date = {end_datetime_str}\n"
            f"bar frequency = {frequency.value}"
        )

        # Reset tracking for this request
        self.ibapi.reset_request(req_id)

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
        self.ibapi.reqHistoricalData(
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

        # Wait for data to arrive
        if not self.ibapi.wait_for_data(req_id, timeout=30.0):
            logger.warning(f"Request {req_id} timed out after 30 seconds")

        # Get accumulated data
        data = self.ibapi.get_data(req_id)

        if not data:
            logger.warning(f"No data received for {contract.symbol}")
            return pl.DataFrame(
                schema=["DateTime", "Open", "High", "Low", "Close", "Volume"]
            )

        # Create DataFrame
        df = pl.DataFrame(
            data, schema=["DateTime", "Open", "High", "Low", "Close", "Volume"]
        )

        # Convert DateTime from epoch seconds to datetime
        try:
            df = df.with_columns(
                [
                    (pl.col("DateTime").cast(pl.Int64) * 1000)
                    .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
                    .alias("DateTime")
                ]
            )
            logger.info(f"Retrieved {len(df)} bars for {contract.symbol}")
        except Exception as e:
            logger.warning(f"Could not convert DateTime to datetime: {e}")

        return df


if __name__ == "__main__":
    # Example usage
    fetcher = HistoricalDataFetcher()
    fetcher.connect()

    try:
        df = fetcher.get_historical_data(
            contract=ContractSpec(symbol="AAPL"),
            # end_date=datetime(2024, 4, 16, 15, 0, 0),
            duration=timedelta(days=365),
            frequency=BarFrequency.ONE_MIN,
        )
        print(df)
    finally:
        fetcher.disconnect()
