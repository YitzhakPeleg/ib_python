"""Historical data fetcher using IBapi wrapper."""

import threading
from datetime import datetime
from typing import Optional

import polars as pl
from ibapi.contract import Contract
from loguru import logger

from src.ibapi_wrapper import IBapi
from src.models import BarFrequency, ContractSpec, DEFAULT_START_DATE, DEFAULT_END_DATE


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
        
        # Wait a bit for connection to establish
        import time
        time.sleep(1)
        self._connected = True
        logger.info(f"Connected to IB at {self.host}:{self.port}")

    def disconnect(self) -> None:
        """Disconnect from IB Gateway/TWS."""
        if self._connected:
            self.ibapi.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    def get_historical_data(
        self,
        contract: ContractSpec,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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
        if start_date is None:
            start_date = DEFAULT_START_DATE
        if end_date is None:
            end_date = DEFAULT_END_DATE()
        
        # Create Contract object from ContractSpec
        ib_contract = Contract()
        ib_contract.symbol = contract.symbol
        ib_contract.secType = contract.sec_type
        ib_contract.exchange = contract.exchange
        ib_contract.currency = contract.currency
        
        # Format dates for IB API (yyyymmdd HH:mm:ss UTC)
        end_datetime_str = end_date.strftime("%Y%m%d %H:%M:%S UTC")
        
        # Calculate duration string
        duration_days = (end_date - start_date).days
        if duration_days <= 0:
            raise ValueError("End date must be after start date")
        
        # IB API limits: use appropriate duration
        if duration_days <= 1:
            duration_str = "1 D"
        elif duration_days <= 7:
            duration_str = f"{duration_days} D"
        elif duration_days <= 30:
            duration_str = f"{duration_days} D"
        elif duration_days <= 365:
            weeks = duration_days // 7
            duration_str = f"{weeks} W" if weeks > 0 else "1 W"
        else:
            years = duration_days // 365
            duration_str = f"{years} Y" if years > 0 else "1 Y"
        
        # Use next available reqId
        req_id = 1
        while req_id in self.ibapi.data:
            req_id += 1
        
        logger.info(
            f"Requesting historical data for {contract.symbol}: "
            f"{start_date.date()} to {end_date.date()} ({frequency})"
        )
        
        # Reset tracking for this request
        self.ibapi.reset_request(req_id)
        
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
        if not self.ibapi.wait_for_data(req_id, timeout=30):
            logger.warning(f"Request {req_id} timed out after 30 seconds")
        
        # Get accumulated data
        data = self.ibapi.get_data(req_id)
        
        if not data:
            logger.warning(f"No data received for {contract.symbol}")
            return pl.DataFrame(schema=["DateTime", "Open", "High", "Low", "Close", "Volume"])
        
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
