import threading
import time

import polars as pl
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
from loguru import logger
from rich import print


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # Initialize variable to store candle
        self.data_ready = threading.Event()  # Signal when data is complete

    def historicalData(self, reqId, bar):
        # print(bar)
        # logger.info(
        #     f"Time: {bar.date} O: {bar.open} H: {bar.high} L: {bar.low} C: {bar.close} V: {bar.volume}"
        # )
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
        # logger.info(f"{len(self.data)} candles received")

    def historicalDataEnd(self, reqId, start, end):
        logger.info(
            f"Historical data request {reqId} completed. Received {len(self.data)} bars."
        )
        self.data_ready.set()  # Signal that data is ready


def run_loop():
    app.run()


if __name__ == "__main__":
    logger.add("hourly_prices.log", rotation="1 MB", level="INFO")
    app = IBapi()
    app.connect(host="127.0.0.1", port=4002, clientId=123)

    # Start the socket in a thread
    api_thread = threading.Thread(target=run_loop, daemon=True)
    api_thread.start()

    time.sleep(1)  # Sleep interval to allow time for connection to server
    logger.info("Connected to IB API")
    # Create contract object
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"

    app.data = []  # Initialize variable to store candle
    logger.info(f"Requesting historical data for {contract.symbol}...")
    # Request historical candles
    app.reqHistoricalData(
        reqId=1,
        contract=contract,
        endDateTime="20260409 23:59:59 UTC",  #  yyyymmdd HH:mm:ss ttt
        durationStr="5 D",
        barSizeSetting="1 min",  # 1 min, 5 mins, 1 hour, etc.
        whatToShow="TRADES",
        useRTH=1,  # Only request data from regular trading hours
        formatDate=2,  # epoch time in seconds
        keepUpToDate=False,
        chartOptions=[],
    )

    # Wait for data to be ready instead of hardcoded sleep
    logger.info("Waiting for historical data...")
    app.data_ready.wait(timeout=30)  # Wait up to 30 seconds for data
    logger.info("Historical data received, processing...")

    # Working with Polars DataFrame
    if app.data:
        df = pl.DataFrame(
            app.data, schema=["DateTime", "Open", "High", "Low", "Close", "Volume"]
        )
        # Convert DateTime from seconds to milliseconds for polars
        try:
            df = df.with_columns(
                [(pl.col("DateTime").cast(pl.Int64) * 1000).alias("DateTime")]
            )
            df = df.with_columns(
                [
                    pl.col("DateTime")
                    .cast(pl.Datetime(time_unit="ms", time_zone="Asia/Jerusalem"))
                    .alias("DateTime")
                ]
            )
        except Exception as e:
            logger.warning(f"Could not convert DateTime to datetime: {e}")
        df.write_csv("AAPL_1min.csv")
        print(df)
    else:
        logger.warning("No data received from IB API.")
    logger.info("Disconnecting from IB API")
    app.disconnect()
