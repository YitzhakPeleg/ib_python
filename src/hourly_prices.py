"""Example: Fetching historical hourly prices using HistoricalDataFetcher."""

from loguru import logger
from rich import print

from historical_data_fetcher import HistoricalDataFetcher
from models import BarFrequency, ContractSpec, Duration

if __name__ == "__main__":
    logger.add("hourly_prices.log", rotation="1 MB", level="INFO")

    # Create fetcher and connect
    fetcher = HistoricalDataFetcher(host="127.0.0.1", port=4002, client_id=123)
    fetcher.connect()

    try:
        # Define contract
        contract = ContractSpec(symbol="AAPL")

        # Fetch hourly data for the last 5 days
        df = fetcher.get_historical_data(
            contract=contract,
            duration=Duration(unit="D", value=10),
            frequency=BarFrequency.ONE_DAY,
            regular_trading_hours=True,
        )

        # Save and display
        if not df.is_empty():
            df.write_csv("AAPL_hourly.csv")
            print(df)
        else:
            logger.warning("No data received")

    finally:
        fetcher.disconnect()
