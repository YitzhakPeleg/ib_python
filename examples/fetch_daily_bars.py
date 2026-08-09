"""Fetch true (non-resampled) daily bars from IB for the tickers already in data/.

Requires IB Gateway/TWS running and reachable at IB_HOST:IB_PORT (see .env).
"""

from datetime import timedelta

from data_fetching.historical_data_fetcher import HistoricalDataFetcher
from models.models import BarFrequency, ContractSpec
from models.paths import get_file
from pydantic_settings import BaseSettings, SettingsConfigDict


class IBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_client_id: int = 123


settings = IBSettings()

TICKERS = ["AAPL", "AMZN", "AVGO", "GOOG", "IBKR", "MSFT", "NVDA", "SPY"]
DURATION = timedelta(days=5 * 365)  # 5 years of daily bars

with HistoricalDataFetcher(host=settings.ib_host, port=settings.ib_port, client_id=settings.ib_client_id) as fetcher:
    for ticker in TICKERS:
        df = fetcher.get_historical_data(
            contract=ContractSpec(symbol=ticker),
            duration=DURATION,
            frequency=BarFrequency.ONE_DAY,
            regular_trading_hours=True,
            timeout=timedelta(seconds=60),
        )
        out_path = get_file(ticker, BarFrequency.ONE_DAY)
        df.write_parquet(out_path)
        print(f"{ticker}: saved {df.height:,} daily bars -> {out_path}")
