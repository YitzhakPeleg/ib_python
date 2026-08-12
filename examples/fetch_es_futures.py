"""Quick smoke test: fetch ES (E-mini S&P 500) continuous-futures daily bars.

Verifies the new SecurityType.CONTINUOUS_FUTURE / Exchange.CME plumbing
against a live IB connection. Requires IB Gateway/TWS running.
"""

from datetime import timedelta

from data_fetching.historical_data_fetcher import HistoricalDataFetcher
from models.models import BarFrequency, ContractSpec, Exchange, SecurityType
from pydantic_settings import BaseSettings, SettingsConfigDict


class IBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_client_id: int = 127


settings = IBSettings()

with HistoricalDataFetcher(
    host=settings.ib_host, port=settings.ib_port, client_id=settings.ib_client_id
) as fetcher:
    df = fetcher.get_historical_data(
        contract=ContractSpec(
            symbol="ES",
            sec_type=SecurityType.CONTINUOUS_FUTURE,
            exchange=Exchange.CME,
        ),
        duration=timedelta(days=30),
        frequency=BarFrequency.ONE_DAY,
        regular_trading_hours=True,
        timeout=timedelta(seconds=60),
    )
    print(df)
