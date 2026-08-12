"""Fetch ES (E-mini S&P 500) continuous-futures daily bars.

Requesting a long duration (5 Y) from IB returns bars going back further
than the account's real market-data permission for CME futures actually
covers — anything before ~2025-01 comes back as flat OHLC with zero volume
(placeholder ticks, not real bars), confirmed by inspecting month-by-month
zero-volume rates. This script fetches the full duration and then drops
everything before the first bar with real volume, rather than hand-coding
a cutoff date that would go stale.

Requires IB Gateway/TWS running (see .env for host/port).
"""

from datetime import timedelta

from data_fetching.historical_data_fetcher import HistoricalDataFetcher
from models.models import BarFrequency, ContractSpec, Exchange, SecurityType
from models.paths import get_file
from pydantic_settings import BaseSettings, SettingsConfigDict


class IBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_client_id: int = 132


settings = IBSettings()

with HistoricalDataFetcher(
    host=settings.ib_host, port=settings.ib_port, client_id=settings.ib_client_id
) as fetcher:
    df = fetcher.get_historical_data(
        contract=ContractSpec(
            symbol="ES", sec_type=SecurityType.CONTINUOUS_FUTURE, exchange=Exchange.CME
        ),
        duration=timedelta(days=5 * 365),
        frequency=BarFrequency.ONE_DAY,
        regular_trading_hours=True,
        timeout=timedelta(seconds=90),
    )

df = df.sort("DateTime")
# The real-data window is contiguous once it starts — find the first bar of
# the longest trailing zero-volume-free run rather than just the first
# nonzero-volume bar, since a handful of early bars can have nonzero volume
# in isolation (noise) before the real, permitted window actually begins.
is_zero = (df["Volume"] == 0).to_numpy()
# last index where a zero-volume bar occurs; real data starts right after it
zero_indices = [i for i, z in enumerate(is_zero) if z]
start_idx = (zero_indices[-1] + 1) if zero_indices else 0
clean = df[start_idx:]

print(f"Fetched {df.height:,} bars ({df['DateTime'].min()} -> {df['DateTime'].max()})")
print(f"Dropping {start_idx:,} leading bars with zero volume / stale placeholder data")
print(
    f"Clean range: {clean.height:,} bars, {clean['DateTime'].min()} -> {clean['DateTime'].max()}"
)

out_path = get_file("ES", BarFrequency.ONE_DAY)
clean.write_parquet(out_path)
print(f"Saved -> {out_path}")
