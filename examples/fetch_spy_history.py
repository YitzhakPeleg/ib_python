"""Chunked fetch of the maximum available SPY 1-min history from IB.

IB caps how much 1-min data a single historicalData request can return, so
this loops backward in fixed-size chunks from the earliest bar we already
have, prepending each new chunk, until IB returns an error or zero bars —
that's how we discover the actual available history empirically rather than
guessing a cutoff. Requires IB Gateway/TWS running (see .env for host/port).
"""

import time
from datetime import timedelta, timezone

import polars as pl
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from data_fetching.historical_data_fetcher import HistoricalDataFetcher
from models.models import BarFrequency, ContractSpec
from models.paths import get_file


class IBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_client_id: int = 124  # different from fetch_daily_bars.py's client id


settings = IBSettings()

TICKER = "SPY"
CHUNK = timedelta(days=30)
SLEEP_BETWEEN_REQUESTS = 2.0  # seconds — stay well under IB's historical-data pacing limits
MAX_CHUNKS = 200  # safety cap so a bug (or unexpectedly deep history) can't loop forever

existing = pl.read_parquet(get_file(TICKER, BarFrequency.ONE_MIN))
earliest_existing = existing["DateTime"].min()
logger.info(f"Existing data starts at {earliest_existing}, {existing.height:,} bars")

# The fetcher formats end_date as "...UTC" regardless of the datetime's own
# tzinfo, so convert explicitly to naive UTC first — otherwise a tz-aware
# US/Eastern timestamp would get mislabeled UTC and silently shift the
# request window by several hours.
end_date = earliest_existing.astimezone(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)

all_new_frames = []
with HistoricalDataFetcher(host=settings.ib_host, port=settings.ib_port, client_id=settings.ib_client_id) as fetcher:
    for i in range(MAX_CHUNKS):
        try:
            chunk_df = fetcher.get_historical_data(
                contract=ContractSpec(symbol=TICKER),
                end_date=end_date,
                duration=CHUNK,
                frequency=BarFrequency.ONE_MIN,
                regular_trading_hours=True,
                timeout=timedelta(seconds=60),
                timezone="US/Eastern",  # match the existing parquet's tz so the two concatenate cleanly
            )
        except RuntimeError as e:
            logger.warning(f"Chunk {i}: request failed/timed out ({e}) — stopping.")
            break

        if chunk_df.height == 0:
            logger.info(f"Chunk {i}: 0 bars returned — reached the start of available history.")
            break

        all_new_frames.append(chunk_df)
        chunk_start = chunk_df["DateTime"].min()
        logger.info(f"Chunk {i}: {chunk_df.height:,} bars, {chunk_start} -> {chunk_df['DateTime'].max()}")

        end_date = chunk_start.astimezone(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

if not all_new_frames:
    logger.warning("No new chunks fetched — nothing to add.")
else:
    new_data = pl.concat(all_new_frames).unique(subset=["DateTime"]).sort("DateTime")
    combined = (
        pl.concat([new_data, existing.select(new_data.columns)])
        .unique(subset=["DateTime"])
        .sort("DateTime")
    )
    combined.write_parquet(get_file(TICKER, BarFrequency.ONE_MIN))
    logger.info(
        f"Combined: {combined.height:,} bars ({existing.height:,} -> {combined.height:,}), "
        f"{combined['DateTime'].min()} -> {combined['DateTime'].max()}"
    )
