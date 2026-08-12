"""Fetch SPY 1-min bars for 2018-01-01 through 2019-12-31 as a separate file.

Second out-of-sample slice, extending the SPY_2020_2021_1_min.parquet idea
two more years back — saved to its own file rather than merged into any
existing one.

Same chunked-backward pattern as fetch_spy_history.py / fetch_spy_2020_2021.py:
IB caps how much 1-min data one request can return, so walk backward in
fixed-size chunks until IB returns zero bars or an error, or we reach the
2018-01-01 target. Requires IB Gateway/TWS running (see .env for host/port).
"""

import time
from datetime import datetime, timedelta, timezone

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
    ib_client_id: int = 128  # distinct from other fetch scripts' client ids


settings = IBSettings()

TICKER = "SPY"
OUT_TICKER_LABEL = "SPY_2018_2019"  # separate file
TARGET_START = datetime(2018, 1, 1)
CHUNK = timedelta(days=30)
SLEEP_BETWEEN_REQUESTS = 2.0
MAX_CHUNKS = 30  # ~2 years / 30-day chunks + margin

# Start just before SPY_2020_2021_1_min.parquet's earliest bar (2020-01-02 09:30 ET).
end_date = datetime(2020, 1, 2, 9, 29)

all_new_frames = []
with HistoricalDataFetcher(
    host=settings.ib_host, port=settings.ib_port, client_id=settings.ib_client_id
) as fetcher:
    for i in range(MAX_CHUNKS):
        try:
            chunk_df = fetcher.get_historical_data(
                contract=ContractSpec(symbol=TICKER),
                end_date=end_date,
                duration=CHUNK,
                frequency=BarFrequency.ONE_MIN,
                regular_trading_hours=True,
                timeout=timedelta(seconds=60),
                timezone="US/Eastern",
            )
        except RuntimeError as e:
            logger.warning(f"Chunk {i}: request failed/timed out ({e}) — stopping.")
            break

        if chunk_df.height == 0:
            logger.info(
                f"Chunk {i}: 0 bars returned — reached the start of available history."
            )
            break

        all_new_frames.append(chunk_df)
        chunk_start = chunk_df["DateTime"].min()
        logger.info(
            f"Chunk {i}: {chunk_df.height:,} bars, {chunk_start} -> {chunk_df['DateTime'].max()}"
        )

        chunk_start_naive = chunk_start.astimezone(timezone.utc).replace(tzinfo=None)
        if chunk_start_naive <= TARGET_START:
            logger.info(f"Reached target start {TARGET_START} — stopping.")
            break

        end_date = chunk_start_naive - timedelta(minutes=1)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

if not all_new_frames:
    logger.warning("No data fetched.")
else:
    new_data = pl.concat(all_new_frames).unique(subset=["DateTime"]).sort("DateTime")
    new_data = new_data.filter(
        pl.col("DateTime") >= pl.lit(TARGET_START).dt.replace_time_zone("US/Eastern")
    )
    out_path = get_file(OUT_TICKER_LABEL, BarFrequency.ONE_MIN)
    new_data.write_parquet(out_path)
    logger.info(
        f"Saved {new_data.height:,} bars -> {out_path}, "
        f"{new_data['DateTime'].min()} -> {new_data['DateTime'].max()}"
    )
