"""Fetch 1-min and daily bars for a new, sector/size-diversified ticker
basket (financials, healthcare, energy, staples, industrials, telecom,
media, plus a mid-cap and a regional bank for size diversity — see
docs/04_range_breakout_strategy.md-adjacent ML work in src/ml/ for why).

1-min bars: chunked-backward fetch targeting ~600 days (~20 months) of
history per ticker, same pattern as examples/fetch_spy_2018_2019.py —
walks backward in 30-day chunks until IB returns zero bars, an error, or
the target start date is reached. Daily bars: single 5-year request per
ticker, same pattern as examples/fetch_daily_bars.py.

Requires IB Gateway/TWS running (see .env for host/port).
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
    ib_client_id: int = 140  # distinct from other fetch scripts' client ids


settings = IBSettings()

TICKERS = ["JPM", "UNH", "XOM", "PG", "HD", "CAT", "VZ", "DIS", "DKNG", "ZION"]
MINUTE_CHUNK = timedelta(days=30)
TARGET_DAYS_BACK = 600  # ~20 months, matching the existing non-SPY tickers' depth
SLEEP_BETWEEN_REQUESTS = 2.0
MAX_CHUNKS_PER_TICKER = 25  # ~600 days / 30-day chunks + margin
DAILY_DURATION = timedelta(days=5 * 365)

target_start = datetime.now() - timedelta(days=TARGET_DAYS_BACK)

with HistoricalDataFetcher(
    host=settings.ib_host, port=settings.ib_port, client_id=settings.ib_client_id
) as fetcher:
    # --- Daily bars (one request per ticker, no chunking needed) ---
    for ticker in TICKERS:
        try:
            daily_df = fetcher.get_historical_data(
                contract=ContractSpec(symbol=ticker),
                duration=DAILY_DURATION,
                frequency=BarFrequency.ONE_DAY,
                regular_trading_hours=True,
                timeout=timedelta(seconds=60),
            )
        except RuntimeError as e:
            logger.warning(f"{ticker}: daily fetch failed ({e}) — skipping.")
            continue
        out_path = get_file(ticker, BarFrequency.ONE_DAY)
        daily_df.write_parquet(out_path)
        logger.info(f"{ticker}: saved {daily_df.height:,} daily bars -> {out_path}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # --- 1-min bars (chunked backward per ticker) ---
    for ticker in TICKERS:
        end_date = datetime.now()
        all_new_frames = []
        for i in range(MAX_CHUNKS_PER_TICKER):
            try:
                chunk_df = fetcher.get_historical_data(
                    contract=ContractSpec(symbol=ticker),
                    end_date=end_date,
                    duration=MINUTE_CHUNK,
                    frequency=BarFrequency.ONE_MIN,
                    regular_trading_hours=True,
                    timeout=timedelta(seconds=60),
                    timezone="US/Eastern",
                )
            except RuntimeError as e:
                logger.warning(
                    f"{ticker} chunk {i}: request failed/timed out ({e}) — stopping."
                )
                break

            if chunk_df.height == 0:
                logger.info(
                    f"{ticker} chunk {i}: 0 bars — reached start of available history."
                )
                break

            all_new_frames.append(chunk_df)
            chunk_start = chunk_df["DateTime"].min()
            logger.info(
                f"{ticker} chunk {i}: {chunk_df.height:,} bars, "
                f"{chunk_start} -> {chunk_df['DateTime'].max()}"
            )

            chunk_start_naive = chunk_start.astimezone(timezone.utc).replace(
                tzinfo=None
            )
            if chunk_start_naive <= target_start:
                logger.info(
                    f"{ticker}: reached target start {target_start} — stopping."
                )
                break

            end_date = chunk_start_naive - timedelta(minutes=1)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        if not all_new_frames:
            logger.warning(f"{ticker}: no 1-min data fetched.")
            continue

        new_data = (
            pl.concat(all_new_frames).unique(subset=["DateTime"]).sort("DateTime")
        )
        out_path = get_file(ticker, BarFrequency.ONE_MIN)
        new_data.write_parquet(out_path)
        logger.info(
            f"{ticker}: saved {new_data.height:,} 1-min bars -> {out_path}, "
            f"{new_data['DateTime'].min()} -> {new_data['DateTime'].max()}"
        )

logger.info("Done.")
