"""ATR-filtered opening-range hammer-reversal strategy — backtest driver.

See src/algo/hammer_reversal.py for the rule itself. This script just loads
every ticker's 1-min + 1-day bars, runs the backtest, and reports results
pooled and per ticker.
"""

import polars as pl
from loguru import logger

from algo.hammer_reversal import run_hammer_backtest
from models.paths import get_file

TICKERS = ["AAPL", "AMZN", "AVGO", "GOOG", "IBKR", "MSFT", "NVDA", "SPY"]

minute_frames, daily_frames = [], []
OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
for ticker in TICKERS:
    minute_frames.append(
        pl.read_parquet(get_file(ticker, "1_min"))
        .select(OHLCV)
        .with_columns(pl.lit(ticker).alias("ticker"))
    )
    daily_frames.append(
        pl.read_parquet(get_file(ticker, "1_day"))
        .select(OHLCV)
        .with_columns(pl.lit(ticker).alias("ticker"))
    )

minute_bars = pl.concat(minute_frames).sort("ticker", "DateTime")
daily_bars = pl.concat(daily_frames).sort("ticker", "DateTime")
logger.info(f"Loaded {minute_bars.height:,} 1-min bars across {len(TICKERS)} tickers")

trades = run_hammer_backtest(minute_bars, daily_bars)
logger.info(f"{trades.height:,} trades found")

if trades.height == 0:
    raise SystemExit("No trades — nothing to report.")


def summarize(t: pl.DataFrame) -> dict:
    r = t["r_multiple"]
    exit_counts = {
        row["exit_reason"]: row["len"]
        for row in t.group_by("exit_reason").len().to_dicts()
    }
    return {
        "n": t.height,
        "win_rate": round(float((r > 0).mean()), 4),
        "mean_r": round(float(r.mean()), 4),
        "exit_counts": str(exit_counts),
    }


pl.Config.set_tbl_width_chars(200)

print("\n" + "=" * 10 + " Pooled " + "=" * 10)
print(pl.DataFrame([{"strategy": "hammer_reversal", **summarize(trades)}]))

print("\n" + "=" * 10 + " Per ticker " + "=" * 10)
per_ticker_rows = [
    {"ticker": ticker, **summarize(trades.filter(pl.col("ticker") == ticker))}
    for ticker in TICKERS
    if trades.filter(pl.col("ticker") == ticker).height > 0
]
print(pl.DataFrame(per_ticker_rows).sort("mean_r", descending=True))
