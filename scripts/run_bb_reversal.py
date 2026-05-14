"""Run BB Consecutive-Bar Reversal signal detection and backtest on SPY bars.

Set TIMEFRAME = "1m" for 1-min bars or "5m" for 5-min bars.
"""

from pathlib import Path

import polars as pl
from loguru import logger

from src.algo.bb_reversal import analyze_bb_reversal, backtest_bb_reversal, detect_bb_reversal_signals
from src.algo.resample_bars import resample_to_timeframe
from src.models import BarFrequency, get_file

TIMEFRAME = "1m"   # "1m" or "5m"

# BB window (bars). 20 works for both timeframes as a starting point.
BB_WINDOW = 20

# Ranging environment: MA stable (< $0.50) AND price actively oscillating (range > $1.50) over 1 hour.
# 1-min: 1 hour = 60 bars.  5-min: 1 hour = 12 bars.
FLAT_LOOKBACK = 60 if TIMEFRAME == "1m" else 12
FLAT_MA_MAX = 0.5
FLAT_RANGE_MIN = 1.5

# Optional price-regime filters (set to None to disable)
MAX_BAND_WIDTH_PCT: float | None = None
MAX_PRICE_RANGE: float | None = None

# TP target: "bb_mid" | "bb_band" (opposite band) | "r1"
TP_TARGET = "bb_band"


def main() -> None:
    ticker = "SPY"
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    tf_label = TIMEFRAME.replace("m", "min")

    logger.info("Loading SPY 1-min data...")
    df_1min = pl.read_parquet(get_file(ticker, BarFrequency.ONE_MIN))
    logger.info(f"Loaded {len(df_1min):,} bars | {df_1min['DateTime'].min()} → {df_1min['DateTime'].max()}")

    if TIMEFRAME == "1m":
        df = df_1min
        logger.info(f"Using 1-min bars directly ({len(df):,} bars)")
    else:
        df = resample_to_timeframe(df_1min, TIMEFRAME)
        logger.info(f"Resampled to {len(df):,} {tf_label} bars")

    df = detect_bb_reversal_signals(
        df,
        window=BB_WINDOW,
        stds=2.0,
        flat_lookback=FLAT_LOOKBACK,
        flat_ma_max=FLAT_MA_MAX,
        flat_range_min=FLAT_RANGE_MIN,
        max_band_width_pct=MAX_BAND_WIDTH_PCT,
        max_price_range=MAX_PRICE_RANGE,
        tp_target=TP_TARGET,
    )

    parquet_path = output_dir / f"SPY_bb_reversal_{tf_label}_signals.parquet"
    df.write_parquet(parquet_path)
    logger.info(f"Full labeled DataFrame saved → {parquet_path}")

    signals_only = df.filter(pl.col("signal_direction").is_not_null())
    csv_path = output_dir / f"SPY_bb_reversal_{tf_label}_signals.csv"
    signals_only.write_csv(csv_path)
    logger.info(f"Signal-only rows ({len(signals_only)}) saved → {csv_path}")

    results_df = backtest_bb_reversal(df)

    results_path = output_dir / f"SPY_bb_reversal_{tf_label}_backtest.csv"
    results_df.write_csv(results_path)
    logger.info(f"Backtest results saved → {results_path}")

    analyze_bb_reversal(results_df)


if __name__ == "__main__":
    main()
