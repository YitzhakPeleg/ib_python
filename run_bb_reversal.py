"""Run BB Consecutive-Bar Reversal signal detection on SPY 1-min data."""

from pathlib import Path

import polars as pl
from loguru import logger

from src.algo.bb_reversal import detect_bb_reversal_signals
from src.models import BarFrequency, get_file


def main() -> None:
    ticker = "SPY"
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    logger.info("Loading SPY 1-min data...")
    df = pl.read_parquet(get_file(ticker, BarFrequency.ONE_MIN))
    logger.info(f"Loaded {len(df):,} bars | {df['DateTime'].min()} → {df['DateTime'].max()}")

    df = detect_bb_reversal_signals(df, window=20, stds=2.0)

    parquet_path = output_dir / "SPY_bb_reversal_signals.parquet"
    df.write_parquet(parquet_path)
    logger.info(f"Full labeled DataFrame saved → {parquet_path}")

    signals = df.filter(pl.col("signal_direction").is_not_null())
    csv_path = output_dir / "SPY_bb_reversal_signals.csv"
    signals.write_csv(csv_path)
    logger.info(f"Signal-only rows ({len(signals)}) saved → {csv_path}")

    logger.info("\nSample signals:")
    with pl.Config(tbl_rows=20):
        print(
            signals.select(
                ["DateTime", "Open", "High", "Low", "Close", "bb_lower", "bb_upper",
                 "signal_direction", "entry_price"]
            ).head(20)
        )


if __name__ == "__main__":
    main()
