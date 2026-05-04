#!/usr/bin/env python3
"""
Run First Bar Breakout Strategy on SPY data.

This script:
1. Loads SPY 1-minute data
2. Runs the first bar breakout strategy
3. Generates performance statistics
4. Exports results to CSV

Usage:
    python run_first_bar_breakout.py
"""

from pathlib import Path

import polars as pl
from loguru import logger

from src.algo.first_bar_breakout import (
    FirstBarBreakoutStrategy,
    export_results_to_csv,
)
from src.models import BarFrequency, get_file


def main():
    """Run the first bar breakout strategy on SPY data."""
    # Configuration
    ticker = "SPY"
    frequency = BarFrequency.ONE_MIN
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("FIRST BAR BREAKOUT STRATEGY - SPY BACKTEST")
    logger.info("=" * 80)

    # Load SPY data
    logger.info(f"\nLoading {ticker} data ({frequency})...")
    data_file = get_file(ticker, frequency)
    df = pl.read_parquet(data_file)
    logger.info(f"Loaded {len(df):,} bars")
    logger.info(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")

    # Ensure we have date column
    if "date" not in df.columns:
        df = df.with_columns(pl.col("DateTime").dt.date().alias("date"))

    # Get unique trading days
    unique_dates = df.select("date").unique().sort("date")
    logger.info(f"Trading days: {len(unique_dates)}")

    # Initialize strategy
    logger.info("\nInitializing First Bar Breakout Strategy...")
    strategy = FirstBarBreakoutStrategy()

    # Run backtest
    logger.info("\nRunning backtest...\n")
    results = strategy.backtest(df)

    if not results:
        logger.error("No trades generated!")
        return

    # Generate summary statistics
    logger.info("\nGenerating performance statistics...")
    stats = strategy.generate_summary_stats(results)
    strategy.print_summary(stats)

    # Export results to CSV
    output_file = output_dir / f"{ticker}_first_bar_breakout_results.csv"
    logger.info(f"\nExporting results to {output_file}...")
    export_results_to_csv(results, str(output_file))

    # Export summary statistics
    summary_file = output_dir / f"{ticker}_first_bar_breakout_summary.txt"
    logger.info(f"Exporting summary to {summary_file}...")
    with open(summary_file, "w") as f:
        f.write("FIRST BAR BREAKOUT STRATEGY - PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Frequency: {frequency}\n")
        f.write(f"Total Trading Days: {len(unique_dates)}\n\n")
        f.write("TRADE STATISTICS\n")
        f.write("-" * 80 + "\n")
        for key, value in stats.items():
            if isinstance(value, float):
                if "rate" in key or "percent" in key:
                    f.write(f"{key}: {value:.2%}\n")
                else:
                    f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")

    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()


# Made with Bob
