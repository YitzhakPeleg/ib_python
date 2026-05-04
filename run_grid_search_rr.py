#!/usr/bin/env python3
"""
Grid Search for Risk-Reward Ratio Optimization

This script runs the First Bar Breakout Strategy with different risk-reward ratios
to find the optimal parameter value.

Usage:
    python run_grid_search_rr.py
"""

from pathlib import Path

import polars as pl
from loguru import logger

from src.algo.first_bar_breakout import FirstBarBreakoutStrategy
from src.models import BarFrequency, get_file


def run_grid_search(
    df: pl.DataFrame,
    rr_start: float = 0.5,
    rr_end: float = 2.0,
    rr_step: float = 0.1,
) -> pl.DataFrame:
    """
    Run grid search over risk-reward ratios.

    Args:
        df: DataFrame with OHLC data
        rr_start: Starting risk-reward ratio
        rr_end: Ending risk-reward ratio
        rr_step: Step size for risk-reward ratio

    Returns:
        DataFrame with results for each risk-reward ratio
    """
    import numpy as np

    # Generate risk-reward ratios to test
    rr_values = np.arange(rr_start, rr_end + rr_step, rr_step)
    rr_values = [round(rr, 1) for rr in rr_values]  # Round to 1 decimal

    logger.info("=" * 80)
    logger.info("GRID SEARCH - RISK-REWARD RATIO OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Testing {len(rr_values)} risk-reward ratios: {rr_values}")
    logger.info("=" * 80)

    results = []

    for rr in rr_values:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing Risk-Reward Ratio: {rr:.1f}R")
        logger.info(f"{'=' * 80}")

        # Initialize strategy with this risk-reward ratio
        strategy = FirstBarBreakoutStrategy(risk_reward_ratio=rr)

        # Run backtest (suppress individual trade logs)
        import sys
        from io import StringIO

        # Capture logs to reduce output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            trades = strategy.backtest(df)
            stats = strategy.generate_summary_stats(trades)
        finally:
            sys.stdout = old_stdout

        # Store results
        result = {
            "risk_reward_ratio": rr,
            "total_trades": stats["total_trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": stats["win_rate"],
            "total_pnl": stats["total_pnl"],
            "avg_pnl": stats["avg_pnl"],
            "avg_win": stats["avg_win"],
            "avg_loss": stats["avg_loss"],
            "profit_factor": stats["profit_factor"],
            "gross_profit": stats["gross_profit"],
            "gross_loss": stats["gross_loss"],
            "tp_exits": stats["tp_exits"],
            "sl_exits": stats["sl_exits"],
            "eod_exits": stats["eod_exits"],
            "avg_bars_held": stats["avg_bars_held"],
        }
        results.append(result)

        # Print summary for this RR
        logger.info(f"\nResults for RR={rr:.1f}:")
        logger.info(f"  Total PnL: ${stats['total_pnl']:.2f}")
        logger.info(f"  Win Rate: {stats['win_rate']:.2%}")
        logger.info(f"  Profit Factor: {stats['profit_factor']:.2f}")
        logger.info(f"  Avg PnL: ${stats['avg_pnl']:.2f}")

    # Convert to DataFrame
    results_df = pl.DataFrame(results)

    return results_df


def main():
    """Run grid search on SPY data."""
    # Configuration
    ticker = "SPY"
    frequency = BarFrequency.ONE_MIN
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("FIRST BAR BREAKOUT - RISK-REWARD RATIO GRID SEARCH")
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

    # Run grid search
    logger.info("\nStarting grid search...")
    results_df = run_grid_search(df, rr_start=0.5, rr_end=2.0, rr_step=0.1)

    # Sort by total PnL
    results_df = results_df.sort("total_pnl", descending=True)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("\nTop 10 Risk-Reward Ratios by Total PnL:")
    logger.info("-" * 80)

    top_10 = results_df.head(10)
    for row in top_10.iter_rows(named=True):
        logger.info(
            f"RR={row['risk_reward_ratio']:.1f}R: "
            f"PnL=${row['total_pnl']:>7.2f} | "
            f"WinRate={row['win_rate']:>5.1%} | "
            f"PF={row['profit_factor']:>4.2f} | "
            f"AvgPnL=${row['avg_pnl']:>5.2f}"
        )

    # Find best by different metrics
    best_pnl = results_df.filter(pl.col("total_pnl") == pl.col("total_pnl").max()).row(
        0, named=True
    )
    best_pf = results_df.filter(
        pl.col("profit_factor") == pl.col("profit_factor").max()
    ).row(0, named=True)
    best_wr = results_df.filter(pl.col("win_rate") == pl.col("win_rate").max()).row(
        0, named=True
    )

    logger.info("\n" + "=" * 80)
    logger.info("BEST PARAMETERS BY METRIC")
    logger.info("=" * 80)
    logger.info(
        f"Best Total PnL: RR={best_pnl['risk_reward_ratio']:.1f}R "
        f"(${best_pnl['total_pnl']:.2f})"
    )
    logger.info(
        f"Best Profit Factor: RR={best_pf['risk_reward_ratio']:.1f}R "
        f"({best_pf['profit_factor']:.2f})"
    )
    logger.info(
        f"Best Win Rate: RR={best_wr['risk_reward_ratio']:.1f}R "
        f"({best_wr['win_rate']:.2%})"
    )

    # Export results
    output_file = output_dir / f"{ticker}_grid_search_rr_results.csv"
    logger.info(f"\nExporting results to {output_file}...")
    results_df.write_csv(output_file)

    # Create summary report
    summary_file = output_dir / f"{ticker}_grid_search_rr_summary.txt"
    logger.info(f"Exporting summary to {summary_file}...")
    with open(summary_file, "w") as f:
        f.write("FIRST BAR BREAKOUT - RISK-REWARD RATIO GRID SEARCH\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Frequency: {frequency}\n")
        f.write(f"Trading Days: {len(unique_dates)}\n")
        f.write("Risk-Reward Ratios Tested: 0.5R to 2.0R (step 0.1)\n\n")

        f.write("BEST PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Best Total PnL: RR={best_pnl['risk_reward_ratio']:.1f}R "
            f"(${best_pnl['total_pnl']:.2f})\n"
        )
        f.write(
            f"Best Profit Factor: RR={best_pf['risk_reward_ratio']:.1f}R "
            f"({best_pf['profit_factor']:.2f})\n"
        )
        f.write(
            f"Best Win Rate: RR={best_wr['risk_reward_ratio']:.1f}R "
            f"({best_wr['win_rate']:.2%})\n\n"
        )

        f.write("TOP 10 BY TOTAL PNL\n")
        f.write("-" * 80 + "\n")
        for row in top_10.iter_rows(named=True):
            f.write(
                f"RR={row['risk_reward_ratio']:.1f}R: "
                f"PnL=${row['total_pnl']:>7.2f} | "
                f"WinRate={row['win_rate']:>5.1%} | "
                f"PF={row['profit_factor']:>4.2f} | "
                f"AvgPnL=${row['avg_pnl']:>5.2f}\n"
            )

    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()


# Made with Bob
