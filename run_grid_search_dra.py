"""
Grid Search for First Bar Breakout Strategy - Daily Range Average (DRA) Variant

This script runs a grid search to find the optimal K coefficient
for the DRA-based TP/SL calculation.

Tests K values from 0.5 to 2.0 in steps of 0.1.
"""

from pathlib import Path

import polars as pl
from loguru import logger

from src.algo.first_bar_breakout_dra import (
    FirstBarBreakoutDRAStrategy,
    export_results_to_csv,
)


def run_grid_search(
    df: pl.DataFrame,
    k_start: float = 0.1,
    k_end: float = 2.0,
    k_step: float = 0.1,
    dra_window: int = 20,
) -> pl.DataFrame:
    """
    Run grid search over K coefficients.

    Args:
        df: DataFrame with OHLC data
        k_start: Starting K coefficient (default: 0.5)
        k_end: Ending K coefficient (default: 2.0)
        k_step: Step size (default: 0.1)
        dra_window: Window for DRA calculation (default: 20 days)

    Returns:
        DataFrame with results for each K coefficient tested
    """
    logger.info("=" * 80)
    logger.info("GRID SEARCH - DAILY RANGE AVERAGE (DRA) VARIANT")
    logger.info(
        f"Testing K coefficients from {k_start:.1f} to {k_end:.1f} (step {k_step:.1f})"
    )
    logger.info(f"DRA Window: {dra_window} days")
    logger.info("=" * 80)

    # Generate list of K coefficients to test
    import numpy as np

    k_values = np.arange(k_start, k_end + k_step / 2, k_step)
    k_values = [round(k, 2) for k in k_values]  # Round to avoid floating point issues

    logger.info(f"Total parameter combinations to test: {len(k_values)}")

    results = []

    for i, k_coef in enumerate(k_values, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Test {i}/{len(k_values)}: K Coefficient = {k_coef:.2f}")
        logger.info(f"{'=' * 80}")

        # Create strategy with this K coefficient
        strategy = FirstBarBreakoutDRAStrategy(
            k_coefficient=k_coef, dra_window=dra_window
        )

        # Run backtest
        trades = strategy.backtest(df)

        # Generate statistics
        stats = strategy.generate_summary_stats(trades)

        if stats:
            # Add K coefficient to stats
            stats["k_coefficient"] = k_coef

            # Store results
            results.append(stats)

            # Print summary
            logger.info(f"\nResults for K={k_coef:.2f}:")
            logger.info(f"  Total Trades: {stats['total_trades']}")
            logger.info(f"  Win Rate: {stats['win_rate']:.2%}")
            logger.info(f"  Total P&L: ${stats['total_pnl']:.2f}")
            logger.info(f"  Profit Factor: {stats['profit_factor']:.2f}")
            logger.info(f"  Avg DRA: ${stats['avg_dra']:.2f}")
        else:
            logger.warning(f"No trades for K={k_coef:.2f}")

    # Convert results to DataFrame
    if results:
        results_df = pl.DataFrame(results)

        # Sort by total P&L descending
        results_df = results_df.sort("total_pnl", descending=True)

        logger.info("\n" + "=" * 80)
        logger.info("GRID SEARCH COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total parameter combinations tested: {len(results)}")
        logger.info("\nTop 10 by Total P&L:")

        # Print results without pandas
        top_10 = results_df.select(
            [
                "k_coefficient",
                "total_trades",
                "win_rate",
                "total_pnl",
                "profit_factor",
                "avg_dra",
            ]
        ).head(10)

        logger.info(
            f"\n{'K Coef':<10} {'Trades':<10} {'Win Rate':<12} {'Total P&L':<12} {'Profit Factor':<15} {'Avg DRA':<12}"
        )
        logger.info("-" * 80)
        for row in top_10.iter_rows(named=True):
            logger.info(
                f"{row['k_coefficient']:<10.2f} {row['total_trades']:<10} "
                f"{row['win_rate']:<11.2%} ${row['total_pnl']:<11.2f} "
                f"{row['profit_factor']:<15.2f} ${row['avg_dra']:<11.2f}"
            )

        return results_df
    else:
        logger.error("No results generated from grid search")
        return pl.DataFrame()


def export_grid_search_results(
    results_df: pl.DataFrame, output_dir: str = "results"
) -> None:
    """
    Export grid search results to CSV and summary text file.

    Args:
        results_df: DataFrame with grid search results
        output_dir: Directory to save results (default: "results")
    """
    if results_df.is_empty():
        logger.warning("No results to export")
        return

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export full results to CSV
    csv_path = f"{output_dir}/SPY_grid_search_dra_results.csv"
    results_df.write_csv(csv_path)
    logger.info(f"Full results exported to {csv_path}")

    # Create summary text file
    summary_path = f"{output_dir}/SPY_grid_search_dra_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GRID SEARCH SUMMARY - DAILY RANGE AVERAGE (DRA) VARIANT\n")
        f.write("=" * 80 + "\n\n")

        # Best by total P&L
        best_pnl = results_df.sort("total_pnl", descending=True).row(0, named=True)
        f.write("Best by Total P&L:\n")
        f.write(f"  K Coefficient: {best_pnl['k_coefficient']:.2f}\n")
        f.write(f"  Total Trades: {best_pnl['total_trades']}\n")
        f.write(f"  Win Rate: {best_pnl['win_rate']:.2%}\n")
        f.write(f"  Total P&L: ${best_pnl['total_pnl']:.2f}\n")
        f.write(f"  Profit Factor: {best_pnl['profit_factor']:.2f}\n")
        f.write(f"  Avg P&L: ${best_pnl['avg_pnl']:.2f}\n")
        f.write(f"  Avg DRA: ${best_pnl['avg_dra']:.2f}\n\n")

        # Best by profit factor
        best_pf = results_df.sort("profit_factor", descending=True).row(0, named=True)
        f.write("Best by Profit Factor:\n")
        f.write(f"  K Coefficient: {best_pf['k_coefficient']:.2f}\n")
        f.write(f"  Total Trades: {best_pf['total_trades']}\n")
        f.write(f"  Win Rate: {best_pf['win_rate']:.2%}\n")
        f.write(f"  Total P&L: ${best_pf['total_pnl']:.2f}\n")
        f.write(f"  Profit Factor: {best_pf['profit_factor']:.2f}\n")
        f.write(f"  Avg P&L: ${best_pf['avg_pnl']:.2f}\n")
        f.write(f"  Avg DRA: ${best_pf['avg_dra']:.2f}\n\n")

        # Best by win rate
        best_wr = results_df.sort("win_rate", descending=True).row(0, named=True)
        f.write("Best by Win Rate:\n")
        f.write(f"  K Coefficient: {best_wr['k_coefficient']:.2f}\n")
        f.write(f"  Total Trades: {best_wr['total_trades']}\n")
        f.write(f"  Win Rate: {best_wr['win_rate']:.2%}\n")
        f.write(f"  Total P&L: ${best_wr['total_pnl']:.2f}\n")
        f.write(f"  Profit Factor: {best_wr['profit_factor']:.2f}\n")
        f.write(f"  Avg P&L: ${best_wr['avg_pnl']:.2f}\n")
        f.write(f"  Avg DRA: ${best_wr['avg_dra']:.2f}\n\n")

        # Top 10 by total P&L
        f.write("=" * 80 + "\n")
        f.write("Top 10 K Coefficients by Total P&L:\n")
        f.write("=" * 80 + "\n\n")

        top_10 = results_df.select(
            [
                "k_coefficient",
                "total_trades",
                "wins",
                "losses",
                "win_rate",
                "total_pnl",
                "profit_factor",
                "avg_pnl",
                "avg_dra",
            ]
        ).head(10)

        # Write header
        f.write(
            f"{'K Coef':<10} {'Trades':<10} {'Wins':<8} {'Losses':<8} {'Win Rate':<12} {'Total P&L':<12} {'Profit Factor':<15} {'Avg P&L':<12} {'Avg DRA':<12}\n"
        )
        f.write("-" * 110 + "\n")

        # Write rows
        for row in top_10.iter_rows(named=True):
            f.write(
                f"{row['k_coefficient']:<10.2f} {row['total_trades']:<10} "
                f"{row['wins']:<8} {row['losses']:<8} "
                f"{row['win_rate']:<11.2%} ${row['total_pnl']:<11.2f} "
                f"{row['profit_factor']:<15.2f} ${row['avg_pnl']:<11.2f} ${row['avg_dra']:<11.2f}\n"
            )

        f.write("\n\n")

        # Statistics across all tests
        f.write("=" * 80 + "\n")
        f.write("Statistics Across All Tests:\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total parameter combinations tested: {len(results_df)}\n")
        f.write(f"Average Total P&L: ${results_df['total_pnl'].mean():.2f}\n")
        f.write(f"Median Total P&L: ${results_df['total_pnl'].median():.2f}\n")
        f.write(f"Best Total P&L: ${results_df['total_pnl'].max():.2f}\n")
        f.write(f"Worst Total P&L: ${results_df['total_pnl'].min():.2f}\n")
        f.write(f"Average Win Rate: {results_df['win_rate'].mean():.2%}\n")
        f.write(f"Average Profit Factor: {results_df['profit_factor'].mean():.2f}\n")
        f.write(f"Average DRA: ${results_df['avg_dra'].mean():.2f}\n")

    logger.info(f"Summary exported to {summary_path}")


def main():
    """Main execution function."""
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="{message}",
        level="INFO",
    )

    # Load SPY data
    logger.info("Loading SPY data...")
    data_path = "data/SPY_1_min.parquet"

    try:
        df = pl.read_parquet(data_path)
        logger.info(f"Loaded {len(df):,} rows of data")
        logger.info(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Run grid search
    results_df = run_grid_search(
        df,
        k_start=0.1,
        k_end=2.0,
        k_step=0.1,
        dra_window=20,
    )

    # Export results
    if not results_df.is_empty():
        export_grid_search_results(results_df)

        # Find and run best parameter
        best_row = results_df.sort("total_pnl", descending=True).row(0, named=True)
        best_k = best_row["k_coefficient"]

        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING BEST PARAMETER: K Coefficient = {best_k:.2f}")
        logger.info("=" * 80)

        # Run with best parameter and export detailed trades
        strategy = FirstBarBreakoutDRAStrategy(k_coefficient=best_k, dra_window=20)
        trades = strategy.backtest(df)
        stats = strategy.generate_summary_stats(trades)
        strategy.print_summary(stats)

        # Export detailed trades
        export_results_to_csv(
            trades, "results/SPY_first_bar_breakout_dra_best_trades.csv"
        )

    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


# Made with Bob
