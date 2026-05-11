"""
Grid Search for First Bar Breakout Strategy - Fixed Dollar TP/SL Variant

This script runs a grid search to find the optimal fixed dollar amount
for take-profit and stop-loss.

Tests fixed amounts from $0.50 to $2.00 in steps of $0.10.
"""

from pathlib import Path

import polars as pl
from loguru import logger

from src.algo.first_bar_breakout_fixed import (
    FirstBarBreakoutFixedStrategy,
    export_results_to_csv,
)


def run_grid_search(
    df: pl.DataFrame,
    amount_start: float = 0.5,
    amount_end: float = 3.0,
    amount_step: float = 0.1,
) -> pl.DataFrame:
    """
    Run grid search over fixed dollar amounts.

    Args:
        df: DataFrame with OHLC data
        amount_start: Starting fixed amount (default: 0.5)
        amount_end: Ending fixed amount (default: 2.0)
        amount_step: Step size (default: 0.1)

    Returns:
        DataFrame with results for each fixed amount tested
    """
    logger.info("=" * 80)
    logger.info("GRID SEARCH - FIXED DOLLAR TP/SL VARIANT")
    logger.info(
        f"Testing fixed amounts from ${amount_start:.2f} to ${amount_end:.2f} (step ${amount_step:.2f})"
    )
    logger.info("=" * 80)

    # Generate list of fixed amounts to test
    import numpy as np

    fixed_amounts = np.arange(amount_start, amount_end + amount_step / 2, amount_step)
    fixed_amounts = [
        round(amt, 2) for amt in fixed_amounts
    ]  # Round to avoid floating point issues

    logger.info(f"Total parameter combinations to test: {len(fixed_amounts)}")

    results = []

    for i, fixed_amount in enumerate(fixed_amounts, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(
            f"Test {i}/{len(fixed_amounts)}: Fixed Amount = ${fixed_amount:.2f}"
        )
        logger.info(f"{'=' * 80}")

        # Create strategy with this fixed amount
        strategy = FirstBarBreakoutFixedStrategy(fixed_amount=fixed_amount)

        # Run backtest
        trades = strategy.backtest(df)

        # Generate statistics
        stats = strategy.generate_summary_stats(trades)

        if stats:
            # Add fixed amount to stats
            stats["fixed_amount"] = fixed_amount

            # Store results
            results.append(stats)

            # Print summary
            logger.info(f"\nResults for Fixed Amount ${fixed_amount:.2f}:")
            logger.info(f"  Total Trades: {stats['total_trades']}")
            logger.info(f"  Win Rate: {stats['win_rate']:.2%}")
            logger.info(f"  Total P&L: ${stats['total_pnl']:.2f}")
            logger.info(f"  Profit Factor: {stats['profit_factor']:.2f}")
        else:
            logger.warning(f"No trades for Fixed Amount ${fixed_amount:.2f}")

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
                "fixed_amount",
                "total_trades",
                "win_rate",
                "total_pnl",
                "profit_factor",
            ]
        ).head(10)

        logger.info(
            f"\n{'Fixed Amount':<15} {'Trades':<10} {'Win Rate':<12} {'Total P&L':<12} {'Profit Factor':<15}"
        )
        logger.info("-" * 70)
        for row in top_10.iter_rows(named=True):
            logger.info(
                f"${row['fixed_amount']:<14.2f} {row['total_trades']:<10} "
                f"{row['win_rate']:<11.2%} ${row['total_pnl']:<11.2f} {row['profit_factor']:<15.2f}"
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
    csv_path = f"{output_dir}/SPY_grid_search_fixed_results.csv"
    results_df.write_csv(csv_path)
    logger.info(f"Full results exported to {csv_path}")

    # Create summary text file
    summary_path = f"{output_dir}/SPY_grid_search_fixed_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GRID SEARCH SUMMARY - FIXED DOLLAR TP/SL VARIANT\n")
        f.write("=" * 80 + "\n\n")

        # Best by total P&L
        best_pnl = results_df.sort("total_pnl", descending=True).row(0, named=True)
        f.write("Best by Total P&L:\n")
        f.write(f"  Fixed Amount: ${best_pnl['fixed_amount']:.2f}\n")
        f.write(f"  Total Trades: {best_pnl['total_trades']}\n")
        f.write(f"  Win Rate: {best_pnl['win_rate']:.2%}\n")
        f.write(f"  Total P&L: ${best_pnl['total_pnl']:.2f}\n")
        f.write(f"  Profit Factor: {best_pnl['profit_factor']:.2f}\n")
        f.write(f"  Avg P&L: ${best_pnl['avg_pnl']:.2f}\n\n")

        # Best by profit factor
        best_pf = results_df.sort("profit_factor", descending=True).row(0, named=True)
        f.write("Best by Profit Factor:\n")
        f.write(f"  Fixed Amount: ${best_pf['fixed_amount']:.2f}\n")
        f.write(f"  Total Trades: {best_pf['total_trades']}\n")
        f.write(f"  Win Rate: {best_pf['win_rate']:.2%}\n")
        f.write(f"  Total P&L: ${best_pf['total_pnl']:.2f}\n")
        f.write(f"  Profit Factor: {best_pf['profit_factor']:.2f}\n")
        f.write(f"  Avg P&L: ${best_pf['avg_pnl']:.2f}\n\n")

        # Best by win rate
        best_wr = results_df.sort("win_rate", descending=True).row(0, named=True)
        f.write("Best by Win Rate:\n")
        f.write(f"  Fixed Amount: ${best_wr['fixed_amount']:.2f}\n")
        f.write(f"  Total Trades: {best_wr['total_trades']}\n")
        f.write(f"  Win Rate: {best_wr['win_rate']:.2%}\n")
        f.write(f"  Total P&L: ${best_wr['total_pnl']:.2f}\n")
        f.write(f"  Profit Factor: {best_wr['profit_factor']:.2f}\n")
        f.write(f"  Avg P&L: ${best_wr['avg_pnl']:.2f}\n\n")

        # Top 10 by total P&L
        f.write("=" * 80 + "\n")
        f.write("Top 10 Fixed Amounts by Total P&L:\n")
        f.write("=" * 80 + "\n\n")

        top_10 = results_df.select(
            [
                "fixed_amount",
                "total_trades",
                "wins",
                "losses",
                "win_rate",
                "total_pnl",
                "profit_factor",
                "avg_pnl",
            ]
        ).head(10)

        # Write header
        f.write(
            f"{'Fixed Amount':<15} {'Trades':<10} {'Wins':<8} {'Losses':<8} {'Win Rate':<12} {'Total P&L':<12} {'Profit Factor':<15} {'Avg P&L':<12}\n"
        )
        f.write("-" * 100 + "\n")

        # Write rows
        for row in top_10.iter_rows(named=True):
            f.write(
                f"${row['fixed_amount']:<14.2f} {row['total_trades']:<10} "
                f"{row['wins']:<8} {row['losses']:<8} "
                f"{row['win_rate']:<11.2%} ${row['total_pnl']:<11.2f} "
                f"{row['profit_factor']:<15.2f} ${row['avg_pnl']:<11.2f}\n"
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
        amount_start=0.5,
        amount_end=3.0,
        amount_step=0.1,
    )

    # Export results
    if not results_df.is_empty():
        export_grid_search_results(results_df)

        # Find and run best parameter
        best_row = results_df.sort("total_pnl", descending=True).row(0, named=True)
        best_amount = best_row["fixed_amount"]

        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING BEST PARAMETER: Fixed Amount = ${best_amount:.2f}")
        logger.info("=" * 80)

        # Run with best parameter and export detailed trades
        strategy = FirstBarBreakoutFixedStrategy(fixed_amount=best_amount)
        trades = strategy.backtest(df)
        stats = strategy.generate_summary_stats(trades)
        strategy.print_summary(stats)

        # Export detailed trades
        export_results_to_csv(
            trades, "results/SPY_first_bar_breakout_fixed_best_trades.csv"
        )

    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


# Made with Bob
