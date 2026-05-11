"""
Run all First Bar Breakout grid searches on 5-minute data.

This script:
1. Loads 1-minute SPY data
2. Resamples to 5-minute bars
3. Runs all 4 grid searches:
   - Fixed Dollar TP/SL
   - Risk-Reward Ratio
   - Daily Range Average (DRA)
   - Mixed: First Bar SL + Fixed TP
4. Exports all results with "_5min" suffix
"""

from pathlib import Path

import polars as pl
from loguru import logger

from src.algo.first_bar_breakout import FirstBarBreakoutStrategy
from src.algo.first_bar_breakout_dra import FirstBarBreakoutDRAStrategy
from src.algo.first_bar_breakout_fixed import FirstBarBreakoutFixedStrategy
from src.algo.first_bar_breakout_mixed_fixed import FirstBarBreakoutMixedFixedStrategy
from src.algo.resample_bars import resample_to_5min


def run_fixed_grid_search(df: pl.DataFrame) -> dict:
    """Run grid search for Fixed Dollar variant."""
    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH 1/4: FIXED DOLLAR TP/SL")
    logger.info("=" * 80)

    import numpy as np

    amounts = np.arange(0.5, 3.0 + 0.05, 0.1)
    amounts = [round(amt, 2) for amt in amounts]

    results = []
    for i, amount in enumerate(amounts, 1):
        logger.info(f"\nTest {i}/{len(amounts)}: Fixed Amount = ${amount:.2f}")
        strategy = FirstBarBreakoutFixedStrategy(fixed_amount=amount)
        trades = strategy.backtest(df)
        stats = strategy.generate_summary_stats(trades)
        if stats:
            stats["fixed_amount"] = amount
            results.append(stats)
            logger.info(
                f"  Total P&L: ${stats['total_pnl']:.2f}, Win Rate: {stats['win_rate']:.2%}"
            )

    results_df = pl.DataFrame(results).sort("total_pnl", descending=True)
    best = results_df.row(0, named=True)

    return {
        "name": "Fixed Dollar",
        "best_param": f"${best['fixed_amount']:.2f}",
        "total_pnl": best["total_pnl"],
        "win_rate": best["win_rate"],
        "profit_factor": best["profit_factor"],
        "results_df": results_df,
    }


def run_rr_grid_search(df: pl.DataFrame) -> dict:
    """Run grid search for Risk-Reward Ratio variant."""
    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH 2/4: RISK-REWARD RATIO")
    logger.info("=" * 80)

    import numpy as np

    rr_values = np.arange(0.5, 4.0 + 0.05, 0.1)
    rr_values = [round(rr, 2) for rr in rr_values]

    results = []
    for i, rr in enumerate(rr_values, 1):
        logger.info(f"\nTest {i}/{len(rr_values)}: RR = {rr:.2f}")
        strategy = FirstBarBreakoutStrategy(risk_reward_ratio=rr)
        trades = strategy.backtest(df)
        stats = strategy.generate_summary_stats(trades)
        if stats:
            stats["risk_reward_ratio"] = rr
            results.append(stats)
            logger.info(
                f"  Total P&L: ${stats['total_pnl']:.2f}, Win Rate: {stats['win_rate']:.2%}"
            )

    results_df = pl.DataFrame(results).sort("total_pnl", descending=True)
    best = results_df.row(0, named=True)

    return {
        "name": "Risk-Reward Ratio",
        "best_param": f"{best['risk_reward_ratio']:.2f}R",
        "total_pnl": best["total_pnl"],
        "win_rate": best["win_rate"],
        "profit_factor": best["profit_factor"],
        "results_df": results_df,
    }


def run_dra_grid_search(df: pl.DataFrame) -> dict:
    """Run grid search for DRA variant."""
    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH 3/4: DAILY RANGE AVERAGE (DRA)")
    logger.info("=" * 80)

    import numpy as np

    k_values = np.arange(0.1, 2.0 + 0.05, 0.1)
    k_values = [round(k, 2) for k in k_values]

    results = []
    for i, k in enumerate(k_values, 1):
        logger.info(f"\nTest {i}/{len(k_values)}: K = {k:.2f}")
        strategy = FirstBarBreakoutDRAStrategy(k_coefficient=k, dra_window=20)
        trades = strategy.backtest(df)
        stats = strategy.generate_summary_stats(trades)
        if stats:
            stats["k_coefficient"] = k
            results.append(stats)
            logger.info(
                f"  Total P&L: ${stats['total_pnl']:.2f}, Win Rate: {stats['win_rate']:.2%}"
            )

    results_df = pl.DataFrame(results).sort("total_pnl", descending=True)
    best = results_df.row(0, named=True)

    return {
        "name": "DRA",
        "best_param": f"K={best['k_coefficient']:.2f}",
        "total_pnl": best["total_pnl"],
        "win_rate": best["win_rate"],
        "profit_factor": best["profit_factor"],
        "results_df": results_df,
    }


def run_mixed_fixed_grid_search(df: pl.DataFrame) -> dict:
    """Run grid search for Mixed Fixed TP variant."""
    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH 4/4: MIXED (FIRST BAR SL + FIXED TP)")
    logger.info("=" * 80)

    import numpy as np

    tp_amounts = np.arange(0.5, 3.0 + 0.05, 0.1)
    tp_amounts = [round(amt, 2) for amt in tp_amounts]

    results = []
    for i, tp in enumerate(tp_amounts, 1):
        logger.info(f"\nTest {i}/{len(tp_amounts)}: Fixed TP = ${tp:.2f}")
        strategy = FirstBarBreakoutMixedFixedStrategy(fixed_tp_amount=tp)
        trades = strategy.backtest(df)
        stats = strategy.generate_summary_stats(trades)
        if stats:
            stats["fixed_tp_amount"] = tp
            results.append(stats)
            logger.info(
                f"  Total P&L: ${stats['total_pnl']:.2f}, Win Rate: {stats['win_rate']:.2%}"
            )

    results_df = pl.DataFrame(results).sort("total_pnl", descending=True)
    best = results_df.row(0, named=True)

    return {
        "name": "Mixed Fixed TP",
        "best_param": f"TP=${best['fixed_tp_amount']:.2f}",
        "total_pnl": best["total_pnl"],
        "win_rate": best["win_rate"],
        "profit_factor": best["profit_factor"],
        "results_df": results_df,
    }


def export_results(all_results: list, output_dir: str = "results"):
    """Export all results to files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export individual CSVs
    for result in all_results:
        name_slug = result["name"].lower().replace(" ", "_").replace("-", "_")
        csv_path = f"{output_dir}/SPY_{name_slug}_5min_results.csv"
        result["results_df"].write_csv(csv_path)
        logger.info(f"Exported {result['name']} results to {csv_path}")

    # Create comparison summary
    summary_path = f"{output_dir}/SPY_all_strategies_5min_comparison.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FIRST BAR BREAKOUT STRATEGIES - 5-MINUTE BARS COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"{'Strategy':<25} {'Best Param':<15} {'Total P&L':<12} {'Win Rate':<12} {'Profit Factor':<15}\n"
        )
        f.write("-" * 80 + "\n")

        for result in sorted(all_results, key=lambda x: x["total_pnl"], reverse=True):
            f.write(
                f"{result['name']:<25} {result['best_param']:<15} "
                f"${result['total_pnl']:<11.2f} {result['win_rate']:<11.2%} "
                f"{result['profit_factor']:<15.2f}\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write(
            "WINNER: " + max(all_results, key=lambda x: x["total_pnl"])["name"] + "\n"
        )
        f.write("=" * 80 + "\n")

    logger.info(f"Exported comparison summary to {summary_path}")


def main():
    """Main execution function."""
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="{message}",
        level="INFO",
    )

    logger.info("=" * 80)
    logger.info("FIRST BAR BREAKOUT STRATEGIES - 5-MINUTE BARS")
    logger.info("=" * 80)

    # Load 1-minute data
    logger.info("\nLoading 1-minute SPY data...")
    data_path = "data/SPY_1_min.parquet"

    try:
        df_1min = pl.read_parquet(data_path)
        logger.info(f"Loaded {len(df_1min):,} rows of 1-minute data")
        logger.info(
            f"Date range: {df_1min['DateTime'].min()} to {df_1min['DateTime'].max()}"
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Resample to 5-minute bars
    df_5min = resample_to_5min(df_1min)

    # Run all grid searches
    all_results = []

    # 1. Fixed Dollar
    result = run_fixed_grid_search(df_5min)
    all_results.append(result)

    # 2. Risk-Reward Ratio
    result = run_rr_grid_search(df_5min)
    all_results.append(result)

    # 3. DRA
    result = run_dra_grid_search(df_5min)
    all_results.append(result)

    # 4. Mixed Fixed TP
    result = run_mixed_fixed_grid_search(df_5min)
    all_results.append(result)

    # Export all results
    export_results(all_results)

    # Print final comparison
    logger.info("\n" + "=" * 80)
    logger.info("FINAL COMPARISON - 5-MINUTE BARS")
    logger.info("=" * 80)
    logger.info(
        f"\n{'Strategy':<25} {'Best Param':<15} {'Total P&L':<12} {'Win Rate':<12} {'Profit Factor':<15}"
    )
    logger.info("-" * 80)

    for result in sorted(all_results, key=lambda x: x["total_pnl"], reverse=True):
        logger.info(
            f"{result['name']:<25} {result['best_param']:<15} "
            f"${result['total_pnl']:<11.2f} {result['win_rate']:<11.2%} "
            f"{result['profit_factor']:<15.2f}"
        )

    winner = max(all_results, key=lambda x: x["total_pnl"])
    logger.info("\n" + "=" * 80)
    logger.info(
        f"🏆 WINNER: {winner['name']} with ${winner['total_pnl']:.2f} total P&L"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


# Made with Bob
