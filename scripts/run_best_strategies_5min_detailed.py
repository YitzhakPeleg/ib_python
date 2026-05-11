"""
Run best strategies on 5-minute bars and export detailed trade journals.

This script runs the best parameter for each strategy variant and exports
detailed trade-by-trade results.
"""

from pathlib import Path

import polars as pl
from loguru import logger

from src.algo.first_bar_breakout import (
    FirstBarBreakoutStrategy,
)
from src.algo.first_bar_breakout import (
    export_results_to_csv as export_rr,
)
from src.algo.first_bar_breakout_dra import (
    FirstBarBreakoutDRAStrategy,
)
from src.algo.first_bar_breakout_dra import (
    export_results_to_csv as export_dra,
)
from src.algo.first_bar_breakout_fixed import (
    FirstBarBreakoutFixedStrategy,
)
from src.algo.first_bar_breakout_fixed import (
    export_results_to_csv as export_fixed,
)
from src.algo.first_bar_breakout_mixed_fixed import (
    FirstBarBreakoutMixedFixedStrategy,
)
from src.algo.first_bar_breakout_mixed_fixed import (
    export_results_to_csv as export_mixed,
)
from src.algo.resample_bars import resample_to_5min


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
    logger.info("GENERATING DETAILED TRADE JOURNALS - 5-MINUTE BARS")
    logger.info("=" * 80)

    # Load 1-minute data
    logger.info("\nLoading 1-minute SPY data...")
    data_path = "data/SPY_1_min.parquet"

    try:
        df_1min = pl.read_parquet(data_path)
        logger.info(f"Loaded {len(df_1min):,} rows of 1-minute data")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Resample to 5-minute bars
    df_5min = resample_to_5min(df_1min)

    # Create results directory
    Path("results/trade_journals_5min").mkdir(parents=True, exist_ok=True)

    # 1. DRA Strategy (K=1.70) - WINNER
    logger.info("\n" + "=" * 80)
    logger.info("1. DRA STRATEGY (K=1.70) - BEST OVERALL")
    logger.info("=" * 80)
    strategy_dra = FirstBarBreakoutDRAStrategy(k_coefficient=1.70, dra_window=20)
    trades_dra = strategy_dra.backtest(df_5min)
    stats_dra = strategy_dra.generate_summary_stats(trades_dra)
    strategy_dra.print_summary(stats_dra)
    export_dra(trades_dra, "results/trade_journals_5min/DRA_K1.70_trades.csv")

    # 2. Fixed Dollar Strategy ($3.00)
    logger.info("\n" + "=" * 80)
    logger.info("2. FIXED DOLLAR STRATEGY ($3.00)")
    logger.info("=" * 80)
    strategy_fixed = FirstBarBreakoutFixedStrategy(fixed_amount=3.00)
    trades_fixed = strategy_fixed.backtest(df_5min)
    stats_fixed = strategy_fixed.generate_summary_stats(trades_fixed)
    strategy_fixed.print_summary(stats_fixed)
    export_fixed(trades_fixed, "results/trade_journals_5min/Fixed_$3.00_trades.csv")

    # 3. Mixed Fixed TP Strategy ($2.10)
    logger.info("\n" + "=" * 80)
    logger.info("3. MIXED FIXED TP STRATEGY ($2.10)")
    logger.info("=" * 80)
    strategy_mixed = FirstBarBreakoutMixedFixedStrategy(fixed_tp_amount=2.10)
    trades_mixed = strategy_mixed.backtest(df_5min)
    stats_mixed = strategy_mixed.generate_summary_stats(trades_mixed)
    strategy_mixed.print_summary(stats_mixed)
    export_mixed(trades_mixed, "results/trade_journals_5min/Mixed_TP$2.10_trades.csv")

    # 4. Risk-Reward Strategy (3.80R)
    logger.info("\n" + "=" * 80)
    logger.info("4. RISK-REWARD STRATEGY (3.80R)")
    logger.info("=" * 80)
    strategy_rr = FirstBarBreakoutStrategy(risk_reward_ratio=3.80)
    trades_rr = strategy_rr.backtest(df_5min)
    stats_rr = strategy_rr.generate_summary_stats(trades_rr)
    strategy_rr.print_summary(stats_rr)
    export_rr(trades_rr, "results/trade_journals_5min/RR_3.80_trades.csv")

    # Create summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY COMPARISON")
    logger.info("=" * 80)

    summary_data = {
        "Strategy": ["DRA K=1.70", "Fixed $3.00", "Mixed TP=$2.10", "RR 3.80"],
        "Total_Trades": [
            stats_dra["total_trades"],
            stats_fixed["total_trades"],
            stats_mixed["total_trades"],
            stats_rr["total_trades"],
        ],
        "Wins": [
            stats_dra["wins"],
            stats_fixed["wins"],
            stats_mixed["wins"],
            stats_rr["wins"],
        ],
        "Losses": [
            stats_dra["losses"],
            stats_fixed["losses"],
            stats_mixed["losses"],
            stats_rr["losses"],
        ],
        "Win_Rate": [
            stats_dra["win_rate"],
            stats_fixed["win_rate"],
            stats_mixed["win_rate"],
            stats_rr["win_rate"],
        ],
        "Total_PnL": [
            stats_dra["total_pnl"],
            stats_fixed["total_pnl"],
            stats_mixed["total_pnl"],
            stats_rr["total_pnl"],
        ],
        "Avg_PnL": [
            stats_dra["avg_pnl"],
            stats_fixed["avg_pnl"],
            stats_mixed["avg_pnl"],
            stats_rr["avg_pnl"],
        ],
        "Profit_Factor": [
            stats_dra["profit_factor"],
            stats_fixed["profit_factor"],
            stats_mixed["profit_factor"],
            stats_rr["profit_factor"],
        ],
    }

    summary_df = pl.DataFrame(summary_data)
    summary_df.write_csv("results/trade_journals_5min/summary_comparison.csv")

    logger.info("\n" + summary_df.__str__())

    logger.info("\n" + "=" * 80)
    logger.info("TRADE JOURNALS EXPORTED")
    logger.info("=" * 80)
    logger.info("\nTrade journals saved to:")
    logger.info("  - results/trade_journals_5min/DRA_K1.70_trades.csv")
    logger.info("  - results/trade_journals_5min/Fixed_$3.00_trades.csv")
    logger.info("  - results/trade_journals_5min/Mixed_TP$2.10_trades.csv")
    logger.info("  - results/trade_journals_5min/RR_3.80_trades.csv")
    logger.info("  - results/trade_journals_5min/summary_comparison.csv")
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()


# Made with Bob
