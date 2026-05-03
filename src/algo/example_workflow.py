"""
Complete workflow example for ML-based trading signal system.

This script demonstrates:
1. Loading and preparing data
2. Training a signal detection model
3. Generating trading signals
4. Backtesting signals
5. Visualizing results
"""

from pathlib import Path

import polars as pl
from loguru import logger

from ..data_fetching.date_converter import add_date_int_column
from .backtester import backtest_trade_setups, generate_performance_report
from .signal_generator import SignalGenerator
from .train_signal_model import (
    load_and_prepare_data,
    save_model,
    train_random_forest_model,
)


def complete_workflow_example(
    data_path: str = "/Users/yitzhakpeleg/Projects/ib_python/AAPL_1_min.parquet",
    model_dir: str = "/Users/yitzhakpeleg/Projects/ib_python/models",
    retrain: bool = False,
):
    """
    Complete workflow from data loading to backtesting.

    Args:
        data_path: Path to 1-minute OHLC data
        model_dir: Directory to save/load model
        retrain: Whether to retrain the model
    """
    logger.info("=" * 80)
    logger.info("ML-BASED TRADING SIGNAL SYSTEM - COMPLETE WORKFLOW")
    logger.info("=" * 80)

    # ========================================================================
    # STEP 1: Load and prepare data
    # ========================================================================
    logger.info("\n[STEP 1] Loading and preparing data...")
    features_df, labels_df = load_and_prepare_data(
        data_path, timezone="US/Eastern", use_timing_labels=False
    )

    # ========================================================================
    # STEP 2: Train model (or load existing)
    # ========================================================================
    model_path = Path(model_dir) / "morning_signal_rf.joblib"

    if retrain or not model_path.exists():
        logger.info("\n[STEP 2] Training Random Forest model...")
        model, metrics = train_random_forest_model(
            features_df,
            labels_df,
            test_size=0.2,
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
        )

        # Save model
        save_model(model, metrics, model_dir, model_name="morning_signal_rf")
    else:
        logger.info(f"\n[STEP 2] Loading existing model from {model_path}")

    # ========================================================================
    # STEP 3: Generate signals
    # ========================================================================
    logger.info("\n[STEP 3] Generating trading signals...")

    # Load full dataset
    df = pl.read_parquet(data_path)
    if "date" not in df.columns:
        df = add_date_int_column(df)

    # Initialize signal generator
    generator = SignalGenerator(model_path)

    # Generate signals with confidence threshold
    signals = generator.generate_signals(
        df,
        timezone="US/Eastern",
        confidence_threshold=0.6,  # Only signals with >60% confidence
        risk_reward_ratio=2.0,  # 2:1 risk-reward ratio
    )

    logger.info(f"\nGenerated {len(signals)} trading signals")
    logger.info("\nSample signals:")
    for setup in signals[:5]:
        logger.info(f"  {setup}")

    # ========================================================================
    # STEP 4: Backtest signals
    # ========================================================================
    logger.info("\n[STEP 4] Backtesting signals...")

    results_df = backtest_trade_setups(df, signals, timezone="US/Eastern")

    # Generate performance report
    performance = generate_performance_report(results_df)

    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE REPORT")
    logger.info("=" * 80)
    logger.info(f"Total Trades: {performance['total_trades']}")
    logger.info(f"Wins: {performance['wins']} | Losses: {performance['losses']}")
    logger.info(f"Win Rate: {performance['win_rate']:.2%}")
    logger.info(f"Average Win: ${performance['avg_win']:.2f}")
    logger.info(f"Average Loss: ${performance['avg_loss']:.2f}")
    logger.info(f"Average PnL per Trade: ${performance['avg_pnl']:.2f}")
    logger.info(f"Total PnL: ${performance['total_pnl']:.2f}")
    logger.info(f"Profit Factor: {performance['profit_factor']:.2f}")
    logger.info(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: ${performance['max_drawdown']:.2f}")
    logger.info(f"Average R-Multiple: {performance['avg_r_multiple']:.2f}R")
    logger.info(f"Average Bars Held: {performance['avg_bars_held']:.1f}")
    logger.info("=" * 80)

    # ========================================================================
    # STEP 5: Save results
    # ========================================================================
    logger.info("\n[STEP 5] Saving results...")

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save backtest results
    results_path = output_dir / "backtest_results.csv"
    results_df.write_csv(results_path)
    logger.info(f"Backtest results saved to {results_path}")

    # Save performance report
    report_path = output_dir / "performance_report.txt"
    with open(report_path, "w") as f:
        f.write("PERFORMANCE REPORT\n")
        f.write("=" * 80 + "\n")
        for key, value in performance.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Performance report saved to {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 80)

    return signals, results_df, performance


def quick_signal_check(
    data_path: str = "/Users/yitzhakpeleg/Projects/ib_python/AAPL_1_min.parquet",
    model_path: str = "/Users/yitzhakpeleg/Projects/ib_python/models/morning_signal_rf.joblib",
    target_date: int = 20260430,
):
    """
    Quick check: Generate signal for a specific date.

    Args:
        data_path: Path to data file
        model_path: Path to trained model
        target_date: Date in YYYYMMDD format
    """
    logger.info(f"Generating signal for date {target_date}...")

    # Load data
    df = pl.read_parquet(data_path)
    if "date" not in df.columns:
        df = add_date_int_column(df)

    # Filter to target date
    date_df = df.filter(pl.col("date") == target_date)

    if len(date_df) == 0:
        logger.error(f"No data found for date {target_date}")
        return None

    # Generate signal
    generator = SignalGenerator(model_path)
    signal = generator.generate_signal_for_date(
        df, target_date, timezone="US/Eastern", risk_reward_ratio=2.0
    )

    if signal:
        logger.info(f"\nSignal for {target_date}:")
        logger.info(f"  {signal}")

        # Show morning price action
        morning_data = date_df.filter(
            (pl.col("DateTime").dt.hour() >= 9) & (pl.col("DateTime").dt.hour() < 11)
        )
        logger.info("\nMorning price action (09:00-11:00):")
        logger.info(f"  Open: ${morning_data['Open'].first():.2f}")
        logger.info(f"  High: ${morning_data['High'].max():.2f}")
        logger.info(f"  Low: ${morning_data['Low'].min():.2f}")
        logger.info(f"  Close: ${morning_data['Close'].last():.2f}")
    else:
        logger.info(f"No signal generated for {target_date}")

    return signal


def analyze_signal_distribution(
    data_path: str = "/Users/yitzhakpeleg/Projects/ib_python/AAPL_1_min.parquet",
    model_path: str = "/Users/yitzhakpeleg/Projects/ib_python/models/morning_signal_rf.joblib",
):
    """
    Analyze the distribution of generated signals.

    Args:
        data_path: Path to data file
        model_path: Path to trained model
    """
    logger.info("Analyzing signal distribution...")

    # Load data
    df = pl.read_parquet(data_path)
    if "date" not in df.columns:
        df = add_date_int_column(df)

    # Generate signals
    generator = SignalGenerator(model_path)
    signals = generator.generate_signals(
        df, timezone="US/Eastern", confidence_threshold=0.5, risk_reward_ratio=2.0
    )

    # Analyze distribution
    buy_signals = [s for s in signals if s.signal.name == "BUY"]
    sell_signals = [s for s in signals if s.signal.name == "SELL"]

    logger.info("\nSignal Distribution:")
    logger.info(f"  Total Signals: {len(signals)}")
    logger.info(
        f"  BUY Signals: {len(buy_signals)} ({len(buy_signals) / len(signals):.1%})"
    )
    logger.info(
        f"  SELL Signals: {len(sell_signals)} ({len(sell_signals) / len(signals):.1%})"
    )

    # Confidence distribution
    confidences = [s.confidence for s in signals]
    logger.info("\nConfidence Statistics:")
    logger.info(f"  Mean: {sum(confidences) / len(confidences):.2%}")
    logger.info(f"  Min: {min(confidences):.2%}")
    logger.info(f"  Max: {max(confidences):.2%}")

    return signals


if __name__ == "__main__":
    # Run complete workflow
    signals, results, performance = complete_workflow_example(retrain=False)

    # Or run quick checks
    # quick_signal_check(target_date=20260430)
    # analyze_signal_distribution()

# Made with Bob
