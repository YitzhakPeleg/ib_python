"""Signal generator using trained ML model to produce trading signals."""

from pathlib import Path
from typing import Optional

import joblib
import polars as pl
from loguru import logger

from algo.bollinger_bands import calculate_bollinger_bands
from algo.feature_engineering import add_technical_indicators, engineer_morning_features
from algo.signal_detector import (
    create_trade_setup,
    filter_morning_window,
)
from data_fetching.date_converter import add_date_int_column
from models import SignalType, TradeSetup


class SignalGenerator:
    """Generate trading signals using a trained ML model."""

    def __init__(self, model_path: str | Path):
        """
        Initialize signal generator with a trained model.

        Args:
            model_path: Path to saved model file (.joblib)
        """
        self.model = joblib.load(model_path)
        self.model_path = model_path
        logger.info(f"Loaded model from {model_path}")

    def generate_signals(
        self,
        df: pl.DataFrame,
        timezone: str = "US/Eastern",
        confidence_threshold: float = 0.5,
        risk_reward_ratio: float = 2.0,
    ) -> list[TradeSetup]:
        """
        Generate trading signals from market data.

        Args:
            df: DataFrame with 1-minute OHLC data
            timezone: Timezone for the data
            confidence_threshold: Minimum confidence to generate signal
            risk_reward_ratio: Risk-reward ratio for TP calculation

        Returns:
            List of TradeSetup objects
        """
        # Prepare data
        if "date" not in df.columns:
            df = add_date_int_column(df)

        # Ensure timezone
        if df["DateTime"].dtype == pl.Datetime:
            df = df.with_columns(
                pl.col("DateTime").dt.convert_time_zone(timezone).alias("DateTime")
            )

        # Calculate indicators
        df = calculate_bollinger_bands(df, window=20, stds=2.0)
        df = add_technical_indicators(df)

        # Filter morning window
        morning_df = filter_morning_window(
            df, start_hour=9, end_hour=11, timezone=timezone
        )

        # Engineer features
        features_df = engineer_morning_features(morning_df, window=20)

        # Get feature columns (exclude date and label-related columns)
        feature_cols = [
            col
            for col in features_df.columns
            if col not in ["date", "label", "entry_price", "max_high", "min_low"]
        ]

        # Prepare features for prediction
        X = features_df.select(feature_cols).to_numpy()

        # Predict signals
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        # Get max probability for each prediction (confidence)
        confidences = probabilities.max(axis=1)

        # Create trade setups
        trade_setups = []
        dates = features_df["date"].to_list()

        for i, (date, pred, conf) in enumerate(zip(dates, predictions, confidences)):
            # Skip if confidence is too low
            if conf < confidence_threshold:
                logger.debug(
                    f"Date {date}: Skipping signal (confidence {conf:.2%} < {confidence_threshold:.2%})"
                )
                continue

            signal = SignalType(pred)

            # Skip HOLD signals
            if signal == SignalType.HOLD:
                continue

            # Get last bar of morning window for this date
            last_bar = morning_df.filter(pl.col("date") == date).tail(1)

            if len(last_bar) == 0:
                logger.warning(f"Date {date}: No morning data found")
                continue

            # Create trade setup
            setup = create_trade_setup(
                date=date,
                signal=signal,
                last_bar=last_bar,
                confidence=conf,
                risk_reward_ratio=risk_reward_ratio,
            )

            if setup:
                trade_setups.append(setup)

        logger.info(
            f"Generated {len(trade_setups)} trade signals from {len(dates)} days"
        )

        return trade_setups

    def generate_signal_for_date(
        self,
        df: pl.DataFrame,
        target_date: int,
        timezone: str = "US/Eastern",
        risk_reward_ratio: float = 2.0,
    ) -> Optional[TradeSetup]:
        """
        Generate signal for a specific date.

        Args:
            df: DataFrame with market data
            target_date: Date in YYYYMMDD format
            timezone: Timezone for the data
            risk_reward_ratio: Risk-reward ratio

        Returns:
            TradeSetup or None
        """
        # Filter to target date
        date_df = df.filter(pl.col("date") == target_date)

        if len(date_df) == 0:
            logger.warning(f"No data found for date {target_date}")
            return None

        # Generate signals for this date
        signals = self.generate_signals(
            date_df, timezone=timezone, risk_reward_ratio=risk_reward_ratio
        )

        if len(signals) == 0:
            logger.info(f"No signal generated for date {target_date}")
            return None

        return signals[0]

    def backtest_signals(
        self,
        df: pl.DataFrame,
        signals: list[TradeSetup],
        timezone: str = "US/Eastern",
    ) -> pl.DataFrame:
        """
        Backtest generated signals against actual price data.

        Args:
            df: Full DataFrame with price data
            signals: List of trade setups to backtest
            timezone: Timezone for the data

        Returns:
            DataFrame with backtest results
        """
        from .backtester import backtest_trade_setups

        return backtest_trade_setups(df, signals, timezone=timezone)


def main_example():
    """Example usage of SignalGenerator."""
    # Load data
    data_path = "/Users/yitzhakpeleg/Projects/ib_python/AAPL_1_min.parquet"
    df = pl.read_parquet(data_path)
    df = add_date_int_column(df)

    # Initialize signal generator
    model_path = (
        "/Users/yitzhakpeleg/Projects/ib_python/models/morning_signal_rf.joblib"
    )
    generator = SignalGenerator(model_path)

    # Generate signals
    signals = generator.generate_signals(
        df, timezone="US/Eastern", confidence_threshold=0.6, risk_reward_ratio=2.0
    )

    # Display signals
    logger.info(f"\nGenerated {len(signals)} signals:")
    for setup in signals[:10]:  # Show first 10
        logger.info(setup)

    # Backtest signals
    results = generator.backtest_signals(df, signals)
    logger.info(f"\nBacktest Results:\n{results}")


if __name__ == "__main__":
    main_example()

# Made with Bob
