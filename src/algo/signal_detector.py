"""Core signal detection logic for morning trading window (09:00-11:00)."""

from typing import Optional

import polars as pl
from loguru import logger

from models import SignalType, TradeSetup


def filter_morning_window(
    df: pl.DataFrame,
    start_hour: int = 9,
    end_hour: int = 11,
    timezone: str = "US/Eastern",
) -> pl.DataFrame:
    """
    Filter dataframe to only include bars within the morning trading window.

    Args:
        df: DataFrame with DateTime column
        start_hour: Start hour (inclusive), default 9 for 09:00
        end_hour: End hour (exclusive), default 11 for before 11:00
        timezone: Timezone for the data, default US/Eastern

    Returns:
        Filtered DataFrame with only morning window data
    """
    # Ensure DateTime is in the correct timezone
    if df["DateTime"].dtype == pl.Datetime:
        df = df.with_columns(
            pl.col("DateTime").dt.convert_time_zone(timezone).alias("DateTime")
        )

    # Filter for the morning window
    morning_df = df.filter(
        (pl.col("DateTime").dt.hour() >= start_hour)
        & (pl.col("DateTime").dt.hour() < end_hour)
    )

    logger.info(
        f"Filtered to morning window ({start_hour}:00-{end_hour}:00): "
        f"{len(morning_df)} bars from {len(df)} total bars"
    )

    return morning_df


def get_post_window_data(
    df: pl.DataFrame, end_hour: int = 11, timezone: str = "US/Eastern"
) -> pl.DataFrame:
    """
    Get data after the morning window for labeling purposes.

    Args:
        df: DataFrame with DateTime column
        end_hour: Hour when morning window ends, default 11
        timezone: Timezone for the data

    Returns:
        DataFrame with only post-window data
    """
    # Ensure DateTime is in the correct timezone
    if df["DateTime"].dtype == pl.Datetime:
        df = df.with_columns(
            pl.col("DateTime").dt.convert_time_zone(timezone).alias("DateTime")
        )

    post_window_df = df.filter(pl.col("DateTime").dt.hour() >= end_hour)

    logger.info(
        f"Post-window data (after {end_hour}:00): "
        f"{len(post_window_df)} bars from {len(df)} total bars"
    )

    return post_window_df


def calculate_entry_stop_tp(
    last_bar: pl.DataFrame,
    signal: SignalType,
    risk_reward_ratio: float = 2.0,
) -> tuple[float, float, float]:
    """
    Calculate entry price, stop-loss, and take-profit levels.

    For BUY signals:
        - Entry: High of last bar (breakout above)
        - Stop-loss: Low of last bar
        - Take-profit: Entry + (risk_reward_ratio * risk)

    For SELL signals:
        - Entry: Low of last bar (breakdown below)
        - Stop-loss: High of last bar
        - Take-profit: Entry - (risk_reward_ratio * risk)

    Args:
        last_bar: DataFrame with single row containing OHLC data
        signal: SignalType (BUY or SELL)
        risk_reward_ratio: Risk-reward ratio for take-profit, default 2.0

    Returns:
        Tuple of (entry_price, stop_loss, take_profit)
    """
    if len(last_bar) != 1:
        raise ValueError(f"Expected single bar, got {len(last_bar)} bars")

    high = last_bar["High"][0]
    low = last_bar["Low"][0]

    if signal == SignalType.BUY:
        entry = high
        stop_loss = low
        risk = entry - stop_loss
        take_profit = entry + (risk_reward_ratio * risk)

    elif signal == SignalType.SELL:
        entry = low
        stop_loss = high
        risk = stop_loss - entry
        take_profit = entry - (risk_reward_ratio * risk)

    else:  # HOLD
        # For HOLD signals, use mid-price with no meaningful SL/TP
        entry = (high + low) / 2
        stop_loss = entry
        take_profit = entry

    return entry, stop_loss, take_profit


def create_trade_setup(
    date: int,
    signal: SignalType,
    last_bar: pl.DataFrame,
    confidence: float,
    risk_reward_ratio: float = 2.0,
) -> Optional[TradeSetup]:
    """
    Create a complete trade setup from signal and bar data.

    Args:
        date: Trading date in YYYYMMDD format
        signal: Predicted signal type
        last_bar: Last bar of the morning window
        confidence: Model confidence score
        risk_reward_ratio: Risk-reward ratio

    Returns:
        TradeSetup object or None if signal is HOLD
    """
    if signal == SignalType.HOLD:
        logger.debug(f"Date {date}: HOLD signal, no trade setup created")
        return None

    entry, stop_loss, take_profit = calculate_entry_stop_tp(
        last_bar, signal, risk_reward_ratio
    )

    setup = TradeSetup(
        date=date,
        signal=signal,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        risk_reward_ratio=risk_reward_ratio,
    )

    logger.info(f"Created trade setup: {setup}")
    return setup


def add_row_number_per_day(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add row number within each trading day.

    Args:
        df: DataFrame with 'date' column

    Returns:
        DataFrame with 'row_nr_day' column added
    """
    return df.with_columns(
        pl.int_range(0, pl.len()).over("date", order_by="DateTime").alias("row_nr_day")
    )


# Made with Bob
