"""BB Consecutive-Bar Reversal signal detection."""

import polars as pl
from loguru import logger

from src.algo.bollinger_bands import calculate_bollinger_bands


def detect_bb_reversal_signals(
    df: pl.DataFrame,
    window: int = 20,
    stds: float = 2.0,
) -> pl.DataFrame:
    """
    Detect Bollinger Band consecutive-bar reversal signals on 1-minute OHLC data.

    Signal logic (requires two consecutive bars both outside the same band):
    - Long:  red bar (Low < lower BB) followed by green bar (Low < lower BB)
             → entry_price = green bar's High
    - Short: green bar (High > upper BB) followed by red bar (High > upper BB)
             → entry_price = red bar's Low

    Previous-bar lookback is scoped per day (`.over("date")`) to prevent
    cross-day bleed at midnight / session open.

    Args:
        df: DataFrame with columns DateTime, Open, High, Low, Close, Volume, date.
            The `date` column must exist (int64 YYYYMMDD or pl.Date — either works
            as a grouping key for `.over()`).
        window: Bollinger Band moving-average period (default 20).
        stds: Number of standard deviations for the bands (default 2.0).

    Returns:
        Input DataFrame with appended columns:
            is_green, is_red,
            bb_mid, bb_upper, bb_lower,
            signal_direction ("long" | "short" | null),
            entry_price (float | null).
    """
    df = calculate_bollinger_bands(df, window=window, stds=stds)

    df = df.with_columns(
        [
            (pl.col("Close") > pl.col("Open")).alias("is_green"),
            (pl.col("Close") < pl.col("Open")).alias("is_red"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("is_green").shift(1).over("date").alias("prev_is_green"),
            pl.col("is_red").shift(1).over("date").alias("prev_is_red"),
            pl.col("High").shift(1).over("date").alias("prev_high"),
            pl.col("Low").shift(1).over("date").alias("prev_low"),
            pl.col("bb_upper").shift(1).over("date").alias("prev_bb_upper"),
            pl.col("bb_lower").shift(1).over("date").alias("prev_bb_lower"),
        ]
    )

    long_cond = (
        pl.col("prev_is_red")
        & (pl.col("prev_low") < pl.col("prev_bb_lower"))
        & pl.col("is_green")
        & (pl.col("Low") < pl.col("bb_lower"))
    )

    short_cond = (
        pl.col("prev_is_green")
        & (pl.col("prev_high") > pl.col("prev_bb_upper"))
        & pl.col("is_red")
        & (pl.col("High") > pl.col("bb_upper"))
    )

    df = df.with_columns(
        [
            pl.when(long_cond)
            .then(pl.lit("long"))
            .when(short_cond)
            .then(pl.lit("short"))
            .otherwise(None)
            .alias("signal_direction"),
            pl.when(long_cond)
            .then(pl.col("High"))
            .when(short_cond)
            .then(pl.col("Low"))
            .otherwise(None)
            .alias("entry_price"),
        ]
    )

    n_long = df.filter(pl.col("signal_direction") == "long").height
    n_short = df.filter(pl.col("signal_direction") == "short").height
    logger.info(f"Signals detected — long: {n_long}, short: {n_short}, total: {n_long + n_short}")

    return df
