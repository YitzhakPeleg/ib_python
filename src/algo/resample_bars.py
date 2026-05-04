"""
Utility functions for resampling OHLC data to different timeframes.
"""

import polars as pl
from loguru import logger


def resample_to_5min(df: pl.DataFrame) -> pl.DataFrame:
    """
    Resample 1-minute OHLC data to 5-minute bars.

    Args:
        df: DataFrame with 1-minute OHLC data including DateTime, Open, High, Low, Close, Volume

    Returns:
        DataFrame with 5-minute OHLC data
    """
    logger.info("Resampling 1-minute data to 5-minute bars...")

    # Ensure DateTime is datetime type
    if df["DateTime"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("DateTime").str.to_datetime())

    # Group by 5-minute intervals and aggregate
    df_5min = (
        df.group_by_dynamic(
            "DateTime",
            every="5m",
            period="5m",
            closed="left",
            label="left",
        )
        .agg(
            [
                pl.col("Open").first().alias("Open"),
                pl.col("High").max().alias("High"),
                pl.col("Low").min().alias("Low"),
                pl.col("Close").last().alias("Close"),
                pl.col("Volume").sum().alias("Volume"),
            ]
        )
        .sort("DateTime")
    )

    # Add date column
    df_5min = df_5min.with_columns(pl.col("DateTime").dt.date().alias("date"))

    logger.info(
        f"Resampled to {len(df_5min):,} 5-minute bars (from {len(df):,} 1-minute bars)"
    )

    return df_5min


def resample_to_timeframe(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Resample OHLC data to specified timeframe.

    Args:
        df: DataFrame with OHLC data including DateTime, Open, High, Low, Close, Volume
        timeframe: Timeframe string (e.g., "5m", "15m", "1h")

    Returns:
        DataFrame with resampled OHLC data
    """
    logger.info(f"Resampling data to {timeframe} bars...")

    # Ensure DateTime is datetime type
    if df["DateTime"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("DateTime").str.to_datetime())

    # Group by timeframe intervals and aggregate
    df_resampled = (
        df.group_by_dynamic(
            "DateTime",
            every=timeframe,
            period=timeframe,
            closed="left",
            label="left",
        )
        .agg(
            [
                pl.col("Open").first().alias("Open"),
                pl.col("High").max().alias("High"),
                pl.col("Low").min().alias("Low"),
                pl.col("Close").last().alias("Close"),
                pl.col("Volume").sum().alias("Volume"),
            ]
        )
        .sort("DateTime")
    )

    # Add date column
    df_resampled = df_resampled.with_columns(pl.col("DateTime").dt.date().alias("date"))

    logger.info(
        f"Resampled to {len(df_resampled):,} {timeframe} bars (from {len(df):,} original bars)"
    )

    return df_resampled


# Made with Bob
