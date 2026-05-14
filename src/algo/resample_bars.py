"""Utility for resampling OHLC data to different timeframes."""

import polars as pl
from loguru import logger


def resample_to_timeframe(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Resample OHLC data to the specified timeframe.

    Args:
        df: DataFrame with columns DateTime, Open, High, Low, Close, Volume.
        timeframe: Polars interval string — e.g. "5m", "15m", "1h", "1d".

    Returns:
        DataFrame resampled to the requested timeframe with a `date` column added.
    """
    logger.info(f"Resampling data to {timeframe} bars...")

    if df["DateTime"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("DateTime").str.to_datetime())

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

    df_resampled = df_resampled.with_columns(pl.col("DateTime").dt.date().alias("date"))

    logger.info(
        f"Resampled to {len(df_resampled):,} {timeframe} bars (from {len(df):,} original bars)"
    )

    return df_resampled
