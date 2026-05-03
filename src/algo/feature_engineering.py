"""Feature engineering for morning trading window (09:00-11:00)."""

import polars as pl
from loguru import logger


def engineer_morning_features(
    morning_df: pl.DataFrame, window: int = 20
) -> pl.DataFrame:
    """
    Engineer features from morning window data for ML model.

    Features include:
    - Normalized OHLC prices (relative to first bar)
    - Price momentum and volatility
    - Bollinger Band metrics
    - Volume patterns
    - Technical indicators

    Args:
        morning_df: DataFrame with morning window data (09:00-11:00)
        window: Window size for rolling calculations

    Returns:
        DataFrame with engineered features per day
    """
    # Ensure we have row numbers per day
    if "row_nr_day" not in morning_df.columns:
        morning_df = morning_df.with_columns(
            pl.int_range(0, pl.len())
            .over("date", order_by="DateTime")
            .alias("row_nr_day")
        )

    # Calculate features within each day
    features_df = morning_df.with_columns(
        [
            # Get first bar's open price for normalization
            pl.col("Open").first().over("date").alias("first_open"),
            # Get last bar's OHLC for entry/SL/TP calculation
            pl.col("High").last().over("date").alias("last_high"),
            pl.col("Low").last().over("date").alias("last_low"),
            pl.col("Close").last().over("date").alias("last_close"),
        ]
    ).with_columns(
        [
            # Normalized prices (relative to first bar's open)
            ((pl.col("Close") / pl.col("first_open")) - 1).alias("norm_close"),
            ((pl.col("High") / pl.col("first_open")) - 1).alias("norm_high"),
            ((pl.col("Low") / pl.col("first_open")) - 1).alias("norm_low"),
            # Intrabar range
            ((pl.col("High") - pl.col("Low")) / pl.col("Open")).alias("bar_range"),
            # Volume relative to first bar
            (pl.col("Volume") / pl.col("Volume").first().over("date")).alias(
                "rel_volume"
            ),
        ]
    )

    # Aggregate features per day
    daily_features = (
        features_df.group_by("date")
        .agg(
            [
                # Price movement features
                pl.col("norm_close").last().alias("total_move"),
                pl.col("norm_high").max().alias("max_high_move"),
                pl.col("norm_low").min().alias("max_low_move"),
                (pl.col("norm_close").last() - pl.col("norm_close").first()).alias(
                    "net_move"
                ),
                # Volatility features
                pl.col("bar_range").mean().alias("avg_bar_range"),
                pl.col("bar_range").max().alias("max_bar_range"),
                pl.col("norm_close").std().alias("price_volatility"),
                # Volume features
                pl.col("rel_volume").mean().alias("avg_rel_volume"),
                pl.col("rel_volume").max().alias("max_rel_volume"),
                pl.col("Volume").sum().alias("total_volume"),
                # Bollinger Band features (if available)
                pl.col("bb_upper").last().alias("bb_upper_last")
                if "bb_upper" in features_df.columns
                else pl.lit(None).alias("bb_upper_last"),
                pl.col("bb_lower").last().alias("bb_lower_last")
                if "bb_lower" in features_df.columns
                else pl.lit(None).alias("bb_lower_last"),
                pl.col("bb_mid").last().alias("bb_mid_last")
                if "bb_mid" in features_df.columns
                else pl.lit(None).alias("bb_mid_last"),
                # Last bar OHLC for entry/SL/TP
                pl.col("last_high").last().alias("last_bar_high"),
                pl.col("last_low").last().alias("last_bar_low"),
                pl.col("last_close").last().alias("last_bar_close"),
                # Count of bars
                pl.len().alias("bar_count"),
            ]
        )
        .sort("date")
    )

    # Add derived Bollinger Band features if available
    if "bb_upper_last" in daily_features.columns:
        daily_features = daily_features.with_columns(
            [
                (
                    (pl.col("bb_upper_last") - pl.col("bb_lower_last"))
                    / pl.col("bb_mid_last")
                ).alias("bb_width_ratio"),
                (
                    (pl.col("last_bar_close") - pl.col("bb_lower_last"))
                    / (pl.col("bb_upper_last") - pl.col("bb_lower_last"))
                ).alias("bb_position"),
            ]
        )

    logger.info(
        f"Engineered {len(daily_features.columns)} features for {len(daily_features)} days"
    )

    return daily_features


def create_sequential_features(
    morning_df: pl.DataFrame, max_bars: int = 120
) -> pl.DataFrame:
    """
    Create sequential features for deep learning models (LSTM/CNN).

    Returns a DataFrame where each row represents a day, and features are
    organized as sequences suitable for time-series models.

    Args:
        morning_df: DataFrame with morning window data
        max_bars: Maximum number of bars to include (default 120 for 2 hours)

    Returns:
        DataFrame with sequential features per day
    """
    # Ensure we have row numbers
    if "row_nr_day" not in morning_df.columns:
        morning_df = morning_df.with_columns(
            pl.int_range(0, pl.len())
            .over("date", order_by="DateTime")
            .alias("row_nr_day")
        )

    # Filter to max_bars per day
    morning_df = morning_df.filter(pl.col("row_nr_day") < max_bars)

    # Normalize prices within each day
    morning_df = morning_df.with_columns(
        [
            pl.col("Open").first().over("date").alias("first_open"),
        ]
    ).with_columns(
        [
            ((pl.col("Open") / pl.col("first_open")) - 1).alias("norm_open"),
            ((pl.col("High") / pl.col("first_open")) - 1).alias("norm_high"),
            ((pl.col("Low") / pl.col("first_open")) - 1).alias("norm_low"),
            ((pl.col("Close") / pl.col("first_open")) - 1).alias("norm_close"),
            (pl.col("Volume") / pl.col("Volume").first().over("date")).alias(
                "norm_volume"
            ),
        ]
    )

    # Pivot to create sequences
    # Each feature becomes columns like: norm_close_0, norm_close_1, ..., norm_close_119
    sequential_features = morning_df.pivot(
        index="date",
        on="row_nr_day",
        values=["norm_open", "norm_high", "norm_low", "norm_close", "norm_volume"],
    )

    logger.info(
        f"Created sequential features: {len(sequential_features)} days, "
        f"{len(sequential_features.columns)} total features"
    )

    return sequential_features


def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add technical indicators to the dataframe.

    Indicators include:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - ATR (Average True Range)

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with technical indicators added
    """
    # RSI calculation (14-period)
    df = (
        df.with_columns(
            [
                # Price changes
                (pl.col("Close") - pl.col("Close").shift(1)).alias("price_change"),
            ]
        )
        .with_columns(
            [
                # Gains and losses
                pl.when(pl.col("price_change") > 0)
                .then(pl.col("price_change"))
                .otherwise(0)
                .alias("gain"),
                pl.when(pl.col("price_change") < 0)
                .then(-pl.col("price_change"))
                .otherwise(0)
                .alias("loss"),
            ]
        )
        .with_columns(
            [
                # Average gains and losses
                pl.col("gain").rolling_mean(window_size=14).alias("avg_gain"),
                pl.col("loss").rolling_mean(window_size=14).alias("avg_loss"),
            ]
        )
        .with_columns(
            [
                # RSI
                (
                    100
                    - (
                        100
                        / (1 + (pl.col("avg_gain") / pl.col("avg_loss").replace(0, 1)))
                    )
                ).alias("rsi"),
            ]
        )
    )

    # MACD calculation
    df = (
        df.with_columns(
            [
                pl.col("Close").ewm_mean(span=12).alias("ema_12"),
                pl.col("Close").ewm_mean(span=26).alias("ema_26"),
            ]
        )
        .with_columns(
            [
                (pl.col("ema_12") - pl.col("ema_26")).alias("macd"),
            ]
        )
        .with_columns(
            [
                pl.col("macd").ewm_mean(span=9).alias("macd_signal"),
            ]
        )
        .with_columns(
            [
                (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram"),
            ]
        )
    )

    # Clean up intermediate columns
    df = df.drop(["price_change", "gain", "loss", "avg_gain", "avg_loss"])

    logger.info("Added technical indicators: RSI, MACD")

    return df


# Made with Bob
