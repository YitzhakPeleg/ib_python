"""Labeling system for generating training labels from post-window price movements."""

import polars as pl
from loguru import logger

from .models import SignalType


def create_labels(
    df: pl.DataFrame,
    morning_end_hour: int = 11,
    tp_threshold: float = 0.005,  # 0.5% move
    sl_threshold: float = 0.005,  # 0.5% move
    use_atr: bool = True,
    atr_multiplier: float = 0.5,
) -> pl.DataFrame:
    """
    Create labels for each trading day based on post-morning-window price movement.

    Labels are determined by which threshold is hit first:
    - BUY (1): Price moves up by tp_threshold before moving down by sl_threshold
    - SELL (-1): Price moves down by sl_threshold before moving up by tp_threshold
    - HOLD (0): Neither threshold is hit, or choppy/unclear movement

    Args:
        df: Full day DataFrame with OHLC data and 'date' column
        morning_end_hour: Hour when morning window ends (default 11)
        tp_threshold: Take-profit threshold as percentage (default 0.005 = 0.5%)
        sl_threshold: Stop-loss threshold as percentage (default 0.005 = 0.5%)
        use_atr: Use ATR-based thresholds instead of fixed percentages
        atr_multiplier: Multiplier for ATR-based thresholds (default 0.5)

    Returns:
        DataFrame with columns: date, label, entry_price, max_high, min_low
    """
    # Ensure DateTime is properly typed
    if df["DateTime"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("DateTime").str.to_datetime())

    # Get the last bar of morning window (entry price reference)
    morning_last = (
        df.filter(pl.col("DateTime").dt.hour() < morning_end_hour)
        .group_by("date")
        .agg(
            [
                pl.col("Close").last().alias("entry_price"),
                pl.col("DateTime").last().alias("entry_time"),
            ]
        )
    )

    # Get post-window data
    post_window = df.filter(pl.col("DateTime").dt.hour() >= morning_end_hour)

    # Calculate ATR if needed
    if use_atr:
        daily_atr = calculate_daily_atr(df)
        morning_last = morning_last.join(daily_atr, on="date")
        # Override thresholds with ATR-based values
        # This will be applied per-row in the labeling logic

    # Join post-window data with entry prices
    labeled_data = (
        post_window.join(morning_last, on="date")
        .with_columns(
            [
                # Calculate percentage moves from entry
                ((pl.col("High") / pl.col("entry_price")) - 1).alias("high_pct_move"),
                ((pl.col("Low") / pl.col("entry_price")) - 1).alias("low_pct_move"),
            ]
        )
        .group_by("date")
        .agg(
            [
                pl.col("entry_price").first(),
                pl.col("High").max().alias("max_high"),
                pl.col("Low").min().alias("min_low"),
                pl.col("high_pct_move").max().alias("max_up_move"),
                pl.col("low_pct_move").min().alias("max_down_move"),
                pl.col("atr").first().alias("atr") if use_atr else pl.lit(None),
            ]
        )
    )

    # Determine labels based on which threshold was hit first
    if use_atr:
        # Use ATR-based thresholds
        labeled_data = labeled_data.with_columns(
            [
                (pl.col("atr") * atr_multiplier / pl.col("entry_price")).alias(
                    "tp_threshold_pct"
                ),
                (pl.col("atr") * atr_multiplier / pl.col("entry_price")).alias(
                    "sl_threshold_pct"
                ),
            ]
        )
    else:
        # Use fixed percentage thresholds
        labeled_data = labeled_data.with_columns(
            [
                pl.lit(tp_threshold).alias("tp_threshold_pct"),
                pl.lit(sl_threshold).alias("sl_threshold_pct"),
            ]
        )

    # Apply labeling logic
    labeled_data = labeled_data.with_columns(
        [
            pl.when(pl.col("max_up_move") > pl.col("tp_threshold_pct"))
            .then(pl.lit(SignalType.BUY.value))
            .when(pl.col("max_down_move") < -pl.col("sl_threshold_pct"))
            .then(pl.lit(SignalType.SELL.value))
            .otherwise(pl.lit(SignalType.HOLD.value))
            .alias("label")
        ]
    )

    # Log label distribution
    label_counts = labeled_data.group_by("label").agg(pl.len().alias("count"))
    logger.info(f"Label distribution:\n{label_counts}")

    return labeled_data.select(["date", "label", "entry_price", "max_high", "min_low"])


def create_labels_with_timing(
    df: pl.DataFrame,
    morning_end_hour: int = 11,
    tp_threshold: float = 0.005,
    sl_threshold: float = 0.005,
) -> pl.DataFrame:
    """
    Create labels considering which threshold is hit FIRST (timing matters).

    This is more sophisticated than simple max/min comparison, as it tracks
    the actual sequence of price movements.

    Args:
        df: Full day DataFrame with OHLC data
        morning_end_hour: Hour when morning window ends
        tp_threshold: Take-profit threshold as percentage
        sl_threshold: Stop-loss threshold as percentage

    Returns:
        DataFrame with labels and timing information
    """
    # Get entry prices
    morning_last = (
        df.filter(pl.col("DateTime").dt.hour() < morning_end_hour)
        .group_by("date")
        .agg([pl.col("Close").last().alias("entry_price")])
    )

    # Get post-window data with cumulative max/min tracking
    post_window = (
        df.filter(pl.col("DateTime").dt.hour() >= morning_end_hour)
        .join(morning_last, on="date")
        .with_columns(
            [
                ((pl.col("High") / pl.col("entry_price")) - 1).alias("high_pct"),
                ((pl.col("Low") / pl.col("entry_price")) - 1).alias("low_pct"),
            ]
        )
        .with_columns(
            [
                # Track cumulative max high and min low
                pl.col("high_pct").cum_max().over("date").alias("cum_max_high"),
                pl.col("low_pct").cum_min().over("date").alias("cum_min_low"),
            ]
        )
    )

    # Find first bar where either threshold is hit
    first_hit = (
        post_window.with_columns(
            [
                pl.when(pl.col("cum_max_high") > tp_threshold)
                .then(pl.lit("TP_HIT"))
                .when(pl.col("cum_min_low") < -sl_threshold)
                .then(pl.lit("SL_HIT"))
                .otherwise(pl.lit("NONE"))
                .alias("hit_type")
            ]
        )
        .filter(pl.col("hit_type") != "NONE")
        .group_by("date")
        .agg(
            [
                pl.col("hit_type").first().alias("first_hit"),
                pl.col("DateTime").first().alias("hit_time"),
            ]
        )
    )

    # Create labels based on first hit
    labels = (
        morning_last.join(first_hit, on="date", how="left")
        .with_columns(
            [
                pl.when(pl.col("first_hit") == "TP_HIT")
                .then(pl.lit(SignalType.BUY.value))
                .when(pl.col("first_hit") == "SL_HIT")
                .then(pl.lit(SignalType.SELL.value))
                .otherwise(pl.lit(SignalType.HOLD.value))
                .alias("label")
            ]
        )
        .select(["date", "label", "entry_price", "first_hit", "hit_time"])
    )

    # Log label distribution
    label_counts = labels.group_by("label").agg(pl.len().alias("count"))
    logger.info(f"Label distribution (with timing):\n{label_counts}")

    return labels


def calculate_daily_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Calculate Average True Range (ATR) for each day.

    Args:
        df: DataFrame with OHLC data
        period: ATR period (default 14 days)

    Returns:
        DataFrame with date and atr columns
    """
    # Calculate daily OHLC
    daily = (
        df.group_by("date")
        .agg(
            [
                pl.col("Open").first().alias("d_open"),
                pl.col("High").max().alias("d_high"),
                pl.col("Low").min().alias("d_low"),
                pl.col("Close").last().alias("d_close"),
            ]
        )
        .sort("date")
    )

    # Calculate True Range
    daily = (
        daily.with_columns(pl.col("d_close").shift(1).alias("prev_close"))
        .with_columns(
            [
                pl.max_horizontal(
                    [
                        (pl.col("d_high") - pl.col("d_low")),
                        (pl.col("d_high") - pl.col("prev_close")).abs(),
                        (pl.col("d_low") - pl.col("prev_close")).abs(),
                    ]
                ).alias("true_range")
            ]
        )
        .with_columns(
            [pl.col("true_range").rolling_mean(window_size=period).alias("atr")]
        )
    )

    return daily.select(["date", "atr"])


def balance_labels(
    features_df: pl.DataFrame, labels_df: pl.DataFrame, method: str = "undersample"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Balance the dataset to handle class imbalance.

    Args:
        features_df: DataFrame with features
        labels_df: DataFrame with labels
        method: Balancing method ('undersample', 'oversample', or 'none')

    Returns:
        Tuple of (balanced_features, balanced_labels)
    """
    # Join features and labels
    combined = features_df.join(labels_df.select(["date", "label"]), on="date")

    if method == "none":
        return features_df, labels_df

    # Get label counts
    label_counts = combined.group_by("label").agg(pl.len().alias("count"))
    min_count = int(label_counts["count"].min() or 0)
    max_count = int(label_counts["count"].max() or 0)

    logger.info(f"Original label distribution: {label_counts}")

    if method == "undersample":
        # Undersample majority classes to match minority class
        balanced_parts = []
        for label_val in combined["label"].unique():
            label_data = combined.filter(pl.col("label") == label_val)
            sampled = label_data.sample(n=min(min_count, len(label_data)), seed=42)
            balanced_parts.append(sampled)
        balanced = pl.concat(balanced_parts)
    elif method == "oversample":
        # Oversample minority classes to match majority class
        balanced_parts = []
        for label_val in combined["label"].unique():
            label_data = combined.filter(pl.col("label") == label_val)
            current_count = len(label_data)
            if current_count < max_count:
                # Oversample with replacement
                oversampled = label_data.sample(
                    n=max_count, with_replacement=True, seed=42
                )
                balanced_parts.append(oversampled)
            else:
                balanced_parts.append(label_data)
        balanced = pl.concat(balanced_parts)
    else:
        raise ValueError(f"Unknown balancing method: {method}")

    # Split back into features and labels
    balanced_features = balanced.drop("label")
    balanced_labels = balanced.select(["date", "label"])

    logger.info(
        f"Balanced dataset: {len(balanced)} samples (original: {len(combined)})"
    )

    return balanced_features, balanced_labels


# Made with Bob
