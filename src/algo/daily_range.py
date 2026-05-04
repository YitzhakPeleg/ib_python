import polars as pl


def daily_range_avg(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """
    Calculates the average high-low range over n periods.

    Args:
        df (pl.DataFrame): DataFrame with 'High' and 'Low' columns.
        n (int): The window size for the moving average.
    """
    # 1. Calculate the high-low range for each unique date
    daily_summary = (
        df.group_by("date")
        .agg((pl.col("High").max() - pl.col("Low").min()).alias("daily_range"))
        .sort("date")
    )

    # 2. Calculate the rolling mean of that daily range
    daily_summary = daily_summary.with_columns(
        pl.col("daily_range").rolling_mean(window_size=n).alias("avg_daily_range")
    )

    # 3. Join the result back to the original intraday dataframe
    return df.join(
        daily_summary.select(["date", "avg_daily_range"]),
        on="date",
        how="left",
    )
