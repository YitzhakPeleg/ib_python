import polars as pl


def calculate_bollinger_bands(
    df: pl.DataFrame, window: int, stds: float
) -> pl.DataFrame:
    """
    Calculates Bollinger Bands for a Polars DataFrame.

    Parameters:
    - df: pl.DataFrame with columns ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    - window: int, the moving average period
    - stds: float, the number of standard deviations for the bands

    Returns:
    - pl.DataFrame with added 'bb_lower', 'bb_mid', and 'bb_upper' columns.
    """

    # 1. Calculate the Middle Band (SMA) and the rolling standard deviation
    # 2. Use those to derive the Upper and Lower bands
    # 3. Drop the temporary standard deviation column

    return (
        df.with_columns(
            [
                pl.col("Close").rolling_mean(window_size=window).alias("bb_mid"),
                pl.col("Close").rolling_std(window_size=window).alias("tmp_std"),
            ]
        )
        .with_columns(
            [
                (pl.col("bb_mid") - (pl.col("tmp_std") * stds)).alias("bb_lower"),
                (pl.col("bb_mid") + (pl.col("tmp_std") * stds)).alias("bb_upper"),
            ]
        )
        .drop("tmp_std")
    )


# Example usage
if __name__ == "__main__":
    df = pl.read_csv("AAPL_1_min.csv")
    df_with_indicators = calculate_bollinger_bands(df, window=20, stds=2.0)
    df_with_indicators.write_csv("AAPL_1_min_bb_20_2.csv")
    print(df_with_indicators)
