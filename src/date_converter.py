import polars as pl


def add_date_int_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parses the 'DateTime' string column and creates a 'date' column
    as an integer in the format YYYYMMDD.

    Parameters:
    - df: pl.DataFrame with a 'DateTime' column (String)

    Returns:
    - pl.DataFrame with an additional 'date' column (Int64)
    """
    return df.with_columns(
        pl.col("DateTime")
        .str.to_datetime()  # 'to_datetime' handles the T separator and timezone offset automatically
        .dt.strftime("%Y%m%d")  # Format Date as "YYYYMMDD" string
        .cast(pl.Int64)  # Cast string to Integer
        .alias("date")
    )


# Example usage:
if __name__ == "__main__":
    df = pl.read_csv("AAPL_1_min.csv")
    df_with_date = add_date_int_column(df)
    df_with_date.write_csv("AAPL_1_min.csv")
    print(df_with_date)
