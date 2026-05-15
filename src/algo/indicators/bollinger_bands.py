import polars as pl

from .base import Indicator


class BollingerBands(Indicator):
    def __init__(
        self,
        window: int = 20,
        std: float = 2.0,
        col: str = "Close",
        reset_per_day: bool = False,
        min_periods: int = 1,
    ):
        self.window = window
        self.std = std
        self.col = col
        self.reset_per_day = reset_per_day
        self.min_periods = min_periods

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        c = pl.col(self.col)
        mean_expr = c.rolling_mean(self.window, min_samples=self.min_periods)
        # fill_nan + fill_null: std of a single sample is NaN/null → treat as 0 (bands = SMA)
        std_expr = (
            c.rolling_std(self.window, min_samples=self.min_periods)
            .fill_nan(0)
            .fill_null(0)
        )

        if self.reset_per_day:
            if "date" not in df.columns:
                raise ValueError(
                    "BollingerBands(reset_per_day=True) requires a 'date' column"
                )
            # Compute SMA and std within each day, then derive bands
            df = df.with_columns(
                mean_expr.over("date").alias("_bb_sma"),
                (std_expr * self.std).over("date").alias("_bb_dev"),
            )
            return df.with_columns(
                pl.col("_bb_sma").alias("bb_mid"),
                (pl.col("_bb_sma") + pl.col("_bb_dev")).alias("bb_upper"),
                (pl.col("_bb_sma") - pl.col("_bb_dev")).alias("bb_lower"),
            ).drop(["_bb_sma", "_bb_dev"])

        dev = std_expr * self.std
        return df.with_columns(
            mean_expr.alias("bb_mid"),
            (mean_expr + dev).alias("bb_upper"),
            (mean_expr - dev).alias("bb_lower"),
        )
