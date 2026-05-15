import polars as pl

from .base import Indicator


class MovingAverage(Indicator):
    def __init__(self, window: int = 20, col: str = "Close", out_col: str | None = None):
        self.window = window
        self.col = col
        self.out_col = out_col or f"ma_{window}"

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.col).rolling_mean(self.window).alias(self.out_col)
        )
