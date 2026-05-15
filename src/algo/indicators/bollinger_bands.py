import polars as pl

from .base import Indicator


class BollingerBands(Indicator):
    def __init__(self, window: int = 20, std: float = 2.0, col: str = "Close"):
        self.window = window
        self.std = std
        self.col = col

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        c = pl.col(self.col)
        mid = c.rolling_mean(self.window).alias("bb_mid")
        dev = c.rolling_std(self.window) * self.std
        upper = (c.rolling_mean(self.window) + dev).alias("bb_upper")
        lower = (c.rolling_mean(self.window) - dev).alias("bb_lower")
        return df.with_columns([mid, upper, lower])
