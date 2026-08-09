import polars as pl

from .base import Indicator


class TrendSMA(Indicator):
    """3-SMA alignment trend indicator.

    trend_signal = +1 when fast > mid > slow (strong bull)
    trend_signal = -1 when fast < mid < slow (strong bear)
    trend_signal =  0 otherwise (mixed order = sideways/neutral)
    """

    def __init__(
        self,
        fast_window: int = 10,
        mid_window: int = 20,
        slow_window: int = 50,
        col: str = "Close",
    ):
        if not (fast_window < mid_window < slow_window):
            raise ValueError(
                f"Windows must be strictly increasing: "
                f"{fast_window} < {mid_window} < {slow_window}"
            )
        self.fast_window = fast_window
        self.mid_window = mid_window
        self.slow_window = slow_window
        self.col = col

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        c = pl.col(self.col)
        fast = c.rolling_mean(self.fast_window)
        mid = c.rolling_mean(self.mid_window)
        slow = c.rolling_mean(self.slow_window)

        signal = (
            pl.when((fast > mid) & (mid > slow))
            .then(1)
            .when((fast < mid) & (mid < slow))
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )

        return df.with_columns(
            fast.alias("trend_sma_fast"),
            mid.alias("trend_sma_mid"),
            slow.alias("trend_sma_slow"),
            signal.alias("trend_signal"),
        )


class TrendSMAGap(Indicator):
    """2-SMA gap-threshold trend indicator.

    trend_signal =  0 when |fast - slow| / close < neutral_band (range-bound)
    trend_signal = +1 when gap is significant AND fast > slow (bullish)
    trend_signal = -1 when gap is significant AND fast < slow (bearish)
    trend_signal = None when fast == slow exactly (should never occur in practice)
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
        neutral_band: float = 0.001,
        col: str = "Close",
    ):
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be less than slow_window ({slow_window})"
            )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.neutral_band = neutral_band
        self.col = col

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        c = pl.col(self.col)
        fast = c.rolling_mean(self.fast_window)
        slow = c.rolling_mean(self.slow_window)
        gap_pct = (fast - slow).abs() / c

        signal = (
            pl.when(gap_pct < self.neutral_band)
            .then(0)
            .when(fast > slow)
            .then(1)
            .when(fast < slow)
            .then(-1)
            .otherwise(None)
            .cast(pl.Int8)
        )

        return df.with_columns(
            fast.alias("trend_sma_fast"),
            slow.alias("trend_sma_slow"),
            signal.alias("trend_signal"),
        )
