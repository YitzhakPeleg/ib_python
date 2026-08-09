import numpy as np
import polars as pl
import talib

from .base import Indicator


class TrendADX(Indicator):
    """ADX/DI-based trend classifier (Wilder's Directional Movement System).

    trend_signal = +1 when ADX > adx_threshold and +DI > -DI (confirmed uptrend)
    trend_signal = -1 when ADX > adx_threshold and -DI > +DI (confirmed downtrend)
    trend_signal =  0 when ADX <= adx_threshold (no trend / ranging — ADX's own
                     "not trending" regime, unlike an SMA crossover which
                     always picks a side even in a flat market)
    """

    def __init__(self, timeperiod: int = 14, adx_threshold: float = 25.0):
        self.timeperiod = timeperiod
        self.adx_threshold = adx_threshold

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["High"].to_numpy().astype(np.float64)
        low = df["Low"].to_numpy().astype(np.float64)
        close = df["Close"].to_numpy().astype(np.float64)

        adx = talib.ADX(high, low, close, timeperiod=self.timeperiod)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.timeperiod)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.timeperiod)

        # fill_nan(None): TA-Lib's warm-up period (~2*timeperiod-1 bars) comes
        # back as float NaN, not a polars null — converting it here means the
        # pl.when() chain below correctly propagates null through warm-up
        # instead of miscomparing NaN as if it were a real ADX value.
        df = df.with_columns(
            pl.Series("adx", adx).fill_nan(None),
            pl.Series("plus_di", plus_di).fill_nan(None),
            pl.Series("minus_di", minus_di).fill_nan(None),
        )

        # polars does NOT propagate null through pl.when(null_condition) — a
        # null condition silently falls through to .otherwise(), it doesn't
        # short-circuit to null — so the warm-up period needs an explicit
        # is_null() guard or it would be mislabeled trend_signal=0 (no trend)
        # instead of "unknown."
        trend_signal = (
            pl.when(pl.col("adx").is_null())
            .then(None)
            .when(pl.col("adx") > self.adx_threshold)
            .then(pl.when(pl.col("plus_di") > pl.col("minus_di")).then(1).otherwise(-1))
            .otherwise(0)
            .cast(pl.Int8)
        )

        return df.with_columns(trend_signal.alias("trend_signal"))
