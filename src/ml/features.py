"""Feature engineering for the day-range predictor.

All OHLC features (daily AND intraday) are expressed relative to the last
known close before that bar/day — open_ret = Open/prev_close - 1, high_ret
= High/prev_close - 1, low_ret = Low/prev_close - 1, close_ret =
Close/prev_close - 1 — kept zero-centered (a return, not a raw ratio)
rather than centered on 1.0: a first version of this centered on 1.0
(open_ratio = Open/prev_close with no "-1") measurably hurt training —
near-constant-scale inputs clustered tightly around a NON-zero mean are
harder for gradient descent to learn from than the same information
expressed as a small deviation from 0, even though it's mathematically
identical information. Volume is expressed as ln(Volume / prior Volume) —
a LOG ratio to the
last day's volume (that row's own preceding day, for daily rows; the
target session's own preceding day — fixed for the whole intraday window
— for intraday rows) — logged specifically because the intraday bars
(each typically a small fraction of a full day's volume) and the daily
bars (a ratio typically near 1.0) would otherwise sit on very different
scales; the log compresses that gap.

Daily and intraday bars now share the EXACT SAME 5-feature schema
(FEATURE_NAMES) — this is what lets them be concatenated into one flat
sequence for a Transformer encoder (src/ml/model.py's DayRangeTransformer),
rather than needing two separately-shaped encoders the way the GRU+MLP
model does.
"""

import numpy as np
import polars as pl

from algo.resample_bars import resample_to_timeframe

DAILY_LOOKBACK_DAYS = 14
INTRADAY_INPUT_BARS = 6
INTRADAY_TIMEFRAME = "5m"
RATIO_EPS = 1e-8

FEATURE_NAMES = [
    "open_ret",
    "high_ret",
    "low_ret",
    "close_ret",
    "log_vol_ratio",
]
# Aliases for backward-compat callers -- daily and intraday bars use an
# identical feature schema now, so both names point at the same list.
DAILY_FEATURE_NAMES = FEATURE_NAMES
INTRADAY_FEATURE_NAMES = FEATURE_NAMES


def prepare_frames(
    minute_bars: pl.DataFrame, intraday_timeframe: str = INTRADAY_TIMEFRAME
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """minute_bars: single-ticker 1-min OHLCV, with a `DateTime` column.

    Casts Volume to Float64 (source parquet stores it as Decimal(38,0)),
    then resamples once to daily bars and once to intraday_timeframe bars
    (both via resample_to_timeframe, which is session-safe for "1d" and
    for any timeframe that evenly divides the 9:30 session open, including
    every "1m"/"5m"/"15m" duration used here). Returns (daily_df,
    intraday_df), each with a `date` (pl.Date) column.
    """
    minute_bars = minute_bars.with_columns(pl.col("Volume").cast(pl.Float64))
    daily_df = resample_to_timeframe(minute_bars, "1d")
    intraday_df = resample_to_timeframe(minute_bars, intraday_timeframe)
    return daily_df, intraday_df


def log_ratio(numerator: np.ndarray, denominator) -> np.ndarray:
    """ln(numerator / denominator), both floored at RATIO_EPS first to
    avoid log(0) or a divide-by-zero on a (rare, but possible) zero-volume
    bar. denominator may be an array (elementwise) or a scalar
    (broadcast) -- used both ways in dataset.py.
    """
    numerator = np.maximum(np.asarray(numerator, dtype=np.float64), RATIO_EPS)
    denominator = np.maximum(np.asarray(denominator, dtype=np.float64), RATIO_EPS)
    return np.log(numerator / denominator)


def add_daily_return_features(daily_df: pl.DataFrame) -> pl.DataFrame:
    """Adds prev_close, prev_volume (Close/Volume.shift(1)) and the 5
    zero-centered features (open_ret, high_ret, low_ret, close_ret,
    log_vol_ratio) — every feature uses only that row's own OHLCV plus its
    own immediate predecessor's close/volume, so no lookahead is possible.
    """
    df = daily_df.with_columns(
        pl.col("Close").shift(1).alias("prev_close"),
        pl.col("Volume").shift(1).alias("prev_volume"),
    )
    return df.with_columns(
        (pl.col("Open") / pl.col("prev_close") - 1).alias("open_ret"),
        (pl.col("High") / pl.col("prev_close") - 1).alias("high_ret"),
        (pl.col("Low") / pl.col("prev_close") - 1).alias("low_ret"),
        (pl.col("Close") / pl.col("prev_close") - 1).alias("close_ret"),
        (
            pl.max_horizontal(pl.col("Volume"), pl.lit(RATIO_EPS))
            / pl.max_horizontal(pl.col("prev_volume"), pl.lit(RATIO_EPS))
        )
        .log()
        .alias("log_vol_ratio"),
    )
