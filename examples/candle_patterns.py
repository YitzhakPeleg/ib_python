"""Assign a candlestick pattern name (or None) to each row of a data file.

When more than one TA-Lib pattern fires on the same bar, the strongest
(highest abs signal) wins; ties break on TA-Lib's function order.
"""

import numpy as np
import polars as pl
import talib

from models.paths import get_file

df = pl.read_parquet(get_file("SPY", "1_min"))

open_ = df["Open"].to_numpy().astype(np.float64)
high = df["High"].to_numpy().astype(np.float64)
low = df["Low"].to_numpy().astype(np.float64)
close = df["Close"].to_numpy().astype(np.float64)

pattern_names = talib.get_function_groups()["Pattern Recognition"]
signals = np.stack(
    [getattr(talib, name)(open_, high, low, close) for name in pattern_names]
)

best_idx = np.argmax(np.abs(signals), axis=0)
best_signal = signals[best_idx, np.arange(signals.shape[1])]

pattern = np.where(
    best_signal != 0,
    np.array(pattern_names)[best_idx],
    None,
)

df = df.with_columns(pl.Series("pattern", pattern))

print(df.select("DateTime", "Open", "High", "Low", "Close", "pattern"))
print()

counts = df["pattern"].value_counts().sort("count", descending=True)
counts = counts.filter(pl.col("pattern").is_not_null())
counts = counts.with_columns(
    (pl.col("count") / df.height * 100).round(2).alias("pct")
)
print(counts.head(10))
