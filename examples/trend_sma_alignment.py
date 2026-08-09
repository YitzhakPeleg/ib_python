"""TrendSMA example — 3-SMA alignment trend detection.

trend_signal: +1 = fast > mid > slow (bull), -1 = fast < mid < slow (bear), 0 = neutral
"""

import subprocess
import tempfile

import polars as pl

from algo.indicators import TrendSMA
from models.paths import get_file
from visualization.plotting import plot_bars

df = pl.read_parquet(get_file("SPY", "1_min"))

indicator = TrendSMA(fast_window=10, mid_window=20, slow_window=50)
df = indicator(df)

print(df["trend_signal"].value_counts().sort("trend_signal"))

fig = plot_bars(df, title="SPY 1-min — TrendSMA (10/20/50)", show_fig=False, return_fig=True)

with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    fig.write_html(f.name)
    subprocess.run(["open", f.name])
