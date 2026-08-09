"""TrendSMAGap example — 2-SMA + gap threshold trend detection.

trend_signal: +1 = bullish, -1 = bearish, 0 = neutral (gap too small)

Tune neutral_band per instrument:
  - tighter (e.g. 0.0005) → less neutral, more directional signals
  - wider  (e.g. 0.002)   → more neutral, only strong trends flagged
"""

import subprocess
import tempfile

import polars as pl

from algo.indicators import TrendSMAGap
from models.paths import get_file
from visualization.plotting import plot_bars

df = pl.read_parquet(get_file("SPY", "1_min"))

indicator = TrendSMAGap(fast_window=20, slow_window=50, neutral_band=0.001)
df = indicator(df)

print(df["trend_signal"].value_counts().sort("trend_signal"))

fig = plot_bars(
    df,
    trend_mid_col=None,
    title="SPY 1-min — TrendSMAGap (20/50, band=0.1%)",
    show_fig=False,
    return_fig=True,
)

with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    fig.write_html(f.name)
    subprocess.run(["open", f.name])
