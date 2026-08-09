"""Does trend context make a candlestick pattern more "meaningful"?

Classical candlestick theory reads patterns contextually: a bullish REVERSAL
only means something if there was a downtrend to reverse; a bullish
CONTINUATION only means something if there was an uptrend to continue. This
script tests that claim directly rather than assuming it.

"meaningful" is a pure OUTCOME definition: did the realized direction after
the pattern match the direction the pattern predicts? Trend context (from
TrendADX) is kept as a SEPARATE label, not folded into that definition — so
we can empirically compare the meaningful-rate WITH the classical context
requirement against the meaningful-rate WITHOUT it, instead of assuming
context matters and only ever looking at the gated subset.

Heavily commented and split into `# %%` (Jupyter/VSCode cell marker)
sections, same convention as examples/pattern_reliability.py.
"""

# %% Imports & configuration
import numpy as np
import polars as pl
import talib
from loguru import logger
from statsmodels.stats.proportion import proportion_confint

from algo.indicators import TrendADX
from algo.patterns import PATTERN_TAXONOMY
from models.paths import get_file

TICKERS = ["AAPL", "AMZN", "AVGO", "GOOG", "IBKR", "MSFT", "NVDA", "SPY"]
FREQUENCY = "1_day"  # real (non-resampled) daily bars fetched via examples/fetch_daily_bars.py
SESSION_BOUNDED = FREQUENCY not in ("1_day", "1_week", "1_month")  # no intraday session concept on daily+ bars

HORIZON = 10  # bars ahead used to check the realized direction (10 trading days ~ 2 weeks, at this frequency)
ADX_TIMEPERIOD = 14
ADX_THRESHOLD = 25.0

# ~1,255 daily bars/ticker (vs >140k for 1-min) means far fewer pattern
# occurrences — lower the per-pattern support bar accordingly. The pooled
# (type, context) and (context) tables below aren't affected by this at all,
# since they aggregate across every pattern regardless of individual support.
MIN_SUPPORT = 10
ALPHA = 0.05

PATTERN_NAMES = list(PATTERN_TAXONOMY)
taxonomy_df = pl.DataFrame(
    [
        {"pattern": info.name, "family": info.family, "bias": info.bias, "type": info.type}
        for info in PATTERN_TAXONOMY.values()
    ]
)


# %% Load data, compute patterns + trend context (per ticker)
# Both the 61 CDL* pattern functions AND TrendADX need each ticker's own
# chronologically-contiguous array — computing across a ticker boundary would
# silently corrupt both the multi-bar patterns and the ADX/DI smoothing.
trend_adx = TrendADX(timeperiod=ADX_TIMEPERIOD, adx_threshold=ADX_THRESHOLD)

per_ticker_frames = []
for ticker in TICKERS:
    df = pl.read_parquet(get_file(ticker, FREQUENCY)).select("DateTime", "Open", "High", "Low", "Close")
    df = df.with_columns(pl.col("DateTime").dt.date().alias("date"))

    o = df["Open"].to_numpy().astype(np.float64)
    h = df["High"].to_numpy().astype(np.float64)
    low = df["Low"].to_numpy().astype(np.float64)
    c = df["Close"].to_numpy().astype(np.float64)

    signal_cols = {name: getattr(talib, name)(o, h, low, c).astype(np.int16) for name in PATTERN_NAMES}
    df = df.with_columns(pl.lit(ticker).alias("ticker"), **{n: pl.Series(v) for n, v in signal_cols.items()})

    df = trend_adx(df)  # adds adx, plus_di, minus_di, trend_signal
    # Trend context must come from BEFORE the pattern's own bar — using the
    # same bar's trend_signal would let the pattern's own candle influence
    # the "prior trend" it's supposedly being read against.
    df = df.with_columns(pl.col("trend_signal").shift(1).alias("trend_signal_prior"))

    # Forward return at HORIZON bars. On intraday data, nulled out across a
    # session boundary (same reasoning as pattern_reliability.py's
    # session_bounded_forward_return) — but on daily+ bars every row IS a
    # different calendar day by construction, so that same check would
    # incorrectly null out every single forward return; SESSION_BOUNDED turns
    # it off for daily/weekly/monthly frequencies.
    fwd_close = pl.col("Close").shift(-HORIZON)
    raw_ret = (fwd_close - pl.col("Close")) / pl.col("Close")
    if SESSION_BOUNDED:
        fwd_date = pl.col("date").shift(-HORIZON)
        ret_expr = pl.when(fwd_date == pl.col("date")).then(raw_ret).otherwise(None)
    else:
        ret_expr = raw_ret
    df = df.with_columns(ret_expr.alias("fwd_ret"))

    per_ticker_frames.append(df)
    logger.info(f"{ticker}: {df.height:,} bars")

bars = pl.concat(per_ticker_frames).sort("ticker", "DateTime")
logger.info(f"Loaded {bars.height:,} total bars across {len(TICKERS)} tickers ({FREQUENCY})")


# %% Reshape to one row per pattern occurrence
occ = bars.unpivot(
    index=["ticker", "DateTime", "trend_signal_prior", "fwd_ret"],
    on=PATTERN_NAMES,
    variable_name="pattern",
    value_name="signal",
).filter(pl.col("signal") != 0)

occ = occ.with_columns(pl.col("signal").sign().cast(pl.Int8).alias("direction")).join(
    taxonomy_df, on="pattern", how="left"
)
occ = occ.filter(pl.col("bias") != "Neutral")  # "meaningful" (direction match) only applies to directional patterns
occ = occ.filter(pl.col("fwd_ret").is_not_null())


# %% Classify trend context per occurrence
# expected_trend = the prior-trend state classical theory says SHOULD have
# preceded this occurrence, given its type:
#   Continuation -> prior trend should already match the pattern's direction
#   Reversal     -> prior trend should be the OPPOSITE of the pattern's direction
#   Indecision   -> no directional context claim to test (excluded above anyway,
#                    since Indecision patterns are all bias="Neutral")
expected_trend = (
    pl.when(pl.col("type") == "Continuation")
    .then(pl.col("direction"))
    .when(pl.col("type") == "Reversal")
    .then(-pl.col("direction"))
    .otherwise(None)
)
occ = occ.with_columns(expected_trend.alias("expected_trend"))

# Some patterns are type="Indecision" but directionally-signed by TA-Lib
# (CDLHIGHWAVE/CDLSHORTLINE/CDLSPINNINGTOP are bias="Both", not "Neutral") —
# they pass the bias filter above but have no context claim to test (their
# expected_trend is null by construction), so they need their own bucket
# rather than falling through to "out_of_context" by accident.
context = (
    pl.when(pl.col("type") == "Indecision")
    .then(pl.lit("n/a"))  # no directional context claim for this pattern type
    .when(pl.col("trend_signal_prior").is_null())
    .then(pl.lit("unknown"))  # ADX warm-up period
    .when(pl.col("trend_signal_prior") == 0)
    .then(pl.lit("no_trend"))  # ADX <= threshold: nothing to reverse or continue
    .when(pl.col("trend_signal_prior") == pl.col("expected_trend"))
    .then(pl.lit("in_context"))  # the classical setup was actually present
    .otherwise(pl.lit("out_of_context"))  # a trend was present, but the wrong one
)
occ = occ.with_columns(context.alias("context"))


# %% "meaningful" = realized direction matches the pattern's predicted direction
# Ties (fwd_ret exactly 0) are excluded, same convention as pattern_reliability.py.
occ = occ.filter(pl.col("fwd_ret") != 0).with_columns(
    (pl.col("fwd_ret").sign() == pl.col("direction")).alias("meaningful")
)


# %% Aggregate: does requiring context actually raise the meaningful-rate?
def meaningful_rate_table(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    stats = df.group_by(group_cols).agg(pl.len().alias("n"), pl.col("meaningful").sum().alias("n_meaningful"))
    n = stats["n"].to_numpy()
    n_meaningful = stats["n_meaningful"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = n_meaningful / n
        ci_low, ci_high = proportion_confint(n_meaningful, n, alpha=ALPHA, method="wilson")
    return stats.with_columns(
        pl.Series("meaningful_rate", rate),
        pl.Series("ci_low", ci_low),
        pl.Series("ci_high", ci_high),
    ).sort(group_cols)


pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_float("mixed")

# Primary answer: pooled across ALL patterns of a type, by context. This is
# the headline comparison — individual-pattern x context bins get thin fast.
print(f"\n=== Meaningful rate by (type, context) — pooled across all patterns, horizon={HORIZON} ===")
print(meaningful_rate_table(occ, ["type", "context"]))

print("\n=== Meaningful rate by context alone (all directional patterns pooled) ===")
print(meaningful_rate_table(occ, ["context"]))

# Secondary breakdown: per pattern, in_context vs out_of_context only (drop
# no_trend/unknown here — those aren't a context claim either way), with the
# same low-support policy used in pattern_reliability.py: below MIN_SUPPORT,
# report support but null the rate rather than show a noisy point estimate.
per_pattern = meaningful_rate_table(
    occ.filter(pl.col("context").is_in(["in_context", "out_of_context"])), ["pattern", "type", "context"]
)
per_pattern = per_pattern.with_columns(
    [
        pl.when(pl.col("n") >= MIN_SUPPORT).then(pl.col(c)).otherwise(None).alias(c)
        for c in ("meaningful_rate", "ci_low", "ci_high")
    ]
)
print(f"\n=== Per-pattern meaningful rate, in_context vs out_of_context (n>={MIN_SUPPORT} to show a rate) ===")
print(per_pattern.sort(["pattern", "context"]))
