"""Phase 4: first-30-minutes trigger -> confirmation -> TP/SL strategies.

Four rule-based, "state it as a sentence" strategies, all restricted to the
first 30 minutes of each session, all using the same trigger -> confirmation
-> 2:1-by-construction TP/SL -> forced-EOD-close engine
(src/algo/intraday_trigger.py):

1. SMA reversion (mean-reversion around a 20-period SMA)
2. Session VWAP reversion (mean-reversion around VWAP anchored to the
   session open — the natural "anchored VWAP" for a first-30-minutes study)
3. Opening range breakout (momentum)
4. Candlestick pattern + confirmation (retests Phase 1's null result, but
   with the confirmation step real TA practice always uses)

Every strategy is also reported PER TICKER, not just pooled — a strategy
that looks flat on average could still be real on one name and canceled out
by another.

See docs/03_pattern_research.md for the full research narrative. Heavily
commented and split into `# %%` sections, same convention as the earlier
scripts in this series.
"""

# %% Imports & configuration
import functools

import numpy as np
import polars as pl
import talib
from loguru import logger
from statsmodels.stats.multitest import multipletests

from algo.intraday_trigger import (
    find_signal_candlestick_confirm,
    find_signal_orb_breakout,
    find_signal_reference_reversion,
    run_backtest,
)
from algo.patterns import PATTERN_TAXONOMY
from models.paths import get_file

TICKERS = ["AAPL", "AMZN", "AVGO", "GOOG", "IBKR", "MSFT", "NVDA", "SPY"]

WINDOW = 30  # first 30 one-minute bars = first 30 minutes of the session
ORB_RANGE_BARS = 5  # opening range is fixed from the first 5 minutes, then scanned for breaks over the rest of WINDOW
SMA_WINDOW = 20  # same window as Phase 4's original Bollinger Bands, for rough comparability

N_BOOTSTRAP = 2000
ALPHA = 0.05
RNG_SEED = 42

# Only directional (non-Neutral-bias) patterns have a predicted direction to confirm against.
DIRECTIONAL_PATTERNS = [name for name, info in PATTERN_TAXONOMY.items() if info.bias != "Neutral"]


# %% Load data, compute the indicator columns each strategy needs (per ticker)
# SMA, session VWAP, and the 61 CDL* pattern functions all need each ticker's
# own chronologically-contiguous array — never compute across a ticker
# boundary (same lesson as every earlier script in this series).
per_ticker_frames = []
for ticker in TICKERS:
    df = pl.read_parquet(get_file(ticker, "1_min")).select(
        "DateTime", "Open", "High", "Low", "Close", "Volume"
    )
    df = df.with_columns(
        pl.col("DateTime").dt.date().alias("date"),
        pl.lit(ticker).alias("ticker"),
        pl.col("Volume").cast(pl.Float64),
    )

    # SMA: continuous across day boundaries (NOT reset per day), same
    # reasoning as Phase 4's original BollingerBands(reset_per_day=False) —
    # the opening window needs an already-warmed-up reference, not one
    # restarting from scratch with only a couple of bars each morning.
    df = df.with_columns(pl.col("Close").rolling_mean(SMA_WINDOW).alias("sma"))

    # Session VWAP: DOES reset every day by definition (that's what makes it
    # VWAP rather than just a volume-weighted SMA) — anchored to the 09:30
    # session open, which is exactly the "anchored VWAP" this first-30-minute
    # study calls for. Typical price = (H+L+C)/3, the standard convention.
    typical_price = (pl.col("High") + pl.col("Low") + pl.col("Close")) / 3
    df = df.with_columns(
        (
            (typical_price * pl.col("Volume")).cum_sum().over("date")
            / pl.col("Volume").cum_sum().over("date")
        ).alias("vwap")
    )

    o = df["Open"].to_numpy().astype(np.float64)
    h = df["High"].to_numpy().astype(np.float64)
    low = df["Low"].to_numpy().astype(np.float64)
    c = df["Close"].to_numpy().astype(np.float64)

    # Candlestick strategy's trigger: for each bar, the sign of whichever
    # directional pattern fired with the largest magnitude (0 if none fired)
    # — same "strongest signal wins" tie-break as examples/candle_patterns.py.
    signals = np.stack([getattr(talib, name)(o, h, low, c) for name in DIRECTIONAL_PATTERNS])
    best_idx = np.argmax(np.abs(signals), axis=0)
    best_val = signals[best_idx, np.arange(signals.shape[1])]
    pattern_direction = np.sign(best_val).astype(np.int8)
    df = df.with_columns(pl.Series("pattern_direction", pattern_direction))

    per_ticker_frames.append(df)
    logger.info(f"{ticker}: {df.height:,} bars")

bars = pl.concat(per_ticker_frames).sort("ticker", "DateTime")
logger.info(f"Loaded {bars.height:,} total bars across {len(TICKERS)} tickers")


# %% Run each strategy's backtest across every (ticker, date) session
# extra_cols as a dict maps find_signal_fn's kwarg name -> bars column name,
# so the one generic find_signal_reference_reversion(reference=...) function
# works for both "sma" and "vwap" without any column-renaming hack.
strategies = {
    "sma_reversion": (
        functools.partial(find_signal_reference_reversion, window=WINDOW),
        {"reference": "sma"},
    ),
    "vwap_reversion": (
        functools.partial(find_signal_reference_reversion, window=WINDOW),
        {"reference": "vwap"},
    ),
    "orb_breakout": (
        functools.partial(find_signal_orb_breakout, range_bars=ORB_RANGE_BARS, window=WINDOW),
        {},
    ),
    "candlestick_confirm": (
        functools.partial(find_signal_candlestick_confirm, window=WINDOW),
        {"pattern_direction": "pattern_direction"},
    ),
}

results = {}
for name, (find_signal_fn, extra_cols) in strategies.items():
    trades = run_backtest(bars, find_signal_fn, extra_cols=extra_cols)
    results[name] = trades
    logger.info(f"{name}: {trades.height:,} trades found")


# %% Block bootstrap on mean R-multiple (blocked by date)
# Two trades on the same calendar date but different tickers are still
# correlated (shared market-wide moves) even though at most one trade exists
# per (ticker, date) — the same cross-ticker same-day concern as every
# earlier script in this series, so block by date rather than treat each
# trade as fully independent. For a single-ticker subset this degenerates to
# an ordinary per-trade bootstrap, which is exactly correct there too.
def block_bootstrap_mean_ci(
    values: np.ndarray, block_ids: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = ALPHA, seed: int = RNG_SEED
) -> tuple[float, float, float]:
    """Returns (ci_low, ci_high, two_sided_p) — the p-value is the bootstrap
    analog of a t-test: 2x the smaller tail fraction of replicates on the
    other side of 0 from the point estimate.
    """
    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    block_sum = np.bincount(inverse, weights=values)
    block_n = np.bincount(inverse)
    k = len(unique_blocks)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boot_means = block_sum[idx].sum(axis=1) / block_n[idx].sum(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())
    return float(ci_low), float(ci_high), float(min(p, 1.0))


def summarize(trades: pl.DataFrame, block_col: str = "date") -> dict:
    r = trades["r_multiple"].to_numpy()
    ci_low, ci_high, p = block_bootstrap_mean_ci(r, trades[block_col].cast(pl.Utf8).to_numpy())
    return {
        "n": trades.height,
        "win_rate": float((r > 0).mean()),
        "mean_r": float(r.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": p,
    }


# %% Report — pooled per strategy, then per-ticker with FDR correction
# Running 4 strategies x 8 tickers = 32 uncorrected 95% CIs (plus 4 pooled
# ones) means ~1.8 "significant-looking" results are expected from PURE
# CHANCE even if nothing here is real. Apply Benjamini-Hochberg FDR
# correction across each hypothesis grid separately (per-ticker grid vs.
# pooled-strategy grid — same reasoning as every earlier script in this
# series: a coarser and a finer grid shouldn't borrow false-discovery budget
# from each other) before treating any single result as a finding.
pl.Config.set_tbl_rows(len(TICKERS) * len(strategies))
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_float("mixed")

BREAKEVEN_WIN_RATE = 1 / 3  # at a fixed 2:1 reward:risk, this is the win rate needed just to break even (pre-costs)

print("\n" + "=" * 10 + " Pooled summary, per strategy " + "=" * 10)
pooled_rows = []
for name, trades in results.items():
    if trades.height == 0:
        continue
    exit_counts = {row["exit_reason"]: row["len"] for row in trades.group_by("exit_reason").len().to_dicts()}
    pooled_rows.append({"strategy": name, "exit_counts": str(exit_counts), **summarize(trades)})
pooled_df = pl.DataFrame(pooled_rows).with_columns([pl.col(c).round(4) for c in ("win_rate", "mean_r", "ci_low", "ci_high", "p")])
reject, _, _, _ = multipletests(pooled_df["p"].to_numpy(), alpha=ALPHA, method="fdr_bh")
pooled_df = pooled_df.with_columns(pl.Series("fdr_significant", reject))
print(pooled_df)

print("\n" + "=" * 10 + " Per-ticker breakdown (all strategies, one FDR pass) " + "=" * 10)
per_ticker_rows = []
for name, trades in results.items():
    for ticker in TICKERS:
        ticker_trades = trades.filter(pl.col("ticker") == ticker)
        if ticker_trades.height == 0:
            continue
        per_ticker_rows.append({"strategy": name, "ticker": ticker, **summarize(ticker_trades)})
per_ticker_df = pl.DataFrame(per_ticker_rows).with_columns(
    [pl.col(c).round(4) for c in ("win_rate", "mean_r", "ci_low", "ci_high", "p")]
)
reject, _, _, _ = multipletests(per_ticker_df["p"].to_numpy(), alpha=ALPHA, method="fdr_bh")
per_ticker_df = per_ticker_df.with_columns(pl.Series("fdr_significant", reject))
print(per_ticker_df.sort("mean_r", descending=True))

n_survive = int(per_ticker_df["fdr_significant"].sum())
print(f"\n{n_survive} / {per_ticker_df.height} per-ticker results survive FDR correction at alpha={ALPHA}.")
