"""Statistical reliability of TA-Lib candlestick patterns on intraday bars.

For every one of TA-Lib's 61 `CDL*` pattern-recognition functions, this script
checks whether the pattern's implied direction (bullish/bearish) actually
predicts the forward return, at several horizons, pooled across every ticker
in data/. Rare patterns get folded into their structural "family" (see
src/algo/patterns.py) so a low-support pattern still contributes to a
statistically meaningful group estimate even when it can't stand on its own.

This script is deliberately commented far more heavily than the rest of the
codebase (near-zero-comment house style) because it's an exploratory
statistics script meant to be read and run section-by-section (the `# %%`
markers are Jupyter/VSCode "code cell" delimiters — each section can be run
independently in an interactive window).
"""

# %% Imports & configuration
import numpy as np
import polars as pl
import talib
from loguru import logger
from scipy.stats import binomtest, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from algo.patterns import PATTERN_TAXONOMY
from models.paths import get_file

# Every ticker we have 1-min data for — pooling across all of them gives rare
# patterns far more support than any single ticker could.
TICKERS = ["AAPL", "AMZN", "AVGO", "GOOG", "IBKR", "MSFT", "NVDA", "SPY"]

# Forward-return horizons to test, in bars (this is 1-min data, so 60 = 1 hour).
HORIZONS = [5, 15, 30, 60]

# A pattern needs at least this many occurrences to get its own report row;
# below this its occurrences still count toward its family's pooled stats.
MIN_SUPPORT = 30

# Significance level used for both the Wilson CI and the bootstrap CI.
ALPHA = 0.05

# Number of resamples for the cluster block-bootstrap (per hypothesis).
N_BOOTSTRAP = 2000

# Forward-return windows that would cross a session (calendar-date) boundary
# are nulled out rather than left as-is, since an overnight gap is a
# different generative process from intraday pattern continuation/reversal.
SESSION_BOUNDED = True

# Fixed seed so bootstrap CIs/p-values and baseline subsampling are reproducible.
RNG_SEED = 42

# Mann-Whitney U on the full baseline population (can be >1M rows) is slow;
# cap the baseline sample size used for the test (not for point estimates).
MWU_MAX_BASELINE_SAMPLE = 200_000

# Triple-barrier take-profit / stop-loss distance, in multiples of that
# ticker's own baseline_std_h ("typical h-bar move"). Symmetric (1.0/1.0) by
# default: this tests whether the pattern predicts direction at all, without
# also encoding a reward:risk opinion. Widen SL relative to PT (or vice versa)
# later once you're sizing an actual strategy, not testing raw reliability.
PT_MULT = 1.0
SL_MULT = 1.0

OUTPUT_DIR = "output/pattern_reliability"


# %% Load pattern taxonomy
# PATTERN_TAXONOMY already asserts (at import time) that it exactly covers
# every CDL* function TA-Lib currently ships — see src/algo/patterns.py.
PATTERN_NAMES = list(PATTERN_TAXONOMY)

# Flatten the taxonomy into a small DataFrame so it can be joined onto the
# long occurrence table later (one row per pattern: name/candles/family/bias/type).
taxonomy_df = pl.DataFrame(
    [
        {
            "pattern": info.name,
            "candles": info.candles,
            "family": info.family,
            "bias": info.bias,
            "type": info.type,
        }
        for info in PATTERN_TAXONOMY.values()
    ]
)


# %% Load data, compute pattern signals & forward returns (per ticker)
# Everything in this loop happens on ONE ticker's own chronologically-
# contiguous array at a time. This matters: both TA-Lib's multi-bar patterns
# and the forward-return shift would silently produce garbage if computed on
# bars from two different tickers stitched together.
def session_bounded_forward_return(
    close: np.ndarray, date_codes: np.ndarray, horizon: int, session_bounded: bool
) -> np.ndarray:
    """(Close[t+h] - Close[t]) / Close[t], NaN'd out where t+h falls off the
    end of the series or (if session_bounded) crosses a session/date boundary.
    """
    n = len(close)
    fwd_close = np.full(n, np.nan)  # start all-NaN; fill in the bars that have a valid t+h
    if horizon < n:
        fwd_close[: n - horizon] = close[horizon:]  # shift close forward by `horizon` bars
    ret = (fwd_close - close) / close
    if session_bounded:
        fwd_date = np.full(n, -1, dtype=date_codes.dtype)
        if horizon < n:
            fwd_date[: n - horizon] = date_codes[horizon:]
        crosses_session = fwd_date != date_codes  # True where t+h is a different calendar day
        ret[crosses_session] = np.nan
    return ret


per_ticker_frames = []
for ticker in TICKERS:
    raw = pl.read_parquet(get_file(ticker, "1_min")).select(
        "DateTime", "Open", "High", "Low", "Close"
    )
    raw = raw.with_columns(pl.col("DateTime").dt.date().alias("date"))  # calendar day, for session boundaries

    # TA-Lib needs contiguous float64 numpy arrays.
    o = raw["Open"].to_numpy().astype(np.float64)
    h = raw["High"].to_numpy().astype(np.float64)
    low = raw["Low"].to_numpy().astype(np.float64)
    c = raw["Close"].to_numpy().astype(np.float64)
    date_codes = raw["date"].to_physical().to_numpy()  # pl.Date -> int32 days-since-epoch, cheap to compare

    # Run every one of the 61 candlestick pattern functions on this ticker's own array.
    signal_cols = {
        name: getattr(talib, name)(o, h, low, c).astype(np.int16) for name in PATTERN_NAMES
    }

    # Forward return at each horizon, session-bounded per the config flag above.
    fwd_ret_cols = {
        f"fwd_ret_{horizon}": session_bounded_forward_return(c, date_codes, horizon, SESSION_BOUNDED)
        for horizon in HORIZONS
    }

    ticker_frame = raw.with_columns(
        pl.lit(ticker).alias("ticker"),
        **{name: pl.Series(values) for name, values in signal_cols.items()},
        **{name: pl.Series(values) for name, values in fwd_ret_cols.items()},
    )
    # np.nan (float NaN) is NOT the same thing as a polars null: NaN survives
    # `.is_not_null()` filters and silently poisons mean()/median() downstream
    # (NaN propagates through arithmetic) while count/boolean-based stats look
    # fine — fill_nan(None) converts the excluded bars to real nulls so every
    # later aggregation and filter treats them as missing, consistently.
    ticker_frame = ticker_frame.with_columns([pl.col(c).fill_nan(None) for c in fwd_ret_cols])
    per_ticker_frames.append(ticker_frame)
    logger.info(f"{ticker}: {raw.height:,} bars, {date_codes.max() - date_codes.min() + 1} calendar days span")

bars = pl.concat(per_ticker_frames).sort("ticker", "DateTime")
logger.info(f"Loaded {bars.height:,} total bars across {len(TICKERS)} tickers")


# %% Per-ticker baseline distribution & z-scores for every bar
# The baseline (mean/std of forward return, with no pattern conditioning) is
# computed once per (ticker, horizon) from ALL bars. Everything downstream
# compares a pattern's occurrences against ITS OWN ticker's baseline before
# pooling across tickers — that's what makes cross-ticker pooling valid
# despite the 8 tickers having very different price levels/volatility.
baseline = bars.group_by("ticker").agg(
    [pl.col(f"fwd_ret_{h}").mean().alias(f"baseline_mean_{h}") for h in HORIZONS]
    + [pl.col(f"fwd_ret_{h}").std().alias(f"baseline_std_{h}") for h in HORIZONS]
)

bars = bars.join(baseline, on="ticker", how="left")
for h in HORIZONS:
    # z = how many baseline-std's away from that ticker's typical h-bar move.
    # By construction this has ~mean 0, ~std 1 across ALL bars of a ticker,
    # so it's directly comparable across tickers and horizons.
    bars = bars.with_columns(
        ((pl.col(f"fwd_ret_{h}") - pl.col(f"baseline_mean_{h}")) / pl.col(f"baseline_std_{h}")).alias(f"z_{h}")
    )


# %% Triple-barrier labels (path-dependent alternative to the plain endpoint return)
# Endpoint fwd_ret only looks at Close[t+h]; this instead scans High/Low
# bar-by-bar for the first touch of a take-profit or stop-loss barrier, sized
# symmetrically off that ticker's own baseline_std_h. Symmetric barriers
# around a roughly driftless price process give a clean ~50% no-edge null for
# the win rate (optional stopping theorem), so the significance test below
# doesn't need the population-baseline machinery the endpoint approach required.
def triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    date_codes: np.ndarray,
    horizon: int,
    sigma: float,
    pt_mult: float,
    sl_mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    """For every bar t, scan (t, t+horizon] for the first bar whose High
    reaches the upper (take-profit) barrier or whose Low reaches the lower
    (stop-loss) barrier. Returns (label, bars_to_outcome):
    label = +1 (upper touched first), -1 (lower touched first — or both
    touched on the same bar; OHLC alone can't tell which came first
    intrabar, so the conservative/adverse outcome wins ties), 0 (neither
    touched before the horizon elapsed — a "timeout"), or NaN (the full
    horizon window wasn't available, same exclusion as the endpoint return).
    """
    n = len(close)
    upper = close * (1 + pt_mult * sigma)  # take-profit price level
    lower = close * (1 - sl_mult * sigma)  # stop-loss price level

    # A bar only gets evaluated if its full horizon window exists AND stays
    # within one session — identical criterion to session_bounded_forward_return,
    # so both outcome measures drop exactly the same set of bars.
    valid_h = np.zeros(n, dtype=bool)
    valid_h[: n - horizon] = True
    if horizon < n:
        fwd_date = np.full(n, -1, dtype=date_codes.dtype)
        fwd_date[: n - horizon] = date_codes[horizon:]
        valid_h &= fwd_date == date_codes
    # Once t+horizon is confirmed same-session as t, every bar in between is
    # too (dates are non-decreasing along one ticker's sorted series) — so the
    # scan below never needs its own mid-window session check.

    label = np.full(n, np.nan)
    bars_to_outcome = np.full(n, np.nan)
    still_open = valid_h.copy()  # bars whose outcome hasn't been resolved yet

    for k in range(1, horizon + 1):
        idx = np.nonzero(still_open)[0]
        if len(idx) == 0:
            break
        fut = idx + k
        hit_upper = high[fut] >= upper[idx]
        hit_lower = low[fut] <= lower[idx]
        resolved = idx[hit_upper | hit_lower]
        if len(resolved) == 0:
            continue
        outcome_upper = idx[hit_upper & ~hit_lower]  # upper touched, lower didn't
        outcome_lower = idx[hit_lower]  # lower touched (includes the same-bar tie case)
        label[outcome_upper] = 1
        label[outcome_lower] = -1
        bars_to_outcome[resolved] = k
        still_open[resolved] = False

    bars_to_outcome[still_open] = horizon  # exhausted the horizon: timeout, at k=horizon
    label[still_open] = 0
    label[~valid_h] = np.nan  # redundant with the initial fill, kept for clarity
    return label, bars_to_outcome


tb_frames = []
for ticker in TICKERS:
    sub = bars.filter(pl.col("ticker") == ticker)  # bars is sorted ticker,DateTime — this stays chronological
    high = sub["High"].to_numpy().astype(np.float64)
    low = sub["Low"].to_numpy().astype(np.float64)
    close = sub["Close"].to_numpy().astype(np.float64)
    date_codes = sub["date"].to_physical().to_numpy()

    tb_cols = {}
    for h in HORIZONS:
        sigma = float(sub[f"baseline_std_{h}"][0])  # constant within this ticker's rows
        label, bars_to_outcome = triple_barrier_labels(close, high, low, date_codes, h, sigma, PT_MULT, SL_MULT)
        tb_cols[f"tb_label_{h}"] = label
        tb_cols[f"tb_bars_{h}"] = bars_to_outcome

    tb_frame = sub.select("ticker", "DateTime").with_columns(**{name: pl.Series(v) for name, v in tb_cols.items()})
    tb_frame = tb_frame.with_columns([pl.col(c).fill_nan(None) for c in tb_cols])  # NaN -> null, same reasoning as fwd_ret
    tb_frames.append(tb_frame)

bars = bars.join(pl.concat(tb_frames), on=["ticker", "DateTime"], how="left")
logger.info(f"Triple-barrier labels computed (PT_MULT={PT_MULT}, SL_MULT={SL_MULT})")


# %% Empirical bias cross-check (diagnostic — doesn't block the analysis)
# The taxonomy's `bias` field was hand-curated; sanity check it against what
# TA-Lib's functions actually emitted on this dataset. This only WARNS — the
# actual analysis below always uses the realized per-occurrence sign, never
# the static taxonomy bias, so a mismatch here can't silently corrupt results.
for name, info in PATTERN_TAXONOMY.items():
    observed_values = bars[name].unique().to_list()
    realized_signs = {int(np.sign(v)) for v in observed_values if v != 0}
    if info.bias == "Bullish" and -1 in realized_signs:
        logger.warning(f"{name}: taxonomy says Bullish but a bearish (-1) signal was observed")
    elif info.bias == "Bearish" and 1 in realized_signs:
        logger.warning(f"{name}: taxonomy says Bearish but a bullish (+1) signal was observed")
    elif info.bias == "Both" and len(realized_signs) < 2:
        logger.info(f"{name}: taxonomy says Both but only {realized_signs} was observed in this dataset")


# %% Reshape wide -> long occurrence table (one row per pattern firing, per horizon)
# Step 1: unpivot the 61 wide pattern columns into (pattern, signal) rows,
# keeping every other column (ticker/date/close/fwd_ret_*/z_*) attached.
# Step 2: drop the "did not fire" rows (signal == 0).
occ_wide = bars.unpivot(
    index=["ticker", "DateTime", "date"]
    + [f"fwd_ret_{h}" for h in HORIZONS]
    + [f"z_{h}" for h in HORIZONS]
    + [f"tb_label_{h}" for h in HORIZONS]
    + [f"tb_bars_{h}" for h in HORIZONS],
    on=PATTERN_NAMES,
    variable_name="pattern",
    value_name="signal",
).filter(pl.col("signal") != 0)

# direction = the SIGN of this specific occurrence's TA-Lib value (not the
# static taxonomy bias) — this is what lets a "Both"-bias pattern like
# CDLENGULFING split into separate bullish-occurrence and bearish-occurrence
# hypotheses instead of being averaged together.
occ_wide = occ_wide.with_columns(pl.col("signal").sign().cast(pl.Int8).alias("direction"))
occ_wide = occ_wide.join(taxonomy_df, on="pattern", how="left")

# Block-membership keys used later by the cluster bootstrap (§ significance):
# "ticker_date" blocks by within-ticker trading day; "date" blocks by
# calendar day only, pooling tickers (proxy for shared market-wide moves).
occ_wide = occ_wide.with_columns(
    pl.concat_str(["ticker", pl.col("date").cast(pl.Utf8)], separator="_").alias("block_tickerdate"),
    pl.col("date").cast(pl.Utf8).alias("block_date"),
)

# Step 3: stack the 4 horizons vertically so each row is a single
# (occurrence, horizon) pair — much easier to group by (pattern, horizon) than
# wrangling 4 parallel sets of columns.
occ_long = pl.concat(
    [
        occ_wide.select(
            "ticker", "date", "pattern", "signal", "direction", "candles", "family", "bias", "type",
            "block_tickerdate", "block_date",
            pl.lit(h).alias("horizon"),
            pl.col(f"fwd_ret_{h}").alias("fwd_ret"),
            pl.col(f"z_{h}").alias("z"),
            pl.col(f"tb_label_{h}").alias("tb_label"),
            pl.col(f"tb_bars_{h}").alias("tb_bars"),
        )
        for h in HORIZONS
    ]
).filter(pl.col("fwd_ret").is_not_null())  # drop rows a session boundary nulled out for that horizon

directional_long = occ_long.filter(pl.col("bias") != "Neutral")  # Bullish/Bearish/Both patterns
neutral_long = occ_long.filter(pl.col("bias") == "Neutral")  # Doji/LongLeggedDoji/RickshawMan


# %% Directional stats: win rate + Wilson CI, mean/median return, edge, effect size
def directional_point_estimates(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Per group: support, mean/median forward return, edge (vs baseline),
    effect size, and win rate with a Wilson score confidence interval.
    Ties (fwd_ret exactly 0) are excluded from the win-rate denominator.
    """
    core = df.group_by(group_cols).agg(
        pl.len().alias("n"),
        pl.col("fwd_ret").mean().alias("mean_forward_return"),
        pl.col("fwd_ret").median().alias("median_forward_return"),
        pl.col("z").mean().alias("effect_size"),  # mean z = edge in baseline-std units, cross-ticker comparable
    )
    # "edge" in raw-return units: mean forward return minus mean baseline
    # return, i.e. effect_size re-scaled back out of standardized units. We
    # get this directly as mean(z) * (population wasn't rescaled per row), so
    # instead compute it directly from z's definition: edge = mean(fwd_ret) - mean(baseline_mean),
    # which we don't have per-row anymore — approximate via effect_size is not
    # equivalent across tickers, so compute edge properly from an explicit column instead.
    wins = (
        df.filter(pl.col("fwd_ret") != 0)
        .with_columns((pl.col("fwd_ret").sign() == pl.col("direction")).alias("is_win"))
        .group_by(group_cols)
        .agg(pl.len().alias("n_win_eval"), pl.col("is_win").sum().alias("n_wins"))
    )
    stats = core.join(wins, on=group_cols, how="left")

    n_wins = stats["n_wins"].fill_null(0).to_numpy()
    n_eval = stats["n_win_eval"].fill_null(0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        win_rate = np.where(n_eval > 0, n_wins / n_eval, np.nan)
        ci_low, ci_high = proportion_confint(n_wins, np.where(n_eval > 0, n_eval, 1), alpha=ALPHA, method="wilson")
        ci_low = np.where(n_eval > 0, ci_low, np.nan)
        ci_high = np.where(n_eval > 0, ci_high, np.nan)

    return stats.with_columns(
        pl.Series("win_rate", win_rate),
        pl.Series("win_rate_ci_low", ci_low),
        pl.Series("win_rate_ci_high", ci_high),
    )


directional_pattern_stats = directional_point_estimates(directional_long, ["pattern", "direction", "horizon"])
directional_family_stats = directional_point_estimates(directional_long, ["family", "direction", "horizon"])


# %% Neutral/indecision stats: does dispersion increase, not direction
def neutral_point_estimates(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Neutral patterns (Doji family) don't imply a direction, so there's no
    win/loss. Instead test the actual claim: does |z| (deviation from the
    typical h-bar move) increase after the pattern fires, vs. the unconditional
    baseline |z| pooled over every bar of every ticker.
    """
    return df.with_columns(pl.col("z").abs().alias("abs_z")).group_by(group_cols).agg(
        pl.len().alias("n"),
        pl.col("fwd_ret").mean().alias("mean_forward_return"),
        pl.col("fwd_ret").median().alias("median_forward_return"),
        pl.col("abs_z").mean().alias("effect_size"),  # mean |z| conditional on the pattern firing
    )


neutral_pattern_stats = neutral_point_estimates(neutral_long, ["pattern", "horizon"])
neutral_family_stats = neutral_point_estimates(neutral_long, ["family", "horizon"])

# Baseline |z| pooled across every bar of every ticker (not just occurrences),
# one number per horizon — this is the "no pattern" reference point that the
# neutral-pattern dispersion effect_size above is compared against.
baseline_abs_z_by_horizon = {
    h: float(bars[f"z_{h}"].drop_nulls().abs().mean()) for h in HORIZONS
}
baseline_abs_z_df = pl.DataFrame(
    {"horizon": list(baseline_abs_z_by_horizon), "baseline_abs_z": list(baseline_abs_z_by_horizon.values())}
)
neutral_pattern_stats = neutral_pattern_stats.join(baseline_abs_z_df, on="horizon", how="left").with_columns(
    (pl.col("effect_size") - pl.col("baseline_abs_z")).alias("edge")  # >0 means dispersion increased
)
neutral_family_stats = neutral_family_stats.join(baseline_abs_z_df, on="horizon", how="left").with_columns(
    (pl.col("effect_size") - pl.col("baseline_abs_z")).alias("edge")
)

# Directional tables don't have "edge" yet either (see the comment left in
# directional_point_estimates) — compute it the same way baseline is
# expressed, by joining each row's ticker-baseline back in. Since directional
# stats are already pooled across tickers, approximate edge as effect_size
# converted back through the pooled baseline std (weighted by occurrence
# count per ticker) — simplest correct approach: recompute edge directly as
# mean(fwd_ret) - mean(baseline_mean_h) weighted by each occurrence's own
# ticker, done via an extra column on the long table before pooling.
def attach_edge(long_df: pl.DataFrame, bars_baseline: pl.DataFrame) -> pl.DataFrame:
    long_with_baseline = long_df.join(
        bars_baseline.unpivot(
            index="ticker",
            on=[f"baseline_mean_{h}" for h in HORIZONS],
            variable_name="horizon_col",
            value_name="baseline_mean",
        ).with_columns(pl.col("horizon_col").str.extract(r"(\d+)$").cast(pl.Int64).alias("horizon")),
        on=["ticker", "horizon"],
        how="left",
    )
    return long_with_baseline.with_columns((pl.col("fwd_ret") - pl.col("baseline_mean")).alias("excess"))


directional_long = attach_edge(directional_long, baseline)


def with_edge(stats_df: pl.DataFrame, group_cols: list[str], long_df: pl.DataFrame) -> pl.DataFrame:
    edge = long_df.group_by(group_cols).agg(pl.col("excess").mean().alias("edge"))
    return stats_df.join(edge, on=group_cols, how="left")


directional_pattern_stats = with_edge(directional_pattern_stats, ["pattern", "direction", "horizon"], directional_long)
directional_family_stats = with_edge(directional_family_stats, ["family", "direction", "horizon"], directional_long)


# %% Significance: Mann-Whitney U + cluster block-bootstrap
def cluster_bootstrap(
    values: np.ndarray, block_ids: np.ndarray, null_value: float, alternative: str
) -> tuple[float, float, float]:
    """Resample whole (ticker,date) or (date) blocks with replacement rather
    than individual rows, so within-block correlation (autocorrelated returns,
    same-day repeated firings, shared market-wide moves) doesn't make the CI
    look artificially tight. Returns (ci_low, ci_high, p_value).
    """
    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    block_sum = np.bincount(inverse, weights=values)
    block_n = np.bincount(inverse)
    k = len(unique_blocks)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, k, size=(N_BOOTSTRAP, k))  # resample block indices with replacement, per replicate
    boot_means = block_sum[idx].sum(axis=1) / block_n[idx].sum(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    if alternative == "two-sided":
        p = 2 * min((boot_means <= null_value).mean(), (boot_means >= null_value).mean())
        p = min(p, 1.0)
    else:  # "greater" — used for the neutral/dispersion hypothesis
        p = (boot_means <= null_value).mean()
    return float(ci_low), float(ci_high), float(p)


rng = np.random.default_rng(RNG_SEED)
baseline_z_sample = {}  # per-horizon subsample of EVERY bar's z (the MWU comparison population)
for h in HORIZONS:
    pool = bars[f"z_{h}"].drop_nulls().to_numpy()
    if len(pool) > MWU_MAX_BASELINE_SAMPLE:
        pool = rng.choice(pool, size=MWU_MAX_BASELINE_SAMPLE, replace=False)
    baseline_z_sample[h] = pool


def compute_significance(
    long_df: pl.DataFrame, group_cols: list[str], value_col: str, alternative: str, null_value_fn
) -> pl.DataFrame:
    """One row per group: Mann-Whitney U p-value (occurrence z vs. baseline z
    population) plus two cluster-bootstrap CIs/p-values on `value_col`
    (blocked by ticker+date, and separately by date only).
    """
    grouped = long_df.group_by(group_cols).agg(
        pl.col(value_col), pl.col("z"), pl.col("block_tickerdate"), pl.col("block_date")
    )
    rows = []
    for row in grouped.iter_rows(named=True):
        values = np.array(row[value_col])
        z_sample = np.array(row["z"])
        horizon = row["horizon"]
        mwu_p = float(mannwhitneyu(z_sample, baseline_z_sample[horizon], alternative=alternative).pvalue)
        null_value = null_value_fn(horizon)
        ci_low_td, ci_high_td, p_td = cluster_bootstrap(
            values, np.array(row["block_tickerdate"]), null_value, alternative
        )
        ci_low_d, ci_high_d, p_d = cluster_bootstrap(values, np.array(row["block_date"]), null_value, alternative)
        rows.append(
            {
                **{c: row[c] for c in group_cols},
                "mwu_pvalue": mwu_p,
                "boot_ci_low_tickerdate": ci_low_td,
                "boot_ci_high_tickerdate": ci_high_td,
                "boot_p_tickerdate": p_td,
                "boot_ci_low_date": ci_low_d,
                "boot_ci_high_date": ci_high_d,
                "boot_p_date": p_d,
            }
        )
    return pl.DataFrame(rows)


logger.info("Computing significance for directional pattern-level hypotheses...")
directional_pattern_sig = compute_significance(
    directional_long, ["pattern", "direction", "horizon"], "excess", "two-sided", lambda h: 0.0
)
logger.info("Computing significance for directional family-level hypotheses...")
directional_family_sig = compute_significance(
    directional_long, ["family", "direction", "horizon"], "excess", "two-sided", lambda h: 0.0
)

neutral_long_abs = neutral_long.with_columns(pl.col("z").abs().alias("abs_z"))
logger.info("Computing significance for neutral pattern-level hypotheses...")
neutral_pattern_sig = compute_significance(
    neutral_long_abs, ["pattern", "horizon"], "abs_z", "greater", lambda h: baseline_abs_z_by_horizon[h]
)
logger.info("Computing significance for neutral family-level hypotheses...")
neutral_family_sig = compute_significance(
    neutral_long_abs, ["family", "horizon"], "abs_z", "greater", lambda h: baseline_abs_z_by_horizon[h]
)

directional_pattern_stats = directional_pattern_stats.join(
    directional_pattern_sig, on=["pattern", "direction", "horizon"], how="left"
)
directional_family_stats = directional_family_stats.join(
    directional_family_sig, on=["family", "direction", "horizon"], how="left"
)
neutral_pattern_stats = neutral_pattern_stats.join(neutral_pattern_sig, on=["pattern", "horizon"], how="left")
neutral_family_stats = neutral_family_stats.join(neutral_family_sig, on=["family", "horizon"], how="left")


# %% Triple-barrier stats: win rate vs. the barrier's own ~50% null
def directional_triple_barrier_stats(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Among occurrences whose barrier actually resolved (tb_label != 0, i.e.
    not a timeout): did the barrier in the PREDICTED direction get touched
    first? Tested against p=0.5 via a plain binomial test — symmetric
    barriers make 0.5 the natural no-edge null, so no population baseline
    sample is needed here (unlike the endpoint-return significance test).
    """
    counts = df.group_by(group_cols).agg(
        pl.len().alias("tb_n"),
        (pl.col("tb_label") == 0).sum().alias("tb_n_timeout"),
        pl.col("tb_bars").mean().alias("tb_mean_bars_to_outcome"),
    )
    resolved = df.filter(pl.col("tb_label") != 0).with_columns(
        (pl.col("tb_label") == pl.col("direction")).alias("is_win")
    )
    win_counts = resolved.group_by(group_cols).agg(
        pl.len().alias("tb_n_resolved"), pl.col("is_win").sum().alias("tb_n_wins")
    )
    stats = counts.join(win_counts, on=group_cols, how="left")

    n_wins = stats["tb_n_wins"].fill_null(0).to_numpy()
    n_resolved = stats["tb_n_resolved"].fill_null(0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        win_rate = np.where(n_resolved > 0, n_wins / n_resolved, np.nan)
        ci_low, ci_high = proportion_confint(n_wins, np.where(n_resolved > 0, n_resolved, 1), alpha=ALPHA, method="wilson")
        ci_low = np.where(n_resolved > 0, ci_low, np.nan)
        ci_high = np.where(n_resolved > 0, ci_high, np.nan)
    p_values = np.array(
        [binomtest(int(w), int(nr), p=0.5).pvalue if nr > 0 else np.nan for w, nr in zip(n_wins, n_resolved)]
    )

    return stats.with_columns(
        pl.Series("tb_win_rate", win_rate),
        pl.Series("tb_win_rate_ci_low", ci_low),
        pl.Series("tb_win_rate_ci_high", ci_high),
        pl.Series("tb_pvalue", p_values),
        (pl.col("tb_n_timeout") / pl.col("tb_n")).alias("tb_timeout_rate"),
    )


def neutral_triple_barrier_stats(
    df: pl.DataFrame, group_cols: list[str], baseline_touch_rate: dict[int, float]
) -> pl.DataFrame:
    """Neutral patterns have no predicted direction, so instead check whether
    EITHER barrier gets touched more often than usual after the pattern fires
    (a breakout), vs. that same-sized barrier's baseline touch rate computed
    over every bar of every ticker (one-sided binomial test).
    """
    stats = df.group_by(group_cols).agg(
        pl.len().alias("tb_n"),
        (pl.col("tb_label") != 0).sum().alias("tb_n_touched"),
        pl.col("tb_bars").mean().alias("tb_mean_bars_to_outcome"),
    )
    n_touched = stats["tb_n_touched"].to_numpy()
    tb_n = stats["tb_n"].to_numpy()
    touch_rate = n_touched / tb_n
    baseline_rate = np.array([baseline_touch_rate[int(h)] for h in stats["horizon"].to_numpy()])
    p_values = np.array(
        [
            binomtest(int(nt), int(ni), p=br, alternative="greater").pvalue
            for nt, ni, br in zip(n_touched, tb_n, baseline_rate)
        ]
    )
    return stats.with_columns(
        pl.Series("tb_touch_rate", touch_rate),
        pl.Series("tb_baseline_touch_rate", baseline_rate),
        pl.Series("tb_edge", touch_rate - baseline_rate),
        pl.Series("tb_pvalue", p_values),
    )


directional_pattern_stats = directional_pattern_stats.join(
    directional_triple_barrier_stats(directional_long, ["pattern", "direction", "horizon"]),
    on=["pattern", "direction", "horizon"],
    how="left",
)
directional_family_stats = directional_family_stats.join(
    directional_triple_barrier_stats(directional_long, ["family", "direction", "horizon"]),
    on=["family", "direction", "horizon"],
    how="left",
)

# Population reference: how often does a same-sized barrier get touched from
# a RANDOM bar (not conditioned on any pattern)? This is what the neutral
# patterns' touch rate is compared against.
baseline_touch_rate_by_horizon = {h: float((bars[f"tb_label_{h}"].drop_nulls() != 0).mean()) for h in HORIZONS}

neutral_pattern_stats = neutral_pattern_stats.join(
    neutral_triple_barrier_stats(neutral_long, ["pattern", "horizon"], baseline_touch_rate_by_horizon),
    on=["pattern", "horizon"],
    how="left",
)
neutral_family_stats = neutral_family_stats.join(
    neutral_triple_barrier_stats(neutral_long, ["family", "horizon"], baseline_touch_rate_by_horizon),
    on=["family", "horizon"],
    how="left",
)


# %% Assemble per-pattern report (apply the MIN_SUPPORT low-support policy)
# Every occurrence always counted toward its family's pooled stats above
# (directional_family_stats / neutral_family_stats are never thresholded).
# A pattern only gets its OWN standalone row here if it individually clears
# MIN_SUPPORT; below that, the row is kept (so you can still see its n) but
# every point estimate is nulled out and `status` points you at its family.
def apply_support_policy(stats_df: pl.DataFrame, taxonomy_df: pl.DataFrame, join_key: str) -> pl.DataFrame:
    df = stats_df.join(taxonomy_df.unique(subset=[join_key]), on=join_key, how="left") if join_key == "pattern" else stats_df
    ok = pl.col("n") >= MIN_SUPPORT
    status = pl.when(ok).then(pl.lit("ok")).otherwise(
        pl.concat_str(
            [pl.lit("insufficient support (n="), pl.col("n").cast(pl.Utf8), pl.lit(") — see family '"), pl.col("family"), pl.lit("'")]
        )
    )
    point_estimate_cols = [
        c
        for c in df.columns
        if c
        not in ("pattern", "family", "direction", "horizon", "n", "candles", "bias", "type", "n_win_eval", "n_wins")
    ]
    df = df.with_columns([pl.when(ok).then(pl.col(c)).otherwise(None).alias(c) for c in point_estimate_cols])
    return df.with_columns(status.alias("status"))


directional_pattern_report = apply_support_policy(
    directional_pattern_stats.join(taxonomy_df.select("pattern", "candles", "bias", "type"), on="pattern", how="left"),
    taxonomy_df,
    "pattern",
).with_columns(pl.col("direction").replace_strict({1: "bullish", -1: "bearish"}, return_dtype=pl.Utf8).alias("direction"))

neutral_pattern_report = apply_support_policy(
    neutral_pattern_stats.join(taxonomy_df.select("pattern", "candles", "bias", "type"), on="pattern", how="left"),
    taxonomy_df,
    "pattern",
).with_columns(pl.lit("neutral").alias("direction"))

pattern_report = pl.concat([directional_pattern_report, neutral_pattern_report], how="diagonal_relaxed")


# %% Assemble per-family report (always fully populated, no threshold)
family_meta = taxonomy_df.group_by("family").agg(pl.col("candles").min().alias("min_candles"))

directional_family_report = directional_family_stats.join(family_meta, on="family", how="left").with_columns(
    pl.col("direction").replace_strict({1: "bullish", -1: "bearish"}, return_dtype=pl.Utf8).alias("direction"), pl.lit("ok").alias("status")
)
neutral_family_report = neutral_family_stats.join(family_meta, on="family", how="left").with_columns(
    pl.lit("neutral").alias("direction"), pl.lit("ok").alias("status")
)
family_report = pl.concat([directional_family_report, neutral_family_report], how="diagonal_relaxed")


# %% Multiple-testing correction (Benjamini-Hochberg FDR)
# Two independent passes: pattern-level hypotheses compete for false-discovery
# budget only against other pattern-level hypotheses, and same for family-level
# — rolling both grids together would let the coarser family grid "borrow"
# significance from the finer pattern grid or vice versa.
def add_fdr(report: pl.DataFrame, pvalue_col: str, output_col: str) -> pl.DataFrame:
    key_col = "pattern" if "pattern" in report.columns else "family"  # pattern_report vs family_report
    join_cols = [key_col, "direction", "horizon"]
    valid = report.filter(pl.col(pvalue_col).is_not_null())
    if valid.height == 0:
        return report.with_columns(pl.lit(None, dtype=pl.Boolean).alias(output_col))
    reject, _, _, _ = multipletests(valid[pvalue_col].to_numpy(), alpha=ALPHA, method="fdr_bh")
    valid = valid.with_columns(pl.Series(output_col, reject))
    return report.join(valid.select(join_cols + [output_col]), on=join_cols, how="left")


# Two separate hypothesis families get their own FDR pass, same reasoning as
# the pattern-grid/family-grid split: the endpoint-return test (mwu_pvalue)
# and the triple-barrier test (tb_pvalue) are different questions, so one
# shouldn't borrow false-discovery budget from the other.
pattern_report = add_fdr(pattern_report, "mwu_pvalue", "fdr_significant")
pattern_report = add_fdr(pattern_report, "tb_pvalue", "tb_fdr_significant")
family_report = add_fdr(family_report, "mwu_pvalue", "fdr_significant")
family_report = add_fdr(family_report, "tb_pvalue", "tb_fdr_significant")


# %% Display final report tables
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_float("mixed")

ROUND_COLS = [
    "win_rate", "win_rate_ci_low", "win_rate_ci_high",
    "mean_forward_return", "median_forward_return", "edge", "effect_size", "mwu_pvalue",
    "tb_win_rate", "tb_win_rate_ci_low", "tb_win_rate_ci_high", "tb_timeout_rate", "tb_pvalue",
    "tb_mean_bars_to_outcome", "tb_touch_rate", "tb_baseline_touch_rate", "tb_edge",
]
# Narrow set of columns for terminal readability — the CSVs written below keep everything.
display_cols = [
    "pattern", "family", "direction", "horizon", "n", "status",
    "win_rate", "win_rate_ci_low", "win_rate_ci_high",
    "edge", "effect_size", "mwu_pvalue", "fdr_significant",
    "tb_win_rate", "tb_timeout_rate", "tb_pvalue", "tb_fdr_significant",
]


def rounded(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.col(c).round(4) for c in ROUND_COLS if c in df.columns])


print("\n=== PATTERN REPORT (ok rows only, sorted by |effect_size|) ===")
ok_patterns = rounded(pattern_report.filter(pl.col("status") == "ok")).sort(pl.col("effect_size").abs(), descending=True)
print(ok_patterns.select([c for c in display_cols if c in ok_patterns.columns]))

print(f"\n=== PATTERNS BELOW MIN_SUPPORT ({MIN_SUPPORT}) — folded into family only ===")
low_support = pattern_report.filter(pl.col("status") != "ok").select("pattern", "family", "direction", "horizon", "n", "status")
print(low_support)

print("\n=== FAMILY REPORT (sorted by |effect_size|) ===")
family_display_cols = [c for c in display_cols if c != "pattern"]
print(
    rounded(family_report)
    .sort(pl.col("effect_size").abs(), descending=True)
    .select([c for c in family_display_cols if c in family_report.columns])
)


# %% Write outputs to disk
from pathlib import Path

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
pattern_report.write_csv(f"{OUTPUT_DIR}/pattern_report.csv")
family_report.write_csv(f"{OUTPUT_DIR}/family_report.csv")
logger.info(f"Wrote reports to {OUTPUT_DIR}/")


# %% Diagnostics / sanity summary
print("\n=== DIAGNOSTICS ===")
total_bars = bars.height
print(f"Total bars analyzed: {total_bars:,} across {len(TICKERS)} tickers")
for h in HORIZONS:
    null_pct = 100 * bars[f"fwd_ret_{h}"].null_count() / total_bars
    print(f"  horizon={h}: {null_pct:.2f}% of bars excluded (session boundary or end of series)")
n_below_threshold = pattern_report.filter(pl.col("status") != "ok")["pattern"].n_unique()
print(f"Patterns with at least one horizon below MIN_SUPPORT={MIN_SUPPORT}: {n_below_threshold} / {len(PATTERN_NAMES)}")
