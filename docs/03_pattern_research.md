# 03 - Candlestick Pattern & Intraday Trigger Research

## Overview

This document tracks an exploratory research thread: does TA-Lib's candlestick
pattern library carry real predictive information on this project's intraday
(1-min) and daily equity data, and — once that turned out to be weak — how to
build and test rule-based ("state it as a sentence," no ML) intraday trigger
strategies with defined entry/stop-loss/take-profit and a target reward:risk.

Everything here is exploratory analysis, not production trading code. Scripts
referenced below live in `examples/`; reusable pieces live in `src/algo/`.

## Table of Contents

- [Phase 1: Candlestick Pattern Reliability](#phase-1-candlestick-pattern-reliability)
- [Phase 2: Trend Context](#phase-2-trend-context)
- [Phase 3: Does Technical Analysis Hold?](#phase-3-does-technical-analysis-hold)
- [Phase 4: First-30-Minutes Trigger Strategies](#phase-4-first-30-minutes-trigger-strategies)

---

## Phase 1: Candlestick Pattern Reliability

### Setup

- Installed TA-Lib (`brew install ta-lib` for the C library, `uv add TA-Lib` for
  the Python wrapper) — gives access to all 61 `CDL*` candlestick pattern
  recognition functions.
- Built a hand-curated taxonomy of all 61 patterns in `src/algo/patterns.py`
  (`PATTERN_TAXONOMY`), classifying each by:
  - `candles` — how many bars the pattern examines (1 to 5, or "3+" for the
    Hikkake variants)
  - `family` — 14 groups by shape/structural similarity (Doji, Marubozu,
    Engulfing, Harami, Star, Soldiers/crows, Hammer-shaped, etc.) — used to
    pool rare patterns for statistical support
  - `bias` — Bullish / Bearish / Both / Neutral, verified empirically against
    the actual sign TA-Lib emits (not just textbook convention)
  - `type` — Reversal / Continuation / Indecision
  - The module asserts at import time that its 61 entries exactly match
    `talib.get_function_groups()["Pattern Recognition"]`, so a TA-Lib version
    bump that changes the function list fails loudly instead of silently
    under/over-counting.
- `examples/candle_patterns.py` — simple per-row pattern labeling (strongest
  signal wins when multiple patterns fire on the same bar).

### Methodology (`examples/pattern_reliability.py`)

Tested reliability across all 8 tickers in `data/` (~1.1M 1-min bars total),
at horizons of 5/15/30/60 bars, two independent ways:

1. **Endpoint return**: `(Close[t+h] - Close[t]) / Close[t]`, session-bounded
   (nulled if the window crosses a calendar-day boundary). Win = realized
   sign matches the pattern's own realized sign (not the static taxonomy
   bias, since "Both"-bias patterns like `CDLENGULFING` fire with either
   sign depending on the specific occurrence).
2. **Triple-barrier** (path-dependent): symmetric take-profit/stop-loss
   barriers at ±1× that ticker's own baseline std for the horizon, scanning
   `High`/`Low` bar-by-bar for the first touch. Symmetric barriers give a
   clean ~50% no-edge null (optional stopping theorem), so this didn't need
   the population-baseline machinery the endpoint test required.

Statistical rigor applied throughout:
- **Low-support policy**: every occurrence always counts toward its family's
  pooled stats; a pattern only gets its own report row if `n >= 30`,
  otherwise it's marked "insufficient support — see family."
- **Wilson score CI** on win rates (`statsmodels.stats.proportion.proportion_confint`).
- **Mann-Whitney U** (endpoint) / **binomial test vs. 0.5** (triple-barrier)
  for significance.
- **Cluster block-bootstrap** (blocked by ticker+date, and separately by date
  only) as a non-iid-robust companion to the parametric tests — 1-min returns
  are autocorrelated and same-day moves across tickers are correlated.
- **Benjamini-Hochberg FDR correction** across the ~300+ hypotheses (separate
  passes for pattern-grid vs. family-grid, and endpoint-pvalue vs.
  triple-barrier-pvalue, so none of these hypothesis families borrow
  false-discovery budget from another).

Outputs: `output/pattern_reliability/{pattern,family}_report.csv`.

### Findings

- **No economically exploitable directional edge.** Patterns that survived
  FDR correction (mostly the Marubozu family: `CDLMARUBOZU`,
  `CDLCLOSINGMARUBOZU`, `CDLBELTHOLD`, `CDLLONGLINE`) were statistically
  airtight (p as low as 10⁻⁴⁶) purely from huge sample size (n in the
  hundreds of thousands) — the actual edges were 0.003%–0.03% per
  occurrence, smaller than the bid-ask spread on any of these 8 names at
  1-min resolution. Several even pointed the *wrong* way relative to their
  textbook direction (e.g. `CDLCLOSINGMARUBOZU` bearish occurrences predicted
  a down-move only 46.7% of the time).
- **The Doji "indecision" claim mostly didn't hold** under the endpoint/MWU
  test, but the triple-barrier's barrier-touch-rate test found `CDLRICKSHAWMAN`
  specifically (not `CDLDOJI` or `CDLLONGLEGGEDDOJI`) does show a real,
  consistent dispersion increase across all 4 horizons (p as low as 3×10⁻⁷ at
  60 bars) — a volatility signal, not a directional one.
- **Best directional candidate found: `CDLHANGINGMAN`** at the 5-bar horizon
  — triple-barrier win rate 53.7% (CI 52.6%–54.8%, p≈6×10⁻¹¹), decaying to a
  coin flip by 60 bars (the decay itself is a mark of a real, if small,
  microstructure effect rather than noise). Converted to actual return terms:
  ~1.1–2.1 bps expected value per resolved trade — still smaller than
  realistic transaction costs, and only ~36% of occurrences resolve within 5
  bars (the other ~64% time out, with no assigned P&L in that calculation).
- **Verdict**: across ~1.1M bars, 8 tickers, 4 horizons, two independent
  outcome definitions, and FDR correction — no tradeable edge from raw
  candlestick shape.

---

## Phase 2: Trend Context

### Question

Classical candlestick theory reads patterns contextually: a bullish
**reversal** only means something if there was a downtrend to reverse; a
bullish **continuation** only means something if there was an uptrend to
continue. Does gating on prior trend context actually improve reliability?

### `TrendADX` indicator (`src/algo/indicators/trend_adx.py`)

Built as a proper `Indicator` subclass (matching `TrendSMA`'s shape) using
TA-Lib's own Directional Movement System, for consistency with the rest of
this analysis (one package for all calculations):

- **`±DM`** (directional movement): `UpMove = High[t]-High[t-1]`,
  `DownMove = Low[t-1]-Low[t]`; `+DM = UpMove` if it's the larger *and*
  positive move, else 0 (symmetric for `-DM`).
- **`TR`** (true range): `max(High-Low, |High-PrevClose|, |Low-PrevClose|)`.
- **`±DI`**: `100 * WilderEMA(±DM, N) / WilderEMA(TR, N)` — the *direction*.
- **`DX`**/**`ADX`**: `DX = 100*|+DI--DI|/(+DI+-DI)`; `ADX = WilderEMA(DX, N)`
  — the *strength* (not direction) of the trend.
- `trend_signal = +1` if `ADX > 25` and `+DI > -DI`; `-1` if `ADX > 25` and
  `-DI > +DI`; `0` if `ADX <= 25` (Wilder's own "not trending" threshold —
  deliberately gives a genuine third state, unlike an SMA crossover which
  always picks a side even in a flat market).
- Implementation note: polars does **not** propagate null through
  `pl.when(null_condition)` — it silently falls through to `.otherwise()`.
  Required an explicit `pl.when(col.is_null()).then(None)` guard first, or
  the ADX warm-up period would have been mislabeled `trend_signal=0`
  ("no trend") instead of "unknown."

### Methodology (`examples/pattern_context.py`)

For every directional-pattern occurrence, computed `trend_signal` on the
**prior** bar (never the pattern's own bar, to avoid circularity), derived
`expected_trend` from the pattern's `type` (Continuation → same direction as
the pattern; Reversal → opposite), and classified each occurrence as
`in_context` / `out_of_context` / `no_trend` / `unknown`. "Meaningful" =
realized direction matches the pattern's predicted direction — kept as a
**pure outcome** definition, with context as a separate label, so the data
could show whether gating on context actually helps rather than assuming it
does.

One subtlety caught during implementation: `CDLHIGHWAVE`/`CDLSHORTLINE`/
`CDLSPINNINGTOP` are `type=Indecision` but directionally-signed
(`bias=Both`, not `Neutral`) — they pass the directional-pattern filter but
have no context claim to test, so they needed their own `n/a` bucket rather
than silently falling into `out_of_context`.

### Findings

**1-min data** (n in the hundreds of thousands per bucket):
| | in_context | out_of_context |
|---|---|---|
| Reversal | 50.55% [50.32%, 50.78%] | 49.01% [48.77%, 49.26%] |
| Continuation | 48.09% [47.79%, 48.39%] | 49.28% [48.98%, 49.58%] |

Reversal-context showed a small but real effect (non-overlapping CIs on
~180k occurrences each) — the one place classical theory checked out.
Continuation-context showed the *opposite* of the classical claim.

**Daily data** (real, non-resampled bars — see below): the reversal-context
effect **did not replicate** (in_context 47.9% vs out_of_context 53.5%, wide
overlapping-ish CIs on ~1,400 occurrences each — ~120x less data than 1-min).
Continuation-context still showed no effect either way, consistent with
1-min. Genuinely inconclusive whether the 1-min reversal effect is real
microstructure or noise that didn't survive a much smaller sample.

### Real daily bars + a fetcher bug fix

- Fetched true (non-resampled) daily bars for all 8 tickers via
  `examples/fetch_daily_bars.py` → `data/{TICKER}_1_day.parquet`
  (~1,255 bars/ticker, Aug 2021–Aug 2026). Requires IB Gateway/TWS running.
- **Bug found and fixed** in `src/data_fetching/historical_data_fetcher.py`:
  IB returns dates for `1 day`/`1 week`/`1 month` bar sizes as plain
  `YYYYMMDD` strings, not epoch seconds — even with `formatDate=2`. The
  fetcher always parsed as epoch seconds, so daily+ bars silently landed in
  1970. Fixed by branching on `frequency` and parsing day/week/month bars
  with `str.strptime(pl.Date, "%Y%m%d")` instead.

---

## Phase 3: Does Technical Analysis Hold?

Zooming out from candlestick shapes specifically:

- **Candlestick shape patterns are the most heavily studied and most
  consistently debunked corner of TA** in the academic literature (e.g.
  Marshall/Young/Rose 2006) — our findings here are consistent with that,
  not an unusually skeptical result.
- **What we tested (shape → forward return) is not what a real TA-based
  strategy does.** Real strategies combine shape + confluence (S/R levels,
  volume, higher-timeframe trend) + asymmetric position sizing (risk 1 to
  make 2–3) + a hard stop + discretion. Our triple-barrier test deliberately
  used *symmetric* barriers specifically to separate "does the shape predict
  direction" from "does good risk management make money" — a trader can be
  profitable with a 35% win rate and 3:1 reward:risk even with zero
  underlying predictive signal.
- **Survivorship/selection bias**: people who report "it works for me" are
  self-selected — losers mostly go quiet.
- **Where TA has more legitimate backing**: trend/momentum-following and
  short-term mean reversion, both about price behavior over time rather than
  pattern recognition on 1–3 candles.
- **Conclusion driving Phase 4**: rather than keep testing raw pattern shape,
  test a *complete* rule-based strategy — trigger + confirmation + defined
  TP/SL + a real reward:risk target — the way an actual discretionary/
  systematic trader would use TA, and let the data show whether that holds
  up.

---

## Phase 4: First-30-Minutes Trigger Strategies

### Goal

Build and test rule-based ("state it as a sentence," no ML) intraday
strategies, restricted to the first 30 minutes of each session:

- A **trigger** condition, checked bar-by-bar across the opening window.
- A **confirmation** condition on the very next bar — no trade without it.
- If confirmation fails, keep scanning the rest of the opening window for a
  fresh trigger (first *confirmed* signal per ticker per day only — once a
  trade is taken, ignore any later trigger that same day).
- No trigger anywhere in the window → no trade that day (`no trigger -> no
  action`).
- **TP is always fixed at exactly 2× the SL distance** — reward:risk ≥ 2 by
  hard construction, not something we hope holds. (Breakeven win rate at
  2:1 is only ~33%, before costs — a materially lower bar than the ~50% the
  symmetric-barrier tests in Phase 1 needed.)
- Exit at TP/SL if touched (`High`/`Low`, same-bar-both-touched counts as
  the conservative/loss outcome), else forced-closed at the **last bar of
  the day** — conservatively at that bar's **Low** for a long, **High** for
  a short (assume the worst reasonable fill rather than the close).
- No overnight risk: every trade is flat by end of day.

### Strategies (built in parallel, to compare setup styles head-to-head)

1. **SMA reversion** (mean-reversion): trigger = a bar trades at/below a
   continuous (not reset-per-day) 20-period SMA; confirm = the next bar
   closes back above it. SL = min(trigger Low, confirm Low). Mirror for
   shorts against the upper side.
2. **Session VWAP reversion** (mean-reversion): same shape as SMA reversion,
   but against VWAP anchored to the session open (resets every day by
   definition — the natural "anchored VWAP" for a first-30-minutes study).
   Both reuse one generic `find_signal_reference_reversion(reference=...)`
   function — the only difference is which column (`sma` vs `vwap`) it's
   given.
3. **Opening range breakout** (momentum): range defined by the first 5
   minutes (`OR_high`/`OR_low`), then scanned for a breakout between minutes
   6–30; confirm = the next bar continues past the break. SL = the opposite
   side of the range. (Deliberately *not* a "running" range recomputed every
   bar — that degenerates into near-guaranteed breaks of a 1-bar range.)
4. **Candlestick + confirmation**: trigger = any directional (non-Neutral-
   bias) `CDL*` pattern fires; confirm = the next bar closes beyond the
   pattern bar's extreme in the predicted direction. SL = the pattern bar's
   opposite extreme. Retests Phase 1's null result, but *with* the
   confirmation step real TA practice always uses and Phase 1 never did.

(An earlier iteration of this phase used a Bollinger Band touch instead of
#1/#2 — replaced once we wanted to compare SMA vs. session VWAP as reference
lines directly, and generalized the BB logic's "two-sided band" shape into
"a single reference line" since neither SMA nor VWAP has an upper/lower
pair. `find_signal_bb_reversal` is still in the module if needed again.)

### Engine (`src/algo/intraday_trigger.py`)

- `Signal` (trigger/confirm bar index, direction, entry/SL/TP) and
  `simulate_trade()` (walks forward from the bar after confirmation,
  same-bar-double-touch resolves as SL) are shared by all four strategies.
- `run_backtest(bars, find_signal_fn, extra_cols)` loops over every
  `(ticker, date)` session and calls `find_signal_fn` once per day —
  strategy-specific parameters (`window`, `range_bars`) are pre-bound via
  `functools.partial`. `extra_cols` accepts a `{kwarg_name: column_name}`
  dict so the same generic `reference=...` kwarg can pull from either the
  `sma` or `vwap` column without a renaming hack.
- Each strategy is a plain function — no class hierarchy, matching the
  low-abstraction style of the rest of `src/algo/`.

### Results (`examples/opening_range_strategies.py`, 8 tickers, 1-min bars)

At a fixed 2:1 reward:risk, breakeven win rate is 1/3 — the bar each
strategy needs to clear before transaction costs even enter the picture.

**Pooled per strategy:**

| Strategy | n trades | tp / sl / eod | win rate (R>0) | mean R (expectancy) | 95% bootstrap CI | p |
|---|---|---|---|---|---|---|
| SMA reversion | 2,701 | 837 / 1,761 / 103 | 33.7% | -0.0163 | [-0.067, 0.041] | 0.55 |
| VWAP reversion | 2,888 | 922 / 1,792 / 174 | 36.0% | +0.0411 | [-0.013, 0.092] | 0.14 |
| ORB breakout | 2,271 | 360 / 1,031 / 880 | 40.5% | -0.0297 | [-0.084, 0.024] | 0.25 |
| Candlestick + confirm | 2,888 | 889 / 1,865 / 134 | 33.7% | -0.0132 | [-0.064, 0.041] | 0.61 |

None significant. VWAP reversion has the best pooled point estimate.

**Per ticker, requested specifically because a strategy could work on one
name and not another** — ran all 4 strategies x 8 tickers = 32 hypotheses.
Two looked individually interesting before correction: `vwap_reversion` on
NVDA (mean R = +0.154, raw p = 0.032) and `sma_reversion`/`orb_breakout`
both notably *negative* on AAPL (raw p = 0.043 and 0.024). **All three
evaporate under Benjamini-Hochberg FDR correction across the 32-hypothesis
grid — 0/32 survive at alpha=0.05.** This is exactly what you'd expect: at
raw 95% CIs, ~1.8 false positives are expected from 32 tests by chance alone
even if nothing here is real, and that's almost exactly how many showed up.
Reported here specifically as a worked example of why the FDR pass matters
— without it, "VWAP reversion works on NVDA!" would have been a tempting
but unsupported headline.

### Takeaway

Restricting to the first 30 minutes, requiring a confirmation bar, and
fixing reward:risk at 2:1 by construction is a materially more favorable
setup than Phase 1's symmetric-barrier test (33% breakeven vs. ~50%) — and
still, none of the four rule-based strategies, at the pooled level or
per-ticker after correction, show a statistically distinguishable-from-zero
edge on this dataset. VWAP reversion is the closest to promising overall
(best pooled point estimate) and the most reasonable candidate for further
work (more history, a different confirmation rule, or a volatility-adjusted
stop) if this thread continues.
