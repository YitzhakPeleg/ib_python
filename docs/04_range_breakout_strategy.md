# 04 - ATR Opening-Range Breakout Strategy (SPY)

## Overview

An intraday momentum strategy on SPY: measure the first 30 minutes of the
session, and if that range is "quiet enough" relative to SPY's own recent
volatility, trade the first clean breakout out of it — in the breakout's own
direction, not faded. Symmetric ATR-scaled take-profit/stop-loss, multiple
sequential trades allowed per day, cut off in the early afternoon.

This document has two purposes:

1. **[Part 1](#part-1--the-rule-human-executable)** — the rule stated plainly
   enough for a person to trade it by hand from a broker platform.
2. **[Part 2](#part-2--implementation-reference)** — the exact code/parameters
   to reproduce or extend the backtest.

Everything here is exploratory research on one ticker's historical data, not
a validated live-trading edge — see [Caveats](#caveats--whats-not-done-yet)
before risking anything on it. That said, this config has now been through a
real out-of-sample and cross-validation study (**[Appendix E](#appendix-e--out-of-sample-and-cross-validation-2018-2025)**),
not just tuned once and reported: **the 30m / `range<ATR/2` / TP=SL=ATR
(divisor 3-5) / multi-trade family is validated as a genuine,
repeatedly-rediscovered signal across 5 of 8 tested years (2018,
2022-2025)** — but it reliably fails on 3 specific years (2019-2021), and
that failure mode is the main open problem, not overfitting to a single
lucky sample. See Appendix E.5 for the honest, unhedged version of this
conclusion, including why "just improve TP/SL sizing" likely isn't the
whole fix.

Engine: `src/algo/range_breakout.py` (`run_breakout_backtest`). Shared
ATR/session-walk infrastructure: `src/algo/hammer_reversal.py`. Visual
browser: `examples/range_breakout_browser.py`.

---

## Part 1 — The Rule (Human-Executable)

**Ticker:** SPY only.

**Before the open (once, using yesterday's data):**

1. Look up SPY's 14-day ATR (daily chart, ATR indicator, period 14, as of
   yesterday's close). Compute **ATR / 2**.

**9:30–10:00 AM ET — just watch:**

2. Track the highest high and lowest low of the first 30 minutes. That's your
   **opening range**.

**10:00 AM ET — check the range:**

3. Range width = high − low. If it's **≥ ATR/2**, stop — no trading today.
   (There's no floor — a quiet morning is fine, as long as it isn't already
   an outsized one.)

**10:00 AM – 1:00 PM ET — watch for a signal:**

4. Watch 5-minute candles. A candle only qualifies as a trigger if **both its
   open AND its close** are outside the opening range on the same side — not
   just a close that pokes through with the open still inside.
   - Whole candle above the range → **go long** at that candle's close.
   - Whole candle below the range → **go short** at that candle's close.
5. No qualifying candle by 1:00 PM? No trade today.

**Once you're in a trade:**

6. Take-profit = entry price **± ATR/5** (in your favor).
   Stop-loss = entry price **∓ ATR/5** (against you — same distance,
   symmetric).
7. Let it run to TP or SL. If neither hits by market close, close it
   manually at the close.

**After that trade closes:**

8. If it's still before 1:00 PM, go back to step 4 and keep watching for
   another qualifying candle — a second, third (etc.) trade the same day is
   allowed, but **only ever one position open at a time**: never open a new
   trade while one is still live.

That's the entire mechanical rule — check the range at 10:00, then watch for
breakout candles until 1:00, taking every qualifying one (one at a time)
until the cutoff.

---

## Part 2 — Implementation Reference

### Code

```python
import polars as pl
from algo.range_breakout import run_breakout_backtest
from models.paths import get_file

OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
ticker = "SPY"

minute_bars = (
    pl.read_parquet(get_file(ticker, "1_min"))
    .select(OHLCV)
    .with_columns(pl.lit(ticker).alias("ticker"))
    .sort("DateTime")
)
daily_bars = (
    pl.read_parquet(get_file(ticker, "1_day"))
    .select(OHLCV)
    .with_columns(pl.lit(ticker).alias("ticker"))
    .sort("DateTime")
)

trades = run_breakout_backtest(
    minute_bars,
    daily_bars,
    signal_timeframe="5m",              # bar size for the trigger scan
    opening_range_minutes=30,           # first 30 minutes sets the range
    signal_window_minutes=210,          # 210 min after 9:30 open = 13:00 cutoff
    range_atr_low_divisor=None,         # no floor on the opening range
    range_atr_high_divisor=2.0,         # ...just range < ATR/2
    tp_atr_divisor=5.0,                 # TP = ATR / 5
    sl_atr_divisor=5.0,                 # SL = ATR / 5 (symmetric)
    require_open_outside=True,          # whole bar's body outside the range
    allow_multiple_trades_per_day=True, # sequential trades until cutoff
)
```

Visual day-by-day browser (5-min candles, opening range shaded, trigger
triangle, TP/SL lines, result + P&L in the title):

```bash
uv run python examples/range_breakout_browser.py
```

### Parameter reference

| Parameter | Value | Meaning |
|---|---|---|
| `opening_range_minutes` | 30 | Duration of the first bar used to set the range |
| `signal_window_minutes` | 210 | Trigger scan window end, minutes after the 9:30 open (210 → 13:00) |
| `range_atr_low_divisor` | `None` | No lower bound on the opening range |
| `range_atr_high_divisor` | 2.0 | Opening range must be `< ATR/2` |
| `tp_atr_divisor` | 5.0 | Take-profit distance `= ATR/5` |
| `sl_atr_divisor` | 5.0 | Stop-loss distance `= ATR/5` (symmetric) |
| `require_open_outside` | `True` | Trigger bar's open AND close both outside the range |
| `allow_multiple_trades_per_day` | `True` | Sequential same-day re-entries after a prior trade closes |
| `signal_timeframe` | `"5m"` | Bar size for the trigger scan itself |

ATR itself: `algo.hammer_reversal.compute_daily_atr` — Wilder's ATR(14) on
daily bars, **shifted one session forward** so the value attached to day *t*
reflects only true ranges through day *t-1*'s close (no lookahead into the
day's own not-yet-complete range).

### Result of the exact config above (SPY, full available history)

Data: SPY 1-min bars, 2022-01-06 → 2026-05-01 — 1,083 total trading sessions
in the dataset (the daily-bars file used for ATR extends further back, to
2021-08-09, so the first ~14 sessions of the 1-min range are already past
ATR warm-up).

| Metric | Value |
|---|---|
| Total sessions in dataset | 1,083 |
| Sessions with ≥1 trade | 910 (**84.0%** of all sessions) |
| Sessions with zero trades | 173 (16.0%) — ATR warm-up, or range ≥ ATR/2 by 10:00 |
| Total trades | 2,101 |
| Trades per trading day (on days that traded) | 2.3 |
| Win rate | 52.2% |
| Mean R-multiple | +0.051 |
| Total R | +107.7R |
| Total $ (1 share/contract, no costs) | **+$147.64** |
| Exit mix | TP 1,063 (50.6%) / SL 938 (44.6%) / EOD 100 (4.8%) |
| Longs | 986 trades, 51.2% win |
| Shorts | 1,115 trades, 53.1% win |

The `range < ATR/2` filter is fairly permissive — it only skips about 1 in 6
sessions, so this isn't a rare-setup strategy; it trades on the large
majority of days, and multiple times on most of them.

---

## Appendix A — Parameter Sweeps and What Won

Every sweep below was run on SPY only, on top of whatever the "current best"
config was at that point in the exploration. Numbers are total $ P&L (1
share/contract, gross of costs) unless noted.

### A.1 Opening-range duration

Tried 5, 10, 15, 30, 60 minutes. 5m and 10m were too short/noisy to define a
meaningful range and lost money outright. 30m and 60m were both viable; **30m
generally beat 60m** at every later stage of tuning (trigger definition, TP/SL
divisor, multi-trade), though 60m stayed a reasonable runner-up throughout.

### A.2 Opening-range filter

Original rule was a two-sided band, `ATR/4 <= range <= ATR/2` (mirroring the
earlier hammer-reversal strategy's filter — see Appendix D). Dropping the
floor and testing the two one-sided alternatives head-to-head (30m, ATR/5,
13:00 cutoff, close-only trigger — before the open+close change in A.3):

| Filter | Total $ |
|---|---|
| `range < ATR/4` | $44.59 |
| **`range < ATR/2` (no floor)** | **$62.28** |

The wider one-sided filter (more admissible days) won outright — dropping
the "day must already be somewhat active" floor was a real improvement, not
just more trades of the same quality; win rate held up even as trade count
grew.

### A.3 Trigger definition: close-only vs. open+close

| Trigger | 30m/ATR4 total $ |
|---|---|
| Close outside range only | $4.22 |
| **Open AND close outside range** | **$90.40** |

Requiring the whole candle body (not just a close that pokes through) outside
the range was one of the single biggest improvements found in the whole
exploration — roughly a 20x jump at that particular divisor, and a
consistent improvement across every divisor tested afterward.

### A.4 TP/SL sizing — fixed dollars, then ATR divisor sweep

Started with a fixed $1 TP against the range's opposite edge as SL (the
original, naive rule) — high win rate (~70%) but flat-to-breakeven, because
the SL (full range width) was usually much wider than the $1 target. Moving
to a **symmetric ATR-scaled stop** fixed this. First divisor sweep
(open+close trigger, 30m, single trade/day):

| TP=SL divisor | Total $ |
|---|---|
| ATR/3 | $75.10 |
| ATR/4 | $90.40 |
| ATR/5 | $78.04 |
| ATR/6 | $80.31 |

Non-monotonic in this narrow window, so pushed further with **asymmetric**
TP/SL (different divisor per side — e.g. `tp=ATR/4, sl=ATR/5` for a 5:4
reward:risk skew), sweeping the pair together from wide to tight:

| tp/sl divisor pair | 30m single total $ |
|---|---|
| 2/3 (widest) | $34.48 |
| 3/4 | $101.79 |
| **4/5** | **$96.88** |
| 5/6 | $70.14 |
| 6/7 | $71.74 |
| 7/8 | $68.44 |
| 8/9 | $55.59 |
| 10/11 (tightest) | $24.84 |

Confirms a real peak around 4/5, not just noise: performance falls off on
both sides — wider targets (2/3, 3/4) leave too many trades stranded at
end-of-day before reaching TP/SL (`eod` exits reached 40% of trades at
2/3); tighter targets (8/9, 10/11) shrink toward tick-level noise, and at
10/11 combined with multi-trade the result actually goes net **negative**
(-$4.19) despite 4,608 trades.

Final decision, symmetric vs. the best asymmetric pair, **under multi-trade
mode** (the mode actually adopted — see A.6):

| Config | Win rate | Mean R | Total $ |
|---|---|---|---|
| TP=SL=ATR/4 | 51.8% | +0.048 | $124.06 |
| TP=ATR/4, SL=ATR/5 | 47.6% | +0.062 | $145.17 |
| **TP=SL=ATR/5** | **52.2%** | +0.051 | **$147.64** |

Symmetric ATR/5 and asymmetric 4/5 are within ~2% of each other on total $;
**symmetric ATR/5 was chosen** for its meaningfully higher win rate (52.2%
vs 47.6%) at essentially the same P&L — fewer, shorter losing streaks for a
human trading it.

### A.5 Signal-window cutoff

Swept 11:00 → 15:30 in 30-minute steps, and also tried removing the cutoff
entirely (trade until 16:00 close).

| Config | Best cutoff | Total $ |
|---|---|---|
| 30m / single trade | 14:30 (flat plateau) | $84.46 |
| **30m / multi-trade** | **13:00** | **$147.64** |
| 60m / single trade | 12:30 (flat plateau) | $50.04 |
| 60m / multi-trade | 12:30 | $85.16 |

Removing the cutoff entirely was the single worst change tested in the whole
exploration when combined with multi-trade mode: 30m/multi went from
**+$147.64 (13:00 cutoff) to -$89.03 (no cutoff)** — late-afternoon
re-entries have too little session time left to resolve cleanly and drag win
rate below breakeven. The cutoff is load-bearing, not an arbitrary
convenience.

### A.6 Single trade vs. multiple trades per day

Allowing sequential re-entry (only after the previous trade has fully
closed, never overlapping) after a 13:00 cutoff:

| Config | Single | Multi |
|---|---|---|
| 30m, ATR/4 symmetric | $90.40 | $124.06 |
| 30m, ATR/5 symmetric | $78.04 | **$147.64** |
| 60m, ATR/4 symmetric | $70.74 | $62.65 (worse) |
| 60m, ATR/5 symmetric | $46.17 | $75.36 |

Multi-trade clearly helps at 30m (both divisors); mixed at 60m (helps at
ATR/5, hurts at ATR/4 — traced to a specifically bad "2nd trade of the day"
cohort at that one config, not a general pattern). Broken down by trade
sequence number, no divisor/range combo shows a *universal* "later trades
are worse" effect except that one cell — most 2nd/3rd/4th-of-the-day trades
stay net positive.

---

## Appendix B — Gap Analysis (Exploratory — Not Yet in the Live Rule)

A secondary investigation, layered on top of the strategy above but **not
currently part of the executed rule** — flagged here as the most promising
direction for a future refinement.

**Setup:** for each trade, computed the overnight gap (today's session open
vs. yesterday's close) as a % of ATR, and its direction (up/down), then
compared against the trade's own direction (long/short).

**Finding 1 — fade beats continuation.** Trades whose direction *opposed* the
overnight gap direction ("fade") outperformed trades that continued the gap
direction ("gap and go") in every cut of the data tried. In one config: fade
trades were 78% of total R despite being a similar trade count to
continuation trades.

**Finding 2 — moderate gaps are the sweet spot, not big ones.** Win rate and
mean-R both peak when the gap is roughly 25-50% of ATR; very small gaps
(0-25%) carry no reliable edge, and very large gaps (>100% of ATR) are
weakly *negative* — plausibly because an outsized gap has already "used up"
the day's likely move before the signal window even starts.

**Finding 3 — the standout cell.** Up-gap + short + 25-50%-of-ATR gap size
was the single strongest, most sample-size-credible subgroup found across
this entire project: **66-73% win rate, +0.22 to +0.56 mean R**, holding up
consistently across multiple trigger definitions, opening-range lengths, and
ATR divisors (n=41-75 trades depending on exact config).

**Finding 4 — prior-day range adds only a mild, structurally-damped signal.**
Bucketing the *prior* day's own range as a % of ATR (reminder: ATR already
includes that day as 1 of its 14 inputs, so this ratio can't stray far from
~100% on average) showed the best cells following a prior day whose range
was close to (just under) its own ATR average — a "normal," not
unusually-quiet or unusually-explosive, prior session. The effect is
directionally consistent but modest (~10-13 percentage points) and gets
noisy fast once crossed with the other dimensions — a `prd_dir` (prior day's
own up/down direction) cut showed no clean separation and was dropped.

**Why this isn't in the live rule yet:** every gap cut was checked
post-hoc against trades the base strategy already took, not as a live
pre-filter re-run through the backtest engine. Folding "only take the trade
if gap is 25-50% of ATR and opposes the gap direction" into
`run_breakout_backtest` as an actual filter (and re-validating the resulting,
smaller trade count) is the natural next step.

---

## Appendix C — Bugs Found and Fixed During This Work

### C.1 60-minute opening range silently measured only 30 minutes

`resample_to_timeframe`'s bins are calendar-anchored (9:00, 10:00, 11:00...),
not anchored to the 9:30 session open. Since 570 (9:30's minutes-since-midnight)
isn't a multiple of 60, a "60m" resample's first bin was `[9:00, 10:00)` —
but real data only starts at 9:30, so that bin silently only ever contained
9:30-10:00 (30 real minutes), and the 10:00-10:30 half was dropped entirely:
neither counted in the range nor scanned for a trigger.

- Affected 89% of sessions (961/1083 checked); average understatement was
  ~27% (mean buggy range $2.22 vs. true $3.07); one day showed a $14.66
  buggy range vs. a true $41.37 range.
- **Fixed** in `algo.hammer_reversal._iter_atr_sessions` by slicing the
  opening range directly off 1-minute bars by elapsed time since session
  open, instead of resampling — correct for any `opening_range_minutes`
  value, not just ones that happen to divide evenly into 570.
- Impact on the (then-current) 60m/ATR-5/range<ATR-2/13:00 config: trade
  count 929→659, win rate 53.6%→51.75%, total $ $106.57→$56.03. All results
  in this document already reflect the fixed version.
- 5m/15m/30m opening ranges were never affected (570 divides evenly into
  each).

---

## Appendix D — Earlier Alternative Strategy: Hammer Reversal (Not Adopted)

Before landing on the breakout-continuation strategy documented above, a
**fade** strategy was built and tuned first: `src/algo/hammer_reversal.py`,
`examples/hammer_reversal_browser.py`. Rule, briefly: 14-day ATR filter on
the first 15-minute bar's range (`> ATR/4`), then watch 5-min bars for a
hammer/shooting-star-shaped reversal candle trading outside that range,
enter a stop order at the candle's extreme, symmetric or opening-range-edge
TP/SL.

Findings there (SPY, various configs): widening the stop-loss from the
candle's own extreme (too tight — 80% of trades round-tripped through it as
ordinary noise) to `sl_multiple=2.5` materially helped (+16R pooled across 8
tickers vs. negative before). Attempts to swap which candle shape gates
which direction, or to drop the shape/direction pairing entirely, both made
results *worse* when checked out of SPY — evidence the directional
hammer/shooting-star pairing was carrying real signal, not noise.

This strategy was **set aside, not disproven** — the breakout-continuation
family (this document) was explored afterward and produced clearly better,
more robust results, so effort concentrated there. `hammer_reversal.py` and
its browser remain in the repo and fully working if picked back up.

---

## Appendix E — Out-of-Sample and Cross-Validation (2018-2025)

Everything in Appendices A-D was tuned and evaluated on the same 2022-2026
sample. This appendix fetched additional history from IB specifically to
check whether any of that survives outside the sample it was built on.

**Data fetched:** `examples/fetch_spy_2020_2021.py` and
`examples/fetch_spy_2018_2019.py` pulled SPY 1-min bars for 2020-01-02 →
2022-01-05 and 2018-01-02 → 2019-12-31 respectively (chunked-backward IB
requests, same pattern as `fetch_spy_history.py`). All available SPY 1-min
history was then merged into one file spanning **2018-01-02 → 2026-05-01**
(813,420 bars) — `data/SPY_full_1_min.parquet` — used for everything below.
Daily bars for ATR were resampled directly from this merged 1-min series
(rather than the separately-fetched true daily file, which only goes back to
2021-08-09) so ATR coverage is uniform across the whole 2018-2025 span.

### E.1 First check: does the existing final config hold up on 2020-2021?

Running the exact config from Part 2 (unchanged) on 2020-2021: **net
negative** — 828 trades, 48.3% win rate, -0.021 mean R, **-$40.04** total.
Broken down by year: 2020 (COVID crash + recovery) was clearly the driver
(-$40.01), 2021 (a calmer melt-up) was roughly flat (-$2.20).

**Why 2020 specifically fails:** ATR as a % of price was actually similar
*on a typical day* between 2020-2021 and 2022-2026 (median 1.24% vs 1.18%),
but 2020-2021's volatility was **far more variable** — standard deviation
roughly double (1.07 vs 0.54) and peak ATR% nearly double (6.80% vs 3.81%,
the COVID spike). The strategy sizes TP/SL off a **14-day trailing ATR**,
which necessarily lags a sudden regime shift — during the crash, trailing
ATR was still reflecting the pre-crash calm for a couple of weeks while
realized volatility had already exploded, producing badly-mis-sized TP/SL
levels and the whipsaw stop-outs that drove 2020's loss.

### E.2 Does *any* config in the searched space work on 2018-2021?

Re-ran the full 32-combo grid (opening range 30m/60m × range filter
`<ATR/4`/`<ATR/2` × TP=SL divisor 3-6 × single/multi-trade) independently on
each 2-year slice, to check whether the *specific* final config was simply
wrong for that period, or whether nothing in the space works there:

| Period | Best config found (re-optimized for that period) | Best total $ |
|---|---|---|
| 2018-2019 | 60m / `<ATR/4` / ATR/6 / single | +$0.70 |
| 2020-2021 | 60m / `<ATR/2` / ATR/6 / single | +$5.58 |
| 2022-2026 (original tuning period) | 30m / `<ATR/2` / ATR/5 / multi | +$147.64 |

Both re-optimized bests are noise-level, not an edge — 30-31 of the 32
configs tried on each period are net negative, several sharply so. This
ruled out "wrong parameters for that period" as the explanation: there was
nothing in this parameter space to find in 2018-2021, even fitting directly
on it.

### E.3 Proper walk-forward: fit on 2022 only, test 2023-2025 forward

A fairer test than "does this generalize to a totally different market
era" is "does a fit on one year hold up on the *next* one, without
retuning." Fit (best of the 32-combo grid by total $) on 2022 alone: **30m /
`<ATR/2` / TP=SL=ATR/3 / multi**, then run that exact, un-touched config
forward:

| Year | n | win_rate | mean_r | total $ |
|---|---|---|---|---|
| 2022 (fit) | 292 | 54.1% | +0.078 | +$53.12 |
| 2023 (forward) | 332 | 53.3% | +0.076 | +$29.30 |
| 2024 (forward) | 351 | 46.4% | -0.049 | -$22.80 |
| 2025 (forward) | 308 | 49.4% | +0.020 | +$12.99 |
| **2023-2025 combined** | 991 | ~50% | +0.024 | **+$19.49** |

2023 held up almost exactly as well as the fit year (even slightly better
in raw R terms). 2024 broke down into a real loss. 2025 partially recovered.
Net across the three forward years: positive, but inconsistent — real
evidence the fit wasn't pure noise, but also real evidence it isn't uniform.

### E.4 Full cross-validation: fit on each year, test on every other year

Extended E.3 into a full grid: fit the 32-combo sweep independently on each
of the 8 full years available (2018-2025, excluding partial 2026), then run
each year's best-fit config against every other year.

**Best-fit config per year:**

| Fit year | Config | In-sample $ |
|---|---|---|
| 2018 | 30m / `<ATR/2` / ATR/4 / multi | $16.86 |
| 2019 | 60m / `<ATR/4` / ATR/5 / single | $2.13 |
| 2020 | 60m / `<ATR/4` / ATR/6 / single | $0.80 |
| 2021 | 30m / `<ATR/4` / ATR/3 / single | $6.72 |
| 2022 | 30m / `<ATR/2` / ATR/3 / multi | $53.10 |
| 2023 | 30m / `<ATR/2` / ATR/4 / multi | $45.05 |
| 2024 | 30m / `<ATR/2` / ATR/5 / multi | $29.58 |
| 2025 | 30m / `<ATR/2` / ATR/5 / multi | $64.99 |

**Cross-test matrix (total $, rows = fit year, columns = tested year):**

| fit\test | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| 2018 | 16.86 | -29.88 | -10.47 | -6.68 | 45.67 | 45.05 | 10.72 | 43.24 |
| 2019 | -5.14 | 2.13 | -0.76 | -5.35 | -9.03 | 0.03 | 6.69 | -2.10 |
| 2020 | -0.79 | 1.49 | 0.80 | -3.15 | -8.98 | -2.77 | 3.95 | -8.91 |
| 2021 | -19.71 | -3.15 | -24.70 | 6.72 | 4.49 | 11.19 | 17.26 | 11.50 |
| 2022 | 1.57 | -26.48 | -3.11 | -25.02 | 53.10 | 29.30 | -23.39 | 9.63 |
| 2023 | 16.86 | -29.88 | -10.47 | -6.68 | 45.67 | 45.05 | 10.72 | 43.24 |
| 2024 | 12.51 | -18.45 | -41.25 | -2.20 | 45.32 | 34.07 | 29.58 | 64.99 |
| 2025 | 12.51 | -18.45 | -41.25 | -2.20 | 45.32 | 34.07 | 29.58 | 64.99 |

**Two findings stand out:**

1. **Independent convergence on a family, not one exact number.** 2018's fit
   and 2023's fit landed on the identical config (30m/`<ATR/2`/ATR-4/multi);
   2024's and 2025's fits independently converged on another identical
   config (ATR-5 instead of ATR-4). But look at the full sequence across all
   four multi-trade years: 2022→ATR/3, 2023→ATR/4, 2024→ATR/5, 2025→ATR/5 —
   the divisor **drifts upward year over year**, it doesn't lock onto one
   value. What's genuinely stable is the family — 30m, `range<ATR/2`,
   multi-trade, divisor somewhere in 3-5 — not a single precise divisor.
   That's still real, non-overfit evidence, but it's evidence for a band,
   not for "ATR/4 or ATR/5" as two specially-privileged numbers. Also worth
   weighing: the search grid itself was only 32 combos with 4 divisor
   choices, so two years agreeing on the best-of-32 is a meaningfully likely
   outcome once 30m/`<ATR/2`/multi is already the dominant pattern in the
   grid — good corroborating evidence, not as statistically striking as
   "agreement out of a huge search space" would be.

2. **A clean regime split, and it isn't simply "before/after 2022."** Both
   robust config families (ATR-4 and ATR-5) are **net positive on 2018,
   2022, 2023, 2024, and 2025** — five years, including 2018, which is not
   adjacent to 2022-2025. Both fail on **2019, 2020, and 2021** — and
   critically, each of those three years' own *in-sample* best fit is barely
   above zero ($2.13, $0.80, $6.72), an order of magnitude weaker than
   2018/2022-2025's in-sample fits ($17-65). There is nothing to find in
   2019-2021 even fitting directly on them, not just a generalization
   failure.

   Pooling either robust config, unchanged, across **all 8 years**: the
   ATR-4 family nets **+$114.51**; the ATR-5 family nets **+$124.57** — both
   net positive across the full 2018-2025 span despite the three bad years,
   because the five good years outweigh them.

### E.5 Conclusion

**30m opening range, `range < ATR/2`, TP=SL=ATR in the 3-5 range,
multi-trade is a validated, good baseline choice** — not a single lucky
fit, but a config family independently rediscovered by fitting on four
different individual years, that transfers cleanly across five of the
eight years tested (2018, 2022-2025) including a non-adjacent year. This
is meaningfully stronger evidence than a single in-sample backtest, and
rules out pure overfitting as the explanation for the original 2022-2026
result — with the caveat that the divisor itself drifts within that band
rather than landing on one fixed number (see the convergence finding
above), and that the search grid
was narrow enough that this shouldn't be read as overwhelming statistical
proof, just real corroborating evidence.

The open problem is **2019-2021**, where nothing in the parameter space
tested has any edge — including each year's own best direct fit. The fixed
14-day-trailing-ATR-scaled TP/SL is the one thing common to every config
tried, and it's tempting to conclude "better TP/SL sizing" is therefore the
whole fix — but that story only cleanly explains **2020**: E.1 showed
trailing ATR specifically lags and mis-sizes trades during a fast
volatility-regime *shift* (the COVID crash). It explains **2019 and 2021**
much less well — both were calm, gradually-grinding melt-up years with no
sudden regime shift, exactly the condition where a 14-day trailing ATR
should track well, not lag. That points to two distinct problems wearing
one label, worth testing as separate, parallel hypotheses rather than
assuming one fix covers both:

1. **Adaptive/regime-aware TP-SL sizing** — a faster-reacting or
   regime-aware volatility estimate, a floor/ceiling on the ATR value used,
   or explicit detection of a stale trailing ATR. Most likely to help
   2020-style volatility-shock years specifically.
2. **A volatility-regime *gate*, separate from sizing** — a pre-trade check
   that skips trading entirely under certain conditions (e.g. a
   persistently low, non-expanding volatility regime), rather than trying
   to size TP/SL correctly for a setup that may just not have real
   continuation behavior in a low-vol grind. Aimed at the 2019/2021-style
   failure mode, which adaptive sizing alone may not fix.

Both are more promising than further tuning the existing fixed-divisor
knobs, which this appendix shows are already close to as good as that
family of rule can get.

One more thing worth weighing before leaning on the "validated" framing in
a live-trading sense: **none of this appendix (or the document as a whole)
models transaction costs.** Multi-trade mode fills 2-3 times on many
trading days, and several of the per-trade mean-R edges found here are
small (+0.02 to +0.08) — real slippage and commissions could erode a
meaningful fraction of that before any TP/SL improvement is even
considered.

---

## Caveats — What's Not Done Yet

- **Single ticker.** Every number in this document is SPY only. Earlier,
  simpler configs were checked across all 8 available tickers and showed
  meaningful cross-ticker variation (some tickers reliably positive, one or
  two reliably negative) — the current final config has not been re-run
  across the full ticker universe.
- **~~No out-of-sample validation~~ — now done, see Appendix E.** The
  30m/`<ATR/2`/multi family (divisor drifting 3-5 across fit years, not one
  fixed number) was independently rediscovered by fitting on 4 different
  individual years and validated net-positive on 5 of 8 years tested (2018,
  2022-2025) via a full leave-one-year-out cross-validation. This is real
  evidence against pure overfitting — but it also surfaced a real,
  unresolved failure mode (next bullet), not a clean bill of health.
- **2019-2021 is an open problem, not just a caveat.** No config in the
  32-combo search space — including each of those years' own best direct
  fit — found any edge in 2019, 2020, or 2021. The fixed 14-day-trailing-ATR
  TP/SL sizing is the common thread across every config tried, but it's only
  a clean explanation for **2020** (it demonstrably mis-sizes trades during a
  fast volatility-regime shift like the COVID crash — Appendix E.1). It
  explains **2019 and 2021** poorly — both were calm, gradually-grinding
  years with no regime shift, exactly the condition a trailing ATR should
  handle fine. Per Appendix E.5, this looks like two distinct problems, not
  one: (a) sizing that lags a volatility shock, fixable with adaptive TP/SL,
  and (b) the breakout-continuation premise possibly just not holding in a
  low-vol grind at all, which better sizing wouldn't fix — likely needs a
  separate regime *gate*, not a resize.
- **No transaction costs, slippage, or spread.** All P&L is gross, 1
  share/contract, fills assumed exactly at signal prices — worth weighing
  seriously here specifically, since multi-trade mode fills 2-3 times on
  many days and several of the validated mean-R edges are small (+0.02 to
  +0.08 per trade); real costs could erode a meaningful share of that before
  any TP/SL work even starts.
- **Gap-fade refinement (Appendix B) not yet folded into the live rule** —
  promising, but untested as an actual pre-trade filter.
- **Data now available for further work:** `data/SPY_full_1_min.parquet`
  (2018-01-02 → 2026-05-01, merged from three separate IB fetches) covers
  the full 8-year span used in Appendix E and can be reused directly for any
  further cross-validation or regime-detection work.
- **Natural next steps**, roughly in priority order: (1) design and test
  adaptive/regime-aware TP-SL sizing *and*, separately, a volatility-regime
  gate (Appendix E.5) — treat these as two hypotheses, not one fix, (2) model
  transaction costs before trusting the small-mean-R years, (3) run the
  validated config across all 8 tickers, (4) implement the gap-fade filter
  (Appendix B) as a real `run_breakout_backtest` parameter and re-validate.
