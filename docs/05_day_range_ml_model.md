# 05 - Day-Range Neural Net & ML-Driven Trade Decisions (SPY + 17 tickers)

## Overview

[Doc 04](04_range_breakout_strategy.md)'s realistic account-level P&L
simulation (`examples/range_breakout_realistic_pnl.py`) showed the
rule-based opening-range breakout — even Appendix G's best config
(+$196.78 gross, 1 share/contract, 8 years) — isn't viable on a $10K
retail account once commissions and share-count constraints are modeled
honestly. That result motivated a pivot: instead of tuning the rule-based
engine further, train a neural net to predict something the rule-based
strategy can actually use — a data-driven TP — and separately test whether
an interpretable model could make the *trade decision itself* better than
a fixed ATR rule.

Two independent threads, both landing here:

1. **A GRU regressor** predicting the day's remaining high/low from 14
   days of history plus the opening 30 minutes — used to replace the fixed
   `tp_atr_divisor` in `run_breakout_backtest` with a day-specific,
   data-driven TP (Appendix C).
2. **An interpretable decision tree** predicting the trade decision
   (long/short/no_trade) and a TP *before* any breakout trigger, from
   hand-engineered daily/opening-range features (Appendix D).

Code: `src/ml/` (features, dataset, losses, model, metrics, conformal),
`examples/train_day_range_nn.py`, `examples/evaluate_ensemble_and_conformal.py`,
`examples/train_stacking_ensemble.py`, `examples/model_driven_breakout_pnl.py`,
`examples/train_direction_tree.py`. The `tp_model_pred` mode added to
`src/algo/range_breakout.py` is the integration point between the GRU and
the existing breakout engine.

No look-ahead discipline throughout: every feature uses only data known
strictly before the value it's predicting (14-day daily window ends the
day before; intraday features come only from the first 6 bars; ATR is
Wilder(14) shifted one session forward, same convention as doc 04).

---

## Part 1 — The Day-Range GRU

### 1.1 Architecture

Two small branches concatenated into an MLP head:

```
daily:     14 days x 5 features -> GRU(5 -> 64) -> last hidden state (64-dim)
intraday:  first 6 x 5-min bars x 5 features -> Flatten(30) -> Linear(30->32) -> ReLU -> Dropout(0.2)
head:      concat(64+32=96) -> Linear(96->64) -> ReLU -> Dropout(0.2) -> Linear(64->2)
output:    [pred_high_ret, pred_low_ret]
```

20,962 parameters. A GRU (Gated Recurrent Unit) reads the 14-day sequence
one step at a time, keeping a running hidden-state summary updated at each
step via learned gates — a lighter cousin of an LSTM (one memory instead
of two), plenty for a 14-step sequence.

A **Transformer** variant was also built (unified 20-token sequence,
segment embeddings for daily vs. intraday, sinusoidal positional encoding,
CLS-token pooling, Pre-LN). It was debugged through three separate,
well-established fixes (input centering, LR warmup, Pre-LN) and never got
past a loss plateau roughly double the GRU's error (train_loss ~0.0365,
mae_high ~0.59-0.61% vs. the GRU's ~0.33%). Diagnosed as likely too
data-hungry for ~8,246 training examples — **parked, not actively pursued
further**, not a bug to keep chasing.

### 1.2 Feature encoding

Daily and intraday bars share an **identical 5-feature schema** (what
makes concatenation into one sequence possible for the Transformer
variant, and keeps the two branches directly comparable for the GRU):

| feature | formula |
|---|---|
| `open_ret` | `Open / prev_close - 1` |
| `high_ret` | `High / prev_close - 1` |
| `low_ret` | `Low / prev_close - 1` |
| `close_ret` | `Close / prev_close - 1` |
| `log_vol_ratio` | `ln(Volume / prev_Volume)` — daily: prior day; intraday: prior **full day** (a fixed reference across the whole intraday window, not each bar's own predecessor) |

**A finding worth keeping**: the first version normalized prices as a
plain ratio (`Open/prev_close`, clustered ~1.00) rather than a return
(`Open/prev_close - 1`, clustered ~0.00). Both the GRU and the Transformer
regressed from ~0.33% MAE to ~0.59-0.60% MAE under the ratio form — same
information, worse-conditioned for gradient descent (near-zero variance
around a non-zero mean). Switching to the return form recovered
performance immediately. Pure representation/conditioning issue, not a
data or architecture problem.

### 1.3 Loss function

An asymmetric ("lean inside, but stay close") loss, plus a range-width
term to stop the model gaming the asymmetry with a uselessly narrow
prediction:

```
diff_high = pred_high - true_high      # >0 = predicted too high (bad)
diff_low  = true_low - pred_low        # >0 = predicted too low (bad)

term(d) = lambda_bad * relu(d)^p + lambda_good * relu(-d)^p

loss = mean[ term(diff_high) + term(diff_low)
             + range_weight * |pred_range - true_range|^p ]
```

Best config: `lambda_bad=1.7, lambda_good=1.0, p=1, range_weight=1.0`
(see Appendix A for the sweep this came from). A plain symmetric L2 loss
was tried as a control and came out worse on every metric (MAE, inside/
outside skew) — the asymmetry is doing real work.

### 1.4 Baseline results (SPY 2025, 250 held-out days)

Trained on 8,246 examples across 18 tickers (SPY 2018-2024; AAPL, AMZN,
AVGO, GOOG, IBKR, MSFT, NVDA, JPM, UNH, XOM, PG, HD, CAT, VZ, DIS, DKNG,
ZION using whatever years each ticker's data covers, restricted so no
ticker leaks SPY's own held-out 2025). Validated on SPY 2025 only — a full
year never seen by the model, for SPY or its own training window.

| | MAE high | MAE low | $ high | $ low | inside | outside |
|---|---|---|---|---|---|---|
| GRU | 0.329% | 0.405% | $1.94 | $2.46 | 14.8% | 6.4% |
| Heuristic (prior-window range + 0.25xATR) | 0.343% | 0.429% | $2.03 | $2.58 | 3.6% | 26.0% |

("inside" = both bounds land conservatively inside the true range; "outside"
= both bounds miss the risky way — the case the asymmetric loss targets.)

Re-running the identical config with a fresh seed later in this work
(Appendix A.2) gave mae_high=0.318%/$1.87, mae_low=0.399%/$2.42,
inside=18.4%, outside=8.0% — a reminder that single-seed runs on a
250-day validation set carry real noise; treat any single run's exact
numbers as approximate, not exact.

---

## Part 2 — Uncertainty: Ensembling + Conformal Prediction

Motivating question: given one prediction, how much should it be
believed? Two complementary answers, both built on 5 GRU copies trained
identically except for random seed (42-46):

**Ensemble spread.** Per-day standard deviation across the 5 models'
predictions. Measured correlation with actual |error| on SPY 2025:
high=0.34, low=0.28 — a real but moderate signal, not a sharp threshold.
The ensemble mean essentially matched the single model on raw accuracy
(mae_high=0.327%/$1.92, mae_low=0.392%/$2.38, inside=16.0%, outside=8.0%)
— averaging didn't change accuracy much here, but the *disagreement*
across members is the useful output.

**Split-conformal prediction** (`src/ml/conformal.py`). Calibrates the
`alpha/2` and `1-alpha/2` quantiles of the **signed** residual
(`pred - true`) on a calibration set, then wraps a future point prediction
in `[pred - q_hi, pred - q_lo]` for a distribution-free coverage
guarantee. Signed (not `|residual|`) deliberately, since the error
distribution is right-skewed (see the per-day error percentiles in the
published presentation artifact — fat right tail, especially on the low
side).

Calibrated on the first half of SPY 2025 (chronologically), tested on the
second half:

| target | coverage high | coverage low | avg width high | avg width low |
|---|---|---|---|---|
| 80% | 91.2% | 88.0% | $6.41 | $8.63 |
| 90% | 96.8% | 95.2% | $8.58 | $11.02 |

Both targets were **over-covered** — the guarantee held, with room to
spare. Likely cause: a chronological split of one calendar year is weaker
than an i.i.d. shuffle, and the calibration half appears to have carried
somewhat more realized volatility than the test half. A rolling or
interleaved calibration scheme is the natural fix if narrower intervals
are ever worth the added complexity.

A full write-up with diagrams (architecture, loss regions, error
percentile strip) was published as an HTML presentation during this work;
not checked into the repo (an Artifact, not a file), but the content above
is the durable summary.

---

## Appendix A — Hyperparameter Sweeps

Per an explicit prioritization early in this work: more intraday bars was
ruled out as a direction *not* worth pursuing ("I don't want to do this,
especially [more intraday bars]"); loss tuning and model capacity were
prioritized instead. Both sweeps below respect that.

### A.1 `intraday_bars` x `range_weight` x `lambda_good` — 36 configs

Single seed (42) per cell, `lambda_good in {0.7, 1.0, 1.3}`,
`range_weight in {0.5, 1.0, 1.5}`, `intraday_bars in {3, 4, 5, 6}`, bars=6
being the existing default. Ranked by MAE$/outside_frac, not raw loss
(loss isn't comparable across different `range_weight`/`lambda_good`
settings, since those change the loss formula's own magnitude).

Group means:

| `intraday_bars` | mean outside% | mean inside% |
|---|---|---|
| 3 | 10.6 | 14.1 |
| 4 | 9.2 | 15.1 |
| 5 | 9.4 | 15.4 |
| 6 (default) | 13.4 | 14.7 |

| `range_weight` | mean outside% | mean inside% |
|---|---|---|
| 0.5 | 8.1 | 17.5 |
| 1.0 | 12.4 | 13.8 |
| 1.5 | 11.5 | 13.2 |

| `lambda_good` | mean outside% | mean inside% |
|---|---|---|
| 0.7 | 7.5 | 18.1 |
| 1.0 | 11.0 | 14.5 |
| 1.3 | 13.5 | 11.8 |

**Conclusion**: `intraday_bars` doesn't matter — 3/4/5/6 are within noise
of each other (confirms the earlier decision not to chase this axis).
`range_weight`/`lambda_good` control a real accuracy-vs-safety tradeoff:
lower values of both push toward fewer risky "outside" days at a modest
accuracy cost.

### A.2 `range_weight` x `lambda_good` focused sweep — 9 configs

`intraday_bars` fixed at 6, same seed (42), `range_weight` and
`lambda_good` both in `{0.5, 1.0, 1.5}`:

| rw | lg | MAE high% | MAE low% | $ high | $ low | combined $ | inside% | outside% |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 0.363 | 0.444 | 2.14 | 2.68 | 4.81 | **23.6** | **6.0** |
| 0.5 | 1.0 | 0.355 | 0.395 | 2.09 | 2.40 | 4.49 | 20.4 | 7.6 |
| 0.5 | 1.5 | 0.341 | 0.401 | 2.00 | 2.43 | 4.43 | 12.4 | 10.0 |
| 1.0 | 0.5 | 0.323 | 0.404 | 1.90 | 2.44 | 4.34 | 13.2 | 10.4 |
| **1.0** | **1.0** | **0.318** | **0.399** | **1.87** | **2.42** | **4.28** | 18.4 | 8.0 |
| 1.0 | 1.5 | 0.627 | 0.662 | 3.74 | 3.99 | 7.73 | 6.0 | 30.8 |
| 1.5 | 0.5 | 0.326 | 0.394 | 1.91 | 2.39 | 4.30 | 11.2 | 13.6 |
| 1.5 | 1.0 | 0.323 | 0.397 | 1.90 | 2.41 | 4.31 | 10.4 | 20.0 |
| 1.5 | 1.5 | 0.342 | 0.407 | 2.02 | 2.47 | 4.49 | 10.0 | 15.2 |

The `rw=1.0, lg=1.5` row is a **broken single-seed run** (MAE roughly
doubled, outside_frac spiked to 30.8% — an outlier vs. every neighboring
cell, not a real finding).

**Conclusion**: no single winner, a clean Pareto frontier instead —
`(1.0, 1.0)` is most accurate (current default), `(0.5, 0.5)` is safest,
`(0.5, 1.0)` a genuine middle point. Feeds directly into Appendix B.

---

## Appendix B — Combining Models (Stacking) — Not Adopted

Tested whether a *learned* combination of the 3 Pareto models from A.2
(`(0.5,0.5)` safe, `(1.0,1.0)` accurate, `(0.5,1.5)` bold) could beat
simple averaging. A small stacking combiner (linear, and a 1-hidden-layer
MLP) took the 3 models' 6 outputs and learned a final `[high, low]`,
trained on the first 60% of SPY 2025 chronologically (with a further
inner train/val split for early stopping) and evaluated on the remaining
100 held-out days:

| model | combined $ MAE | inside% | outside% |
|---|---|---|---|
| stacked_linear | 4.71 | 7 | 28 |
| stacked_mlp | 8.00 | 29 | 0 |
| **simple_average** | **3.58** | 8 | 8 |
| **best_single (1.0,1.0)** | **3.56** | 12 | 12 |
| base (0.5,0.5) | 3.95 | 13 | 8 |
| base (0.5,1.5) | 3.70 | 6 | 15 |

Both learned combiners lost clearly to simple averaging, even with proper
early stopping (the MLP variant collapsed to a degenerate "always predict
a huge range" strategy — 0% outside, but ~2x everyone else's MAE).
**Conclusion**: with only ~150 combiner-training days and 3 highly
correlated base predictions (all predicting the same underlying
quantity), there isn't enough independent signal for a learned combiner
to beat an unweighted average. Simple averaging of the 3 mildly beats the
single best model on "outside" rate (8% vs. 12%) at flat accuracy cost —
worth using, but not worth the extra complexity of a learned combiner at
this data scale.

---

## Appendix C — Model-Driven TP for the Opening-Range Breakout

The GRU's target — the remainder of the session's high/low **after the
same 30-minute opening window** — lines up exactly with
`opening_range_minutes=30` in `src/algo/range_breakout.py`. Added
`tp_model_pred` as a fifth TP-sizing mode: TP becomes the model's
predicted remaining extreme (long: `pred_high`, short: `pred_low`)
instead of a fixed `ATR/N`. SL is left untouched (still the opposite
range edge) so the comparison isolates the TP change. A trade is skipped
if the model predicts no further room beyond entry — doubling as a
filter, not just a sizing rule.

Fair-comparison constraint: the model only has one genuinely
out-of-sample year (SPY 2025), so **both** the model-driven and the
fixed-ATR baseline below are restricted to 2025 only — comparing against
doc 04/the realistic-P&L script's own 2018-2025 numbers would be
in-sample vs. out-of-sample, not a fair test.

`examples/model_driven_breakout_pnl.py`, same cost assumptions as
`examples/range_breakout_realistic_pnl.py` ($10K start, 1% risk/trade,
IBKR-style commissions, slippage):

| | fixed-ATR baseline (TP=ATR/7, SL=ATR/5) | model-driven TP (SL=ATR/5, unchanged) |
|---|---|---|
| trades | 799 | 822 |
| win rate | 60.2% | **23.0%** |
| gross P&L | $280.84 | **$498.07** (+77%) |
| commission | $3,995 | $4,110 |
| net P&L | -$3,714.16 | -$3,611.93 |
| max drawdown | 37.2% | 36.1% |

**Two findings.** (1) Model-driven TP captures real edge — gross P&L up
77% with essentially the same trade count and SL, so the gain is purely
from smarter TP placement. Win rate collapsing 60%→23% is the mechanism:
the model's TP sits farther out on average than ATR/7, trading frequent
small wins for fewer, bigger ones (consistent with the right-skewed error
distribution from Part 2). (2) **Both configs are still deeply
net-negative** — commissions (~$4,000 on ~800 trades) dominate a gross
edge of only $280-$498 either way. This reframes doc 04's original
"not viable on $10K" finding: it **isn't a TP-sizing problem**, a smarter
TP alone can't fix it — it's a trade-frequency-vs-account-size problem.
~800 trades/year at $2.50+/leg minimum commission needs a much bigger
edge or a much bigger account to amortize.

**Natural next step** (not yet built): turn the "skip if no predicted
room" check into a real high-conviction filter (require room beyond some
$ or ATR-relative threshold, not just >0) to cut trade count sharply — a
direct test of whether *filtering*, not just TP sizing, closes the
commission gap.

---

## Appendix D — Interpretable Direction+TP Tree Model — Not Adopted

A structurally different approach: instead of a GRU that only sizes TP
after a breakout has already triggered, train an interpretable model
(`examples/train_direction_tree.py`) to make the trading decision itself
— direction (long/short/no_trade) and TP — **before** any trigger, from
~9 hand-engineered daily/opening-range features (prior-day/5-day returns,
ATR%, volume ratio, gap%, opening-range width relative to ATR, opening
candle color, relative volume). SL stays mechanical (opposite opening-
range edge, same convention as elsewhere).

**Labels, honestly built.** For every day, the actual remainder-of-session
bars (5-minute, after the same 30-minute opening window) are walked
bar-by-bar in both directions, tracking the max favorable excursion (in
R-multiples of the opening range's own width) reached *before* the
opposite edge would have stopped the trade out (`_max_favorable_r`) —
respects first-touch order, unlike deriving a label from session-level
high/low alone. `direction = long/short` if that side reached >=1R before
being stopped, else `no_trade`.

**Execution-style caveat**: this model enters immediately at
`or_high`/`or_low` the instant the opening window closes, rather than
waiting for a bar to *close* beyond the range like the trigger-scan
engine in `algo.range_breakout` does — a materially more aggressive entry
rule, so trade counts between this and Appendix C aren't directly
comparable.

Trained on SPY 2018-2024 (1,745 days), validated on SPY 2025 (250 days).
Label balance: long 439 (21%), no_trade 1,203 (58%), short 436 (21%).

| | accuracy | notes |
|---|---|---|
| Decision tree (max_depth=4, balanced) | 41% | below the 58% majority-class ("always no_trade") baseline |
| Random forest (300 trees) | 37% | same pattern |

Feature importance (forest): `opening_direction` dominates (0.445), then
`or_width_over_atr` (0.110), `relative_volume` (0.081) — the tree's
top splits are genuinely interpretable ("green open → long unless the
range is already too wide relative to ATR; red open → short unless
volume is unusually quiet"), a sensible momentum-with-an-exhaustion-guard
heuristic, even though the overall fit is weak.

TP regressor (predicted R-multiple): MAE 0.944R against a mean actual
outcome of 2.167R — a ~44% relative error, too noisy to trust the
magnitude prediction much.

**Realistic P&L** (SPY 2025 only, same $10K/cost assumptions as Appendix
C): 193 trades (far fewer — one decision per day, no intraday scanning),
win rate 32.1%, **gross P&L -$826.13** (negative, unlike Appendix C's
fixed-ATR and model-driven variants, which were both gross-positive),
commission $965.00, net -$1,791.13, max drawdown 18.4% (much lower than
Appendix C's ~36-37%, a direct consequence of trading ~4x less often).

**Conclusion**: predicting *direction* from these features is a harder
problem than predicting *range magnitude* was — accuracy is below the
trivial "don't trade" baseline, precision on individual long/short calls
is only ~33-35% (`class_weight="balanced"` traded precision for recall to
avoid the tree collapsing to always-no_trade), and critically, gross P&L
is negative — this isn't a costs problem like Appendix C, the underlying
directional signal itself isn't there yet at this feature/model scale.
The lower drawdown (fewer, filtered trades) is a genuine upside of the
approach's selectivity, but doesn't offset the weak core signal.

**Not yet tried, in rough priority order**: (1) drop
`class_weight="balanced"` for a precision-favoring version (fewer,
higher-conviction calls); (2) feed the GRU's own `pred_high`/`pred_low`
in as additional engineered features, letting the tree access the
already-validated range signal instead of working from raw daily stats
alone; (3) raise `MIN_R_FOR_TRADE` to demand a bigger edge before
labeling a day tradeable, thinning out marginal/noisy labels.

---

## Caveats — What's Not Done Yet

- **Every P&L number in Appendices C and D is SPY-only, 2025-only** (the
  model's one genuinely out-of-sample year) — not re-run across the other
  17 training tickers or a longer window.
- **Single-seed noise is real and repeatedly observed** — Appendix A.2's
  broken `(rw=1.0, lg=1.5)` cell, and the ~$0.02-0.07 MAE swings between
  identically-configured re-runs (Part 1.4's baseline vs. Appendix A.2's
  re-run of the same config) both point the same way: any single number
  in this document should be read as approximate, and multi-seed
  confirmation (as done for the 5-model ensemble in Part 2) is the fix
  where it matters.
- **Conformal intervals over-covered** on the one chronological
  calibration/test split tried — a rolling or interleaved calibration
  scheme hasn't been tried yet and would likely tighten the intervals.
- **Appendix C's positive finding (model-driven TP captures real gross
  edge) is not yet followed up with the filtering idea** that would
  actually test whether it can close the net-P&L gap — the natural
  immediate next step.
- **Appendix D's tree model has three concrete untried improvements**
  (see above) before it should be considered a settled negative result
  rather than a first pass.
- **Cross-sectional / regime features** (deferred early in this work,
  "we can discuss later") — still not explored; could plausibly help both
  the GRU and the tree model, especially the latter given how weak its
  current daily/opening-range-only feature set turned out to be.
- **The two threads (GRU-TP and tree-direction) haven't been combined** —
  e.g. using the tree's direction call to gate which side the GRU's TP
  applies to, instead of always taking whichever side breaks out first.
