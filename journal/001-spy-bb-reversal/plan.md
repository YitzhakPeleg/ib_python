# 001 — SPY BB Consecutive-Bar Reversal Signal

**Date:** 2026-05-11
**Branch:** feat/001-spy-bb-reversal
**Issue:** #2

## Context

New intraday strategy on SPY 1-min data using Bollinger Bands (20, 2).
Signal fires when two consecutive bars both overshoot a band and the second bar reverses direction:
- **Long**: red bar (Low < lower BB) → green bar (Low < lower BB) → entry at green bar's High
- **Short**: green bar (High > upper BB) → red bar (High > upper BB) → entry at red bar's Low

TP/SL are deferred — this phase produces a labeled DataFrame to study signal frequency and distribution before sizing decisions.

## Planned Approach

1. Apply `calculate_bollinger_bands(df, 20, 2.0)` on full SPY 1-min data
2. Use `.shift(1).over("date")` for previous-bar lookback (prevents cross-day bleed)
3. Detect long/short signals, attach `signal_direction` and `entry_price` columns
4. Save full labeled DataFrame to `results/SPY_bb_reversal_signals.parquet`
5. Save signal-only rows to `results/SPY_bb_reversal_signals.csv` for inspection

## Open Questions

- Where to place TP? Options: fixed R, ATR multiple, band midpoint (mean reversion target), or daily range average (DRA)
- Where to place SL? Options: first bar of the signal (opposite band), entry ± ATR
- Filter by time of day? Many strategies avoid first 30 min and final hour
- Should we require minimum first-bar range to filter low-volatility signals?
