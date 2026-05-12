# 002 — Core Utilities: Resampling, Plotting & Trade Journal

**Date:** 2026-05-12
**Branch:** feat/002-core-utilities
**Issue:** #5

## Context

Shared utility layer needed across all strategy scripts. Three areas identified:
1. **Resampling** — already complete (`resample_to_timeframe(df, "5m")` etc. in `src/algo/resample_bars.py`)
2. **Plotting** — `plot_bars` exists but has no trade marker support
3. **Trade journal** — no Python module exists; only markdown stubs

## Planned Approach

### Plotting enhancement (`src/visualization/plotting.py`)
Add optional `trades` parameter to `plot_bars`:
- Entry markers: triangle-up/down at entry_price by direction
- Exit markers: X at exit_price, green/red by outcome
- Dashed connecting line per trade

### Trade journal module (`src/trade_journal/`)
- `write_trades(trades_df, path)` — append/create CSV
- `read_trades(path) -> pl.DataFrame` — read back with correct types
- `plot_journal(trades_df, title)` — 2-panel: cumulative PnL + per-trade bar chart

## Open Questions

- Should trade markers on `plot_bars` also accept the BB reversal `BBReversalTradeResult` format directly, or only the generic CSV schema?
- For `plot_journal`: should it show long/short split as separate series?
- Additional utilities to consider: `src/utils/stats.py` (shared performance metrics), `src/utils/session.py` (RTH filter)
