# 003 — Clean Repo: Keep Core Utilities Only

**Date:** 2026-05-14
**Branch:** feat/003-clean-repo-keep-core-utilities
**Issue:** #7

## Context

After building two strategy experiments (BB Reversal, First Bar Breakout) and a core-utilities layer, we're reverting the repo to a clean foundation. Keep only the IB data download infrastructure and two generic utilities (plotting, resampling). All strategy-specific algorithms, indicators, results files, and one-off scripts are deleted.

## Planned Approach

Remove all strategy/indicator code from src/algo/, src/trade_journal/, and scripts/.
Delete all results/. Edit resample_bars.py (drop resample_to_5min), models.py (drop strategy types), CLAUDE.md.

## Open Questions
