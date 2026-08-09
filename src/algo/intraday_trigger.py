"""Trigger -> confirmation -> entry/SL/TP intraday backtest engine.

Every strategy here follows the same shape, restricted to a session's
opening window (first N bars):

1. Scan bar-by-bar for a strategy-specific TRIGGER condition.
2. Require the very NEXT bar to CONFIRM it. If it doesn't, keep scanning the
   rest of the window for a fresh trigger — the first CONFIRMED signal wins.
3. Enter at the confirmation bar's close. SL is strategy-specific; TP is
   always fixed at exactly 2x the SL distance, so reward:risk >= 2 holds by
   construction rather than by hope.
4. No trigger anywhere in the window -> no trade that day.
5. After entry, watch High/Low for the first bar that touches TP or SL
   (same-bar double-touch counts as SL — conservative, since OHLC alone
   can't tell which was touched first intrabar). If neither is touched by
   the last bar of the day, force-close there: that bar's Low for a long,
   High for a short (assume the worst reasonable fill, not the close).
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import polars as pl


@dataclass
class Signal:
    trigger_idx: int
    confirm_idx: int
    direction: int  # +1 long, -1 short
    entry_price: float
    sl: float
    tp: float


def simulate_trade(high: np.ndarray, low: np.ndarray, close: np.ndarray, signal: Signal) -> dict:
    """Walk forward from the bar after confirmation to the end of the day."""
    n = len(close)
    for i in range(signal.confirm_idx + 1, n):
        if signal.direction == 1:
            hit_tp = high[i] >= signal.tp
            hit_sl = low[i] <= signal.sl
        else:
            hit_tp = low[i] <= signal.tp
            hit_sl = high[i] >= signal.sl
        if hit_sl:  # both-touched tie and SL-only both resolve as SL — check SL first
            return _make_trade(signal, i, signal.sl, "sl")
        if hit_tp:
            return _make_trade(signal, i, signal.tp, "tp")

    last = n - 1  # neither touched by end of day: forced, conservative EOD close
    exit_price = low[last] if signal.direction == 1 else high[last]
    return _make_trade(signal, last, exit_price, "eod")


def _make_trade(signal: Signal, exit_idx: int, exit_price: float, reason: str) -> dict:
    risk = abs(signal.entry_price - signal.sl)
    r_multiple = (exit_price - signal.entry_price) / risk * signal.direction
    return {
        "trigger_idx": signal.trigger_idx,
        "confirm_idx": signal.confirm_idx,
        "direction": signal.direction,
        "entry_price": signal.entry_price,
        "sl": signal.sl,
        "tp": signal.tp,
        "exit_idx": exit_idx,
        "exit_price": exit_price,
        "exit_reason": reason,
        "r_multiple": r_multiple,
    }


# %% Strategy 1 — Bollinger Band touch + reversal confirmation (mean-reversion)
def find_signal_bb_reversal(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bb_lower: np.ndarray,
    bb_upper: np.ndarray,
    window: int,
) -> Optional[Signal]:
    """Long: a bar's Low touches the lower band, next bar closes higher.
    Short: a bar's High touches the upper band, next bar closes lower.
    """
    n = len(close)
    for t in range(window):
        c = t + 1
        if c >= n:
            break
        if np.isnan(bb_lower[t]) or np.isnan(bb_upper[t]):
            continue  # indicator warm-up
        if low[t] <= bb_lower[t] and close[c] > close[t]:
            entry, sl = close[c], min(low[t], low[c])
            risk = entry - sl
            if risk > 0:
                return Signal(t, c, 1, entry, sl, entry + 2 * risk)
        if high[t] >= bb_upper[t] and close[c] < close[t]:
            entry, sl = close[c], max(high[t], high[c])
            risk = sl - entry
            if risk > 0:
                return Signal(t, c, -1, entry, sl, entry - 2 * risk)
    return None


# %% Strategy 1b — Reference-line reversion (mean-reversion around SMA / session VWAP / any single line)
def find_signal_reference_reversion(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    reference: np.ndarray,
    window: int,
) -> Optional[Signal]:
    """Generalizes find_signal_bb_reversal to a single reference line instead
    of a two-sided band — the natural shape for an SMA or a session VWAP,
    neither of which has an "upper/lower" pair the way Bollinger Bands do.

    Long: a bar trades AT OR BELOW the reference, next bar closes back ABOVE
    it (a round trip across the line, not just "closed higher than the
    trigger bar" — the confirmation is specifically a reversion back to fair
    value/trend, which is the actual mean-reversion claim being tested).
    Short: the mirror image.
    """
    n = len(close)
    for t in range(window):
        c = t + 1
        if c >= n:
            break
        if np.isnan(reference[t]) or np.isnan(reference[c]):
            continue  # indicator warm-up
        if low[t] <= reference[t] and close[c] > reference[c]:
            entry, sl = close[c], min(low[t], low[c])
            risk = entry - sl
            if risk > 0:
                return Signal(t, c, 1, entry, sl, entry + 2 * risk)
        if high[t] >= reference[t] and close[c] < reference[c]:
            entry, sl = close[c], max(high[t], high[c])
            risk = sl - entry
            if risk > 0:
                return Signal(t, c, -1, entry, sl, entry - 2 * risk)
    return None


# %% Strategy 2 — Opening range breakout + continuation confirmation (momentum)
def find_signal_orb_breakout(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    range_bars: int,
    window: int,
) -> Optional[Signal]:
    """The first `range_bars` bars fix OR_high/OR_low (no trades taken yet —
    a "running" range recomputed every bar would degenerate into near-
    guaranteed breaks of a 1-bar range). Trigger = a later bar trades outside
    that fixed range; confirm = the next bar closes even further past it.
    """
    n = len(close)
    if range_bars >= n:
        return None
    or_high = float(np.max(high[:range_bars]))
    or_low = float(np.min(low[:range_bars]))
    for t in range(range_bars, window):
        c = t + 1
        if c >= n:
            break
        if high[t] > or_high and close[c] > high[t]:
            entry, sl = close[c], or_low
            risk = entry - sl
            if risk > 0:
                return Signal(t, c, 1, entry, sl, entry + 2 * risk)
        if low[t] < or_low and close[c] < low[t]:
            entry, sl = close[c], or_high
            risk = sl - entry
            if risk > 0:
                return Signal(t, c, -1, entry, sl, entry - 2 * risk)
    return None


# %% Strategy 3 — Candlestick pattern + follow-through confirmation
def find_signal_candlestick_confirm(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pattern_direction: np.ndarray,
    window: int,
) -> Optional[Signal]:
    """Trigger = any directional CDL* pattern fires (pattern_direction != 0,
    precomputed upstream as the sign of the strongest pattern firing that
    bar). Confirm = the next bar closes beyond the trigger bar's extreme in
    the predicted direction — real TA practice never trades the raw pattern
    alone, which is exactly the step Phase 1's pattern_reliability.py never
    tested.
    """
    n = len(close)
    for t in range(window):
        c = t + 1
        if c >= n:
            break
        d = pattern_direction[t]
        if d == 0:
            continue
        if d == 1 and close[c] > high[t]:
            entry, sl = close[c], low[t]
            risk = entry - sl
            if risk > 0:
                return Signal(t, c, 1, entry, sl, entry + 2 * risk)
        if d == -1 and close[c] < low[t]:
            entry, sl = close[c], high[t]
            risk = sl - entry
            if risk > 0:
                return Signal(t, c, -1, entry, sl, entry - 2 * risk)
    return None


# %% Orchestration — run one strategy across every (ticker, date) session
def run_backtest(
    bars: pl.DataFrame,
    find_signal_fn: Callable[..., Optional[Signal]],
    extra_cols: Optional[list[str] | dict[str, str]] = None,
) -> pl.DataFrame:
    """bars: multi-ticker, multi-day 1-min OHLC (+ any indicator columns the
    strategy needs, e.g. bb_lower/bb_upper or pattern_direction — already
    computed on each ticker's own continuous series upstream), sorted by
    ticker/DateTime, with a `date` column. find_signal_fn's strategy-specific
    parameters (window, range_bars, ...) should already be bound via
    functools.partial before being passed in here.

    extra_cols maps find_signal_fn keyword arguments to bars columns — a
    plain list assumes kwarg name == column name (e.g. `pattern_direction`);
    a dict lets a generically-named kwarg (e.g. `reference`) pull from a
    differently-named column per strategy (e.g. `sma` or `vwap`), so the same
    find_signal_reference_reversion function works for either without a
    column-renaming hack at the call site.
    """
    col_by_kwarg = extra_cols if isinstance(extra_cols, dict) else {c: c for c in (extra_cols or [])}

    trades = []
    for (ticker, date), day in bars.group_by(["ticker", "date"], maintain_order=True):
        high = day["High"].to_numpy().astype(np.float64)
        low = day["Low"].to_numpy().astype(np.float64)
        close = day["Close"].to_numpy().astype(np.float64)
        extra = {kwarg: day[col].to_numpy() for kwarg, col in col_by_kwarg.items()}

        signal = find_signal_fn(high=high, low=low, close=close, **extra)
        if signal is None:
            continue

        trade = simulate_trade(high, low, close, signal)
        trade["ticker"] = ticker
        trade["date"] = date
        trades.append(trade)

    return pl.DataFrame(trades)
