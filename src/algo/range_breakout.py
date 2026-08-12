"""ATR-band-filtered opening-range breakout strategy.

Rule, stated as a sentence:

1. Compute a 14-session Wilder ATR from daily bars, known as of the PRIOR
   session's close (see algo.hammer_reversal.compute_daily_atr — same
   no-lookahead reasoning applies here).
2. Take the first 15-minute bar's range (High - Low), unrounded. If it is
   NOT between ATR/4 and ATR/2 (inclusive), skip the day — this setup wants
   a moderately-active open: quieter than ATR/4 and there's nothing to
   break out of, wider than ATR/2 and the day's likely already spent its
   move before the signal window even starts.
3. Otherwise, switch to 5-minute bars and scan the rest of the session
   (bars starting >=15 and <90 minutes after the open) for the first bar
   that CLOSES outside the opening range: closing above -> go long, closing
   below -> go short. This is a continuation/momentum trigger (trade WITH
   the breakout), the mirror image of hammer_reversal's fade. Optionally
   (require_open_outside), the bar's OPEN must also already be outside the
   range on the same side — the whole bar's body outside it, not just a
   close that pokes through.
4. Entry at that bar's close (immediate — no stop-order fill phase, unlike
   hammer_reversal, since the close itself is the confirmation). TP defaults
   to a fixed $1 in the trade's favor, and SL defaults to the OPPOSITE edge
   of the opening 15-minute range (long: SL = or_low, short: SL = or_high) —
   both are configurable on run_breakout_backtest: TP as a fixed dollar
   amount or as ATR/N (tp_atr_divisor), and SL as a fixed dollar amount or
   mirroring the TP distance (sl_equals_tp), instead of the range edge.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import polars as pl

from algo.hammer_reversal import (
    OPENING_RANGE_MINUTES,
    SIGNAL_WINDOW_MINUTES,
    _iter_atr_sessions,
)
from algo.intraday_trigger import Signal, simulate_trade

TP_DOLLARS = 1.0
RANGE_ATR_LOW_DIVISOR = 4  # opening range must be >= ATR / this
RANGE_ATR_HIGH_DIVISOR = 2  # opening range must be <= ATR / this


def _iter_band_sessions(
    minute_bars: pl.DataFrame,
    daily_bars: pl.DataFrame,
    signal_timeframe: str,
    opening_range_minutes: int = OPENING_RANGE_MINUTES,
    range_atr_low_divisor: Optional[float] = RANGE_ATR_LOW_DIVISOR,
    range_atr_high_divisor: Optional[float] = RANGE_ATR_HIGH_DIVISOR,
) -> Iterator[tuple[str, object, float, float, float, pl.DataFrame]]:
    """_iter_atr_sessions filtered to this strategy's rule: the opening
    range must fall within [ATR/range_atr_low_divisor,
    ATR/range_atr_high_divisor] (unrounded) — moderately active, not dead
    and not already-spent, by default. Either bound can be relaxed to None
    for a one-sided filter (e.g. low_divisor=None, high_divisor=4 tests
    "range < ATR/4" with no floor).
    """
    for ticker, date, or_high, or_low, atr_prior, day_signal in _iter_atr_sessions(
        minute_bars, daily_bars, signal_timeframe, opening_range_minutes
    ):
        opening_range = or_high - or_low
        lo = 0.0 if range_atr_low_divisor is None else atr_prior / range_atr_low_divisor
        hi = (
            np.inf
            if range_atr_high_divisor is None
            else atr_prior / range_atr_high_divisor
        )
        if not (lo <= opening_range <= hi):
            continue
        yield ticker, date, or_high, or_low, atr_prior, day_signal


@dataclass
class BreakoutTrigger:
    trigger_idx: int
    direction: int  # +1 long, -1 short
    entry_price: float  # the trigger bar's own close
    sl: float
    tp: float


def _find_breakout_trigger(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    minutes_since_open: np.ndarray,
    or_high: float,
    or_low: float,
    tp_distance: float,
    sl_distance: Optional[float],
    opening_range_minutes: int = OPENING_RANGE_MINUTES,
    signal_window_minutes: int = SIGNAL_WINDOW_MINUTES,
    require_open_outside: bool = False,
    start_idx: int = 0,
    tp_range_projection: bool = False,
    tp_sl_trigger_bar_multiple: Optional[float] = None,
) -> Optional[BreakoutTrigger]:
    """First-signal-wins scan for a bar closing outside [or_low, or_high],
    starting no earlier than start_idx (used to resume scanning after a
    prior trade on the same day has already closed — see
    run_breakout_backtest's allow_multiple_trades_per_day).

    require_open_outside, if set, additionally requires the bar's OPEN to
    already be outside the range (same side as the close) — the whole bar's
    body outside the range, not just a close that pokes through it.

    tp_distance: TP sits this far from entry, in the trade's favor. Ignored
    if tp_range_projection or tp_sl_trigger_bar_multiple is set.
    sl_distance: if given, SL sits this far from entry, against the trade;
    if None, SL is the opposite edge of the opening range (or_low for a
    long, or_high for a short) — the strategy's original rule. Ignored if
    tp_sl_trigger_bar_multiple is set.

    tp_range_projection, if set, overrides tp_distance entirely: TP is
    placed the same distance beyond entry as entry itself sits from the
    OPPOSITE edge of the opening range — i.e. mirror entry's distance to
    the far edge, projected forward. Long: entry + (entry - or_low) =
    2*entry - or_low. Short: entry - (or_high - entry) = 2*entry - or_high.
    Same-day-reactive (scales with today's own realized range), unlike the
    trailing-ATR-based tp_distance which only updates once per day from a
    14-day-old average.

    tp_sl_trigger_bar_multiple, if set, overrides BOTH tp_distance/
    tp_range_projection and sl_distance: both TP and SL sit
    tp_sl_trigger_bar_multiple * (the trigger bar's own High - Low) from
    entry, symmetrically. The most local/reactive sizing tried yet — scales
    with a single 5-min bar's own realized range rather than the 14-day
    ATR or the 30-min opening range.
    """
    in_window = (
        (minutes_since_open >= opening_range_minutes)
        & (minutes_since_open < signal_window_minutes)
        & (np.arange(len(close)) >= start_idx)
    )
    for i in np.flatnonzero(in_window):
        if close[i] > or_high and (not require_open_outside or open_[i] > or_high):
            entry = close[i]
            if tp_sl_trigger_bar_multiple is not None:
                d = tp_sl_trigger_bar_multiple * (high[i] - low[i])
                return BreakoutTrigger(i, 1, entry, entry - d, entry + d)
            sl = or_low if sl_distance is None else entry - sl_distance
            tp = (2 * entry - or_low) if tp_range_projection else entry + tp_distance
            return BreakoutTrigger(i, 1, entry, sl, tp)
        if close[i] < or_low and (not require_open_outside or open_[i] < or_low):
            entry = close[i]
            if tp_sl_trigger_bar_multiple is not None:
                d = tp_sl_trigger_bar_multiple * (high[i] - low[i])
                return BreakoutTrigger(i, -1, entry, entry + d, entry - d)
            sl = or_high if sl_distance is None else entry + sl_distance
            tp = (2 * entry - or_high) if tp_range_projection else entry - tp_distance
            return BreakoutTrigger(i, -1, entry, sl, tp)
    return None


def run_breakout_backtest(
    minute_bars: pl.DataFrame,
    daily_bars: pl.DataFrame,
    signal_timeframe: str = "5m",
    opening_range_minutes: int = OPENING_RANGE_MINUTES,
    signal_window_minutes: int = SIGNAL_WINDOW_MINUTES,
    range_atr_low_divisor: Optional[float] = RANGE_ATR_LOW_DIVISOR,
    range_atr_high_divisor: Optional[float] = RANGE_ATR_HIGH_DIVISOR,
    tp_dollars: float = TP_DOLLARS,
    tp_atr_divisor: Optional[float] = None,
    sl_dollars: Optional[float] = None,
    sl_atr_divisor: Optional[float] = None,
    sl_equals_tp: bool = False,
    require_open_outside: bool = False,
    allow_multiple_trades_per_day: bool = False,
    tp_range_projection: bool = False,
    tp_sl_trigger_bar_multiple: Optional[float] = None,
) -> pl.DataFrame:
    """minute_bars, daily_bars: multi-ticker OHLC (1-min and 1-day resp.),
    each with a `ticker` column, sorted by ticker/DateTime.

    opening_range_minutes: duration of the first bar used to set
    or_high/or_low (default 15, matching the original rule).

    signal_window_minutes: bars starting >= opening_range_minutes and <
    this are eligible to trigger (default 90, i.e. trading can happen until
    90 minutes after the open — pass e.g. 150 for "until 12:00" on a 9:30
    open, 210 for "until 13:00").

    range_atr_low_divisor / range_atr_high_divisor: the opening range must
    fall within [ATR/low_divisor, ATR/high_divisor] (defaults 4 and 2,
    matching the original rule). Pass None to drop either bound — e.g.
    low_divisor=None, high_divisor=4 tests "range < ATR/4" with no floor.

    tp_atr_divisor, if given, overrides tp_dollars: TP = ATR / tp_atr_divisor
    (e.g. 4 -> ATR/4), computed fresh per session from that day's prior-ATR
    instead of a fixed dollar amount.

    sl_atr_divisor, if given (and sl_equals_tp is not set), makes SL its own
    independent ATR fraction — SL = ATR / sl_atr_divisor — so TP and SL can
    use different divisors (e.g. tp_atr_divisor=4, sl_atr_divisor=5 gives a
    5:4 reward:risk ATR-scaled trade instead of a symmetric one).

    sl_equals_tp, if set, makes SL mirror whatever TP distance was resolved
    above (fixed $ or ATR/N) — a symmetric stop instead of the range edge.
    Takes priority over sl_atr_divisor and sl_dollars. If none of
    sl_equals_tp, sl_atr_divisor, sl_dollars is set, SL falls back to the
    original opposite-range-edge rule.

    require_open_outside: see _find_breakout_trigger — requires the whole
    trigger bar's body (open AND close) outside the range, not just the
    close.

    allow_multiple_trades_per_day: if set, once a trade closes (TP, SL, or
    EOD) the scan resumes on the bar right after its exit, looking for
    another trigger in the same session (still bounded by
    signal_window_minutes) — sequential trades only, never overlapping
    positions, since the next scan can't start until the previous trade has
    already exited.

    tp_range_projection: see _find_breakout_trigger — overrides tp_dollars/
    tp_atr_divisor with a same-day-reactive measured-move target instead.

    tp_sl_trigger_bar_multiple: see _find_breakout_trigger — overrides both
    TP and SL sizing with a symmetric multiple of the trigger bar's own
    High-Low range.
    """
    trades = []
    for ticker, date, or_high, or_low, atr_prior, day_signal in _iter_band_sessions(
        minute_bars,
        daily_bars,
        signal_timeframe,
        opening_range_minutes,
        range_atr_low_divisor,
        range_atr_high_divisor,
    ):
        session_open = day_signal["DateTime"][0]
        minutes_since_open = (
            (day_signal["DateTime"] - session_open).dt.total_minutes().to_numpy()
        )
        open_ = day_signal["Open"].to_numpy().astype(np.float64)
        close = day_signal["Close"].to_numpy().astype(np.float64)

        tp_distance = (
            atr_prior / tp_atr_divisor if tp_atr_divisor is not None else tp_dollars
        )
        if sl_equals_tp:
            sl_distance = tp_distance
        elif sl_atr_divisor is not None:
            sl_distance = atr_prior / sl_atr_divisor
        else:
            sl_distance = sl_dollars
        high = day_signal["High"].to_numpy().astype(np.float64)
        low = day_signal["Low"].to_numpy().astype(np.float64)

        start_idx = 0
        while True:
            trigger = _find_breakout_trigger(
                open_,
                high,
                low,
                close,
                minutes_since_open,
                or_high,
                or_low,
                tp_distance,
                sl_distance,
                opening_range_minutes,
                signal_window_minutes,
                require_open_outside,
                start_idx,
                tp_range_projection,
                tp_sl_trigger_bar_multiple,
            )
            if trigger is None:
                break

            signal = Signal(
                trigger.trigger_idx,
                trigger.trigger_idx,
                trigger.direction,
                trigger.entry_price,
                trigger.sl,
                trigger.tp,
            )
            trade = simulate_trade(high, low, close, signal)
            trade["ticker"] = ticker
            trade["date"] = date
            trade["trigger_time"] = day_signal["DateTime"][int(trigger.trigger_idx)]
            trade["or_high"] = or_high
            trade["or_low"] = or_low
            trades.append(trade)

            if not allow_multiple_trades_per_day:
                break
            start_idx = int(trade["exit_idx"]) + 1

    return pl.DataFrame(trades)
