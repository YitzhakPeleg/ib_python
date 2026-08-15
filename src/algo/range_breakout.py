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
VOLUME_LOOKBACK_DAYS = 14


def _compute_opening_context(
    minute_bars: pl.DataFrame,
    opening_range_minutes: int,
    volume_lookback: int = VOLUME_LOOKBACK_DAYS,
) -> pl.DataFrame:
    """Per (ticker, date): opening_direction (+1 if the opening range's own
    close > its own open — a "green" open, -1 for "red") and
    relative_volume — the opening range's own volume AS A FRACTION OF A
    TYPICAL WHOLE DAY'S volume: opening_volume / the trailing
    volume_lookback-session mean of that day's OWN FULL-SESSION volume
    (shifted one session forward — no lookahead, same convention as ATR).
    E.g. 0.20 means the opening range carried 20% of this ticker's recent
    typical full-day volume. The first volume_lookback sessions of each
    ticker have relative_volume = null (warm-up, mirroring ATR's own
    warm-up window).
    """
    rows = []
    for (ticker,), ticker_minutes in minute_bars.group_by(
        "ticker", maintain_order=True
    ):
        ticker_minutes = ticker_minutes.sort("DateTime").with_columns(
            pl.col("DateTime").dt.date().alias("date")
        )
        for (date,), day_minutes in ticker_minutes.group_by(
            "date", maintain_order=True
        ):
            session_open = day_minutes["DateTime"][0]
            minutes_since_open = (
                (day_minutes["DateTime"] - session_open).dt.total_minutes().to_numpy()
            )
            opening = day_minutes.filter(minutes_since_open < opening_range_minutes)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "opening_direction": 1
                    if float(opening["Close"][-1]) > float(opening["Open"][0])
                    else -1,
                    "opening_volume": float(opening["Volume"].sum()),
                    "daily_volume": float(day_minutes["Volume"].sum()),
                }
            )

    context = pl.DataFrame(rows).sort(["ticker", "date"])
    return context.with_columns(
        (
            pl.col("opening_volume")
            / pl.col("daily_volume")
            .shift(1)
            .rolling_mean(window_size=volume_lookback)
            .over("ticker")
        ).alias("relative_volume")
    )


def _iter_band_sessions(
    minute_bars: pl.DataFrame,
    daily_bars: pl.DataFrame,
    signal_timeframe: str,
    opening_range_minutes: int = OPENING_RANGE_MINUTES,
    range_atr_low_divisor: Optional[float] = RANGE_ATR_LOW_DIVISOR,
    range_atr_high_divisor: Optional[float] = RANGE_ATR_HIGH_DIVISOR,
    min_relative_volume: Optional[float] = None,
) -> Iterator[tuple[str, object, float, float, float, int, float, pl.DataFrame]]:
    """_iter_atr_sessions filtered to this strategy's rule: the opening
    range must fall within [ATR/range_atr_low_divisor,
    ATR/range_atr_high_divisor] (unrounded) — moderately active, not dead
    and not already-spent, by default. Either bound can be relaxed to None
    for a one-sided filter (e.g. low_divisor=None, high_divisor=4 tests
    "range < ATR/4" with no floor).

    min_relative_volume, if given, additionally requires the opening
    range's own relative_volume (see _compute_opening_context) to be at
    least this — skips both volume warm-up sessions (relative_volume is
    null) and quiet-volume sessions.

    Yields (ticker, date, or_high, or_low, atr_prior, opening_direction,
    relative_volume, day_signal) — opening_direction and relative_volume
    are always computed and yielded (regardless of min_relative_volume)
    so callers can use opening_direction as a trigger-direction gate even
    when they don't want a volume filter.
    """
    context_by_key = {
        (row["ticker"], row["date"]): (row["opening_direction"], row["relative_volume"])
        for row in _compute_opening_context(
            minute_bars, opening_range_minutes
        ).iter_rows(named=True)
    }
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
        opening_direction, relative_volume = context_by_key[(ticker, date)]
        if min_relative_volume is not None and (
            relative_volume is None or relative_volume < min_relative_volume
        ):
            continue
        yield (
            ticker,
            date,
            or_high,
            or_low,
            atr_prior,
            opening_direction,
            relative_volume,
            day_signal,
        )


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
    atr_prior: Optional[float] = None,
    sl_at_trigger_extreme: bool = False,
    tp_bar_multiple: Optional[float] = None,
    large_bar_atr_divisor: Optional[float] = None,
    sl_range_edge: Optional[str] = None,
    tp_atr_z: Optional[float] = None,
    require_open_inside: bool = False,
    opening_direction: Optional[int] = None,
    require_direction: Optional[str] = None,
    max_open_excess_atr_z: Optional[float] = None,
    model_pred: Optional[tuple[float, float]] = None,
) -> Optional[BreakoutTrigger]:
    """First-signal-wins scan for a bar closing outside [or_low, or_high],
    starting no earlier than start_idx (used to resume scanning after a
    prior trade on the same day has already closed — see
    run_breakout_backtest's allow_multiple_trades_per_day).

    require_open_outside, if set, additionally requires the bar's OPEN to
    already be outside the range (same side as the close) — the whole bar's
    body outside the range, not just a close that pokes through it.

    require_direction ("with" or "against"), if given (requires
    opening_direction), gates which trade directions are even considered
    based on the OPENING RANGE CANDLE's own color (green if its close >
    its open, red otherwise) — not the trigger bar. "with" only allows a
    long if the opening candle was green / a short if it was red (trade in
    the same direction as the early momentum); "against" only allows a
    long if the opening candle was red / a short if it was green (fade the
    early momentum, mirroring the Quick Flip Scalper's own directional
    gate — see algo.quick_flip_scalper).

    require_open_inside, if set, additionally requires the bar's OPEN to
    still be INSIDE [or_low, or_high] — the opposite of require_open_outside:
    only the single bar that actually crosses from inside to outside counts
    as the trigger, not a later bar that was already fully outside (e.g. a
    continuation bar on a day where the range was breached earlier). Mutually
    exclusive with require_open_outside in practice (both True never matches).
    A hard, all-or-nothing version of max_open_excess_atr_z below.

    max_open_excess_atr_z, if given, is a SOFTER alternative to
    require_open_inside: a bar whose open is already outside the range
    (same side as the close) is only rejected if it's gone too FAR beyond
    the edge — specifically if (open - or_high) for a long / (or_low -
    open) for a short exceeds atr_prior / max_open_excess_atr_z. A bar
    whose open is inside the range always passes regardless (same as
    today); a bar whose open is outside but only marginally so (within
    that ATR fraction of the edge) still counts as a valid trigger, unlike
    require_open_inside which would reject it outright. Smaller z (e.g. 5)
    is a tighter/stricter cap than larger z (e.g. 10). Requires atr_prior.
    Typically used instead of, not together with, require_open_inside/
    require_open_outside (combining them is redundant, not additive).

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

    tp_atr_z, if given (requires atr_prior), adds a plausibility filter
    applied to whichever TP was resolved above (any mode): a candidate is
    skipped (scanning continues for a later bar) if TP falls outside
    [or_low + tp_atr_z * ATR, or_high - tp_atr_z * ATR] — measured from
    the OPPOSITE edge of the opening range for each direction, not from
    entry. E.g. or_low=10, ATR=2, tp_atr_z=1.0: a long's TP > 12 is
    skipped, since or_low to TP is already a bigger move than a typical
    full day (1.0x ATR) makes plausible; tp_atr_z=0.5 would instead cap it
    at TP > 11 (half a typical day's range). Mirror for shorts: TP <
    or_high - tp_atr_z * ATR is skipped. This is a plausibility check on
    the TP level itself, not a comparison of the SL/TP *distance* to ATR.
    tp_atr_z=1.0 reproduces the original (unparameterized) version of this
    filter.

    sl_at_trigger_extreme: a third sizing mode, checked before
    tp_sl_trigger_bar_multiple. SL is the trigger bar's own extreme (low
    for a long, high for a short — NOT scaled, an exact anchor, unlike
    every other SL rule here). If tp_bar_multiple is also given, TP is
    entry +/- tp_bar_multiple * (the trigger bar's own High - Low) — since
    entry is that bar's CLOSE (which can sit inside a wick, not at the
    extreme), the resulting SL distance is generally smaller than the
    bar's full range while TP is scaled off the full range, an
    intentionally asymmetric reward:risk. If tp_bar_multiple is None, TP
    instead mirrors the SL distance from entry (reward:risk 1:1) — same
    idea as sl_range_edge below, but anchored to the trigger bar's own
    extreme instead of an opening-range edge. A candidate bar that closed
    exactly at its own extreme (entry == sl, zero risk) is skipped rather
    than returned, since that's a degenerate trade, not a real signal.

    large_bar_atr_divisor, if given, adds an extra trigger condition on top
    of whichever close-outside-range / require_open_outside check is
    active: the trigger bar's own High-Low must exceed atr_prior /
    large_bar_atr_divisor. A candidate bar that closes outside the range
    but isn't "large" by this measure is skipped, not returned — scanning
    continues for a later, bigger bar. Requires atr_prior to be set.

    sl_range_edge ("near" or "far"), if given, is a fourth sizing mode
    (checked after the three above, before the plain default): SL sits
    exactly at an opening-range edge rather than being scaled off ATR or
    the trigger bar. "near" = the edge that was just broken (or_high for a
    long, or_low for a short) — a tight stop right at the breakout level.
    "far" = the opposite edge (or_low for a long, or_high for a short) —
    the same level the original default rule already uses. TP is then
    forced to mirror that SL distance from entry (reward:risk 1:1),
    overriding tp_distance/tp_range_projection. A candidate whose entry
    already sits at the near edge (zero risk) is skipped.

    model_pred, if given, is a fifth TP mode: (pred_high, pred_low) in
    dollars, from src/ml -- a neural net trained to predict the REMAINDER
    of the session's own high/low after exactly this same opening window
    (see src/ml/dataset.py), from 14 days of prior history plus the
    opening range's own bars. TP becomes that prediction directly (long:
    pred_high, short: pred_low) instead of tp_distance/tp_range_projection.
    SL is untouched -- still resolved by sl_distance/sl_atr_divisor/the
    default opposite-range-edge rule, exactly as when model_pred is None --
    so this isolates the TP-sizing question from everything else. A
    candidate where the model predicts NO further room beyond entry (long:
    pred_high <= entry, short: pred_low >= entry) is skipped rather than
    taken with a non-positive target -- this doubles as a trade filter,
    not just a sizing rule.
    """
    in_window = (
        (minutes_since_open >= opening_range_minutes)
        & (minutes_since_open < signal_window_minutes)
        & (np.arange(len(close)) >= start_idx)
    )
    long_allowed = short_allowed = True
    if require_direction == "with":
        long_allowed, short_allowed = opening_direction == 1, opening_direction == -1
    elif require_direction == "against":
        long_allowed, short_allowed = opening_direction == -1, opening_direction == 1

    for i in np.flatnonzero(in_window):
        if (
            long_allowed
            and close[i] > or_high
            and (not require_open_outside or open_[i] > or_high)
            and (not require_open_inside or or_low <= open_[i] <= or_high)
        ):
            if (
                max_open_excess_atr_z is not None
                and open_[i] > or_high
                and (open_[i] - or_high) > atr_prior / max_open_excess_atr_z
            ):
                continue
            if large_bar_atr_divisor is not None and (high[i] - low[i]) <= (
                atr_prior / large_bar_atr_divisor
            ):
                continue
            entry = close[i]
            if sl_at_trigger_extreme:
                if entry <= low[i]:  # entry closed at its own low: zero risk, skip
                    continue
                sl = low[i]
                tp = (
                    entry + tp_bar_multiple * (high[i] - low[i])
                    if tp_bar_multiple is not None
                    else entry + (entry - sl)
                )
            elif tp_sl_trigger_bar_multiple is not None:
                d = tp_sl_trigger_bar_multiple * (high[i] - low[i])
                sl, tp = entry - d, entry + d
            elif sl_range_edge is not None:
                sl = or_high if sl_range_edge == "near" else or_low
                if entry <= sl:  # zero/negative risk, skip
                    continue
                tp = entry + (entry - sl)
            elif model_pred is not None:
                sl = or_low if sl_distance is None else entry - sl_distance
                tp = model_pred[0]
                if tp <= entry:  # model predicts no further room: skip, not a signal
                    continue
            else:
                sl = or_low if sl_distance is None else entry - sl_distance
                tp = (
                    (2 * entry - or_low) if tp_range_projection else entry + tp_distance
                )
            if tp_atr_z is not None and tp > or_low + tp_atr_z * atr_prior:
                continue
            return BreakoutTrigger(i, 1, entry, sl, tp)
        if (
            short_allowed
            and close[i] < or_low
            and (not require_open_outside or open_[i] < or_low)
            and (not require_open_inside or or_low <= open_[i] <= or_high)
        ):
            if (
                max_open_excess_atr_z is not None
                and open_[i] < or_low
                and (or_low - open_[i]) > atr_prior / max_open_excess_atr_z
            ):
                continue
            if large_bar_atr_divisor is not None and (high[i] - low[i]) <= (
                atr_prior / large_bar_atr_divisor
            ):
                continue
            entry = close[i]
            if sl_at_trigger_extreme:
                if entry >= high[i]:  # entry closed at its own high: zero risk, skip
                    continue
                sl = high[i]
                tp = (
                    entry - tp_bar_multiple * (high[i] - low[i])
                    if tp_bar_multiple is not None
                    else entry - (sl - entry)
                )
            elif tp_sl_trigger_bar_multiple is not None:
                d = tp_sl_trigger_bar_multiple * (high[i] - low[i])
                sl, tp = entry + d, entry - d
            elif sl_range_edge is not None:
                sl = or_low if sl_range_edge == "near" else or_high
                if entry >= sl:  # zero/negative risk, skip
                    continue
                tp = entry - (sl - entry)
            elif model_pred is not None:
                sl = or_high if sl_distance is None else entry + sl_distance
                tp = model_pred[1]
                if tp >= entry:  # model predicts no further room: skip, not a signal
                    continue
            else:
                sl = or_high if sl_distance is None else entry + sl_distance
                tp = (
                    (2 * entry - or_high)
                    if tp_range_projection
                    else entry - tp_distance
                )
            if tp_atr_z is not None and tp < or_high - tp_atr_z * atr_prior:
                continue
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
    tp_atr_z: Optional[float] = None,
    sl_at_trigger_extreme: bool = False,
    tp_bar_multiple: Optional[float] = None,
    large_bar_atr_divisor: Optional[float] = None,
    sl_range_edge: Optional[str] = None,
    require_open_inside: bool = False,
    min_relative_volume: Optional[float] = None,
    require_direction: Optional[str] = None,
    max_open_excess_atr_z: Optional[float] = None,
    tp_model_pred: Optional[dict[tuple[str, object], tuple[float, float]]] = None,
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

    tp_atr_z: see _find_breakout_trigger — applies to whichever TP mode is
    active; skips a candidate whose TP would require more than tp_atr_z *
    ATR's worth of range from the opposite edge of the opening range (e.g.
    0.5, 0.75, 1.0), instead of taking it.

    sl_at_trigger_extreme: see _find_breakout_trigger — a third sizing
    mode: SL is the trigger bar's own extreme (not scaled). TP is
    tp_bar_multiple * the trigger bar's own range from entry if
    tp_bar_multiple is given, else TP mirrors the SL distance (1:1).

    large_bar_atr_divisor: see _find_breakout_trigger — an extra trigger
    condition requiring the trigger bar's own High-Low to exceed
    atr_prior / large_bar_atr_divisor, i.e. only "large" breakout bars
    count as a signal.

    sl_range_edge ("near" or "far"): see _find_breakout_trigger — a fourth
    sizing mode: SL sits exactly at an opening-range edge (near = the
    broken edge, far = the opposite edge) and TP mirrors that distance
    (reward:risk 1:1).

    require_open_inside: see _find_breakout_trigger — requires the trigger
    bar's open to still be inside the range, so only the single bar that
    actually crosses from inside to outside can trigger (combine with
    large_bar_atr_divisor for "a large bar that breaks out of the range").

    min_relative_volume: see _compute_opening_context — requires the
    opening range's own volume to be at least this FRACTION OF A TYPICAL
    WHOLE DAY'S volume (trailing 14-session mean full-day volume, no
    lookahead) — skips both volume-warm-up and quiet-open sessions. E.g.
    0.2 requires the opening range to have already carried at least 20% of
    this ticker's recent typical full-day volume.

    require_direction ("with" or "against"): see _find_breakout_trigger —
    gates which trade direction is even considered, based on the opening
    range candle's own color (green/red), independent of the trigger bar.
    "with" trades in the same direction as the early move; "against" fades
    it (mirrors algo.quick_flip_scalper's own directional gate).

    max_open_excess_atr_z: see _find_breakout_trigger — a softer
    alternative to require_open_inside. A trigger bar whose open is
    already outside the range is only rejected if the open sits more than
    ATR/max_open_excess_atr_z beyond the edge; a bar whose open is inside
    the range, or only marginally outside it, still counts.

    tp_model_pred: see _find_breakout_trigger — a dict {(ticker, date):
    (pred_high_dollar, pred_low_dollar)} from a trained src/ml day-range
    model, overriding TP sizing (SL untouched). A session whose (ticker,
    date) isn't in the dict is skipped entirely (no prediction available —
    e.g. outside the model's warm-up window or validation years), rather
    than falling back to the default TP rule.
    """
    trades = []
    for (
        ticker,
        date,
        or_high,
        or_low,
        atr_prior,
        opening_direction,
        relative_volume,
        day_signal,
    ) in _iter_band_sessions(
        minute_bars,
        daily_bars,
        signal_timeframe,
        opening_range_minutes,
        range_atr_low_divisor,
        range_atr_high_divisor,
        min_relative_volume,
    ):
        if tp_model_pred is not None and (ticker, date) not in tp_model_pred:
            continue
        model_pred = (
            tp_model_pred[(ticker, date)] if tp_model_pred is not None else None
        )
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
                atr_prior,
                sl_at_trigger_extreme,
                tp_bar_multiple,
                large_bar_atr_divisor,
                sl_range_edge,
                tp_atr_z,
                require_open_inside,
                opening_direction,
                require_direction,
                max_open_excess_atr_z,
                model_pred,
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
            trade["atr_prior"] = atr_prior
            trade["opening_direction"] = "green" if opening_direction == 1 else "red"
            trade["relative_volume"] = relative_volume
            trades.append(trade)

            if not allow_multiple_trades_per_day:
                break
            start_idx = int(trade["exit_idx"]) + 1

    return pl.DataFrame(trades)
