"""ATR-filtered opening-range hammer-reversal strategy.

Rule, stated as a sentence:

1. Compute a 14-session Wilder ATR from daily bars, known as of the PRIOR
   session's close (no lookahead — a session's own range isn't final until
   it ends, so the filter run at today's open may only see ATR through
   yesterday).
2. If today's first 15-minute bar's range (High - Low) is <= ATR/4 (rounded
   UP to the nearest whole point), skip the day — too quiet an open for
   this reversal setup to be worth watching.
3. Otherwise, switch to 5-minute bars and scan the rest of the session
   (bars starting >=15 and <90 minutes after the open) for the first bar
   that trades fully outside the opening 15-minute range AND has a
   reversal-candle shape on the correct side:
   - trading above the opening range -> look for a bearish (shooting-star
     shaped) candle -> SHORT setup
   - trading below the opening range -> look for a bullish (hammer shaped)
     candle -> LONG setup
4. That candle becomes a stop order: enter on a break of its low (short) /
   high (long), stop loss at its high (short) / low (long), take profit at
   the opening 15-minute bar's low (short) / high (long) — the FAR edge of
   the range, i.e. a full round trip back across it, not just to the near
   edge that was broken.
   If price breaks the stop-loss level before the entry level ever
   triggers, the setup is abandoned (no trade that day). A single bar that
   triggers both levels at once is treated conservatively as filled and
   immediately stopped out.
"""

import math
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import polars as pl
import talib

from algo.intraday_trigger import Signal, simulate_trade
from algo.resample_bars import resample_to_timeframe

ATR_PERIOD = 14
OPENING_RANGE_MINUTES = 15
SIGNAL_WINDOW_MINUTES = 90
RANGE_ATR_DIVISOR = 4
LONG_WICK_BODY_MULTIPLE = 2  # reversal-side wick must be >= this many times the body
SMALL_WICK_BODY_RATIO = 0.5  # opposite-side wick must be <= this fraction of the body


def compute_daily_atr(daily: pl.DataFrame, period: int = ATR_PERIOD) -> pl.DataFrame:
    """daily: single-ticker daily OHLC bars, sorted by DateTime.

    Adds `date` and `atr_prior` (Wilder ATR(period) shifted one session
    forward, so the value attached to day t reflects only true ranges
    through day t-1's close).
    """
    high = daily["High"].to_numpy().astype(np.float64)
    low = daily["Low"].to_numpy().astype(np.float64)
    close = daily["Close"].to_numpy().astype(np.float64)
    atr = talib.ATR(high, low, close, timeperiod=period)
    atr_prior = np.concatenate(([np.nan], atr[:-1]))
    return daily.with_columns(
        pl.col("DateTime").dt.date().alias("date"),
        pl.Series("atr_prior", atr_prior),
    )


def _is_reversal_candle(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: int,
) -> np.ndarray:
    """Classic hammer/shooting-star shape test: the wick opposite the
    reversal direction is at least LONG_WICK_BODY_MULTIPLE times the real
    body, and the wick on the reversal side is at most SMALL_WICK_BODY_RATIO
    times the body.

    direction=+1 (hammer, bullish): long lower wick, small/no upper wick.
    direction=-1 (shooting star, bearish): long upper wick, small/no lower wick.
    """
    body = np.abs(close - open_)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    # strict "> 0" on the long-wick side excludes zero-range bars (O=H=L=C,
    # e.g. a no-trade minute on thin 1-min data): body=0 and both wicks=0
    # would otherwise trivially satisfy the ">=" and "<=" comparisons at
    # once and false-positive as BOTH a hammer and a shooting star.
    if direction == 1:
        return (
            (lower_wick >= LONG_WICK_BODY_MULTIPLE * body)
            & (upper_wick <= SMALL_WICK_BODY_RATIO * body)
            & (lower_wick > 0)
        )
    return (
        (upper_wick >= LONG_WICK_BODY_MULTIPLE * body)
        & (lower_wick <= SMALL_WICK_BODY_RATIO * body)
        & (upper_wick > 0)
    )


@dataclass
class HammerTrigger:
    trigger_idx: int
    direction: int  # +1 long, -1 short
    entry_price: float  # stop-entry level: trigger bar's high (long) / low (short)
    sl: float  # entry -/+ sl_multiple * trigger bar's own range
    tp: float  # opening 15-min bar's high (long) / low (short) — the FAR edge


def _find_hammer_trigger(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    minutes_since_open: np.ndarray,
    or_high: float,
    or_low: float,
    tp_edge: str = "far",
    rr_multiple: Optional[float] = None,
    sl_multiple: float = 1.0,
    swap_candle_roles: bool = False,
    any_candle_shape: bool = False,
) -> Optional[HammerTrigger]:
    """First-signal-wins scan of a day's signal-timeframe bars for a reversal
    candle trading fully outside [or_low, or_high], within
    SIGNAL_WINDOW_MINUTES of the session open.

    Entry is always the trigger bar's own high (long) / low (short). SL is
    sl_multiple * the trigger bar's own range beyond entry — sl_multiple=1
    (the default) puts it exactly at the trigger bar's opposite extreme, the
    original rule; sl_multiple>1 gives the trade more room, since a path
    analysis showed most trades (including most eventual winners) round-trip
    through that 1x level as ordinary intraday noise on these
    already-wide-range days, well before any real reversal/continuation
    plays out.

    tp_edge selects which side of the opening range the take-profit sits on
    (ignored if rr_multiple is given): "far" = a full round trip back across
    the range (long TP = or_high, short TP = or_low); "near" = just back to
    the edge that was broken (long TP = or_low, short TP = or_high).

    rr_multiple, if given, overrides tp_edge entirely: TP is instead placed
    at a fixed reward:risk multiple of the (possibly widened) SL distance —
    entry + rr_multiple * risk for a long, entry - rr_multiple * risk for a
    short.

    swap_candle_roles flips which candle shape gates which direction: by
    default, price below the range needs a hammer shape (long lower wick)
    to go long, and price above needs a shooting-star shape (long upper
    wick) to go short — the textbook reversal-candle pairing. Swapped, price
    below the range instead needs a shooting-star shape to go long, and
    price above needs a hammer shape to go short. Ignored if
    any_candle_shape is set.

    any_candle_shape drops the shape/direction pairing entirely: either
    shape (hammer OR shooting star) qualifies for either direction — the
    only thing that still matters is location (trading outside the opening
    range) and having ONE long, one-sided wick, not which side it's on.
    """
    in_window = (minutes_since_open >= OPENING_RANGE_MINUTES) & (
        minutes_since_open < SIGNAL_WINDOW_MINUTES
    )
    hammer_long = _is_reversal_candle(open_, high, low, close, 1)
    hammer_short = _is_reversal_candle(open_, high, low, close, -1)
    if any_candle_shape:
        candle_for_long = candle_for_short = hammer_long | hammer_short
    else:
        candle_for_long = hammer_short if swap_candle_roles else hammer_long
        candle_for_short = hammer_long if swap_candle_roles else hammer_short
    long_tp = or_high if tp_edge == "far" else or_low
    short_tp = or_low if tp_edge == "far" else or_high

    for i in np.flatnonzero(in_window):
        if high[i] < or_low and candle_for_long[i]:
            risk = sl_multiple * (high[i] - low[i])
            sl = high[i] - risk
            tp = high[i] + rr_multiple * risk if rr_multiple is not None else long_tp
            return HammerTrigger(i, 1, high[i], sl, tp)
        if low[i] > or_high and candle_for_short[i]:
            risk = sl_multiple * (high[i] - low[i])
            sl = low[i] + risk
            tp = low[i] - rr_multiple * risk if rr_multiple is not None else short_tp
            return HammerTrigger(i, -1, low[i], sl, tp)
    return None


def _simulate_hammer_setup(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, trigger: HammerTrigger
) -> Optional[dict]:
    """Walk forward from the trigger bar for the stop-entry fill, then hand
    off to the shared SL/TP/EOD walk (algo.intraday_trigger.simulate_trade)
    once filled.
    """
    n = len(close)
    for i in range(trigger.trigger_idx + 1, n):
        if trigger.direction == 1:
            hit_entry = high[i] >= trigger.entry_price
            hit_invalidate = low[i] <= trigger.sl
        else:
            hit_entry = low[i] <= trigger.entry_price
            hit_invalidate = high[i] >= trigger.sl

        if not hit_entry:
            if hit_invalidate:
                return None  # setup invalidated before ever filling
            continue

        if (
            hit_invalidate
        ):  # same-bar double touch: conservative, filled then immediately stopped
            return {
                "trigger_idx": trigger.trigger_idx,
                "confirm_idx": i,
                "direction": trigger.direction,
                "entry_price": trigger.entry_price,
                "sl": trigger.sl,
                "tp": trigger.tp,
                "exit_idx": i,
                "exit_price": trigger.sl,
                "exit_reason": "sl",
                "r_multiple": -1.0,
            }

        signal = Signal(
            trigger.trigger_idx,
            i,
            trigger.direction,
            trigger.entry_price,
            trigger.sl,
            trigger.tp,
        )
        return simulate_trade(high, low, close, signal)

    return None  # never filled by end of day


def _atr_threshold(atr_prior: float) -> int:
    """ATR/4 rounded UP to the nearest whole point (e.g. ATR/4=1.42 -> 2) —
    the actual gate the opening range must clear, kept as a single function
    so the filter and any display of it can't drift apart.
    """
    return math.ceil(atr_prior / RANGE_ATR_DIVISOR)


def _iter_atr_sessions(
    minute_bars: pl.DataFrame,
    daily_bars: pl.DataFrame,
    signal_timeframe: str,
    opening_range_minutes: int = OPENING_RANGE_MINUTES,
) -> Iterator[tuple[str, object, float, float, float, pl.DataFrame]]:
    """Shared per-(ticker, date) session walk: resamples once per ticker,
    applies only the ATR warm-up filter, and yields EVERY session with a
    valid prior-day ATR — (ticker, date, or_high, or_low, atr_prior,
    day_signal) where day_signal is that session's signal-timeframe bars.
    Callers apply their own opening-range-vs-ATR filter on top; see
    _iter_sessions (this module's own >ATR/4 filter) and
    algo.range_breakout's ATR/4..ATR/2 band filter for two different ones
    layered on the same underlying walk.

    opening_range_minutes controls the duration of the first slice of 1-min
    bars used to set or_high/or_low — the strategy's own default (15) is the
    constant's default here too, but callers (e.g. range_breakout's sweeps)
    can pass a different duration (5, 30, 60, ...) to test the setup at a
    different opening-range length without changing what "opening range"
    means everywhere else.

    The opening range is measured directly off the 1-min bars (a per-day
    elapsed-minutes slice), NOT via resample_to_timeframe: that function's
    bins are calendar-anchored (e.g. "60m" bins fall on the hour: 9:00,
    10:00, ...), and 9:30 isn't a multiple of every possible
    opening_range_minutes (570 minutes-since-midnight isn't divisible by
    60) — resampling to "60m" would silently give a 9:30-10:00 bin (really
    just 30 real minutes of data) instead of the intended 9:30-10:30. Slicing
    1-min bars by elapsed-time-since-session-open sidesteps that entirely,
    for any duration.
    """
    for (ticker,), ticker_minutes in minute_bars.group_by(
        "ticker", maintain_order=True
    ):
        ticker_daily = compute_daily_atr(
            daily_bars.filter(pl.col("ticker") == ticker).sort("DateTime")
        )
        atr_by_date = dict(
            zip(ticker_daily["date"].to_list(), ticker_daily["atr_prior"].to_list())
        )

        ticker_minutes = ticker_minutes.sort("DateTime").with_columns(
            pl.col("DateTime").dt.date().alias("date")
        )
        signal_bars = resample_to_timeframe(ticker_minutes, signal_timeframe)

        for (date,), day_minutes in ticker_minutes.group_by(
            "date", maintain_order=True
        ):
            atr_prior = atr_by_date.get(date)
            if atr_prior is None or np.isnan(atr_prior):
                continue  # ATR warm-up (first ATR_PERIOD sessions)

            session_open = day_minutes["DateTime"][0]
            minutes_since_open = (
                (day_minutes["DateTime"] - session_open).dt.total_minutes().to_numpy()
            )
            opening = day_minutes.filter(minutes_since_open < opening_range_minutes)
            or_high = float(opening["High"].max())
            or_low = float(opening["Low"].min())
            day_signal = signal_bars.filter(pl.col("date") == date).sort("DateTime")
            yield ticker, date, or_high, or_low, atr_prior, day_signal


def _iter_sessions(
    minute_bars: pl.DataFrame, daily_bars: pl.DataFrame, signal_timeframe: str
) -> Iterator[tuple[str, object, float, float, float, pl.DataFrame]]:
    """_iter_atr_sessions filtered to this strategy's own rule: the opening
    range must exceed ATR/4 (rounded up — see _atr_threshold).
    """
    for ticker, date, or_high, or_low, atr_prior, day_signal in _iter_atr_sessions(
        minute_bars, daily_bars, signal_timeframe
    ):
        if (or_high - or_low) <= _atr_threshold(atr_prior):
            continue  # opening bar too quiet for this setup
        yield ticker, date, or_high, or_low, atr_prior, day_signal


def _trigger_for_session(
    day_signal: pl.DataFrame,
    or_high: float,
    or_low: float,
    tp_edge: str,
    rr_multiple: Optional[float],
    sl_multiple: float,
    swap_candle_roles: bool,
    any_candle_shape: bool,
) -> Optional[HammerTrigger]:
    session_open = day_signal["DateTime"][0]
    minutes_since_open = (
        (day_signal["DateTime"] - session_open).dt.total_minutes().to_numpy()
    )
    open_ = day_signal["Open"].to_numpy().astype(np.float64)
    high = day_signal["High"].to_numpy().astype(np.float64)
    low = day_signal["Low"].to_numpy().astype(np.float64)
    close = day_signal["Close"].to_numpy().astype(np.float64)
    return _find_hammer_trigger(
        open_,
        high,
        low,
        close,
        minutes_since_open,
        or_high,
        or_low,
        tp_edge=tp_edge,
        rr_multiple=rr_multiple,
        sl_multiple=sl_multiple,
        swap_candle_roles=swap_candle_roles,
        any_candle_shape=any_candle_shape,
    )


def run_hammer_backtest(
    minute_bars: pl.DataFrame,
    daily_bars: pl.DataFrame,
    signal_timeframe: str = "5m",
    tp_edge: str = "far",
    rr_multiple: Optional[float] = None,
    sl_multiple: float = 1.0,
    swap_candle_roles: bool = False,
    any_candle_shape: bool = False,
) -> pl.DataFrame:
    """minute_bars, daily_bars: multi-ticker OHLC (1-min and 1-day resp.),
    each with a `ticker` column, sorted by ticker/DateTime.

    signal_timeframe: bar size used for the post-opening-range hammer scan
    and trade simulation (the strategy's original spec calls for "5m"; "1m"
    runs the identical rules on 1-min bars instead). tp_edge, rr_multiple,
    sl_multiple, swap_candle_roles, any_candle_shape: see
    _find_hammer_trigger.
    """
    trades = []
    for ticker, date, or_high, or_low, _atr_prior, day_signal in _iter_sessions(
        minute_bars, daily_bars, signal_timeframe
    ):
        trigger = _trigger_for_session(
            day_signal,
            or_high,
            or_low,
            tp_edge,
            rr_multiple,
            sl_multiple,
            swap_candle_roles,
            any_candle_shape,
        )
        if trigger is None:
            continue

        high = day_signal["High"].to_numpy().astype(np.float64)
        low = day_signal["Low"].to_numpy().astype(np.float64)
        close = day_signal["Close"].to_numpy().astype(np.float64)
        trade = _simulate_hammer_setup(high, low, close, trigger)
        if trade is None:
            continue
        trade["ticker"] = ticker
        trade["date"] = date
        trades.append(trade)

    return pl.DataFrame(trades)


def list_hammer_triggers(
    minute_bars: pl.DataFrame,
    daily_bars: pl.DataFrame,
    signal_timeframe: str = "5m",
    tp_edge: str = "far",
    rr_multiple: Optional[float] = None,
    sl_multiple: float = 1.0,
    swap_candle_roles: bool = False,
    any_candle_shape: bool = False,
) -> pl.DataFrame:
    """Diagnostic listing: one row per (ticker, date) session where a hammer
    trigger fired — regardless of whether the stop-entry ever filled — with
    the context needed to inspect the day: ticker, date, trigger_time,
    direction, entry_price, sl, tp, atr_prior, atr_quarter (the ATR/4
    threshold the opening range had to clear, rounded UP to a whole point —
    see _atr_threshold), or_high, or_low, opening_range. Same parameters as
    run_hammer_backtest.
    """
    rows = []
    for ticker, date, or_high, or_low, atr_prior, day_signal in _iter_sessions(
        minute_bars, daily_bars, signal_timeframe
    ):
        trigger = _trigger_for_session(
            day_signal,
            or_high,
            or_low,
            tp_edge,
            rr_multiple,
            sl_multiple,
            swap_candle_roles,
            any_candle_shape,
        )
        if trigger is None:
            continue

        rows.append(
            {
                "ticker": ticker,
                "date": date,
                "trigger_time": day_signal["DateTime"][int(trigger.trigger_idx)],
                "direction": "long" if trigger.direction == 1 else "short",
                "entry_price": trigger.entry_price,
                "sl": trigger.sl,
                "tp": trigger.tp,
                "atr_prior": atr_prior,
                "atr_quarter": _atr_threshold(atr_prior),
                "or_high": or_high,
                "or_low": or_low,
                "opening_range": or_high - or_low,
            }
        )

    return pl.DataFrame(rows)
