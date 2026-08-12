"""Quick Flip Scalper — opening-range "liquidity grab" reversal strategy.

Sourced from a YouTube walkthrough (see run_scalper_backtest's docstring
for the link); rules, stated as a sentence:

1. The first 15-minute candle of the session defines a box (its absolute
   high/low, wicks included).
2. That box only qualifies as a "liquidity candle" if its range (High -
   Low) is >= 25% of the daily ATR(14) known as of yesterday's close —
   otherwise skip the day.
3. Within the first 90 minutes of the open, switch to 5-minute candles and
   watch for a reversal trigger — but only on the side implied by the
   OPENING CANDLE'S OWN COLOR: a green (up) opening candle only looks for a
   SHORT reversal (inverted hammer or bearish engulfing) trading above the
   box; a red (down) opening candle only looks for a LONG reversal (hammer
   or bullish engulfing) trading below the box. (The green-above/red-below
   combos are the only two traded — a green open never signals long-below,
   a red open never signals short-above.)
4. Entry differs by pattern type: hammer/inverted-hammer enters on a stop
   break of the TRIGGER candle's own extreme (its high for a long, low for
   a short); engulfing instead enters at the ENGULFED (immediately prior)
   candle's own extreme.
5. Stop-loss: the trigger candle's own opposite extreme. Take-profit: the
   box's far (opposite) edge.

This module is intentionally separate from hammer_reversal.py — despite
sharing infra (ATR, the trade-fill/simulate walk, the HammerTrigger shape)
— because the direction gate (keyed off the opening candle's own color,
not just price location) and the pattern-dependent entry price make the
trigger-finding logic genuinely different from that module's own rule.
"""

from typing import Iterator, Optional

import numpy as np
import polars as pl
import talib

from algo.hammer_reversal import (
    HammerTrigger,
    _simulate_hammer_setup,
    compute_daily_atr,
)
from algo.resample_bars import resample_to_timeframe

OPENING_RANGE_MINUTES = 15
SIGNAL_WINDOW_MINUTES = 90
RANGE_ATR_DIVISOR = 4  # 25% of ATR


def _iter_scalper_sessions(
    minute_bars: pl.DataFrame,
    daily_bars: pl.DataFrame,
    signal_timeframe: str,
    opening_range_minutes: int = OPENING_RANGE_MINUTES,
    range_atr_divisor: float = RANGE_ATR_DIVISOR,
) -> Iterator[tuple[str, object, float, float, float, int, pl.DataFrame]]:
    """Per-(ticker, date) session walk, filtered to the "liquidity candle"
    rule and yielding the opening candle's own direction (+1 green, -1
    red) alongside the usual (or_high, or_low, atr_prior, day_signal) —
    (ticker, date, or_high, or_low, atr_prior, opening_direction,
    day_signal).

    The ATR gate here is a plain `range >= ATR/range_atr_divisor` float
    comparison (no whole-point rounding) — the strategy's literal "25% of
    ATR" rule, unlike hammer_reversal.py's own (unrelated) rounded-up
    version.

    The opening range/candle is measured directly off 1-min bars (a
    per-day elapsed-minutes slice), not via resample_to_timeframe, for the
    same reason as algo.hammer_reversal._iter_atr_sessions: resample's
    calendar-anchored bins would silently truncate ranges that don't
    divide evenly into the session's minutes-since-midnight.
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
                continue  # ATR warm-up

            session_open = day_minutes["DateTime"][0]
            minutes_since_open = (
                (day_minutes["DateTime"] - session_open).dt.total_minutes().to_numpy()
            )
            opening = day_minutes.filter(minutes_since_open < opening_range_minutes)
            or_high = float(opening["High"].max())
            or_low = float(opening["Low"].min())
            opening_open = float(opening["Open"][0])
            opening_close = float(opening["Close"][-1])

            if (or_high - or_low) < atr_prior / range_atr_divisor:
                continue  # not a "liquidity candle"

            opening_direction = 1 if opening_close > opening_open else -1
            day_signal = signal_bars.filter(pl.col("date") == date).sort("DateTime")
            yield (
                ticker,
                date,
                or_high,
                or_low,
                atr_prior,
                opening_direction,
                day_signal,
            )


def _find_scalper_trigger(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    minutes_since_open: np.ndarray,
    or_high: float,
    or_low: float,
    opening_direction: int,
    opening_range_minutes: int = OPENING_RANGE_MINUTES,
    signal_window_minutes: int = SIGNAL_WINDOW_MINUTES,
    tp_edge: str = "far",
) -> Optional[HammerTrigger]:
    """First-signal-wins scan for the trigger described in this module's
    docstring. Only one side is ever considered per session, gated by
    opening_direction: -1 (red open) scans for a long trigger below
    or_low; +1 (green open) scans for a short trigger above or_high.

    Pattern detection uses TA-Lib's raw SHAPE recognizers: CDLHAMMER
    (long lower wick) for the long/hammer case and CDLINVERTEDHAMMER (long
    upper wick) for the short/"inverted hammer" case — TA-Lib labels both
    of these patterns "bullish" by name, but mechanically they're just a
    bottom-wick-rejection and a top-wick-rejection shape respectively, and
    the video uses "inverted hammer" colloquially for the top-rejection
    shape (elsewhere often called a shooting star), not TA-Lib's specific
    bullish-context meaning. CDLENGULFING covers both directions natively
    (100 bullish / -100 bearish).

    tp_edge selects which side of the box the take-profit sits on: "far"
    (the video's own rule, and the default here) = a full round trip back
    across the box (long TP = or_high, short TP = or_low); "near" = just
    back to the edge that was broken (long TP = or_low, short TP =
    or_high) — a materially closer, easier-to-reach target.
    """
    in_window = (minutes_since_open >= opening_range_minutes) & (
        minutes_since_open < signal_window_minutes
    )
    hammer = talib.CDLHAMMER(open_, high, low, close)
    inv_hammer = talib.CDLINVERTEDHAMMER(open_, high, low, close)
    engulfing = talib.CDLENGULFING(open_, high, low, close)
    long_tp = or_high if tp_edge == "far" else or_low
    short_tp = or_low if tp_edge == "far" else or_high

    for i in np.flatnonzero(in_window):
        if i == 0:
            continue  # engulfing's entry needs a preceding bar
        if opening_direction == -1 and high[i] < or_low:
            if hammer[i] == 100:
                return HammerTrigger(i, 1, high[i], low[i], long_tp)
            if engulfing[i] == 100:
                return HammerTrigger(i, 1, high[i - 1], low[i], long_tp)
        if opening_direction == 1 and low[i] > or_high:
            if inv_hammer[i] == 100:
                return HammerTrigger(i, -1, low[i], high[i], short_tp)
            if engulfing[i] == -100:
                return HammerTrigger(i, -1, low[i - 1], high[i], short_tp)
    return None


def run_scalper_backtest(
    minute_bars: pl.DataFrame,
    daily_bars: pl.DataFrame,
    signal_timeframe: str = "5m",
    opening_range_minutes: int = OPENING_RANGE_MINUTES,
    signal_window_minutes: int = SIGNAL_WINDOW_MINUTES,
    range_atr_divisor: float = RANGE_ATR_DIVISOR,
    tp_edge: str = "far",
) -> pl.DataFrame:
    """ "Quick Flip Scalper" strategy, as specified from
    https://www.youtube.com/watch?v=XFtayhPIdEs : 15-min opening range/box
    that must be >= ATR/range_atr_divisor ("liquidity candle"), only trade
    the side implied by the opening candle's own color (green -> short
    fade above the box, red -> long fade below it), trigger =
    hammer/inverted-hammer (stop-entry on a break of its own extreme) or
    engulfing (stop-entry at the ENGULFED prior candle's own extreme)
    within signal_window_minutes of the open, SL at the trigger candle's
    own opposite extreme, TP at the box's far edge by default (see
    _find_scalper_trigger's tp_edge for the "near" alternative).

    minute_bars, daily_bars: multi-ticker OHLC (1-min and 1-day resp.),
    each with a `ticker` column, sorted by ticker/DateTime.
    """
    trades = []
    for (
        ticker,
        date,
        or_high,
        or_low,
        _atr_prior,
        opening_direction,
        day_signal,
    ) in _iter_scalper_sessions(
        minute_bars,
        daily_bars,
        signal_timeframe,
        opening_range_minutes,
        range_atr_divisor,
    ):
        session_open = day_signal["DateTime"][0]
        minutes_since_open = (
            (day_signal["DateTime"] - session_open).dt.total_minutes().to_numpy()
        )
        open_ = day_signal["Open"].to_numpy().astype(np.float64)
        high = day_signal["High"].to_numpy().astype(np.float64)
        low = day_signal["Low"].to_numpy().astype(np.float64)
        close = day_signal["Close"].to_numpy().astype(np.float64)

        trigger = _find_scalper_trigger(
            open_,
            high,
            low,
            close,
            minutes_since_open,
            or_high,
            or_low,
            opening_direction,
            opening_range_minutes,
            signal_window_minutes,
            tp_edge,
        )
        if trigger is None:
            continue

        trade = _simulate_hammer_setup(high, low, close, trigger)
        if trade is None:
            continue
        trade["ticker"] = ticker
        trade["date"] = date
        trade["trigger_time"] = day_signal["DateTime"][int(trigger.trigger_idx)]
        trade["confirm_time"] = day_signal["DateTime"][int(trade["confirm_idx"])]
        trade["or_high"] = or_high
        trade["or_low"] = or_low
        trade["atr_prior"] = _atr_prior
        trade["opening_direction"] = "green" if opening_direction == 1 else "red"
        trades.append(trade)

    return pl.DataFrame(trades)


def list_scalper_triggers(
    minute_bars: pl.DataFrame,
    daily_bars: pl.DataFrame,
    signal_timeframe: str = "5m",
    opening_range_minutes: int = OPENING_RANGE_MINUTES,
    signal_window_minutes: int = SIGNAL_WINDOW_MINUTES,
    range_atr_divisor: float = RANGE_ATR_DIVISOR,
    tp_edge: str = "far",
) -> pl.DataFrame:
    """Diagnostic listing: one row per (ticker, date) session where a
    trigger fired — regardless of whether the stop-entry ever filled.
    Same parameters as run_scalper_backtest.
    """
    rows = []
    for (
        ticker,
        date,
        or_high,
        or_low,
        atr_prior,
        opening_direction,
        day_signal,
    ) in _iter_scalper_sessions(
        minute_bars,
        daily_bars,
        signal_timeframe,
        opening_range_minutes,
        range_atr_divisor,
    ):
        session_open = day_signal["DateTime"][0]
        minutes_since_open = (
            (day_signal["DateTime"] - session_open).dt.total_minutes().to_numpy()
        )
        open_ = day_signal["Open"].to_numpy().astype(np.float64)
        high = day_signal["High"].to_numpy().astype(np.float64)
        low = day_signal["Low"].to_numpy().astype(np.float64)
        close = day_signal["Close"].to_numpy().astype(np.float64)

        trigger = _find_scalper_trigger(
            open_,
            high,
            low,
            close,
            minutes_since_open,
            or_high,
            or_low,
            opening_direction,
            opening_range_minutes,
            signal_window_minutes,
            tp_edge,
        )
        if trigger is None:
            continue

        rows.append(
            {
                "ticker": ticker,
                "date": date,
                "trigger_time": day_signal["DateTime"][int(trigger.trigger_idx)],
                "direction": "long" if trigger.direction == 1 else "short",
                "opening_direction": "green" if opening_direction == 1 else "red",
                "entry_price": trigger.entry_price,
                "sl": trigger.sl,
                "tp": trigger.tp,
                "atr_prior": atr_prior,
                "or_high": or_high,
                "or_low": or_low,
                "opening_range": or_high - or_low,
            }
        )

    return pl.DataFrame(rows)
