"""Windowing/alignment: turns (daily_df, intraday_df) into fixed-size
tensors for the day-range predictor, with strict no-lookahead alignment.
See this repo's plan doc / examples/train_day_range_nn.py for the full rule.
"""

from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
import polars as pl
import torch

from algo.hammer_reversal import compute_daily_atr
from ml.features import DAILY_LOOKBACK_DAYS, INTRADAY_INPUT_BARS, log_ratio


@dataclass
class DayRangeDataset:
    daily: torch.Tensor  # [N, daily_lookback, 5]
    intraday: torch.Tensor  # [N, intraday_bars, 5]
    target: torch.Tensor  # [N, 2] = [high_ret, low_ret]
    meta: pl.DataFrame  # ticker, date, prev_close, true_high, true_low,
    # input_window_high, input_window_low, atr_prior


def extract_examples(
    daily_df: pl.DataFrame,
    intraday_df: pl.DataFrame,
    target_years: Union[int, Iterable[int]],
    daily_lookback: int = DAILY_LOOKBACK_DAYS,
    intraday_bars: int = INTRADAY_INPUT_BARS,
    ticker: str = "",
) -> DayRangeDataset:
    """daily_df: output of features.add_daily_return_features (has
    open_ret/high_ret/low_ret/close_ret/log_vol_ratio/prev_close/
    Close/Volume/date). intraday_df: output of features.prepare_frames's
    second return value, at whatever timeframe was chosen (has
    Open/High/Low/Close/Volume/date/DateTime).

    A target day t (0-indexed into the sorted unique trading dates) is
    usable iff t >= daily_lookback + 1 (each of the daily_lookback window
    days needs its own prior close, so the window's earliest day, t -
    daily_lookback, needs t - daily_lookback - 1 >= 0) AND its session has
    more than intraday_bars intraday bars (at least one bar left to form
    the target). Only days whose OWN year is in target_years are kept
    (target_years may be a single int or any collection of ints, e.g.
    range(2018, 2025)), but the daily lookback window is allowed to span
    into an earlier year not in target_years (e.g. an early-January target
    day pulling a December daily window) — that's correct no-lookahead
    behavior, not a bug. `ticker` is stamped onto every meta row (purely
    for bookkeeping when combining datasets across tickers — see
    combine_datasets — not used in any feature computation).

    Intraday features are normalized against day t-1's own close AND full
    session volume (fixed for the whole intraday window, unlike each daily
    row's own day-over-day reference) — the one reference point genuinely
    known before day t's session opens.
    """
    target_year_set = (
        {target_years} if isinstance(target_years, int) else set(target_years)
    )
    daily_df = daily_df.sort("date")
    daily_with_atr = compute_daily_atr(daily_df)
    atr_by_date = dict(
        zip(daily_with_atr["date"].to_list(), daily_with_atr["atr_prior"].to_list())
    )
    trading_dates = daily_df["date"].to_list()

    daily_features = daily_df.select(
        "open_ret", "high_ret", "low_ret", "close_ret", "log_vol_ratio"
    ).to_numpy()
    daily_close = daily_df["Close"].to_numpy().astype(np.float64)
    daily_volume = daily_df["Volume"].to_numpy().astype(np.float64)

    intraday_by_date = {
        date: day.sort("DateTime")
        for (date,), day in intraday_df.group_by("date", maintain_order=True)
    }

    daily_rows: list[np.ndarray] = []
    intraday_rows: list[np.ndarray] = []
    targets: list[list[float]] = []
    meta_rows: list[dict] = []

    for t in range(daily_lookback + 1, len(trading_dates)):
        date = trading_dates[t]
        if date.year not in target_year_set:
            continue

        atr_prior = atr_by_date.get(date)
        if atr_prior is None or (isinstance(atr_prior, float) and np.isnan(atr_prior)):
            continue  # ATR warm-up (shouldn't bind given t's own bound, but guard anyway)

        day_bars = intraday_by_date.get(date)
        if day_bars is None or day_bars.height <= intraday_bars:
            continue  # not enough intraday bars for input + >=1 target bar

        window = slice(t - daily_lookback, t)
        daily_matrix = daily_features[window]
        if not np.isfinite(daily_matrix).all():
            continue  # defensive: shouldn't trigger given t's own bound

        prev_close = daily_close[t - 1]
        prev_volume = daily_volume[t - 1]
        input_bars = day_bars.head(intraday_bars)
        target_bars = day_bars.slice(intraday_bars, day_bars.height - intraday_bars)

        i_open = input_bars["Open"].to_numpy() / prev_close - 1
        i_high = input_bars["High"].to_numpy() / prev_close - 1
        i_low = input_bars["Low"].to_numpy() / prev_close - 1
        i_close = input_bars["Close"].to_numpy() / prev_close - 1
        i_log_vol = log_ratio(input_bars["Volume"].to_numpy(), prev_volume)
        intraday_matrix = np.stack([i_open, i_high, i_low, i_close, i_log_vol], axis=1)

        true_high = float(target_bars["High"].max())
        true_low = float(target_bars["Low"].min())

        daily_rows.append(daily_matrix)
        intraday_rows.append(intraday_matrix)
        targets.append([true_high / prev_close - 1, true_low / prev_close - 1])
        meta_rows.append(
            {
                "ticker": ticker,
                "date": date,
                "prev_close": prev_close,
                "true_high": true_high,
                "true_low": true_low,
                "input_window_high": float(input_bars["High"].max()),
                "input_window_low": float(input_bars["Low"].min()),
                "atr_prior": float(atr_prior),
            }
        )

    daily_arr = np.stack(daily_rows).astype(np.float32)
    intraday_arr = np.stack(intraday_rows).astype(np.float32)
    target_arr = np.array(targets, dtype=np.float32)

    assert np.isfinite(daily_arr).all(), "NaN/Inf in daily feature tensor"
    assert np.isfinite(intraday_arr).all(), "NaN/Inf in intraday feature tensor"
    assert np.isfinite(target_arr).all(), "NaN/Inf in target tensor"

    return DayRangeDataset(
        daily=torch.from_numpy(daily_arr),
        intraday=torch.from_numpy(intraday_arr),
        target=torch.from_numpy(target_arr),
        meta=pl.DataFrame(meta_rows),
    )


def combine_datasets(datasets: list[DayRangeDataset]) -> DayRangeDataset:
    """Concatenates multiple DayRangeDatasets along the batch dimension —
    e.g. one per ticker, each built by its own extract_examples call (each
    ticker needs its own daily/intraday frames since prices/dates differ,
    so this happens post-hoc rather than inside extract_examples itself).
    All datasets must share the same intraday_bars (tensor shapes must
    match) — asserted here rather than silently producing a ragged batch.
    """
    intraday_shapes = {ds.intraday.shape[1:] for ds in datasets}
    assert len(intraday_shapes) == 1, (
        f"combine_datasets requires matching intraday_bars across all inputs, got {intraday_shapes}"
    )
    return DayRangeDataset(
        daily=torch.cat([ds.daily for ds in datasets], dim=0),
        intraday=torch.cat([ds.intraday for ds in datasets], dim=0),
        target=torch.cat([ds.target for ds in datasets], dim=0),
        meta=pl.concat([ds.meta for ds in datasets]),
    )
