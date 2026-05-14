"""BB Consecutive-Bar Reversal signal detection, backtest, and analysis."""

from dataclasses import dataclass
from typing import Literal

import polars as pl
from loguru import logger
from tqdm import tqdm

from src.algo.bollinger_bands import calculate_bollinger_bands


@dataclass
class BBReversalTradeResult:
    """Result of a single BB reversal trade, with analysis dimensions."""

    date: int
    direction: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float
    exit_reason: Literal["take_profit", "stop_loss", "end_of_day"]
    outcome: Literal["win", "loss", "breakeven"]
    pnl: float
    r_multiple: float
    bars_held: int
    # Analysis dimensions
    signal_hour: int           # hour of the signal bar (9–15)
    band_width_pct: float      # (bb_upper - bb_lower) / bb_mid × 100
    bar1_range: float          # 1st bar High - Low
    bar2_range: float          # 2nd bar High - Low
    bar_size_ratio: float      # bar2_range / bar1_range
    ma_change: float           # |bb_mid[t] − bb_mid[t−T]| in dollars over flat_lookback bars
    price_range_T: float       # rolling (High_max − Low_min) in dollars over flat_lookback bars


def detect_bb_reversal_signals(
    df: pl.DataFrame,
    window: int = 20,
    stds: float = 2.0,
    flat_lookback: int = 12,
    flat_ma_max: float = 0.5,
    flat_range_min: float = 1.5,
    max_band_width_pct: float | None = None,
    max_price_range: float | None = None,
    tp_target: Literal["bb_mid", "bb_band", "r1"] = "bb_mid",
) -> pl.DataFrame:
    """
    Detect BB consecutive-bar reversal signals and attach TP/SL levels.

    Signal logic (two consecutive bars both outside the same band):
    - Long:  red bar (Low < lower BB) → green bar (Low < lower BB, Low > prev Low)
             entry = green bar High, SL = min(bar-2 Low, bar-1 Low)
    - Short: green bar (High > upper BB) → red bar (High > upper BB, High < prev High)
             entry = red bar Low, SL = max(bar-2 High, bar-1 High)

    TP options (tp_target):
    - "bb_mid":  TP = BB middle band (half mean-reversion). Tight TP, R < 1 for longs.
    - "bb_band": TP = opposite band — bb_upper for longs, bb_lower for shorts.
                 Full mean-reversion target; improves R significantly for longs.
    - "r1":      TP = entry ± SL distance (R=1). Guarantees R=1 regardless of band position.

    Additional filters applied:
    - Time: signal bar must be between 10:00 and 15:25 (excludes first/last 30 min of RTH)
    - Bar-2 quality: bar-2 Low > bar-1 Low (long) / bar-2 High < bar-1 High (short)
    - Flat environment: MA change < flat_ma_max AND price range > flat_range_min over T bars
    - TP validity: entry must be on the correct side of BB mid (long: entry < bb_mid)
    - max_band_width_pct: exclude wide-band (trending) regime if set
    - max_price_range: exclude over-active sessions if set

    Args:
        df: DataFrame with columns DateTime, Open, High, Low, Close, Volume, date.
        window: BB moving-average period (default 20).
        stds: Number of standard deviations for the bands (default 2.0).
        flat_lookback: Number of bars to look back for the environment check (default 12 = 1 hour on 5-min bars).
        flat_ma_max: Max allowed |bb_mid change| over lookback bars in dollars (default $0.50).
        flat_range_min: Min required high-low range over lookback bars in dollars (default $1.50).
        max_band_width_pct: If set, exclude bars where (bb_upper−bb_lower)/bb_mid×100 exceeds this value.
        max_price_range: If set, exclude bars where rolling price_range_T exceeds this value.
        tp_target: TP sizing method — "bb_mid" (default), "bb_band" (opposite band), or "r1".

    Returns:
        Input DataFrame with appended columns:
            is_green, is_red, bb_mid, bb_upper, bb_lower, band_width_pct,
            prev_is_green, prev_is_red, prev_high, prev_low,
            ma_change, price_range_T,
            signal_direction ("long" | "short" | null),
            entry_price, stop_loss, take_profit (float | null).
    """
    df = calculate_bollinger_bands(df, window=window, stds=stds)

    df = df.with_columns(
        [
            (pl.col("Close") > pl.col("Open")).alias("is_green"),
            (pl.col("Close") < pl.col("Open")).alias("is_red"),
            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_mid") * 100).alias("band_width_pct"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("is_green").shift(1).over("date").alias("prev_is_green"),
            pl.col("is_red").shift(1).over("date").alias("prev_is_red"),
            pl.col("High").shift(1).over("date").alias("prev_high"),
            pl.col("Low").shift(1).over("date").alias("prev_low"),
            pl.col("bb_upper").shift(1).over("date").alias("prev_bb_upper"),
            pl.col("bb_lower").shift(1).over("date").alias("prev_bb_lower"),
        ]
    )

    # Flat environment: |bb_mid change| OR high-low range over last T bars < K dollars.
    # Rolling high/low computed per day to prevent cross-day contamination.
    parts = []
    for part in df.sort("DateTime").partition_by("date", maintain_order=True):
        part = part.with_columns([
            pl.col("High").rolling_max(window_size=flat_lookback, min_periods=flat_lookback).alias("roll_high_T"),
            pl.col("Low").rolling_min(window_size=flat_lookback, min_periods=flat_lookback).alias("roll_low_T"),
        ])
        parts.append(part)
    df = pl.concat(parts).sort("DateTime")

    df = df.with_columns([
        (pl.col("bb_mid") - pl.col("bb_mid").shift(flat_lookback).over("date")).abs().alias("ma_change"),
        (pl.col("roll_high_T") - pl.col("roll_low_T")).alias("price_range_T"),
    ])

    # Ranging environment: MA is stable AND price is actively oscillating
    is_flat = (pl.col("ma_change") < flat_ma_max) & (pl.col("price_range_T") > flat_range_min)

    # No trades in the first or last 30 minutes of the session (9:30–9:59, 15:30–15:59)
    # Cast to Int32 before multiplying — dt.hour()/minute() return i8 which overflows at 127
    bar_minutes = (
        pl.col("DateTime").dt.hour().cast(pl.Int32) * 60
        + pl.col("DateTime").dt.minute().cast(pl.Int32)
    )
    is_valid_time = (bar_minutes >= 600) & (bar_minutes < 930)  # 10:00 – 15:25

    # Optional price-regime filters (exclude trending/over-active markets)
    is_narrow_band = (
        pl.col("band_width_pct") < max_band_width_pct
        if max_band_width_pct is not None
        else pl.lit(True)
    )
    is_bounded_range = (
        pl.col("price_range_T") < max_price_range
        if max_price_range is not None
        else pl.lit(True)
    )

    long_cond = (
        pl.col("prev_is_red")
        & (pl.col("prev_low") < pl.col("prev_bb_lower"))
        & pl.col("is_green")
        & (pl.col("Low") < pl.col("bb_lower"))
        & (pl.col("Low") > pl.col("prev_low"))   # bar-2 low above bar-1 low
        & (pl.col("High") < pl.col("bb_mid"))     # entry (bar-2 High) below TP (bb_mid)
        & is_flat
        & is_valid_time
        & is_narrow_band
        & is_bounded_range
    )

    short_cond = (
        pl.col("prev_is_green")
        & (pl.col("prev_high") > pl.col("prev_bb_upper"))
        & pl.col("is_red")
        & (pl.col("High") > pl.col("bb_upper"))
        & (pl.col("High") < pl.col("prev_high"))  # bar-2 high below bar-1 high
        & (pl.col("Low") > pl.col("bb_mid"))      # entry (bar-2 Low) above TP (bb_mid)
        & is_flat
        & is_valid_time
        & is_narrow_band
        & is_bounded_range
    )

    df = df.with_columns(
        [
            pl.when(long_cond)
            .then(pl.lit("long"))
            .when(short_cond)
            .then(pl.lit("short"))
            .otherwise(None)
            .alias("signal_direction"),
            pl.when(long_cond)
            .then(pl.col("High"))
            .when(short_cond)
            .then(pl.col("Low"))
            .otherwise(None)
            .alias("entry_price"),
            pl.when(long_cond)
            .then(pl.min_horizontal(pl.col("Low"), pl.col("prev_low")))
            .when(short_cond)
            .then(pl.max_horizontal(pl.col("High"), pl.col("prev_high")))
            .otherwise(None)
            .alias("stop_loss"),
        ]
    )

    # TP sizing — three modes
    if tp_target == "bb_mid":
        # Half mean-reversion: TP = BB middle band
        df = df.with_columns(
            pl.when(pl.col("signal_direction").is_not_null())
            .then(pl.col("bb_mid"))
            .otherwise(None)
            .alias("take_profit")
        )
    elif tp_target == "bb_band":
        # Full mean-reversion: long → bb_upper, short → bb_lower
        df = df.with_columns(
            pl.when(pl.col("signal_direction") == "long")
            .then(pl.col("bb_upper"))
            .when(pl.col("signal_direction") == "short")
            .then(pl.col("bb_lower"))
            .otherwise(None)
            .alias("take_profit")
        )
    else:  # "r1"
        # TP = entry ± SL distance (R=1)
        df = df.with_columns(
            pl.when(pl.col("signal_direction") == "long")
            .then(pl.col("entry_price") + (pl.col("entry_price") - pl.col("stop_loss")))
            .when(pl.col("signal_direction") == "short")
            .then(pl.col("entry_price") - (pl.col("stop_loss") - pl.col("entry_price")))
            .otherwise(None)
            .alias("take_profit")
        )

    n_long = df.filter(pl.col("signal_direction") == "long").height
    n_short = df.filter(pl.col("signal_direction") == "short").height
    logger.info(f"Signals — long: {n_long}, short: {n_short}, total: {n_long + n_short}")

    return df


def backtest_bb_reversal(df: pl.DataFrame) -> pl.DataFrame:
    """
    Backtest all BB reversal signals, capturing analysis dimensions per trade.

    Args:
        df: Output of detect_bb_reversal_signals.

    Returns:
        DataFrame with one row per trade including trade metrics and:
            signal_hour, band_width_pct, bar1_range, bar2_range, bar_size_ratio.
    """
    signals = df.filter(pl.col("signal_direction").is_not_null())
    logger.info(f"Backtesting {len(signals):,} signals...")

    daily_dfs: dict[int, pl.DataFrame] = {}
    for partition in df.sort("DateTime").partition_by("date"):
        daily_dfs[partition["date"][0]] = partition

    results: list[BBReversalTradeResult] = []

    for row in tqdm(signals.iter_rows(named=True), total=len(signals), desc="Backtest"):
        date = row["date"]
        direction = row["signal_direction"]
        entry_dt = row["DateTime"]
        entry_price = float(row["entry_price"])
        stop_loss = float(row["stop_loss"])
        take_profit = float(row["take_profit"])

        bar2_range = float(row["High"]) - float(row["Low"])
        bar1_high = row["prev_high"]
        bar1_low = row["prev_low"]
        bar1_range = (float(bar1_high) - float(bar1_low)) if (bar1_high is not None and bar1_low is not None) else 0.0
        bar_size_ratio = bar2_range / bar1_range if bar1_range > 0 else 0.0

        bb_mid = float(row["bb_mid"])
        band_width_pct = ((float(row["bb_upper"]) - float(row["bb_lower"])) / bb_mid * 100) if bb_mid else 0.0
        ma_change = float(row.get("ma_change") or 0.0)
        price_range_T = float(row.get("price_range_T") or 0.0)

        daily_df = daily_dfs.get(date)
        if daily_df is None:
            continue

        trade_data = daily_df.filter(pl.col("DateTime") > entry_dt)

        exit_price = entry_price
        exit_reason: Literal["take_profit", "stop_loss", "end_of_day"] = "end_of_day"
        bars_held = 0
        hit = False

        for i, bar in enumerate(trade_data.iter_rows(named=True)):
            bars_held = i + 1
            if direction == "long":
                if bar["Low"] <= stop_loss:
                    exit_price, exit_reason, hit = stop_loss, "stop_loss", True
                    break
                if bar["High"] >= take_profit:
                    exit_price, exit_reason, hit = take_profit, "take_profit", True
                    break
            else:
                if bar["High"] >= stop_loss:
                    exit_price, exit_reason, hit = stop_loss, "stop_loss", True
                    break
                if bar["Low"] <= take_profit:
                    exit_price, exit_reason, hit = take_profit, "take_profit", True
                    break

        if not hit:
            bars_held = len(trade_data)
            exit_price = float(trade_data["Close"][-1]) if bars_held > 0 else entry_price

        pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        risk = abs(entry_price - stop_loss)
        r_multiple = pnl / risk if risk > 0 else 0.0

        if pnl > 0:
            outcome: Literal["win", "loss", "breakeven"] = "win"
        elif pnl < 0:
            outcome = "loss"
        else:
            outcome = "breakeven"

        results.append(
            BBReversalTradeResult(
                date=date,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                exit_price=exit_price,
                exit_reason=exit_reason,
                outcome=outcome,
                pnl=pnl,
                r_multiple=r_multiple,
                bars_held=bars_held,
                signal_hour=entry_dt.hour,
                band_width_pct=band_width_pct,
                bar1_range=bar1_range,
                bar2_range=bar2_range,
                bar_size_ratio=bar_size_ratio,
                ma_change=ma_change,
                price_range_T=price_range_T,
            )
        )

    results_df = pl.DataFrame(
        {
            "date": [r.date for r in results],
            "direction": [r.direction for r in results],
            "entry_price": [r.entry_price for r in results],
            "stop_loss": [r.stop_loss for r in results],
            "take_profit": [r.take_profit for r in results],
            "exit_price": [r.exit_price for r in results],
            "exit_reason": [r.exit_reason for r in results],
            "outcome": [r.outcome for r in results],
            "pnl": [r.pnl for r in results],
            "r_multiple": [r.r_multiple for r in results],
            "bars_held": [r.bars_held for r in results],
            "signal_hour": [r.signal_hour for r in results],
            "band_width_pct": [r.band_width_pct for r in results],
            "bar1_range": [r.bar1_range for r in results],
            "bar2_range": [r.bar2_range for r in results],
            "bar_size_ratio": [r.bar_size_ratio for r in results],
            "ma_change": [r.ma_change for r in results],
            "price_range_T": [r.price_range_T for r in results],
        }
    )

    _log_summary(results_df)
    return results_df


def analyze_bb_reversal(results_df: pl.DataFrame) -> None:
    """Print win-rate / avg-PnL breakdowns by time of day, band width, bar size ratio, and flatness."""

    def _print_breakdown(df: pl.DataFrame, group_col: str, label: str) -> None:
        tbl = (
            df.group_by(group_col)
            .agg(
                [
                    pl.len().alias("trades"),
                    (pl.col("outcome") == "win").mean().alias("win_rate"),
                    pl.col("pnl").mean().alias("avg_pnl"),
                    pl.col("r_multiple").mean().alias("avg_r"),
                ]
            )
            .sort(group_col)
        )
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        with pl.Config(tbl_rows=30, float_precision=3):
            print(tbl)

    # ── Time of day (by hour) ──────────────────────────────────────
    _print_breakdown(results_df, "signal_hour", "Win rate by hour of day")

    # ── Band width quartiles ───────────────────────────────────────
    bw_labels = ["narrow (Q1)", "mid-low (Q2)", "mid-high (Q3)", "wide (Q4)"]
    results_with_bw = results_df.with_columns(
        pl.col("band_width_pct")
        .qcut(4, labels=bw_labels, allow_duplicates=True)
        .alias("band_width_bucket")
    )
    _print_breakdown(results_with_bw, "band_width_bucket", "Win rate by band width (quartiles)")

    # ── Bar size ratio quartiles ───────────────────────────────────
    ratio_labels = ["small (Q1)", "mid-low (Q2)", "mid-high (Q3)", "large (Q4)"]
    results_with_ratio = results_df.filter(pl.col("bar_size_ratio") > 0).with_columns(
        pl.col("bar_size_ratio")
        .qcut(4, labels=ratio_labels, allow_duplicates=True)
        .alias("bar_ratio_bucket")
    )
    _print_breakdown(results_with_ratio, "bar_ratio_bucket", "Win rate by 2nd/1st bar size ratio (quartiles)")

    # ── Flatness: MA change quartiles ─────────────────────────────
    flat_labels = ["very flat (Q1)", "flat (Q2)", "moderate (Q3)", "active (Q4)"]
    results_with_mac = results_df.filter(pl.col("ma_change").is_not_null()).with_columns(
        pl.col("ma_change")
        .qcut(4, labels=flat_labels, allow_duplicates=True)
        .alias("ma_change_bucket")
    )
    _print_breakdown(results_with_mac, "ma_change_bucket", "Win rate by MA change over T bars (quartiles)")

    # ── Flatness: price range quartiles ───────────────────────────
    results_with_rng = results_df.filter(pl.col("price_range_T").is_not_null()).with_columns(
        pl.col("price_range_T")
        .qcut(4, labels=flat_labels, allow_duplicates=True)
        .alias("price_range_bucket")
    )
    _print_breakdown(results_with_rng, "price_range_bucket", "Win rate by price range over T bars (quartiles)")

    # ── Cross-tab: band width × price range ───────────────────────
    cross = (
        results_with_bw.join(results_with_rng.select(["date", "direction", "entry_price", "price_range_bucket"]),
                             on=["date", "direction", "entry_price"], how="inner")
        .group_by(["band_width_bucket", "price_range_bucket"])
        .agg(
            [
                pl.len().alias("trades"),
                (pl.col("outcome") == "win").mean().alias("win_rate"),
                pl.col("pnl").mean().alias("avg_pnl"),
                pl.col("pnl").sum().alias("total_pnl"),
            ]
        )
        .sort(["band_width_bucket", "price_range_bucket"])
    )
    print(f"\n{'='*60}")
    print("  Cross-tab: band_width × price_range")
    print(f"{'='*60}")
    with pl.Config(tbl_rows=30, float_precision=3):
        print(cross)


def _log_summary(results_df: pl.DataFrame) -> None:
    total = len(results_df)
    wins = results_df.filter(pl.col("outcome") == "win").height
    losses = results_df.filter(pl.col("outcome") == "loss").height
    win_rate = wins / total if total else 0

    gross_profit = results_df.filter(pl.col("pnl") > 0)["pnl"].sum()
    gross_loss = abs(results_df.filter(pl.col("pnl") < 0)["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY — BB Reversal (TP = BB mid)")
    logger.info("=" * 60)
    logger.info(f"Total trades : {total}")
    logger.info(f"Wins / Losses: {wins} / {losses}  ({win_rate:.2%} win rate)")
    logger.info(f"Avg PnL/trade: ${results_df['pnl'].mean():.3f}")
    logger.info(f"Total PnL    : ${results_df['pnl'].sum():.2f}")
    logger.info(f"Profit factor: {profit_factor:.2f}")
    logger.info(f"Avg R-multiple: {results_df['r_multiple'].mean():.3f}")
    logger.info("=" * 60)
    for direction in ("long", "short"):
        sub = results_df.filter(pl.col("direction") == direction)
        w = sub.filter(pl.col("outcome") == "win").height
        logger.info(
            f"  {direction:5s}: {len(sub)} trades, {w/len(sub):.2%} win rate, "
            f"avg PnL ${sub['pnl'].mean():.3f}"
        )
    logger.info("=" * 60)
