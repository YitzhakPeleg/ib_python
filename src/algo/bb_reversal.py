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


def detect_bb_reversal_signals(
    df: pl.DataFrame,
    window: int = 20,
    stds: float = 2.0,
) -> pl.DataFrame:
    """
    Detect BB consecutive-bar reversal signals and attach TP/SL levels.

    Signal logic (two consecutive bars both outside the same band):
    - Long:  red bar (Low < lower BB) → green bar (Low < lower BB)
             entry = green bar High, SL = green bar Low, TP = entry + (entry - SL)
    - Short: green bar (High > upper BB) → red bar (High > upper BB)
             entry = red bar Low, SL = red bar High, TP = entry - (SL - entry)

    TP is sized at R=1 (same distance as SL from entry).
    Previous-bar lookback uses `.over("date")` to prevent cross-day bleed.

    Args:
        df: DataFrame with columns DateTime, Open, High, Low, Close, Volume, date.
        window: BB moving-average period (default 20).
        stds: Number of standard deviations for the bands (default 2.0).

    Returns:
        Input DataFrame with appended columns:
            is_green, is_red, bb_mid, bb_upper, bb_lower,
            prev_is_green, prev_is_red, prev_high, prev_low,
            signal_direction ("long" | "short" | null),
            entry_price, stop_loss, take_profit (float | null).
    """
    df = calculate_bollinger_bands(df, window=window, stds=stds)

    df = df.with_columns(
        [
            (pl.col("Close") > pl.col("Open")).alias("is_green"),
            (pl.col("Close") < pl.col("Open")).alias("is_red"),
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

    long_cond = (
        pl.col("prev_is_red")
        & (pl.col("prev_low") < pl.col("prev_bb_lower"))
        & pl.col("is_green")
        & (pl.col("Low") < pl.col("bb_lower"))
    )

    short_cond = (
        pl.col("prev_is_green")
        & (pl.col("prev_high") > pl.col("prev_bb_upper"))
        & pl.col("is_red")
        & (pl.col("High") > pl.col("bb_upper"))
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
            .then(pl.col("Low"))
            .when(short_cond)
            .then(pl.col("High"))
            .otherwise(None)
            .alias("stop_loss"),
        ]
    )

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
        }
    )

    _log_summary(results_df)
    return results_df


def analyze_bb_reversal(results_df: pl.DataFrame) -> None:
    """Print win-rate / avg-PnL breakdowns by time of day, band width, and bar size ratio."""

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


def _log_summary(results_df: pl.DataFrame) -> None:
    total = len(results_df)
    wins = results_df.filter(pl.col("outcome") == "win").height
    losses = results_df.filter(pl.col("outcome") == "loss").height
    win_rate = wins / total if total else 0

    gross_profit = results_df.filter(pl.col("pnl") > 0)["pnl"].sum()
    gross_loss = abs(results_df.filter(pl.col("pnl") < 0)["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY — BB Reversal (R=1)")
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
