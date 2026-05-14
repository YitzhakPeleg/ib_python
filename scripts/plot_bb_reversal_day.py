"""
Plot 5-min BB reversal trades for a specific date.

Usage:
    uv run python -m scripts.plot_bb_reversal_day 2026-04-29
    uv run python -m scripts.plot_bb_reversal_day          # defaults to last trading day in data
"""

import sys
from datetime import timedelta
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots

SIGNALS_PATH = Path("results/SPY_bb_reversal_5min_signals.parquet")
BACKTEST_PATH = Path("results/SPY_bb_reversal_5min_backtest.csv")


def plot_day(target_date: str) -> None:
    # ── Load data ─────────────────────────────────────────────────
    if not SIGNALS_PATH.exists():
        logger.error(
            f"Signals file not found: {SIGNALS_PATH}. Run scripts/run_bb_reversal.py first."
        )
        return

    signals_df = pl.read_parquet(SIGNALS_PATH)

    year, month, day = map(int, target_date.split("-"))
    day_bars = signals_df.filter(
        pl.col("DateTime").dt.date() == pl.date(year, month, day)
    )

    if len(day_bars) == 0:
        logger.error(f"No bar data for {target_date}")
        return

    signal_bars = day_bars.filter(pl.col("signal_direction").is_not_null())
    logger.info(f"{target_date}: {len(day_bars)} bars, {len(signal_bars)} signals")

    # Join signal DateTime with backtest outcomes
    day_signals = signal_bars.select(
        ["DateTime", pl.col("signal_direction").alias("direction"), "entry_price"]
    )
    backtest = pl.read_csv(BACKTEST_PATH, infer_schema_length=1000)
    day_results = backtest.filter(pl.col("date") == target_date).select(
        [
            "direction",
            "entry_price",
            "stop_loss",
            "take_profit",
            "exit_price",
            "exit_reason",
            "outcome",
            "pnl",
            "bars_held",
        ]
    )
    trades = day_signals.join(day_results, on=["direction", "entry_price"], how="inner")

    # Compute exit DateTime from bars_held (each bar = 5 min)
    exit_dts = [
        row["DateTime"] + timedelta(minutes=5 * row["bars_held"])
        for row in trades.iter_rows(named=True)
    ]
    trades = trades.with_columns(pl.Series("exit_dt", exit_dts))

    last_dt = day_bars["DateTime"][-1]
    n = len(trades)
    wins = (trades["outcome"] == "win").sum()
    total_pnl = trades["pnl"].sum()

    # ── Subplots ──────────────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("Price", "Volume"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=day_bars["DateTime"],
            open=day_bars["Open"],
            high=day_bars["High"],
            low=day_bars["Low"],
            close=day_bars["Close"],
            name="OHLC",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1,
        col=1,
    )

    # BB bands
    has_bb = all(c in day_bars.columns for c in ["bb_upper", "bb_mid", "bb_lower"])
    if has_bb:
        fig.add_trace(
            go.Scatter(
                x=day_bars["DateTime"],
                y=day_bars["bb_upper"],
                line=dict(color="rgba(173,216,230,0.5)", width=1),
                name="BB Upper",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=day_bars["DateTime"],
                y=day_bars["bb_mid"],
                line=dict(color="orange", width=1, dash="dash"),
                name="BB Mid",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=day_bars["DateTime"],
                y=day_bars["bb_lower"],
                line=dict(color="rgba(173,216,230,0.5)", width=1),
                name="BB Lower",
            ),
            row=1,
            col=1,
        )

    # Volume
    vol_colors = [
        "green" if c >= o else "red"
        for c, o in zip(day_bars["Close"], day_bars["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=day_bars["DateTime"],
            y=day_bars["Volume"],
            marker_color=vol_colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # ── Signal bar highlights ─────────────────────────────────────
    for row in signal_bars.iter_rows(named=True):
        color = (
            "rgba(30,144,255,0.12)"
            if row["signal_direction"] == "long"
            else "rgba(255,140,0,0.12)"
        )
        fig.add_vrect(
            x0=row["DateTime"],
            x1=row["DateTime"] + timedelta(minutes=5),
            fillcolor=color,
            opacity=1.0,
            line_width=0,
            row=1,
            col=1,
        )

    # ── Per-trade markers and lines ───────────────────────────────
    for row in trades.iter_rows(named=True):
        entry_dt: object = row["DateTime"]
        exit_dt: object = row["exit_dt"]
        direction = row["direction"]
        outcome = row["outcome"]
        exit_reason = row["exit_reason"]

        is_long = direction == "long"

        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[entry_dt],
                y=[row["entry_price"]],
                mode="markers",
                marker=dict(
                    symbol="triangle-up" if is_long else "triangle-down",
                    size=14,
                    color="#1e90ff" if is_long else "#ff8c00",
                    line=dict(width=1, color="white"),
                ),
                hovertext=(
                    f"{'LONG ▲' if is_long else 'SHORT ▼'}<br>"
                    f"Entry: {row['entry_price']:.2f}<br>"
                    f"SL: {row['stop_loss']:.2f}<br>"
                    f"TP: {row['take_profit']:.2f}"
                ),
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # SL line — ends where SL was hit, otherwise stretches to end of day
        sl_end = exit_dt if exit_reason == "stop_loss" else last_dt
        fig.add_trace(
            go.Scatter(
                x=[entry_dt, sl_end],
                y=[row["stop_loss"], row["stop_loss"]],
                mode="lines",
                line=dict(color="#ff4444", dash="dot", width=1.5),
                showlegend=False,
                hoverinfo="none",
            ),
            row=1,
            col=1,
        )

        # TP line — ends where TP was hit, otherwise stretches to end of day
        tp_end = exit_dt if exit_reason == "take_profit" else last_dt
        fig.add_trace(
            go.Scatter(
                x=[entry_dt, tp_end],
                y=[row["take_profit"], row["take_profit"]],
                mode="lines",
                line=dict(color="#00cc44", dash="dot", width=1.5),
                showlegend=False,
                hoverinfo="none",
            ),
            row=1,
            col=1,
        )

        # Exit marker
        exit_color = (
            "#00cc44"
            if outcome == "win"
            else "#ff3333"
            if outcome == "loss"
            else "#aaaaaa"
        )
        exit_label = (
            "TP ✓"
            if exit_reason == "take_profit"
            else "SL ✗"
            if exit_reason == "stop_loss"
            else "EOD"
        )
        fig.add_trace(
            go.Scatter(
                x=[exit_dt],
                y=[row["exit_price"]],
                mode="markers",
                marker=dict(
                    symbol="x", size=13, color=exit_color, line=dict(width=2.5)
                ),
                hovertext=(
                    f"{exit_label}<br>"
                    f"Exit: {row['exit_price']:.2f}<br>"
                    f"PnL: {row['pnl']:+.2f}"
                ),
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # ── Layout ────────────────────────────────────────────────────
    fig.update_layout(
        title=(
            f"SPY 5-min  —  {target_date}  |  "
            f"{n} trades  |  {wins}W / {n - wins}L  |  PnL: ${total_pnl:+.2f}"
        ),
        xaxis_rangeslider_visible=False,
        height=850,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    logger.info(
        f"Showing chart — {n} trades, {wins}W/{n - wins}L, PnL ${total_pnl:+.2f}"
    )
    fig.show(renderer="browser")


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if date_arg is None:
        # Default to last trading day in the data
        _df = pl.read_parquet(SIGNALS_PATH)
        date_arg = str(
            _df.filter(pl.col("signal_direction").is_not_null())["DateTime"]
            .dt.date()
            .max()
        )
        logger.info(f"No date given — defaulting to last signal date: {date_arg}")

    plot_day(date_arg)
