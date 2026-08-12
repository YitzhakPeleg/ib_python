"""Quick Flip Scalper strategy — trade browser.

Config kept from the sweep: 30-minute opening range/box, range >= ATR/4
("liquidity candle"), signal window 210 minutes (13:00 cutoff on a 9:30
open), TP at the box's far edge — the best cross-validated cell found
(+$40.00 over 8 years). See src/algo/quick_flip_scalper.py for the rule
and examples/range_breakout_browser.py for the browser this mirrors.

Launches a day-by-day Dash browser over every triggered trade, showing
5-min candles with the opening box shaded, the trigger candle marked with
a directional triangle, the confirm (stop-entry fill) bar marked, and
SL/TP drawn.
"""

import argparse

import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, ctx, dcc, html
from loguru import logger

from algo.quick_flip_scalper import run_scalper_backtest
from algo.resample_bars import resample_to_timeframe
from models.paths import get_file
from visualization.plotting import plot_bars

TICKER = "SPY"
DATA_LABEL = "SPY_full"  # merged 2018-01-02 -> 2026-05-01 file
YEAR = 2018
OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]

OPENING_RANGE_MINUTES = 30
SIGNAL_WINDOW_MINUTES = 210
RANGE_ATR_DIVISOR = 4
SIGNAL_TIMEFRAME = "5m"


def load_trades() -> tuple[pl.DataFrame, pl.DataFrame]:
    minute_bars_full = (
        pl.read_parquet(get_file(DATA_LABEL, "1_min"))
        .select(OHLCV)
        .with_columns(pl.lit(TICKER).alias("ticker"))
        .sort("DateTime")
    )
    daily_bars = (
        resample_to_timeframe(minute_bars_full.drop("ticker"), "1d")
        .with_columns(pl.lit(TICKER).alias("ticker"))
        .select(OHLCV + ["ticker"])
    )

    minute_bars = minute_bars_full.filter(pl.col("DateTime").dt.year() == YEAR)
    trades = run_scalper_backtest(
        minute_bars,
        daily_bars,
        signal_timeframe=SIGNAL_TIMEFRAME,
        opening_range_minutes=OPENING_RANGE_MINUTES,
        signal_window_minutes=SIGNAL_WINDOW_MINUTES,
        range_atr_divisor=RANGE_ATR_DIVISOR,
    )
    bars_signal = resample_to_timeframe(minute_bars, SIGNAL_TIMEFRAME)
    return trades, bars_signal


RESULT_LABELS = {"tp": "Take Profit", "sl": "Stop Loss", "eod": "End of Day"}


def _build_figure(day_bars: pl.DataFrame, trade_row: dict) -> go.Figure:
    direction = "long" if trade_row["direction"] == 1 else "short"
    pnl_dollars = (trade_row["exit_price"] - trade_row["entry_price"]) * trade_row[
        "direction"
    ]
    result_label = RESULT_LABELS[trade_row["exit_reason"]]
    or_low, or_high = trade_row["or_low"], trade_row["or_high"]
    title = (
        f"{TICKER} — {trade_row['date']} ({direction}, "
        f"{trade_row['opening_direction']} open)  |  "
        f"box=[{or_low:.2f}, {or_high:.2f}] ({or_high - or_low:.2f})  |  "
        f"ATR={trade_row['atr_prior']:.2f}  |  "
        f"Result: {result_label}  |  P&L: {pnl_dollars:+.2f} $ (r={trade_row['r_multiple']:.2f})"
    )
    fig = plot_bars(
        day_bars,
        bb_upper_col=None,
        bb_mid_col=None,
        bb_lower_col=None,
        title=title,
        show_fig=False,
        return_fig=True,
    )
    assert fig is not None

    fig.add_hrect(
        y0=trade_row["or_low"],
        y1=trade_row["or_high"],
        fillcolor="rgba(255, 255, 0, 0.08)",
        line_width=1,
        line_color="rgba(255, 255, 0, 0.4)",
        row=1,
        col=1,
    )
    for y, name, color in [
        (trade_row["sl"], "SL", "#ff3333"),
        (trade_row["tp"], "TP", "#00cc44"),
    ]:
        fig.add_hline(
            y=y,
            line_dash="dash",
            line_color=color,
            annotation_text=name,
            annotation_position="right",
            row=1,
            col=1,
        )

    # Trigger candle: directional triangle. Confirm (stop-entry fill) bar:
    # a vertical dotted line, since it's a different bar from the trigger
    # whenever the fill doesn't happen on the very next candle.
    day_range = float(day_bars["High"].max() - day_bars["Low"].min())
    offset = day_range * 0.03
    trigger_row = day_bars.filter(pl.col("DateTime") == trade_row["trigger_time"])
    if trigger_row.height:
        bar = trigger_row.row(0, named=True)
        if trade_row["direction"] == 1:
            marker_y, symbol = bar["High"] + offset, "triangle-up"
        else:
            marker_y, symbol = bar["Low"] - offset, "triangle-down"
        fig.add_trace(
            go.Scatter(
                x=[trade_row["trigger_time"]],
                y=[marker_y],
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=14,
                    color="yellow",
                    line=dict(width=1, color="black"),
                ),
                name="Trigger",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.add_vline(
        x=trade_row["confirm_time"],
        line_dash="dot",
        line_color="cyan",
        row=1,
        col=1,
    )

    return fig


def run(port: int = 8053) -> None:
    trades, bars_signal = load_trades()

    out_path = f"output/hammer_reversal/spy_quick_flip_scalper_trades_{YEAR}.csv"
    trades.write_csv(out_path)
    logger.info(f"{trades.height} trades found — saved to {out_path}")

    trades = trades.sort("trigger_time")
    trade_rows = trades.to_dicts()
    labels = [
        f"{row['date']} {row['trigger_time'].strftime('%H:%M')} "
        f"({'long' if row['direction'] == 1 else 'short'})"
        for row in trade_rows
    ]

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Button("◀ Prev", id="prev-btn", n_clicks=0),
                    dcc.Dropdown(
                        id="trade-dropdown",
                        options=[
                            {"label": lbl, "value": i} for i, lbl in enumerate(labels)
                        ],
                        value=0,
                        clearable=False,
                        style={"width": "260px", "display": "inline-block"},
                    ),
                    html.Button("Next ▶", id="next-btn", n_clicks=0),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "padding": "8px 12px",
                },
            ),
            dcc.Graph(id="chart", style={"height": "85vh"}),
        ]
    )

    @app.callback(
        Output("trade-dropdown", "value"),
        Input("prev-btn", "n_clicks"),
        Input("next-btn", "n_clicks"),
        State("trade-dropdown", "value"),
    )
    def navigate(_prev: int, _nxt: int, current: int) -> int:
        trigger = ctx.triggered_id
        idx = current if current is not None else 0
        if trigger == "prev-btn":
            idx = max(0, idx - 1)
        elif trigger == "next-btn":
            idx = min(len(trade_rows) - 1, idx + 1)
        return idx

    @app.callback(Output("chart", "figure"), Input("trade-dropdown", "value"))
    def update_chart(idx: int) -> go.Figure:
        trade_row = trade_rows[idx if idx is not None else 0]
        day_bars = bars_signal.filter(pl.col("date") == trade_row["date"]).sort(
            "DateTime"
        )
        return _build_figure(day_bars, trade_row)

    logger.info(f"Starting browser at http://localhost:{port}")
    app.run(debug=False, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Flip Scalper trade browser")
    parser.add_argument("--port", type=int, default=8053)
    args = parser.parse_args()
    run(port=args.port)
