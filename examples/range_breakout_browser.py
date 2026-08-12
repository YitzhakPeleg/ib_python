"""ATR-band-filtered opening-range breakout strategy — trade browser.

Config: 30-minute opening range, range < ATR/2 (no floor), trigger requires
the whole bar (open AND close) outside the range, trading allowed until
13:00, TP=SL=ATR/4, SPY only — the best config found so far. See
src/algo/range_breakout.py for the rule and
examples/hammer_reversal_browser.py for the browser this mirrors.

Launches a day-by-day Dash browser over every triggered trade, showing
5-minute candles with the opening range shaded, the trigger bar marked with
a directional triangle (up above the bar for a long, down below for a
short), and SL/TP drawn.
"""

import argparse

import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, ctx, dcc, html
from loguru import logger

from algo.range_breakout import run_breakout_backtest
from algo.resample_bars import resample_to_timeframe
from models.paths import get_file
from visualization.plotting import plot_bars

TICKER = "SPY"
OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]

OPENING_RANGE_MINUTES = 30
SIGNAL_WINDOW_MINUTES = 210  # trading allowed until 13:00 on a 9:30 open
TP_ATR_DIVISOR = 4.0


def load_trades() -> tuple[pl.DataFrame, pl.DataFrame]:
    minute_bars = (
        pl.read_parquet(get_file(TICKER, "1_min"))
        .select(OHLCV)
        .with_columns(pl.lit(TICKER).alias("ticker"))
        .sort("DateTime")
    )
    daily_bars = (
        pl.read_parquet(get_file(TICKER, "1_day"))
        .select(OHLCV)
        .with_columns(pl.lit(TICKER).alias("ticker"))
        .sort("DateTime")
    )
    trades = run_breakout_backtest(
        minute_bars,
        daily_bars,
        opening_range_minutes=OPENING_RANGE_MINUTES,
        signal_window_minutes=SIGNAL_WINDOW_MINUTES,
        range_atr_low_divisor=None,
        range_atr_high_divisor=2.0,
        tp_atr_divisor=TP_ATR_DIVISOR,
        sl_equals_tp=True,
        require_open_outside=True,
    )
    bars5 = resample_to_timeframe(minute_bars, "5m")
    return trades, bars5


RESULT_LABELS = {"tp": "Take Profit", "sl": "Stop Loss", "eod": "End of Day"}


def _build_figure(day_bars: pl.DataFrame, trade_row: dict) -> go.Figure:
    direction = "long" if trade_row["direction"] == 1 else "short"
    pnl_dollars = (trade_row["exit_price"] - trade_row["entry_price"]) * trade_row[
        "direction"
    ]
    result_label = RESULT_LABELS[trade_row["exit_reason"]]
    title = (
        f"{TICKER} — {trade_row['date']} ({direction})  |  "
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

    # Directional trigger marker: triangle-up ABOVE the bar for a long,
    # triangle-down BELOW the bar for a short — offset by a small fraction
    # of the day's own range so it doesn't sit on top of the candle.
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

    return fig


def run(port: int = 8052) -> None:
    trades, bars5 = load_trades()

    out_path = "output/hammer_reversal/spy_range_breakout_trades.csv"
    trades.write_csv(out_path)
    logger.info(f"{trades.height} trades found — saved to {out_path}")

    dates = trades["date"].to_list()
    str_dates = [str(d) for d in dates]
    trades_by_date = {row["date"]: row for row in trades.iter_rows(named=True)}

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Button("◀ Prev", id="prev-btn", n_clicks=0),
                    dcc.Dropdown(
                        id="date-dropdown",
                        options=[{"label": s, "value": s} for s in str_dates],
                        value=str_dates[0],
                        clearable=False,
                        style={"width": "220px", "display": "inline-block"},
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
        Output("date-dropdown", "value"),
        Input("prev-btn", "n_clicks"),
        Input("next-btn", "n_clicks"),
        State("date-dropdown", "value"),
    )
    def navigate(_prev: int, _nxt: int, current: str) -> str:
        trigger = ctx.triggered_id
        idx = str_dates.index(current) if current in str_dates else 0
        if trigger == "prev-btn":
            idx = max(0, idx - 1)
        elif trigger == "next-btn":
            idx = min(len(str_dates) - 1, idx + 1)
        return str_dates[idx]

    @app.callback(Output("chart", "figure"), Input("date-dropdown", "value"))
    def update_chart(date_str: str) -> go.Figure:
        matching = [d for d in dates if str(d) == date_str]
        day = matching[0] if matching else dates[0]
        day_bars = bars5.filter(pl.col("date") == day).sort("DateTime")
        return _build_figure(day_bars, trades_by_date[day])

    logger.info(f"Starting browser at http://localhost:{port}")
    app.run(debug=False, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Range-breakout trade browser")
    parser.add_argument("--port", type=int, default=8052)
    args = parser.parse_args()
    run(port=args.port)
