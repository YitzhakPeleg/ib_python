"""ATR-filtered opening-range hammer-reversal strategy — trigger list + browser.

Original rule (no swap_candle_roles, no any_candle_shape, tp_edge="far",
sl_multiple=1.0), SPY only. See src/algo/hammer_reversal.py for the rule.

1. Prints/saves the full trigger list (every session a hammer fired,
   whether or not the stop-entry ever filled): date, trigger time, ATR/4,
   first-15-minute-bar range.
2. Launches a day-by-day Dash browser over just those trigger days, showing
   5-minute candles with the opening 15-min range shaded, the trigger bar
   marked, and entry/SL/TP drawn — so each trigger can be inspected visually.
"""

import argparse

import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, ctx, dcc, html
from loguru import logger

from algo.hammer_reversal import list_hammer_triggers
from algo.resample_bars import resample_to_timeframe
from models.paths import get_file
from visualization.plotting import plot_bars

TICKER = "SPY"
OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]


def load_triggers() -> tuple[pl.DataFrame, pl.DataFrame]:
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
    triggers = list_hammer_triggers(minute_bars, daily_bars)
    bars5 = resample_to_timeframe(minute_bars, "5m")
    return triggers, bars5


def _build_figure(day_bars: pl.DataFrame, trigger_row: dict) -> go.Figure:
    trade = pl.DataFrame(
        [
            {
                "DateTime": trigger_row["trigger_time"],
                "entry_price": trigger_row["entry_price"],
                "direction": trigger_row["direction"],
                "stop_loss": trigger_row["sl"],
                "take_profit": trigger_row["tp"],
            }
        ]
    )
    title = (
        f"{TICKER} — {trigger_row['date']} ({trigger_row['direction']})  |  "
        f"ATR/4 threshold={trigger_row['atr_threshold']}  |  "
        f"first-15m range={trigger_row['opening_range']:.2f}"
    )
    fig = plot_bars(
        day_bars,
        bb_upper_col=None,
        bb_mid_col=None,
        bb_lower_col=None,
        trades=trade,
        title=title,
        show_fig=False,
        return_fig=True,
    )
    assert fig is not None

    fig.add_hrect(
        y0=trigger_row["or_low"],
        y1=trigger_row["or_high"],
        fillcolor="rgba(255, 255, 0, 0.08)",
        line_width=1,
        line_color="rgba(255, 255, 0, 0.4)",
        row=1,
        col=1,
    )
    fig.add_vline(
        x=trigger_row["trigger_time"],
        line_dash="dot",
        line_color="yellow",
        row=1,
        col=1,
    )
    for y, name, color in [
        (trigger_row["sl"], "SL", "#ff3333"),
        (trigger_row["tp"], "TP", "#00cc44"),
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

    return fig


def run(port: int = 8051) -> None:
    triggers, bars5 = load_triggers()

    out_path = "output/hammer_reversal/spy_triggers.csv"
    triggers.write_csv(out_path)
    logger.info(f"{triggers.height} triggers found — saved to {out_path}")
    pl.Config.set_tbl_rows(triggers.height)
    pl.Config.set_tbl_width_chars(200)
    print(
        triggers.select(
            "date",
            "trigger_time",
            "direction",
            "atr_threshold",
            "opening_range",
        )
    )

    dates = triggers["date"].to_list()
    str_dates = [str(d) for d in dates]
    triggers_by_date = {row["date"]: row for row in triggers.iter_rows(named=True)}

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
        return _build_figure(day_bars, triggers_by_date[day])

    logger.info(f"Starting browser at http://localhost:{port}")
    app.run(debug=False, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hammer-reversal trigger browser")
    parser.add_argument("--port", type=int, default=8051)
    args = parser.parse_args()
    run(port=args.port)
