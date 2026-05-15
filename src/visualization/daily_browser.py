"""Day-by-day candlestick chart browser with configurable indicator overlays."""

from __future__ import annotations

import argparse
from typing import Optional

import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, ctx, dcc, html
from loguru import logger

from src.algo.indicators.base import Indicator
from src.algo.indicators.bollinger_bands import BollingerBands
from src.models.models import BarFrequency
from src.models.paths import get_file
from src.visualization.plotting import plot_bars


def _apply_indicators(
    df: pl.DataFrame,
    indicators: list[Indicator],
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Apply each indicator; return augmented df and map of label → added columns."""
    indicator_cols: dict[str, list[str]] = {}
    for ind in indicators:
        before = set(df.columns)
        df = ind(df)
        label = type(ind).__name__
        indicator_cols[label] = sorted(set(df.columns) - before)
    return df, indicator_cols


def _get_dates(df: pl.DataFrame) -> list:
    """Return sorted list of unique dates from the `date` column."""
    return sorted(df["date"].unique().to_list())


def _build_figure(
    df: pl.DataFrame,
    day: object,
    active_labels: list[str],
    indicator_cols: dict[str, list[str]],
    ticker: str,
) -> go.Figure:
    day_df = df.filter(pl.col("date") == day)
    bb_on = "BollingerBands" in active_labels

    fig = plot_bars(
        day_df,
        bb_upper_col="bb_upper" if (bb_on and "bb_upper" in day_df.columns) else None,
        bb_mid_col="bb_mid" if (bb_on and "bb_mid" in day_df.columns) else None,
        bb_lower_col="bb_lower" if (bb_on and "bb_lower" in day_df.columns) else None,
        title=f"{ticker} — {day}",
        price_range_half=10,
        volume_min=10_000,
        volume_max=1_000_000,
        show_fig=False,
        return_fig=True,
    )
    assert fig is not None

    # Add non-BB indicator traces to the price panel
    for label, cols in indicator_cols.items():
        if label == "BollingerBands" or label not in active_labels:
            continue
        for col in cols:
            if col not in day_df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=day_df["DateTime"],
                    y=day_df[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1),
                ),
                row=1,
                col=1,
            )

    return fig


def run(
    ticker: str,
    frequency: str | BarFrequency = BarFrequency.ONE_MIN,
    indicators: Optional[list[Indicator]] = None,
    port: int = 8050,
) -> None:
    """Launch the day-by-day chart browser for *ticker*.

    Args:
        ticker: Ticker symbol — must have a Parquet file in data/.
        frequency: Bar frequency used when the data was fetched.
        indicators: Indicators to overlay (default: [BollingerBands()]).
        port: Local port for the Dash server.
    """
    if indicators is None:
        indicators = [BollingerBands(reset_per_day=True)]

    logger.info(f"Loading {ticker} data…")
    df = pl.read_parquet(get_file(ticker, frequency))
    if "date" not in df.columns:
        df = df.with_columns(pl.col("DateTime").dt.date().alias("date"))
    df, indicator_cols = _apply_indicators(df, indicators)

    dates = _get_dates(df)
    if not dates:
        raise ValueError(f"No data found for {ticker}")

    str_dates = [str(d) for d in dates]
    all_labels = list(indicator_cols.keys())
    logger.info(f"Loaded {len(dates)} trading days for {ticker}")

    app = Dash(__name__)
    app.layout = html.Div(
        [
            # Navigation bar
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
                    html.Span(
                        "  (← → arrow keys to navigate)",
                        style={
                            "color": "#888",
                            "fontSize": "12px",
                            "marginLeft": "12px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "padding": "8px 12px",
                },
            ),
            # Indicator toggles + volume scale
            html.Div(
                [
                    dcc.Checklist(
                        id="indicator-toggles",
                        options=[{"label": f"  {lbl}", "value": lbl} for lbl in all_labels],
                        value=all_labels,
                        inline=True,
                    ),
                    dcc.Checklist(
                        id="volume-log-toggle",
                        options=[{"label": "  Log volume", "value": "log"}],
                        value=[],
                        inline=True,
                        style={"marginLeft": "24px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "padding": "0 12px 8px"},
            ),
            # Chart
            dcc.Graph(id="chart", style={"height": "80vh"}),
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

    @app.callback(
        Output("chart", "figure"),
        Input("date-dropdown", "value"),
        Input("indicator-toggles", "value"),
        Input("volume-log-toggle", "value"),
    )
    def update_chart(
        date_str: str,
        active_labels: list[str] | None,
        volume_log: list[str] | None,
    ) -> go.Figure:
        matching = [d for d in dates if str(d) == date_str]
        day = matching[0] if matching else dates[0]
        fig = _build_figure(df, day, active_labels or [], indicator_cols, ticker)
        if "log" in (volume_log or []):
            # log10 range [4, 6] → [10 000, 1 000 000]
            fig.update_yaxes(type="log", range=[4, 6], title_text="Volume", row=2, col=1)
        return fig

    logger.info(f"Starting browser at http://localhost:{port}")
    app.run(debug=False, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day-by-day chart browser")
    parser.add_argument(
        "ticker", help="Ticker symbol (must have a Parquet file in data/)"
    )
    parser.add_argument(
        "--frequency", default="1 min", help='Bar frequency, e.g. "1 min", "5 min"'
    )
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    run(ticker=args.ticker, frequency=args.frequency, port=args.port)
