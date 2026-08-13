"""Day-range predictor — prediction visualization.

Reads a predictions_val_*.csv (written by examples/train_day_range_nn.py)
and produces two plots:

1. Scatter: predicted vs actual high (and low), colored by outcome
   (inside/outside/mixed), with a y=x reference line — shows calibration:
   points below the line on the high panel / above the line on the low
   panel are the "safe" (inside) direction the asymmetric loss rewards.
2. Per-day range chart: every validation day along the x-axis, with the
   REAL day's range (true_low..true_high) and the PREDICTED range
   (pred_low..pred_high) drawn as two side-by-side vertical segments, in
   return-% terms (relative to prev_close) so days/years with different
   price levels are directly comparable.
"""

import argparse
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots

OUTCOME_COLORS = {"inside": "#00cc44", "outside": "#ff3333", "mixed": "#cccc00"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot day-range predictions vs actuals")
    p.add_argument(
        "--predictions-csv",
        default="output/day_range_nn/predictions_val_SPY_2025-2025.csv",
    )
    p.add_argument(
        "--output-dir",
        default="output/day_range_nn",
    )
    return p.parse_args()


def plot_scatter(df: pl.DataFrame, output_path: Path) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("High: predicted vs actual", "Low: predicted vs actual"),
    )
    for outcome, color in OUTCOME_COLORS.items():
        sub = df.filter(pl.col("outcome") == outcome)
        if sub.height == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["true_high"],
                y=sub["pred_high"],
                mode="markers",
                name=outcome,
                marker=dict(color=color, size=6),
                legendgroup=outcome,
                text=sub["date"].cast(pl.Utf8),
                hovertemplate="%{text}<br>true=%{x:.2f} pred=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["true_low"],
                y=sub["pred_low"],
                mode="markers",
                name=outcome,
                marker=dict(color=color, size=6),
                legendgroup=outcome,
                showlegend=False,
                text=sub["date"].cast(pl.Utf8),
                hovertemplate="%{text}<br>true=%{x:.2f} pred=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    for col, price_col in [(1, "true_high"), (2, "true_low")]:
        lo = float(
            min(df[price_col].min(), df[f"pred_{'high' if col == 1 else 'low'}"].min())
        )
        hi = float(
            max(df[price_col].max(), df[f"pred_{'high' if col == 1 else 'low'}"].max())
        )
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="gray", dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

    fig.update_xaxes(title_text="actual price ($)", row=1, col=1)
    fig.update_xaxes(title_text="actual price ($)", row=1, col=2)
    fig.update_yaxes(title_text="predicted price ($)", row=1, col=1)
    fig.update_yaxes(title_text="predicted price ($)", row=1, col=2)
    fig.update_layout(
        height=500,
        title=(
            "Predicted vs actual — points on the dotted line are exact; below it on "
            "the High panel / above it on the Low panel is the 'safe' (inside) side"
        ),
    )
    fig.write_html(str(output_path))
    logger.info(f"Wrote {output_path}")


def plot_per_day_ranges(df: pl.DataFrame, output_path: Path) -> None:
    df = df.sort("date")
    prev_close = df["prev_close"]
    true_high_ret = (df["true_high"] / prev_close - 1) * 100
    true_low_ret = (df["true_low"] / prev_close - 1) * 100
    pred_high_ret = (df["pred_high"] / prev_close - 1) * 100
    pred_low_ret = (df["pred_low"] / prev_close - 1) * 100
    dates = df["date"].cast(pl.Utf8).to_list()
    outcomes = df["outcome"].to_list()
    n = df.height

    # One trace per range-type, using None-separated segments so each day's
    # vertical bracket is independent (not visually connected to the next
    # day's, which a single continuous line would misleadingly imply).
    true_x, true_y = [], []
    pred_x, pred_y = [], []
    for i in range(n):
        x_true, x_pred = i - 0.15, i + 0.15
        true_x += [x_true, x_true, None]
        true_y += [float(true_low_ret[i]), float(true_high_ret[i]), None]
        pred_x += [x_pred, x_pred, None]
        pred_y += [float(pred_low_ret[i]), float(pred_high_ret[i]), None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=true_x,
            y=true_y,
            mode="lines",
            line=dict(color="#1f77b4", width=3),
            name="actual range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_x,
            y=pred_y,
            mode="lines",
            line=dict(color="#ff7f0e", width=3, dash="dot"),
            name="predicted range",
        )
    )
    # Colored markers at the bottom showing outcome per day, for quick scanning.
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=[min(true_low_ret.min(), pred_low_ret.min()) - 0.3] * n,
            mode="markers",
            marker=dict(
                color=[OUTCOME_COLORS[o] for o in outcomes],
                size=5,
                symbol="square",
            ),
            name="outcome",
            text=dates,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    tick_step = max(n // 20, 1)
    fig.update_layout(
        height=550,
        title=(
            "Per-day: actual range (blue, solid) vs predicted range (orange, dashed) "
            "— return % from prior close. Bottom row: outcome (green=inside, "
            "red=outside, yellow=mixed)"
        ),
        xaxis=dict(
            title="date",
            tickmode="array",
            tickvals=list(range(0, n, tick_step)),
            ticktext=[dates[i] for i in range(0, n, tick_step)],
            tickangle=-45,
        ),
        yaxis=dict(title="return from prior close (%)"),
    )
    fig.write_html(str(output_path))
    logger.info(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    df = pl.read_csv(args.predictions_csv, try_parse_dates=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scatter(df, output_dir / "predictions_scatter.html")
    plot_per_day_ranges(df, output_dir / "predictions_per_day.html")


if __name__ == "__main__":
    main()
