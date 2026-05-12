"""Trade journal: write, read, and plot backtest trade logs."""

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots


def write_trades(trades_df: pl.DataFrame, path: str | Path, append: bool = False) -> None:
    """
    Write a trades DataFrame to a CSV journal file.

    Args:
        trades_df: DataFrame with trade results (any columns).
        path: Destination CSV path. Parent directories are created if needed.
        append: If True and the file exists, append rows (no header written).
                If False (default), overwrite the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        existing = pl.read_csv(path)
        combined = pl.concat([existing, trades_df], how="diagonal")
        combined.write_csv(path)
        logger.info(f"Appended {len(trades_df)} trades to {path} (total: {len(combined)})")
    else:
        trades_df.write_csv(path)
        logger.info(f"Wrote {len(trades_df)} trades to {path}")


def read_trades(path: str | Path) -> pl.DataFrame:
    """
    Read a trade journal CSV back into a Polars DataFrame.

    Numeric columns are inferred automatically by Polars.

    Args:
        path: Path to the CSV file written by write_trades.

    Returns:
        DataFrame with trade records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trade journal not found: {path}")

    df = pl.read_csv(path, infer_schema_length=1000)
    logger.info(f"Read {len(df)} trades from {path}")
    return df


def plot_journal(
    trades_df: pl.DataFrame,
    title: str = "Trade Journal",
    theme: str = "plotly_dark",
    height: int = 700,
    show_fig: bool = True,
    return_fig: bool = False,
) -> go.Figure | None:
    """
    Plot a 2-panel trade journal figure.

    Top panel:  cumulative PnL curve over trade sequence.
    Bottom panel: per-trade PnL bars, colored green (win) / red (loss).

    Args:
        trades_df: DataFrame with at least ``pnl`` and ``outcome`` columns.
                   An ``r_multiple`` column is used for hover if present.
        title: Figure title.
        theme: Plotly theme (default ``"plotly_dark"``).
        height: Figure height in pixels.
        show_fig: Display the figure immediately.
        return_fig: Return the figure object.

    Returns:
        go.Figure if return_fig=True, else None.
    """
    required = {"pnl", "outcome"}
    missing = required - set(trades_df.columns)
    if missing:
        raise ValueError(f"trades_df missing required columns: {missing}")

    n = len(trades_df)
    cum_pnl = trades_df["pnl"].cum_sum().to_list()
    trade_idx = list(range(1, n + 1))
    outcomes = trades_df["outcome"].to_list()
    pnls = trades_df["pnl"].to_list()

    bar_colors = [
        "#00cc44" if o == "win" else "#ff3333" if o == "loss" else "#aaaaaa"
        for o in outcomes
    ]

    has_r = "r_multiple" in trades_df.columns
    has_direction = "direction" in trades_df.columns

    hover_texts = []
    for i in range(n):
        parts = [f"Trade #{i+1}", f"PnL: ${pnls[i]:.2f}", f"Outcome: {outcomes[i]}"]
        if has_r:
            parts.append(f"R: {trades_df['r_multiple'][i]:.2f}")
        if has_direction:
            parts.append(f"Dir: {trades_df['direction'][i]}")
        hover_texts.append("<br>".join(parts))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.45],
        subplot_titles=("Cumulative PnL", "Per-Trade PnL"),
    )

    # Cumulative PnL line
    final_pnl = cum_pnl[-1] if cum_pnl else 0
    line_color = "#00cc44" if final_pnl >= 0 else "#ff3333"
    fig.add_trace(
        go.Scatter(
            x=trade_idx,
            y=cum_pnl,
            mode="lines",
            line=dict(color=line_color, width=2),
            fill="tozeroy",
            fillcolor=line_color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in line_color
                else f"rgba(0, 204, 68, 0.15)" if final_pnl >= 0 else "rgba(255, 51, 51, 0.15)",
            name="Cumulative PnL",
            hovertext=[f"Trade #{i+1}<br>Cum PnL: ${v:.2f}" for i, v in enumerate(cum_pnl)],
            hoverinfo="text",
        ),
        row=1, col=1,
    )

    # Zero reference line
    fig.add_hline(y=0, line=dict(color="white", width=1, dash="dot"), row=1, col=1)

    # Per-trade bars
    fig.add_trace(
        go.Bar(
            x=trade_idx,
            y=pnls,
            marker_color=bar_colors,
            name="Trade PnL",
            hovertext=hover_texts,
            hoverinfo="text",
        ),
        row=2, col=1,
    )

    # Summary annotation
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    win_rate = wins / n if n else 0
    fig.add_annotation(
        text=f"Trades: {n} | Win rate: {win_rate:.1%} | W: {wins} L: {losses} | Total PnL: ${final_pnl:.2f}",
        xref="paper", yref="paper",
        x=0, y=1.08, showarrow=False,
        font=dict(size=12, color="white"),
        align="left",
    )

    fig.update_layout(
        title=title,
        template=theme,
        height=height,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Trade #", row=2, col=1)
    fig.update_yaxes(title_text="$ PnL", row=1, col=1)
    fig.update_yaxes(title_text="$ PnL", row=2, col=1)

    logger.info(f"Journal plot created: {n} trades, {win_rate:.1%} win rate, ${final_pnl:.2f} total PnL")

    if show_fig:
        fig.show()
    if return_fig:
        return fig
    return None
