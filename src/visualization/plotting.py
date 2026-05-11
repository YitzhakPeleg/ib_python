"""Interactive candlestick charts with technical indicators using Plotly."""

from typing import Optional

import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots


def plot_bars(
    df: pl.DataFrame,
    *,
    # Required columns
    datetime_col: str = "DateTime",
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    volume_col: str = "Volume",
    # Bollinger Bands columns (optional, pre-calculated)
    bb_upper_col: Optional[str] = "bb_upper",
    bb_mid_col: Optional[str] = "bb_mid",
    bb_lower_col: Optional[str] = "bb_lower",
    # Subplot configuration
    show_volume: bool = True,
    # Visual configuration
    title: Optional[str] = None,
    height: int = 800,
    theme: str = "plotly_dark",
    # Display options
    show_fig: bool = True,
    return_fig: bool = False,
) -> Optional[go.Figure]:
    """
    Create an interactive candlestick chart with Bollinger Bands and volume.

    This function creates a professional trading chart with:
    - Candlestick chart for OHLC data
    - Bollinger Bands overlay (if columns provided)
    - Volume subplot (optional)

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing OHLC data and pre-calculated indicators.
        Must have columns for DateTime, Open, High, Low, Close, and Volume.
    datetime_col : str, default "DateTime"
        Column name for datetime/timestamp data.
    open_col : str, default "Open"
        Column name for opening prices.
    high_col : str, default "High"
        Column name for high prices.
    low_col : str, default "Low"
        Column name for low prices.
    close_col : str, default "Close"
        Column name for closing prices.
    volume_col : str, default "Volume"
        Column name for volume data.
    bb_upper_col : str | None, default "bb_upper"
        Column name for Bollinger Bands upper band. Set to None to skip.
    bb_mid_col : str | None, default "bb_mid"
        Column name for Bollinger Bands middle band. Set to None to skip.
    bb_lower_col : str | None, default "bb_lower"
        Column name for Bollinger Bands lower band. Set to None to skip.
    show_volume : bool, default True
        Whether to show volume subplot below the price chart.
    title : str | None, default None
        Chart title. If None, auto-generates from data.
    height : int, default 800
        Total chart height in pixels.
    theme : str, default "plotly_dark"
        Plotly theme name. Options: 'plotly_dark', 'plotly_white', 'plotly', etc.
    show_fig : bool, default True
        Whether to display the figure immediately.
    return_fig : bool, default False
        Whether to return the figure object.

    Returns
    -------
    go.Figure | None
        Plotly figure object if return_fig=True, otherwise None.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.

    Examples
    --------
    Basic usage with Bollinger Bands:

    >>> import polars as pl
    >>> from src.visualization.plotting import plot_bars
    >>> from src.algo.bollinger_bands import calculate_bollinger_bands
    >>>
    >>> df = pl.read_parquet("AAPL_1_min.parquet")
    >>> df = calculate_bollinger_bands(df, window=20, stds=2.0)
    >>> plot_bars(df, title="AAPL - 1 Minute Bars")

    Without volume subplot:

    >>> plot_bars(df, show_volume=False, title="AAPL - Price Only")

    Filter to specific date range:

    >>> day_df = df.filter(pl.col("date") == 20260417)
    >>> plot_bars(day_df, title="AAPL - April 17, 2026")

    Notes
    -----
    - Bollinger Bands must be pre-calculated using `calculate_bollinger_bands()`
    - The chart is interactive with zoom, pan, and hover tooltips
    - Volume bars are color-coded: green (up days) and red (down days)
    """
    # Validate required columns
    required_cols = [datetime_col, open_col, high_col, low_col, close_col]
    if show_volume:
        required_cols.append(volume_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    logger.info(f"Creating chart with {len(df)} bars")

    # Check for Bollinger Bands
    has_bb = all(
        col is not None and col in df.columns
        for col in [bb_upper_col, bb_mid_col, bb_lower_col]
    )
    if not has_bb and any(
        col is not None for col in [bb_upper_col, bb_mid_col, bb_lower_col]
    ):
        logger.warning(
            "Some Bollinger Band columns missing. Skipping BB overlay. "
            "Use calculate_bollinger_bands() to add them."
        )

    # Create subplots
    if show_volume:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume"),
        )
        volume_row = 2
    else:
        fig = make_subplots(rows=1, cols=1)
        volume_row = None

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df[datetime_col],
            open=df[open_col],
            high=df[high_col],
            low=df[low_col],
            close=df[close_col],
            name="OHLC",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1,
        col=1,
    )

    # Add Bollinger Bands if available
    if has_bb:
        # Type narrowing: we know these are not None when has_bb is True
        assert bb_upper_col is not None
        assert bb_mid_col is not None
        assert bb_lower_col is not None

        # Upper band
        fig.add_trace(
            go.Scatter(
                x=df[datetime_col],
                y=df[bb_upper_col],
                line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
                name="BB Upper",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Middle band
        fig.add_trace(
            go.Scatter(
                x=df[datetime_col],
                y=df[bb_mid_col],
                line=dict(color="orange", width=1, dash="dash"),
                name="BB Mid",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Lower band
        fig.add_trace(
            go.Scatter(
                x=df[datetime_col],
                y=df[bb_lower_col],
                line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
                name="BB Lower",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        logger.info("Added Bollinger Bands overlay")

    # Add volume subplot
    if show_volume:
        # Determine volume bar colors (green for up days, red for down days)
        colors = [
            "green" if close >= open_ else "red"
            for close, open_ in zip(df[close_col], df[open_col])
        ]

        fig.add_trace(
            go.Bar(
                x=df[datetime_col],
                y=df[volume_col],
                name="Volume",
                marker_color=colors,
                showlegend=False,
            ),
            row=volume_row,
            col=1,
        )

        logger.info("Added volume subplot")

    # Generate title if not provided
    if title is None:
        date_range = f"{df[datetime_col].min()} to {df[datetime_col].max()}"
        title = f"Candlestick Chart ({len(df)} bars)"
        logger.debug(f"Auto-generated title: {title}")

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=height,
        template=theme,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # Update x-axis labels
    if show_volume:
        fig.update_xaxes(title_text="Time", row=2, col=1)
    else:
        fig.update_xaxes(title_text="Time", row=1, col=1)

    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    logger.info(f"Chart created successfully with theme '{theme}'")

    # Display the figure
    if show_fig:
        fig.show()
        logger.debug("Figure displayed")

    # Return the figure if requested
    if return_fig:
        return fig

    return None


# Made with Bob
