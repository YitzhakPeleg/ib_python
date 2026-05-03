"""Example usage of the visualization module with real data."""

import polars as pl
from loguru import logger

from algo.bollinger_bands import calculate_bollinger_bands
from data_fetching.date_converter import add_date_int_column
from models import BarFrequency, get_file
from visualization import plot_bars

# Configure logger
logger.info("Starting visualization example")


def example_1_basic_chart():
    """Example 1: Basic chart with all features."""
    logger.info("Example 1: Basic chart with Bollinger Bands and Volume")

    # Load data
    ticker = "AAPL"
    freqency = BarFrequency.ONE_MIN
    df = pl.read_parquet(get_file(ticker, freqency))

    # Add date column for filtering
    df = add_date_int_column(df)

    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df, window=20, stds=2.0)

    # Create chart
    fig = plot_bars(
        df,
        title=f"{ticker} - 1 Minute Bars with Bollinger Bands",
        height=900,
        return_fig=True,
        show_fig=False,
    )
    if fig:
        fig.show(renderer="browser")


def example_2_single_day():
    """Example 2: Plot a single trading day."""
    logger.info("Example 2: Single day chart")

    # Load data
    df = pl.read_parquet("AAPL_1_min.parquet")
    df = add_date_int_column(df)
    df = calculate_bollinger_bands(df, window=20, stds=2.0)

    # Filter to specific day
    target_date = 20260417
    day_df = df.filter(pl.col("date") == target_date)

    logger.info(f"Plotting {len(day_df)} bars for date {target_date}")

    plot_bars(
        day_df,
        title=f"AAPL - {target_date}",
        height=700,
    )


def example_3_date_range():
    """Example 3: Plot a date range."""
    logger.info("Example 3: Date range chart")

    # Load data
    df = pl.read_parquet("AAPL_1_min.parquet")
    df = add_date_int_column(df)
    df = calculate_bollinger_bands(df, window=20, stds=2.0)

    # Filter to date range
    start_date = 20260410
    end_date = 20260417
    range_df = df.filter(pl.col("date").is_between(start_date, end_date))

    logger.info(f"Plotting {len(range_df)} bars from {start_date} to {end_date}")

    plot_bars(
        range_df,
        title="AAPL - Week of April 10-17, 2026",
        height=800,
    )


def example_4_no_volume():
    """Example 4: Chart without volume subplot."""
    logger.info("Example 4: Price chart only (no volume)")

    # Load data
    df = pl.read_parquet("AAPL_1_min.parquet")
    df = add_date_int_column(df)
    df = calculate_bollinger_bands(df, window=20, stds=2.0)

    # Filter to single day
    day_df = df.filter(pl.col("date") == 20260417)

    plot_bars(
        day_df,
        title="AAPL - Price Action Only",
        show_volume=False,
        height=600,
    )


def example_5_no_bollinger_bands():
    """Example 5: Chart without Bollinger Bands."""
    logger.info("Example 5: Candlesticks and volume only")

    # Load data (no BB calculation)
    df = pl.read_parquet("AAPL_1_min.parquet")
    df = add_date_int_column(df)

    # Filter to single day
    day_df = df.filter(pl.col("date") == 20260417)

    plot_bars(
        day_df,
        title="AAPL - Basic Candlesticks",
        bb_upper_col=None,
        bb_mid_col=None,
        bb_lower_col=None,
        height=700,
    )


def example_6_morning_window():
    """Example 6: Morning trading window (9:00-11:00)."""
    logger.info("Example 6: Morning trading window")

    from src.algo.signal_detector import filter_morning_window

    # Load data
    df = pl.read_parquet("AAPL_1_min.parquet")
    df = add_date_int_column(df)

    # Filter to morning window
    morning_df = filter_morning_window(df, start_hour=9, end_hour=11)
    morning_df = calculate_bollinger_bands(morning_df, window=20, stds=2.0)

    plot_bars(
        morning_df,
        title="AAPL - Morning Trading Window (9:00-11:00)",
        height=800,
    )


def example_7_light_theme():
    """Example 7: Light theme for presentations."""
    logger.info("Example 7: Light theme")

    # Load data
    df = pl.read_parquet("AAPL_1_min.parquet")
    df = add_date_int_column(df)
    df = calculate_bollinger_bands(df, window=20, stds=2.0)

    # Filter to single day
    day_df = df.filter(pl.col("date") == 20260417)

    plot_bars(
        day_df,
        title="AAPL - Light Theme",
        theme="plotly_white",
        height=700,
    )


def example_8_save_to_file():
    """Example 8: Save chart to HTML file."""
    logger.info("Example 8: Save to file")

    # Load data
    df = pl.read_parquet("AAPL_1_min.parquet")
    df = add_date_int_column(df)
    df = calculate_bollinger_bands(df, window=20, stds=2.0)

    # Filter to single day
    day_df = df.filter(pl.col("date") == 20260417)

    # Get figure without displaying
    fig = plot_bars(
        day_df,
        title="AAPL - Saved Chart",
        show_fig=False,
        return_fig=True,
    )

    # Save to HTML
    if fig is not None:
        output_file = "aapl_chart.html"
        fig.write_html(output_file)
        logger.info(f"Chart saved to {output_file}")
    else:
        logger.error("Failed to create figure")


if __name__ == "__main__":
    # Run examples
    # Uncomment the examples you want to run

    example_1_basic_chart()
    # example_2_single_day()
    # example_3_date_range()
    # example_4_no_volume()
    # example_5_no_bollinger_bands()
    # example_6_morning_window()
    # example_7_light_theme()
    # example_8_save_to_file()

    # Default: run basic example
    # logger.info("Running default example (basic chart)")
    # logger.info("Uncomment other examples in the script to try them")
    # example_2_single_day()


# Made with Bob
