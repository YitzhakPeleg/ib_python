# Visualization Module

Interactive candlestick charts with technical indicators using Plotly.

## Overview

The visualization module provides a simple, powerful interface for creating professional trading charts with:
- **Candlestick charts** for OHLC (Open, High, Low, Close) data
- **Bollinger Bands overlay** for volatility analysis
- **Volume subplot** with color-coded bars
- **Interactive features** including zoom, pan, and hover tooltips

## Installation

The module uses Plotly, which is already included in the project dependencies:

```bash
# Already installed via pyproject.toml
plotly[express]>=6.2.0
```

## Quick Start

### Basic Usage

```python
import polars as pl
from src.visualization import plot_bars
from src.algo.bollinger_bands import calculate_bollinger_bands

# Load your data
df = pl.read_parquet("AAPL_1_min.parquet")

# Calculate Bollinger Bands
df = calculate_bollinger_bands(df, window=20, stds=2.0)

# Create the chart
plot_bars(df, title="AAPL - 1 Minute Bars")
```

## Function Reference

### `plot_bars()`

Create an interactive candlestick chart with Bollinger Bands and volume.

#### Parameters

**Data Columns:**
- `df` (pl.DataFrame): Polars DataFrame with OHLC data
- `datetime_col` (str, default="DateTime"): Column name for timestamps
- `open_col` (str, default="Open"): Column name for opening prices
- `high_col` (str, default="High"): Column name for high prices
- `low_col` (str, default="Low"): Column name for low prices
- `close_col` (str, default="Close"): Column name for closing prices
- `volume_col` (str, default="Volume"): Column name for volume data

**Indicator Columns (Optional):**
- `bb_upper_col` (str | None, default="bb_upper"): Bollinger Bands upper band
- `bb_mid_col` (str | None, default="bb_mid"): Bollinger Bands middle band
- `bb_lower_col` (str | None, default="bb_lower"): Bollinger Bands lower band

**Display Options:**
- `show_volume` (bool, default=True): Show volume subplot
- `title` (str | None, default=None): Chart title (auto-generated if None)
- `height` (int, default=800): Total chart height in pixels
- `theme` (str, default="plotly_dark"): Plotly theme name
- `show_fig` (bool, default=True): Display the figure immediately
- `return_fig` (bool, default=False): Return the figure object

#### Returns

- `go.Figure | None`: Plotly figure object if `return_fig=True`, otherwise None

## Usage Examples

### Example 1: Full Chart with All Features

```python
import polars as pl
from src.visualization import plot_bars
from src.algo.bollinger_bands import calculate_bollinger_bands

# Load and prepare data
df = pl.read_parquet("AAPL_1_min.parquet")
df = calculate_bollinger_bands(df, window=20, stds=2.0)

# Create comprehensive chart
plot_bars(
    df,
    title="AAPL - Complete Analysis",
    height=900,
    theme="plotly_dark",
)
```

### Example 2: Price Chart Only (No Volume)

```python
# Show only price action with Bollinger Bands
plot_bars(
    df,
    title="AAPL - Price Action",
    show_volume=False,
    height=600,
)
```

### Example 3: Without Bollinger Bands

```python
# Just candlesticks and volume
plot_bars(
    df,
    title="AAPL - Basic Chart",
    bb_upper_col=None,
    bb_mid_col=None,
    bb_lower_col=None,
)
```

### Example 4: Filter to Specific Date Range

```python
# Plot a single day
day_df = df.filter(pl.col("date") == 20260417)
plot_bars(
    day_df,
    title="AAPL - April 17, 2026",
    height=700,
)

# Plot a date range
week_df = df.filter(pl.col("date").is_between(20260410, 20260417))
plot_bars(
    week_df,
    title="AAPL - Week of April 10-17, 2026",
)
```

### Example 5: Morning Trading Window

```python
from src.algo.signal_detector import filter_morning_window

# Filter to morning window (9:00-11:00)
morning_df = filter_morning_window(df, start_hour=9, end_hour=11)
morning_df = calculate_bollinger_bands(morning_df, window=20, stds=2.0)

plot_bars(
    morning_df,
    title="AAPL - Morning Trading Window (9:00-11:00)",
)
```

### Example 6: Light Theme

```python
# Use light theme for presentations
plot_bars(
    df,
    title="AAPL - Light Theme",
    theme="plotly_white",
)
```

### Example 7: Save Figure to File

```python
# Return figure and save to HTML
fig = plot_bars(
    df,
    title="AAPL Analysis",
    show_fig=False,  # Don't display
    return_fig=True,  # Return figure object
)

# Save to HTML file
fig.write_html("aapl_chart.html")

# Or save as static image (requires kaleido)
# fig.write_image("aapl_chart.png", width=1920, height=1080)
```

### Example 8: Custom Column Names

```python
# If your DataFrame has different column names
plot_bars(
    df,
    datetime_col="timestamp",
    open_col="open_price",
    high_col="high_price",
    low_col="low_price",
    close_col="close_price",
    volume_col="trade_volume",
    title="Custom Column Names",
)
```

## Visual Features

### Candlestick Colors
- **Green**: Closing price higher than opening price (bullish)
- **Red**: Closing price lower than opening price (bearish)

### Bollinger Bands
- **Light Blue Lines**: Upper and lower bands (volatility envelope)
- **Orange Dashed Line**: Middle band (20-period SMA)
- **Transparency**: Bands use 50% opacity for better visibility

### Volume Bars
- **Green Bars**: Volume on up days (close > open)
- **Red Bars**: Volume on down days (close < open)

### Interactive Features
- **Zoom**: Click and drag to zoom into a region
- **Pan**: Hold shift and drag to pan
- **Reset**: Double-click to reset view
- **Hover**: Hover over bars to see detailed OHLC data
- **Legend**: Click legend items to show/hide traces

## Integration with Existing Code

### With Bollinger Bands Module

```python
from src.algo.bollinger_bands import calculate_bollinger_bands
from src.visualization import plot_bars

df = calculate_bollinger_bands(df, window=20, stds=2.0)
plot_bars(df)
```

### With Signal Detection System

```python
from src.algo.signal_detector import filter_morning_window
from src.algo.bollinger_bands import calculate_bollinger_bands
from src.visualization import plot_bars

# Get morning window data
morning_df = filter_morning_window(df)
morning_df = calculate_bollinger_bands(morning_df, window=20, stds=2.0)

# Visualize
plot_bars(morning_df, title="Morning Trading Signals")
```

### With Date Converter

```python
from src.data_fetching.date_converter import add_date_int_column
from src.visualization import plot_bars

# Add date column for filtering
df = add_date_int_column(df)

# Filter and plot specific dates
target_date = 20260417
day_df = df.filter(pl.col("date") == target_date)
plot_bars(day_df, title=f"Trading Day {target_date}")
```

## Themes

Available Plotly themes:
- `plotly_dark` (default) - Dark background, ideal for trading
- `plotly_white` - Light background, good for presentations
- `plotly` - Default Plotly theme
- `ggplot2` - ggplot2 style
- `seaborn` - Seaborn style
- `simple_white` - Minimal white theme
- `none` - No theme

## Performance Tips

1. **Large Datasets**: For datasets with >10,000 bars, consider filtering to specific date ranges
2. **Memory**: Plotly creates interactive HTML, which can be memory-intensive for very large datasets
3. **Rendering**: First render may be slow; subsequent interactions are fast
4. **File Size**: HTML exports can be large (5-10MB for 10,000 bars)

## Troubleshooting

### Missing Bollinger Bands

If you see a warning about missing Bollinger Band columns:

```python
# Calculate them first
from src.algo.bollinger_bands import calculate_bollinger_bands
df = calculate_bollinger_bands(df, window=20, stds=2.0)
```

### Empty Chart

If the chart appears empty:
- Check that your DataFrame is not empty: `len(df) > 0`
- Verify column names match the parameters
- Check for NaN values in OHLC columns

### Type Errors

If you get type errors about column names:
- Ensure all column name parameters are strings, not None
- Or explicitly set to None to skip: `bb_upper_col=None`

## Advanced Usage

### Multiple Charts in Jupyter

```python
# Create multiple charts without blocking
fig1 = plot_bars(df1, title="Chart 1", show_fig=False, return_fig=True)
fig2 = plot_bars(df2, title="Chart 2", show_fig=False, return_fig=True)

# Display them
fig1.show()
fig2.show()
```

### Customize After Creation

```python
# Get the figure object
fig = plot_bars(df, show_fig=False, return_fig=True)

# Add custom annotations
fig.add_annotation(
    x="2026-04-17 10:30:00",
    y=150.0,
    text="Important Event",
    showarrow=True,
    arrowhead=2,
)

# Show the modified figure
fig.show()
```

## API Design Philosophy

The module follows these principles:

1. **Pre-calculated Indicators**: Separates calculation from visualization for flexibility
2. **Sensible Defaults**: Works out-of-the-box with standard column names
3. **Type Safety**: Full type hints for IDE support
4. **Logging**: Uses loguru for debugging and monitoring
5. **Polars-First**: Optimized for Polars DataFrames (project standard)

## See Also

- [Bollinger Bands Module](../algo/bollinger_bands.py) - Calculate Bollinger Bands
- [Signal Detection](../algo/signal_detector.py) - Morning window filtering
- [Feature Engineering](../algo/feature_engineering.py) - Technical indicators
- [Plotly Documentation](https://plotly.com/python/) - Full Plotly reference

---

**Made with Bob**
