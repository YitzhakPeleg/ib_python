# %% 1. Imports and Data Loading
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from bollinger_bands import calculate_bollinger_bands
from date_converter import add_date_int_column

# Load the dataset
ticker = "AAPL"
path = f"/Users/yitzhakpeleg/Projects/ib_python/{ticker}_1_min.parquet"
df = pl.read_parquet(path)
df = add_date_int_column(df)
df = calculate_bollinger_bands(df, window=20, stds=2.0)
# %% 2. Feature Engineering
# Adding date parsing, Bollinger Band metrics, and row numbering
df = (
    df.with_columns(
        [
            # Calculate BB metrics
            (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_size"),
            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_mid")).alias(
                "bb_ratio"
            ),
        ]
    )
    .with_columns(
        [
            # Advanced BB metrics (scaled to integers for easier distribution analysis)
            (pl.col("bb_ratio") * 1000).cast(pl.Int32).alias("bb_ratio_percent"),
            # Helper for grouping/plotting
            pl.col("date").cast(pl.String).alias("date_str"),
        ]
    )
    .with_columns(
        [
            # Row number within each day (0...N)
            pl.int_range(0, pl.len())
            .over("date", order_by="DateTime")
            .alias("row_nr_day"),
            # Max bb_ratio_percent in the first 30 mins of each day
            pl.col("bb_ratio_percent")
            .sort_by("DateTime")
            .head(30)
            .max()
            .over("date")
            .alias("max_bb_ratio_opening"),
        ]
    )
)

# %% 3. Visualizations - Time Series & Intraday
# Viewing a specific day's bands
target_date = 20260417
px.line(
    df.filter(pl.col("date") == target_date),
    x="DateTime",
    y=["bb_lower", "bb_mid", "bb_upper"],
    title=f"Bollinger Bands for {target_date}",
).show()

# Intraday BB Size across multiple days
px.scatter(
    df.filter(pl.col("date") >= 20260401),
    x="row_nr_day",
    y="bb_ratio_percent",
    color="date_str",
    title="Intraday BB Ratio Evolution",
).show()

# %% 4. Visualizations - Distributions
# Histogram of BB Ratios
px.histogram(
    df, x="bb_ratio_percent", title="Distribution of BB Ratio Percent", nbins=30
).show()

# %% 5. Analysis of Opening Volatility
# Scatter plot of the maximum ratio found in the first 30 rows of each day
px.scatter(
    df.unique("date"),  # Only one point per day needed
    x="date_str",
    y="max_bb_ratio_opening",
    title="Maximum BB Ratio in Opening 30 Mins",
).show()

# %% Candlestick Chart for a Specific Day
# 1. Filter for a specific day
day_df = df.filter(pl.col("date").is_between(20260410, 20260416))
# .filter(pl.col("row_nr_day") < 30)
# 2. Create the Candlestick object
fig = go.Figure(
    data=[
        go.Candlestick(
            x=day_df["DateTime"],
            open=day_df["Open"],
            high=day_df["High"],
            low=day_df["Low"],
            close=day_df["Close"],
            name=ticker,
        )
    ]
)
fig.add_trace(
    go.Scatter(
        x=day_df["DateTime"],
        y=day_df["bb_upper"],
        line=dict(color="rgba(173, 216, 230, 0.5)"),
        name="Upper Band",
    )
)

fig.add_trace(
    go.Scatter(
        x=day_df["DateTime"],
        y=day_df["bb_mid"],
        line=dict(color="orange", dash="dash"),
        name="Mid Band",
    )
)

fig.add_trace(
    go.Scatter(
        x=day_df["DateTime"],
        y=day_df["bb_lower"],
        line=dict(color="rgba(173, 216, 230, 0.5)"),
        name="Lower Band",
    )
)

fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark")
fig.show()
# %%
