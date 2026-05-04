# %%


import polars as pl

from algo import daily_range_avg
from models import get_file
from models.models import BarFrequency
from models.paths import DATA_PATH

# %%
ticker = "SPY"
frequency = BarFrequency.ONE_MIN
print(f"{ticker = }")
df = pl.read_parquet(get_file(ticker, frequency))
print(df)
df = daily_range_avg(df, 14)
df.tail()

# %%

df = df.with_columns(
    time=df["DateTime"].dt.time(),
    date=df["DateTime"].dt.date(),
)
df.head()
# %%
first_hour = df.filter(pl.col("time") < pl.time(10, 30, 0))
first_hour.tail()
# %%
# Calculate daily statistics
daily_stats = df.group_by("date").agg(
    [
        # First bar high and low
        pl.col("High").first().alias("first_bar_high"),
        pl.col("Low").first().alias("first_bar_low"),
        # Whole day high and low
        pl.col("High").max().alias("day_high"),
        pl.col("Low").min().alias("day_low"),
    ]
)
# %%
# 1. Join the daily stats onto the main dataframe
df_enriched = df.join(daily_stats, on="date")
# %%
# 2. Find the first row where High crosses the first bar high
high_breakout_rows = (
    df_enriched.filter(pl.col("High") > pl.col("first_bar_high"))
    .group_by("date")
    .first()
    .sort("date")
    .select(
        "date",
        "Volume",
        "avg_daily_range",
        "first_bar_high",
        pl.col("DateTime").alias("time_at_breakout_high"),
    )
)
low_breakout_rows = (
    df_enriched.filter(pl.col("Low") < pl.col("first_bar_low"))
    .group_by("date")
    .first()
    .sort("date")
    .select(
        "date",
        "first_bar_low",
        pl.col("DateTime").alias("time_at_breakout_low"),
    )
)
# %%
# 3. Find the row(s) where High equals the day's high
# Note: If a high is touched multiple times, .first() picks the earliest
day_high_rows = (
    df_enriched.filter(pl.col("High") == pl.col("day_high"))
    .group_by("date")
    .first()
    .sort("date")
    .select(
        "date",
        "day_high",
        pl.col("DateTime").alias("time_at_day_high"),
    )
)
day_low_rows = (
    df_enriched.filter(pl.col("Low") == pl.col("day_low"))
    .group_by("date")
    .first()
    .sort("date")
    .select(
        "date",
        "day_low",
        pl.col("DateTime").alias("time_at_day_low"),
    )
)
# %%
first_bar = (
    df_enriched.group_by("date")
    .agg(pl.col("DateTime").min().alias("first_bar_time"))
    .sort("date")
    .select("date", pl.col("first_bar_time"))
)


# 2. Join the breakout info with the slimmed-down day high info
high_results = high_breakout_rows.join(day_high_rows, on="date")
low_results = low_breakout_rows.join(day_low_rows, on="date")
final_results = high_results.join(low_results, on="date").join(first_bar, on="date")
final_results
# %%
final_results = final_results.with_columns(
    time_until_high_breakout=pl.col("time_at_breakout_high") - pl.col("first_bar_time"),
    time_until_low_breakout=pl.col("time_at_breakout_low") - pl.col("first_bar_time"),
    time_until_day_high=pl.col("time_at_day_high") - pl.col("time_at_breakout_high"),
    time_until_day_low=pl.col("time_at_day_low") - pl.col("time_at_breakout_low"),
)
final_results
# %%
# A more "Polars-native" way to handle nulls and logic without map_elements
is_clean_high = (
    (pl.col("time_at_breakout_high") <= pl.col("time_at_day_high"))
    & (pl.col("time_at_day_high") < pl.col("time_at_breakout_low"))
    & (pl.col("time_at_day_high") < pl.col("time_at_day_low"))
)
results = final_results.with_columns(is_clean_high.alias("is_clean_high_run"))
results
# %%
results["is_clean_high_run"].mean()

# %%
results.filter(
    pl.col("time_until_high_breakout") > 1
    # pl.col("time_until_low_breakout") > 1
)
# %%
results.filter(
    (pl.col("time_until_high_breakout") < pl.duration(minutes=30))
    | (pl.col("time_until_low_breakout") < pl.duration(minutes=30))
)
# %%
# Define the mapping of columns to labels
events = {
    "time_at_breakout_high": "BH",
    "time_at_breakout_low": "BL",
    "time_at_day_high": "DH",
    "time_at_day_low": "DL",
}

results = results.with_columns(
    _event_order=pl.concat_list(
        [
            pl.struct(time=pl.col(col_name), label=pl.lit(label))
            for col_name, label in events.items()
        ]
    )
    .list.drop_nulls()
    .list.sort(descending=False)  # Sort by 'time' field in the struct
    .list.eval(pl.element().struct.field("label"))
    .list.join(" -> ")
)
# %%
results = results.with_columns(
    engulfing=pl.col("time_at_breakout_high") == pl.col("time_at_breakout_low"),
)
# %%
results = results.with_columns(
    event_order=pl.when(~pl.col("engulfing")).then(pl.col("_event_order"))
).drop("_event_order")
# %%
results = results.with_columns(
    pl.when(
        # BH -> DH... OR BH -> DL -> DH
        pl.col("event_order").str.contains(r"^BH -> (DH|DL -> DH)")
    )
    .then(1)
    .when(
        # BL -> DL... OR BL -> DH -> DL
        pl.col("event_order").str.contains(r"^BL -> (DL|DH -> DL)")
    )
    .then(-1)
    .otherwise(0)
    .alias("signal")
)
results
# %%
print(results.get_column("event_order").value_counts().sort("event_order"))
print(results.filter())
# %%
results = results.with_columns(
    until_extreme_of_first_breakout=pl.when(
        pl.col("time_until_high_breakout") < pl.col("time_until_low_breakout"),
    )
    .then(pl.col("time_until_day_high"))
    .otherwise(pl.col("time_until_day_low"))
).with_columns(
    extreme_size=pl.when(
        pl.col("time_until_high_breakout") < pl.col("time_until_low_breakout"),
    )
    .then(pl.col("day_high") - pl.col("first_bar_high"))
    .otherwise(pl.col("day_low") - pl.col("first_bar_low"))
)
results

# %%
counts = (
    results.group_by("event_order")
    .agg(
        mean=pl.col("extreme_size").mean(),
        std=pl.col("extreme_size").std(),
    )
    .sort("event_order")
)
print(counts)

# %%
results.filter(
    (pl.col("extreme_size") < 0) & (pl.col("event_order").str.starts_with("BH"))
)

# %%
results.filter(pl.col("date") == pl.date(2024, 12, 6))

# %%
# The fast, native, and correct way
summary = (
    results
    # .drop_nulls("event_order")
    .group_by("event_order").agg(
        [
            pl.len().alias("count"),
            pl.col("extreme_size").mean().alias("mean"),
            pl.col("extreme_size").std().alias("std"),
            pl.col("extreme_size").min().alias("min"),
            pl.col("extreme_size").quantile(0.1).alias("10%"),
            pl.col("extreme_size").quantile(0.25).alias("25%"),
            pl.col("extreme_size").quantile(0.5).alias("50%"),
            pl.col("extreme_size").quantile(0.75).alias("75%"),
            pl.col("extreme_size").quantile(0.9).alias("90%"),
            pl.col("extreme_size").max().alias("max"),
        ]
    )
).sort("event_order")
summary
# %%
# This keeps the full dataframe but groups sequences together chronologically
dates_detailed = (
    results.select(["event_order", "date", "extreme_size", "signal"])
    .drop_nulls("event_order")
    .sort(["event_order", "date"])
)
dates_detailed.write_csv(f"{ticker}_dates_detailed.csv")
print(dates_detailed)
# %%
dates_detailed.filter(pl.col("event_order") == "BH -> BL -> DL -> DH").tail()

# %%
cols_to_drop = [c for c in results.columns if "until" in c]
results.drop(*cols_to_drop).write_csv((DATA_PATH / f"{ticker}_results.csv"))
# %%
