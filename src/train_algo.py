# %%
"""
Trading ML Pipeline: Decision Tree for Intraday Directional Prediction.
Target: Predict day direction based on first 10 minutes of trading.
Author: Gemini
Python: 3.14+
"""

# %%
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from loguru import logger
from rich.pretty import pprint
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text

from bollinger_bands import calculate_bollinger_bands
from date_converter import add_date_int_column

pio.renderers.default = "browser"

# --- 2. FEATURE ENGINEERING & NORMALIZATION ---


def prepare_ml_dataset(
    df: pl.DataFrame, bar_count: int = 10, bar_wait: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Normalizes OHLC data relative to the day's open and pivots for ML.
    Returns (Features, Labels).
    """
    # Add row numbering within each day
    df = df.with_columns(
        pl.int_range(0, pl.len()).over("date", order_by="DateTime").alias("row_nr_day")
    )

    # Calculate BB metrics
    df = df.with_columns(
        [
            (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_size"),
            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_mid") * 1000)
            .cast(pl.Int32)
            .alias("bb_ratio_percent"),
        ]
    )

    # Get Daily Open for normalization
    daily_opens = df.filter(pl.col("row_nr_day") == 0).select(
        ["date", pl.col("Open").alias("day_open")]
    )

    # --- Feature Extraction (First N Bars) ---
    features_raw = (
        df.filter(pl.col("row_nr_day") < bar_count)
        .join(daily_opens, on="date")
        .with_columns(
            [
                ((pl.col("Close") / pl.col("day_open")) - 1).alias("rel_close"),
                ((pl.col("High") / pl.col("day_open")) - 1).alias("rel_high"),
                ((pl.col("Low") / pl.col("day_open")) - 1).alias("rel_low"),
                (pl.col("Volume") / pl.col("Volume").mean().over("date")).alias(
                    "rel_volume"
                ),
            ]
        )
    )

    # Pivot: Flatten 10 bars into one row per day
    features_pivoted = features_raw.pivot(
        index="date",
        on="row_nr_day",
        values=["rel_close", "rel_high", "rel_low", "rel_volume", "bb_ratio_percent"],
    ).drop_nulls()

    # --- Label Extraction (The Target) ---
    # We define Long(1) if price moves +1% before EOD, Short(2) if -1%, else Nothing(0)
    # Thresholds can be adjusted
    tp_threshold = 0.01
    sl_threshold = 0.005

    labels = create_labels_no_leakage(
        df,
        bar_count=bar_count + bar_wait,
        tp_threshold=tp_threshold,
        sl_threshold=sl_threshold,
    )

    return features_pivoted, labels


def create_labels_no_leakage(
    df: pl.DataFrame, bar_count: int, tp_threshold: float, sl_threshold: float
):
    # 1. Get the entry price (the Open of the VERY NEXT bar after features)
    entry_prices = df.filter(pl.col("row_nr_day") == bar_count).select(
        ["date", pl.col("Open").alias("entry_price")]
    )

    # 2. Filter for data ONLY after the feature window
    # This ensures the model doesn't 'see' the 1% move that happened in the first 10 mins
    future_only = df.filter(pl.col("row_nr_day") >= bar_count)

    labels = (
        future_only.join(entry_prices, on="date")
        .group_by("date")
        .agg(
            [
                # We compare the MAX/MIN of the FUTURE to the ENTRY price
                (
                    ((pl.col("High").max() / pl.col("entry_price").first()) - 1)
                    > tp_threshold
                ).alias("hit_tp"),
                (
                    ((pl.col("Low").min() / pl.col("entry_price").first()) - 1)
                    < -sl_threshold
                ).alias("hit_sl"),
            ]
        )
        .with_columns(
            label=pl.when(pl.col("hit_tp"))
            .then(1)
            .when(pl.col("hit_sl"))
            .then(2)
            .otherwise(0)
        )
        .select(["date", "label"])
    )
    return labels


# --- 3. MAIN EXECUTION FLOW ---


def main(max_depth: int = 4, min_samples_leaf: int = 10) -> None:
    # A. Load and Basic Prep
    ticker = "AAPL"
    path = f"/Users/yitzhakpeleg/Projects/ib_python/{ticker}_1_min.parquet"

    # Mock data loading (Replace with actual read_parquet)
    try:
        df = pl.read_parquet(path)
    except Exception as e:
        logger.error(f"Error loading file: {e}. Ensure path is correct.")
        return

    logger.info("Data loaded. Preprocessing...")
    df = add_date_int_column(df)
    df = calculate_bollinger_bands(df, window=20, stds=2.0)

    # B. Feature Engineering
    features, labels = prepare_ml_dataset(df, bar_count=10)
    full_dataset = features.join(labels, on="date").drop_nulls().sort("date")

    # C. Time-Series Train/Test Split
    # Split at 80% mark to prevent lookahead bias
    split_idx = int(len(full_dataset) * 0.8)
    train_df = full_dataset.head(split_idx)
    test_df = full_dataset.tail(len(full_dataset) - split_idx)

    X_train = train_df.drop(["date", "label"]).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test = test_df.drop(["date", "label"]).to_numpy()
    y_test = test_df["label"].to_numpy()

    # D. Model Training (Decision Tree)
    # Use class_weight='balanced' to handle market bias (more ups than downs)
    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight="balanced"
    )
    logger.info("Training Decision Tree Classifier...")
    clf.fit(X_train, y_train)
    logger.info("Model training completed.")

    # E. Evaluation
    logger.info("Evaluating model on test set...")
    y_pred = clf.predict(X_test)
    performace_report = classification_report(
        y_test, y_pred, target_names=["Neutral", "Long", "Short"]
    )
    logger.info(f"Model Performance on Test Set:\n{performace_report}")

    # F. Visualization: Feature Importance
    feat_importance = (
        pl.DataFrame(
            {
                "feature": train_df.drop(["date", "label"]).columns,
                "importance": clf.feature_importances_,
            }
        )
        .sort("importance", descending=True)
        .head(10)
    )

    fig_imp = px.bar(
        feat_importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Features (Normalized First 10 Bars)",
    )
    # fig_imp.show()

    # G. Visualization: Cumulative Return Comparison (Backtest Lite)
    # Simple check: If we trade every 'Long' signal, how does it look?
    test_results = test_df.with_columns(prediction=y_pred).join(
        # Join with actual close prices to calculate returns
        df.group_by("date").agg(
            ((pl.col("Close").last() / pl.col("Open").first()) - 1).alias("daily_ret")
        ),
        on="date",
    )

    test_results = test_results.with_columns(
        strategy_ret=pl.when(pl.col("prediction") == 1)
        .then(pl.col("daily_ret"))
        .when(pl.col("prediction") == 2)
        .then(-pl.col("daily_ret"))
        .otherwise(0)
    )

    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(
            x=test_results["date"].cast(pl.String),
            y=test_results["daily_ret"].cum_sum(),
            name="Buy & Hold",
        )
    )
    fig_cum.add_trace(
        go.Scatter(
            x=test_results["date"].cast(pl.String),
            y=test_results["strategy_ret"].cum_sum(),
            name="ML Strategy",
        )
    )
    fig_cum.update_layout(
        title="Strategy Performance vs Buy & Hold (Test Set)", template="plotly_dark"
    )
    # fig_cum.show()

    print_tree_rules(clf, feature_names=train_df.drop(["date", "label"]).columns)
    return performace_report


def print_tree_rules(clf, feature_names: list[str]):
    tree_rules = export_text(clf, feature_names=feature_names, show_weights=True)
    print(f"--- Decision Tree Rules ---\n{tree_rules}")


# Usage:
# print_tree_rules(clf, feature_cols)

# %%
if __name__ == "__main__":
    result = {}
    for max_depth in [4, 6, 8]:
        for min_samples_leaf in [10, 20, 30]:
            result[(max_depth, min_samples_leaf)] = main(
                max_depth=max_depth, min_samples_leaf=min_samples_leaf
            )
    pprint(result, expand_all=True)
# %%
