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
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text

from src.algo.bollinger_bands import calculate_bollinger_bands
from src.data_fetching.date_converter import add_date_int_column

pio.renderers.default = "browser"

# --- 2. FEATURE ENGINEERING & NORMALIZATION ---


def add_daily_context(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates Daily ATR and Overnight Gap."""
    # 1. Get Daily OHLC
    daily = (
        df.group_by("date")
        .agg(
            [
                pl.col("Open").first().alias("d_open"),
                pl.col("High").max().alias("d_high"),
                pl.col("Low").min().alias("d_low"),
                pl.col("Close").last().alias("d_close"),
            ]
        )
        .sort("date")
    )

    # 2. Calculate ATR (Simplified True Range over 14 days)
    daily = (
        daily.with_columns(prev_close=pl.col("d_close").shift(1))
        .with_columns(
            tr=pl.max_horizontal(
                [
                    (pl.col("d_high") - pl.col("d_low")),
                    (pl.col("d_high") - pl.col("prev_close")).abs(),
                    (pl.col("d_low") - pl.col("prev_close")).abs(),
                ]
            )
        )
        .with_columns(atr=pl.col("tr").rolling_mean(window_size=14))
    )

    # 3. Calculate Gap %
    daily = daily.with_columns(gap_pct=(pl.col("d_open") / pl.col("prev_close") - 1))

    return daily.select(["date", "atr", "gap_pct", "prev_close"])


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


def prepare_ml_dataset(
    df: pl.DataFrame, bar_count: int = 10
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Add row numbering
    df = df.with_columns(
        pl.int_range(0, pl.len()).over("date", order_by="DateTime").alias("row_nr_day")
    )

    # Get Daily Context
    daily_context = add_daily_context(df)

    # --- Feature Extraction ---
    # We join the ATR and Gap into the first 10 bars
    features_raw = (
        df.filter(pl.col("row_nr_day") < bar_count)
        .join(daily_context, on="date")
        .with_columns(
            [
                # Normalized Price
                ((pl.col("Close") / pl.col("Open").first().over("date")) - 1).alias(
                    "rel_close"
                ),
                # BB Diff normalized by ATR (How much of the daily 'budget' is the squeeze?)
                ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("atr")).alias(
                    "bb_atr_ratio"
                ),
                # Keep Gap as a static feature for all 10 bars (pivoting will handle this)
                pl.col("gap_pct"),
            ]
        )
    )

    # Pivot: We include our new metrics
    features_pivoted = features_raw.pivot(
        index="date", on="row_nr_day", values=["rel_close", "bb_atr_ratio", "gap_pct"]
    ).drop_nulls()

    # --- Label Extraction (No Leakage) ---
    entry_prices = df.filter(pl.col("row_nr_day") == bar_count).select(
        ["date", pl.col("Open").alias("entry_price")]
    )

    # We use ATR-based Targets! (e.g., Target = 1.0 * ATR)
    # This is much smarter than a fixed 1%
    labels = (
        df.filter(pl.col("row_nr_day") >= bar_count)
        .join(entry_prices, on="date")
        .join(daily_context, on="date")
        .group_by("date")
        .agg(
            [
                # Is the future move > 0.5 * ATR?
                (
                    (pl.col("High").max() - pl.col("entry_price").first())
                    > (pl.col("atr").first() * 0.5)
                ).alias("hit_tp"),
                (
                    (pl.col("entry_price").first() - pl.col("Low").min())
                    > (pl.col("atr").first() * 0.5)
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

    return features_pivoted, labels


# --- 3. MAIN EXECUTION FLOW ---


def main(*, ticker: str, max_depth: int = 4, min_samples_leaf: int = 10) -> None:
    # A. Load and Basic Prep
    path = f"/Users/yitzhakpeleg/Projects/ib_python/{ticker}_1_min.parquet"

    # Mock data loading (Replace with actual read_parquet)
    try:
        df = pl.read_parquet(path)
    except Exception as e:
        logger.error(f"Error loading file: {e}. Ensure path is correct.")
        return

    logger.info(f"Data loaded. Preprocessing {max_depth = }, {min_samples_leaf = }...")
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
    return classification_report(
        y_test, y_pred, target_names=["Neutral", "Long", "Short"], output_dict=True
    ) | {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf}


def print_tree_rules(clf, feature_names: list[str]):
    tree_rules = export_text(clf, feature_names=feature_names, show_weights=True)
    print(f"--- Decision Tree Rules ---\n{tree_rules}")


# %% Deep learning and more complex models could be added in the future, but this is a solid baseline to start with.
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Check if Apple Silicon GPU (MPS) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")


# --- 1. THE MODEL ARCHITECTURE ---
class TradingNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TradingNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes),  # 3 outputs: Neutral, Long, Short
        )

    def forward(self, x):
        return self.network(x)


# --- 2. THE TRAINING FUNCTION ---
def train_deep_learning(X_train, y_train, X_test, y_test):
    # Scale features (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    # Create DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True
    )

    # Initialize Model
    model = TradingNN(X_train.shape[1], 3).to(device)
    criterion = nn.CrossEntropyLoss()  # Handles class imbalance well
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    model.train()
    for epoch in range(50):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs, 1)

    return predicted.cpu().numpy(), y_test


# --- 3. MAIN_2 PIPELINE ---
def main_2(ticker: str = "AAPL"):
    # Reuse your existing preprocessing
    # (Assuming functions from your train_algo.py are in scope)
    df = pl.read_csv(f"data/{ticker}_1m.csv")
    df = add_date_int_column(df)
    df = calculate_bollinger_bands(df)  # Ensure this adds bb_upper/lower

    features, labels = prepare_ml_dataset(df)

    # Merge and split
    data = features.join(labels, on="date").drop("date")
    X = data.drop("label").to_numpy()
    y = data.select("label").to_numpy().flatten()

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info("Starting Deep Learning Training...")
    y_pred, y_true = train_deep_learning(X_train, y_train, X_test, y_test)

    report = classification_report(
        y_true, y_pred, target_names=["Neutral", "Long", "Short"]
    )
    print("--- Deep Learning Performance ---")
    print(report)


# Usage:
# print_tree_rules(clf, feature_cols)
main_2("AAPL")

# # %%
# if __name__ == "__main__":
#     ticker = "AAPL"
#     results = []
#     for max_depth in [2, 3, 4, 5]:
#         for min_samples_leaf in [10]:
#             results.append(
#                 main(
#                     ticker=ticker,
#                     max_depth=max_depth,
#                     min_samples_leaf=min_samples_leaf,
#                 )
#             )
#     flattened_data = []
#     for entry in results:
#         row = {
#             "max_depth": entry["max_depth"],
#             "min_samples_leaf": entry["min_samples_leaf"],
#             "accuracy": entry["accuracy"],
#             # Extract specific class metrics
#             "long_precision": entry["Long"]["precision"],
#             "long_recall": entry["Long"]["recall"],
#             "short_precision": entry["Short"]["precision"],
#             "short_recall": entry["Short"]["recall"],
#             "neutral_precision": entry["Neutral"]["precision"],
#             "neutral_recall": entry["Neutral"]["recall"],
#         }
#         flattened_data.append(row)

#     # 2. Create the DataFrame
#     df_results = pl.DataFrame(flattened_data)

#     # 3. Sort by accuracy or long_precision to see winners
#     df_results = df_results.sort("accuracy", descending=True)
#     df_results.write_csv(f"{ticker}_model_performance_summary.csv")
#     print(df_results)
# # %%
