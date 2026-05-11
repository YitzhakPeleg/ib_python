"""Training pipeline for ML-based trading signal detection."""

from pathlib import Path
from typing import Optional

import joblib
import polars as pl
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from algo.bollinger_bands import calculate_bollinger_bands
from algo.feature_engineering import (
    add_technical_indicators,
    engineer_morning_features,
)
from algo.labeling import create_labels, create_labels_with_timing
from algo.signal_detector import filter_morning_window
from data_fetching.date_converter import add_date_int_column


def load_and_prepare_data(
    data_path: str | Path,
    timezone: str = "US/Eastern",
    use_timing_labels: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load data and prepare features and labels.

    Args:
        data_path: Path to parquet file with 1-minute data
        timezone: Timezone for the data
        use_timing_labels: Use timing-aware labeling (which threshold hit first)

    Returns:
        Tuple of (features_df, labels_df)
    """
    logger.info(f"Loading data from {data_path}")
    df = pl.read_parquet(data_path)

    # Add date column if not present
    if "date" not in df.columns:
        df = add_date_int_column(df)

    # Ensure timezone is set
    if df["DateTime"].dtype == pl.Datetime:
        df = df.with_columns(
            pl.col("DateTime").dt.convert_time_zone(timezone).alias("DateTime")
        )

    logger.info(f"Loaded {len(df)} bars across {df['date'].n_unique()} days")

    # Calculate technical indicators
    logger.info("Calculating technical indicators...")
    df = calculate_bollinger_bands(df, window=20, stds=2.0)
    df = add_technical_indicators(df)

    # Filter morning window (09:00-11:00)
    logger.info("Filtering morning window (09:00-11:00)...")
    morning_df = filter_morning_window(df, start_hour=9, end_hour=11, timezone=timezone)

    # Engineer features from morning window
    logger.info("Engineering features from morning window...")
    features_df = engineer_morning_features(morning_df, window=20)

    # Create labels from post-window data
    logger.info("Creating labels from post-window price movements...")
    if use_timing_labels:
        labels_df = create_labels_with_timing(
            df, morning_end_hour=11, tp_threshold=0.005, sl_threshold=0.005
        )
    else:
        labels_df = create_labels(
            df,
            morning_end_hour=11,
            tp_threshold=0.005,
            sl_threshold=0.005,
            use_atr=True,
            atr_multiplier=0.5,
        )

    # Join features and labels
    full_dataset = features_df.join(labels_df.select(["date", "label"]), on="date")
    full_dataset = full_dataset.drop_nulls()

    logger.info(f"Prepared dataset: {len(full_dataset)} days with complete data")

    return features_df, labels_df


def train_random_forest_model(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    min_samples_leaf: int = 5,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict]:
    """
    Train a Random Forest model for signal prediction.

    Args:
        features_df: DataFrame with engineered features
        labels_df: DataFrame with labels
        test_size: Fraction of data to use for testing
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        min_samples_leaf: Minimum samples required at leaf node
        class_weight: Class weighting strategy
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Join features and labels
    full_dataset = features_df.join(labels_df.select(["date", "label"]), on="date")
    full_dataset = full_dataset.drop_nulls().sort("date")

    # Time-series split (no shuffling to prevent lookahead bias)
    split_idx = int(len(full_dataset) * (1 - test_size))
    train_df = full_dataset.head(split_idx)
    test_df = full_dataset.tail(len(full_dataset) - split_idx)

    logger.info(f"Train set: {len(train_df)} days, Test set: {len(test_df)} days")

    # Prepare X and y
    feature_cols = [col for col in train_df.columns if col not in ["date", "label"]]
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df["label"].to_numpy()

    logger.info(f"Training Random Forest with {len(feature_cols)} features...")

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
    )

    model.fit(X_train, y_train)

    logger.info("Model training complete. Evaluating...")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Classification report
    target_names = ["SELL", "HOLD", "BUY"]
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )

    logger.info("Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Feature importance
    feature_importance = pl.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort("importance", descending=True)

    logger.info("Top 10 Most Important Features:")
    logger.info(f"\n{feature_importance.head(10)}")

    # Compile metrics
    metrics = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
        "test_accuracy": report["accuracy"],
        "train_size": len(train_df),
        "test_size": len(test_df),
        "n_features": len(feature_cols),
    }

    return model, metrics


def save_model(
    model: RandomForestClassifier,
    metrics: dict,
    output_dir: str | Path,
    model_name: str = "signal_model",
) -> None:
    """
    Save trained model and metrics to disk.

    Args:
        model: Trained model
        metrics: Dictionary of metrics
        output_dir: Directory to save model
        model_name: Base name for model files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save feature importance
    if "feature_importance" in metrics:
        fi_path = output_dir / f"{model_name}_feature_importance.csv"
        metrics["feature_importance"].write_csv(fi_path)
        logger.info(f"Feature importance saved to {fi_path}")

    # Save metrics summary
    metrics_path = output_dir / f"{model_name}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Test Accuracy: {metrics['test_accuracy']:.4f}\n")
        f.write(f"Train Size: {metrics['train_size']}\n")
        f.write(f"Test Size: {metrics['test_size']}\n")
        f.write(f"Number of Features: {metrics['n_features']}\n\n")
        f.write("Classification Report:\n")
        f.write(str(metrics["classification_report"]))
    logger.info(f"Metrics saved to {metrics_path}")


def load_model(model_path: str | Path) -> RandomForestClassifier:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded model
    """
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def main(
    data_path: str = "/Users/yitzhakpeleg/Projects/ib_python/AAPL_1_min.parquet",
    output_dir: str = "/Users/yitzhakpeleg/Projects/ib_python/models",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> None:
    """
    Main training pipeline.

    Args:
        data_path: Path to data file
        output_dir: Directory to save model
        test_size: Test set size
        n_estimators: Number of trees
        max_depth: Maximum tree depth
    """
    logger.info("=" * 80)
    logger.info("TRAINING SIGNAL DETECTION MODEL")
    logger.info("=" * 80)

    # Load and prepare data
    features_df, labels_df = load_and_prepare_data(
        data_path, timezone="US/Eastern", use_timing_labels=False
    )

    # Train model
    model, metrics = train_random_forest_model(
        features_df,
        labels_df,
        test_size=test_size,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # Save model
    save_model(model, metrics, output_dir, model_name="morning_signal_rf")

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
