# Trading Signal Detection System

ML-based system for detecting intraday trading signals from morning price action (09:00-11:00).

## Quick Start

### 1. Train a Model

```python
from src.algo.train_signal_model import main

main(
    data_path="AAPL_1_min.parquet",
    output_dir="models/",
    test_size=0.2,
    n_estimators=100,
    max_depth=10
)
```

### 2. Generate Signals

```python
from src.algo.signal_generator import SignalGenerator
import polars as pl

# Load data
df = pl.read_parquet("AAPL_1_min.parquet")

# Generate signals
generator = SignalGenerator("models/morning_signal_rf.joblib")
signals = generator.generate_signals(df, confidence_threshold=0.6)

# Display
for setup in signals[:5]:
    print(setup)
```

### 3. Backtest Signals

```python
from src.algo.backtester import backtest_trade_setups, generate_performance_report

# Backtest
results = backtest_trade_setups(df, signals)

# Performance report
performance = generate_performance_report(results)
print(performance)
```

### 4. Complete Workflow

```python
from src.algo.example_workflow import complete_workflow_example

signals, results, performance = complete_workflow_example(
    data_path="AAPL_1_min.parquet",
    model_dir="models/",
    retrain=False
)
```

## System Overview

### How It Works

1. **Morning Window Analysis** (09:00-11:00)
   - Extract 120 bars of 1-minute data
   - Calculate technical indicators (Bollinger Bands, RSI, MACD)
   - Engineer features (price patterns, volume, volatility)

2. **ML Prediction**
   - Random Forest model predicts: BUY, SELL, or HOLD
   - Confidence score for each prediction
   - Filter by confidence threshold

3. **Trade Setup Generation**
   - **Entry**: High/Low of last bar (11:00)
   - **Stop-Loss**: Opposite extreme of last bar
   - **Take-Profit**: 2:1 risk-reward ratio

4. **Backtesting**
   - Simulate trades on historical data
   - Track wins, losses, PnL
   - Calculate performance metrics

### Key Features

- ✅ Time-series aware (no lookahead bias)
- ✅ ATR-based adaptive thresholds
- ✅ Confidence-based filtering
- ✅ Comprehensive backtesting
- ✅ Feature importance analysis
- ✅ Balanced class handling

## Module Reference

### Core Modules

- **`models.py`**: Data models (SignalType, TradeSetup, SignalResult)
- **`signal_detector.py`**: Time filtering, entry/SL/TP calculation
- **`feature_engineering.py`**: Feature extraction pipeline
- **`labeling.py`**: Label generation from price movements
- **`train_signal_model.py`**: Model training pipeline
- **`signal_generator.py`**: Signal generation from trained model
- **`backtester.py`**: Backtesting framework
- **`example_workflow.py`**: Complete workflow examples

### Supporting Modules

- **`bollinger_bands.py`**: Bollinger Bands indicator
- **`train_algo.py`**: Original ML experiments

## Configuration

### Default Parameters

```python
# Time Window
START_HOUR = 9      # 09:00
END_HOUR = 11       # 11:00
TIMEZONE = "US/Eastern"

# Risk Management
RISK_REWARD_RATIO = 2.0  # 2:1 TP:SL

# Model
N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 5

# Labeling
TP_THRESHOLD = 0.005  # 0.5% or 0.5×ATR
USE_ATR = True
ATR_MULTIPLIER = 0.5

# Signal Generation
CONFIDENCE_THRESHOLD = 0.6  # 60%
```

## Performance Metrics

The system tracks:
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Average R-Multiple**: Average profit in terms of risk
- **Average Bars Held**: Trade duration

## Examples

### Check Signal for Specific Date

```python
from src.algo.example_workflow import quick_signal_check

signal = quick_signal_check(
    data_path="AAPL_1_min.parquet",
    model_path="models/morning_signal_rf.joblib",
    target_date=20260430
)
```

### Analyze Signal Distribution

```python
from src.algo.example_workflow import analyze_signal_distribution

signals = analyze_signal_distribution(
    data_path="AAPL_1_min.parquet",
    model_path="models/morning_signal_rf.joblib"
)
```

### Custom Feature Engineering

```python
from src.algo.feature_engineering import engineer_morning_features
from src.algo.signal_detector import filter_morning_window

# Filter morning window
morning_df = filter_morning_window(df, start_hour=9, end_hour=11)

# Engineer features
features = engineer_morning_features(morning_df, window=20)
```

### Custom Labeling

```python
from src.algo.labeling import create_labels

labels = create_labels(
    df,
    morning_end_hour=11,
    tp_threshold=0.01,  # 1% move
    sl_threshold=0.01,
    use_atr=True,
    atr_multiplier=0.75
)
```

## Best Practices

1. **Always backtest** before live trading
2. **Use confidence thresholds** to filter low-quality signals
3. **Monitor feature importance** to understand model behavior
4. **Retrain periodically** as market conditions change
5. **Use ATR-based thresholds** for adaptive risk management
6. **Validate on out-of-sample data** to avoid overfitting

## Troubleshooting

### No signals generated
- Check confidence threshold (try lowering it)
- Verify data has morning window (09:00-11:00)
- Check model is trained on similar data

### Poor backtest performance
- Increase confidence threshold
- Adjust risk-reward ratio
- Retrain with more data
- Check for data quality issues

### Model training fails
- Verify data has required columns (DateTime, OHLC, Volume)
- Check for missing values
- Ensure sufficient data (>100 days recommended)

## Documentation

See [docs/03_signal_detection_system.md](../../docs/03_signal_detection_system.md) for comprehensive documentation.

## License

MIT License - See main project README
