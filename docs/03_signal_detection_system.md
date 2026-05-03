# Trading Signal Detection System

## Overview

The ML-based trading signal detection system analyzes the morning trading window (09:00-11:00) to predict whether to BUY, SELL, or HOLD for the rest of the trading day. The system uses machine learning to discover patterns in price action, volume, and technical indicators.

## Architecture

```
Data (1-min OHLC) 
    ↓
Morning Window Filter (09:00-11:00)
    ↓
Feature Engineering (price patterns, indicators, volume)
    ↓
ML Model (Random Forest)
    ↓
Signal Prediction (BUY/SELL/HOLD)
    ↓
Trade Setup (Entry, Stop-Loss, Take-Profit)
    ↓
Backtesting & Evaluation
```

## Key Components

### 1. Data Models (`models.py`)

**SignalType**: Enum for signal types
- `BUY = 1`: Long signal
- `SELL = -1`: Short signal  
- `HOLD = 0`: No trade

**TradeSetup**: Complete trade specification
```python
@dataclass
class TradeSetup:
    date: int              # YYYYMMDD
    signal: SignalType     # BUY/SELL/HOLD
    entry_price: float     # Entry price
    stop_loss: float       # Stop-loss price
    take_profit: float     # Take-profit price
    confidence: float      # Model confidence (0-1)
    risk_reward_ratio: float = 2.0  # Default 2:1
```

**SignalResult**: Backtest result for a trade
```python
@dataclass
class SignalResult:
    setup: TradeSetup
    outcome: Literal["win", "loss", "breakeven", "open"]
    exit_price: float
    pnl: float            # Profit/Loss in dollars
    pnl_percent: float    # P/L as percentage
    bars_held: int        # Duration of trade
```

### 2. Signal Detection (`signal_detector.py`)

**Time Window Filtering**
```python
filter_morning_window(df, start_hour=9, end_hour=11, timezone="US/Eastern")
```
Filters data to only include the morning trading window.

**Entry/Stop-Loss/Take-Profit Calculation**
```python
calculate_entry_stop_tp(last_bar, signal, risk_reward_ratio=2.0)
```

For BUY signals:
- Entry: High of last bar (11:00)
- Stop-Loss: Low of last bar
- Take-Profit: Entry + 2 × (Entry - Stop-Loss)

For SELL signals:
- Entry: Low of last bar
- Stop-Loss: High of last bar
- Take-Profit: Entry - 2 × (Stop-Loss - Entry)

### 3. Feature Engineering (`feature_engineering.py`)

**Morning Window Features**
- **Price Movement**: Normalized OHLC relative to first bar
- **Volatility**: Bar ranges, price standard deviation
- **Volume**: Relative volume, total volume
- **Technical Indicators**: Bollinger Bands, RSI, MACD
- **Momentum**: Net price change, max high/low moves

**Key Functions**
```python
engineer_morning_features(morning_df, window=20)
add_technical_indicators(df)
create_sequential_features(morning_df, max_bars=120)  # For DL models
```

### 4. Labeling System (`labeling.py`)

**Label Generation**
Labels are created by analyzing price movement AFTER 11:00:

```python
create_labels(
    df,
    morning_end_hour=11,
    tp_threshold=0.005,    # 0.5% move
    sl_threshold=0.005,
    use_atr=True,          # Use ATR-based thresholds
    atr_multiplier=0.5
)
```

**Labeling Logic**
- **BUY (1)**: Price moves up by threshold before moving down
- **SELL (-1)**: Price moves down by threshold before moving up
- **HOLD (0)**: No significant directional move

**ATR-Based Thresholds**
Instead of fixed percentages, the system can use Average True Range (ATR) for adaptive thresholds:
```
threshold = 0.5 × ATR
```

### 5. Model Training (`train_signal_model.py`)

**Random Forest Classifier**
```python
train_random_forest_model(
    features_df,
    labels_df,
    test_size=0.2,
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced"
)
```

**Training Pipeline**
1. Load and prepare data
2. Calculate technical indicators
3. Filter morning window (09:00-11:00)
4. Engineer features
5. Create labels from post-window data
6. Time-series train/test split (80/20)
7. Train Random Forest model
8. Evaluate on test set
9. Save model and metrics

**Key Features**
- Time-series split (no shuffling to prevent lookahead bias)
- Balanced class weights (handles imbalanced data)
- Feature importance analysis
- Comprehensive evaluation metrics

### 6. Signal Generation (`signal_generator.py`)

**SignalGenerator Class**
```python
generator = SignalGenerator(model_path)

signals = generator.generate_signals(
    df,
    timezone="US/Eastern",
    confidence_threshold=0.6,  # Only signals with >60% confidence
    risk_reward_ratio=2.0
)
```

**Workflow**
1. Load trained model
2. Prepare data (indicators, morning window)
3. Engineer features
4. Predict signals with confidence scores
5. Filter by confidence threshold
6. Calculate entry/SL/TP for each signal
7. Return list of TradeSetup objects

### 7. Backtesting (`backtester.py`)

**Backtest Trade Setups**
```python
results_df = backtest_trade_setups(df, signals, timezone="US/Eastern")
```

**Simulation Logic**
- For each trade setup, simulate execution on actual price data
- Check if stop-loss or take-profit is hit first
- Track exit price, PnL, and duration
- Calculate outcome (win/loss/breakeven)

**Performance Metrics**
```python
performance = generate_performance_report(results_df)
```

Metrics include:
- Win rate
- Average win/loss
- Total PnL
- Profit factor (gross profit / gross loss)
- Sharpe ratio
- Maximum drawdown
- Average R-multiple
- Average bars held

## Usage Examples

### Complete Workflow

```python
from src.algo.example_workflow import complete_workflow_example

# Run complete workflow: train → generate signals → backtest
signals, results, performance = complete_workflow_example(
    data_path="AAPL_1_min.parquet",
    model_dir="models/",
    retrain=False  # Set True to retrain model
)
```

### Training a Model

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

### Generating Signals

```python
from src.algo.signal_generator import SignalGenerator
import polars as pl

# Load data
df = pl.read_parquet("AAPL_1_min.parquet")

# Initialize generator
generator = SignalGenerator("models/morning_signal_rf.joblib")

# Generate signals
signals = generator.generate_signals(
    df,
    timezone="US/Eastern",
    confidence_threshold=0.6,
    risk_reward_ratio=2.0
)

# Display signals
for setup in signals:
    print(setup)
```

### Backtesting Signals

```python
from src.algo.backtester import backtest_trade_setups, generate_performance_report

# Backtest
results_df = backtest_trade_setups(df, signals)

# Generate report
performance = generate_performance_report(results_df)
print(performance)
```

### Quick Signal Check for Specific Date

```python
from src.algo.example_workflow import quick_signal_check

signal = quick_signal_check(
    data_path="AAPL_1_min.parquet",
    model_path="models/morning_signal_rf.joblib",
    target_date=20260430
)
```

## Configuration

### Time Window
- **Start**: 09:00 (market open)
- **End**: 11:00 (2 hours of data)
- **Timezone**: US/Eastern (for US stocks)

### Risk Management
- **Risk-Reward Ratio**: 2:1 (default)
- **Stop-Loss**: Opposite extreme of last bar
- **Take-Profit**: 2× the risk distance

### Model Parameters
- **Algorithm**: Random Forest
- **Trees**: 100
- **Max Depth**: 10
- **Min Samples per Leaf**: 5
- **Class Weight**: Balanced

### Labeling Thresholds
- **Fixed**: 0.5% move (tp_threshold=0.005)
- **ATR-Based**: 0.5 × ATR (adaptive to volatility)

## Performance Expectations

Based on backtesting:
- **Win Rate**: Target >55%
- **Profit Factor**: Target >1.5
- **Sharpe Ratio**: Target >1.0
- **Average R-Multiple**: Target >0.5R

## Best Practices

1. **Always use time-series split** for train/test to prevent lookahead bias
2. **Filter by confidence threshold** to trade only high-confidence signals
3. **Use ATR-based thresholds** for adaptive risk management
4. **Backtest thoroughly** before live trading
5. **Monitor feature importance** to understand what drives signals
6. **Retrain periodically** as market conditions change

## Limitations

1. **Intraday only**: System designed for same-day trades
2. **Single symbol**: Trained on one stock (AAPL)
3. **No market regime detection**: Doesn't adapt to bull/bear markets
4. **No position sizing**: Fixed risk per trade
5. **Simplified execution**: Assumes fills at exact prices

## Future Enhancements

1. **Deep Learning Models**: LSTM/Transformer for sequential patterns
2. **Multi-Symbol Training**: Train on multiple stocks
3. **Regime Detection**: Adapt strategy to market conditions
4. **Position Sizing**: Dynamic risk allocation
5. **Real-Time Signals**: Live signal generation
6. **Advanced Features**: Order flow, market microstructure
7. **Ensemble Models**: Combine multiple models

## File Structure

```
src/algo/
├── models.py                    # Data models (SignalType, TradeSetup, SignalResult)
├── signal_detector.py           # Time filtering, entry/SL/TP calculation
├── feature_engineering.py       # Feature extraction from morning window
├── labeling.py                  # Label generation from post-window data
├── train_signal_model.py        # Model training pipeline
├── signal_generator.py          # Signal generation using trained model
├── backtester.py               # Backtesting framework
├── example_workflow.py         # Complete workflow examples
├── bollinger_bands.py          # Bollinger Bands indicator
└── train_algo.py               # Original ML experiments
```

## References

- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Polars DataFrame](https://pola-rs.github.io/polars/)
- [Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)
