# First Bar Breakout Strategy

## Overview

The First Bar Breakout Strategy is a simple intraday trading strategy that trades breakouts from the first bar of the trading day. It aims to capture momentum moves that occur after the market opens.

## Strategy Logic

### Entry Rules

1. **First Bar Definition**: The first 1-minute bar of the trading day (9:30 AM ET)
2. **Breakout Detection**: 
   - **Long Entry**: Price breaks above the first bar high
   - **Short Entry**: Price breaks below the first bar low
3. **Trade Limit**: Maximum 1 trade per day (whichever breakout happens first)
4. **Trading Hours**: 9:30 AM - 4:00 PM ET

### Exit Rules

1. **Stop Loss**:
   - Long trades: First bar low
   - Short trades: First bar high

2. **Take Profit**:
   - Long trades: Entry price + first bar range (high - low)
   - Short trades: Entry price - first bar range (high - low)

3. **End of Day**: Close position at 4:00 PM if TP/SL not hit

## Implementation

### Core Components

#### `FirstBarBreakoutStrategy` Class

Main strategy class with the following methods:

- `identify_first_bars(df)`: Identifies the 9:30 AM bar for each trading day
- `detect_breakout(df, date, first_bar_high, first_bar_low)`: Detects first breakout (high or low)
- `execute_trade(df, date, direction, entry_time, entry_price, stop_loss, take_profit)`: Simulates trade execution
- `backtest(df)`: Runs complete backtest on historical data
- `generate_summary_stats(results)`: Generates performance statistics
- `print_summary(stats)`: Prints formatted summary

#### `TradeResult` Dataclass

Stores complete trade information:
- Date, direction (long/short)
- First bar details (time, high, low, range)
- Entry details (breakout time, entry price)
- Exit levels (stop loss, take profit)
- Exit details (time, price, reason)
- Performance (P&L, P&L%, bars held)

### Usage

```python
from src.algo.first_bar_breakout import FirstBarBreakoutStrategy
import polars as pl

# Load data
df = pl.read_parquet("SPY_1_min.parquet")

# Initialize strategy
strategy = FirstBarBreakoutStrategy()

# Run backtest
results = strategy.backtest(df)

# Generate statistics
stats = strategy.generate_summary_stats(results)
strategy.print_summary(stats)

# Export results
from src.algo.first_bar_breakout import export_results_to_csv
export_results_to_csv(results, "results.csv")
```

### Running the Backtest Script

```bash
# Activate virtual environment
source .venv/bin/activate

# Run backtest on SPY data
python run_first_bar_breakout.py
```

## Results

The backtest generates two output files in the `results/` directory:

1. **`SPY_first_bar_breakout_results.csv`**: Detailed trade-by-trade results with all logged information
2. **`SPY_first_bar_breakout_summary.txt`**: Performance summary statistics

### CSV Columns

- `date`: Trade date (YYYY-MM-DD)
- `direction`: Trade direction (long/short)
- `first_bar_time`: DateTime of first bar
- `first_bar_high`: First bar high price
- `first_bar_low`: First bar low price
- `first_bar_range`: First bar range (high - low)
- `breakout_time`: DateTime when breakout occurred
- `entry_price`: Entry price
- `stop_loss`: Stop loss price
- `take_profit`: Take profit price
- `exit_time`: DateTime when trade exited
- `exit_price`: Exit price
- `exit_reason`: Reason for exit (take_profit, stop_loss, end_of_day)
- `pnl`: Profit/Loss in dollars
- `pnl_percent`: Profit/Loss as percentage
- `bars_held`: Number of 1-minute bars held

## Performance Metrics

The strategy calculates the following metrics:

- **Total Trades**: Number of trades executed
- **Wins/Losses**: Count of winning and losing trades
- **Win Rate**: Percentage of winning trades
- **Average Win/Loss**: Average profit/loss per winning/losing trade
- **Total P&L**: Cumulative profit/loss
- **Profit Factor**: Gross profit / Gross loss
- **Exit Breakdown**: Distribution of exits by reason (TP/SL/EOD)
- **Direction Breakdown**: Performance by trade direction (long/short)
- **Average Bars Held**: Average holding period in 1-minute bars

## Example Results (SPY, Jan 2025 - May 2026)

```
Total Trades: 333
Wins: 169 | Losses: 164 | Breakeven: 0
Win Rate: 50.75%
Average Win: $0.74
Average Loss: $-0.78
Average PnL per Trade: $-0.01
Total PnL: $-2.81
Profit Factor: 0.98
Gross Profit: $124.37
Gross Loss: $127.18

Exit Breakdown:
  Take Profit: 169 (50.8%)
  Stop Loss: 164 (49.2%)
  End of Day: 0 (0.0%)

Direction Breakdown:
  Long: 170 trades, 51.76% win rate
  Short: 163 trades, 49.69% win rate

Average Bars Held: 8.6
```

## Strategy Characteristics

### Strengths
- Simple and easy to understand
- Clear entry and exit rules
- Risk-reward ratio of 1:1 (first bar range)
- No overnight risk (intraday only)
- Captures early momentum moves

### Weaknesses
- Profit factor close to 1.0 (breakeven)
- Small average P&L per trade
- Sensitive to first bar range size
- No filtering for market conditions
- Fixed risk-reward ratio

## Potential Improvements

1. **Market Condition Filter**: Only trade on trending days
2. **Volume Filter**: Require minimum volume on first bar
3. **Time Filter**: Avoid trading during low volatility periods
4. **Dynamic Risk-Reward**: Adjust TP based on ATR or recent volatility
5. **Position Sizing**: Scale position size based on first bar range
6. **Multiple Timeframes**: Use 5-minute or 15-minute first bar
7. **Trend Filter**: Only trade in direction of longer-term trend

## Files

- `src/algo/first_bar_breakout.py`: Strategy implementation
- `run_first_bar_breakout.py`: Backtest execution script
- `results/SPY_first_bar_breakout_results.csv`: Detailed trade results
- `results/SPY_first_bar_breakout_summary.txt`: Performance summary

## Dependencies

- polars: DataFrame operations
- loguru: Logging
- Python 3.10+

## License

Made with Bob
