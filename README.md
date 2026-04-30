# ib-python

A comprehensive Python toolkit for algorithmic trading with Interactive Brokers. This project provides tools for fetching historical market data, calculating technical indicators, and training machine learning models for trading strategies.

## 🚀 Features

- **IB API Integration**: Robust wrapper for Interactive Brokers API with multi-request support
- **Historical Data Fetching**: Easy-to-use interface for downloading market data
- **Technical Indicators**: Bollinger Bands and other technical analysis tools
- **Machine Learning**: Decision Tree and Neural Network models for trading strategy development
- **Data Visualization**: Interactive Plotly charts for market analysis
- **High Performance**: Built with Polars for fast data processing

## 📋 Requirements

- Python >=3.12
- Interactive Brokers TWS or IB Gateway (for data fetching)
- UV package manager (recommended) or pip

## 🔧 Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ib_python

# Install dependencies
uv sync

# Install in editable mode
uv pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd ib_python

# Install dependencies
pip install -e .
```

## 🏗️ Project Structure

```
ib_python/
├── src/
│   ├── data_fetching/      # IB API data fetching utilities
│   ├── algo/               # Trading algorithms and ML models
│   └── notebooks/          # Analysis and visualization scripts
├── docs/                   # Comprehensive documentation
├── main.py                 # Main entry point
└── pyproject.toml          # Project configuration
```

## 📚 Quick Start

### 1. Fetch Historical Data

```python
from datetime import datetime, timedelta
from src.data_fetching import HistoricalDataFetcher
from src.algo.models import ContractSpec, BarFrequency

# Create a contract specification
contract = ContractSpec(symbol="AAPL")

# Fetch historical data
with HistoricalDataFetcher(host="127.0.0.1", port=4002) as fetcher:
    df = fetcher.get_historical_data(
        contract=contract,
        end_date=datetime.now(),
        duration=timedelta(days=30),
        frequency=BarFrequency.ONE_HOUR,
        timezone="US/Eastern"
    )
    
print(df)
```

### 2. Calculate Technical Indicators

```python
import polars as pl
from src.algo.bollinger_bands import calculate_bollinger_bands

# Load your data
df = pl.read_parquet("AAPL_1_min.parquet")

# Calculate Bollinger Bands
df_with_bb = calculate_bollinger_bands(df, window=20, stds=2.0)
print(df_with_bb)
```

### 3. Train ML Models

```python
from src.algo.train_algo import main

# Train a decision tree model for AAPL
main(ticker="AAPL", max_depth=4, min_samples_leaf=10)
```

### 4. Visualize Data

```python
# Run the visualization script
python src/notebooks/plot_data.py
```

## 🔑 Key Components

### Data Fetching Module (`src/data_fetching/`)

- **IBapi Wrapper**: Enhanced wrapper with multi-request support and automatic connection management
- **Historical Data Fetcher**: High-level interface for fetching market data from Interactive Brokers
- **Date Converter**: Utilities for date/time format conversions

### Algorithms Module (`src/algo/`)

- **Models**: Type-safe Pydantic models for contracts, bar frequencies, and security types
- **Bollinger Bands**: Technical indicator calculation with Polars
- **ML Pipeline**: Complete machine learning pipeline including:
  - Feature engineering with normalized prices and ATR-based targets
  - Decision Tree classifier for directional prediction
  - PyTorch Neural Network for advanced modeling
  - Backtesting and performance visualization

### Notebooks Module (`src/notebooks/`)

- **Plot Data**: Interactive Plotly visualizations including candlestick charts, Bollinger Bands, and intraday analysis

## 📖 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[00_developer_guide.md](docs/00_developer_guide.md)** - Development setup and guidelines
- **[01_architecture.md](docs/01_architecture.md)** - System design and data flow
- **[02_data_fetching.md](docs/02_data_fetching.md)** - Data fetching module documentation
- **[03_algorithms.md](docs/03_algorithms.md)** - Trading algorithms and ML models
- **[04_api_reference.md](docs/04_api_reference.md)** - Detailed API documentation
- **[05_examples.md](docs/05_examples.md)** - Practical usage examples

## ⚙️ Configuration

### IB Gateway/TWS Connection

Default connection settings:
- **Host**: `127.0.0.1`
- **Port**: `4002` (paper trading) or `4001` (live trading)
- **Client ID**: `1`

These can be configured when creating a `HistoricalDataFetcher` instance.

### Environment Variables

You can use a `.env` file for configuration (optional):

```bash
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1
```

## 🧪 Development

### Setup Development Environment

```bash
# Install dependencies
uv sync

# Install pre-commit hooks (if available)
pre-commit install

# Run linting
ruff check src/

# Format code
ruff format src/
```

### Running Tests

Ensure IB Gateway or TWS is running before testing data fetching functionality:

```bash
# Start IB Gateway on port 4002 (paper trading)
# Then run your tests
python -m pytest tests/
```

## 📊 Data Storage

- Historical data is stored in **Parquet format** for efficient storage and fast retrieval
- Default data directory: `./data/`
- File naming convention: `{SYMBOL}_{FREQUENCY}.parquet`

## 🤖 Machine Learning

The ML pipeline includes:

1. **Feature Engineering**: Normalized prices, Bollinger Band ratios, ATR-based targets
2. **Time-Series Split**: Prevents lookahead bias with 80/20 train/test split
3. **Models**:
   - Decision Tree Classifier (scikit-learn)
   - Neural Network (PyTorch with MPS support for Apple Silicon)
4. **Evaluation**: Classification reports and cumulative return comparisons

## ⚠️ Important Notes

- **Paper Trading**: Always test with paper trading account before live trading
- **Data Limits**: IB API has rate limits and historical data restrictions
- **Time Zones**: Be aware of timezone conversions when working with market data
- **ATR Targets**: Models use ATR-based targets instead of fixed percentages for better adaptability

## 🛠️ Dependencies

Key dependencies:
- `ibapi` - Interactive Brokers API
- `polars` - High-performance DataFrames
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning
- `torch` - Deep learning
- `loguru` - Logging
- `pydantic-settings` - Configuration management

See `pyproject.toml` for complete dependency list.

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please see the development guide in `docs/00_developer_guide.md` for guidelines.

## 📧 Support

For issues and questions:
- Check the documentation in `docs/`
- Review the [IB API documentation](https://interactivebrokers.github.io/tws-api/)
- Open an issue on the project repository

## 🔗 Resources

- [Interactive Brokers API Documentation](https://interactivebrokers.github.io/tws-api/)
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Plotly Documentation](https://plotly.com/python/)

---

**Note**: This project is for educational and research purposes. Always test thoroughly before using in live trading environments.
