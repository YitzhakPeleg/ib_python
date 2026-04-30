# CLAUDE.md - Project Context & Guidelines

## Project Overview
- **Name**: ib-python
- **Purpose**: Interactive Brokers API wrapper, market data utilities, and algorithmic trading tools
- **Python Version**: >=3.12
- **Description**: A comprehensive toolkit for fetching historical market data from Interactive Brokers, calculating technical indicators (Bollinger Bands), and training machine learning models for trading strategies.

## Package Management
- **Tool**: UV (uv) - NOT pip
- **Commands**:
  - Install dependencies: `uv sync`
  - Add package: `uv add <package>`
  - Remove package: `uv remove <package>`
  - Install project in editable mode: `uv pip install -e .`

## Project Structure
```
ib_python/
├── src/                          # Source code modules
│   ├── data_fetching/           # IB API data fetching utilities
│   │   ├── __init__.py          # Module exports
│   │   ├── ibapi_wrapper.py     # Enhanced IBapi wrapper with multi-request support
│   │   ├── historical_data_fetcher.py  # High-level historical data fetcher
│   │   └── date_converter.py    # Date/time conversion utilities
│   ├── algo/                    # Trading algorithms and ML models
│   │   ├── models.py            # Pydantic data models (ContractSpec, BarFrequency, etc.)
│   │   ├── bollinger_bands.py   # Bollinger Bands indicator calculation
│   │   └── train_algo.py        # ML pipeline for trading strategy (Decision Tree & Neural Network)
│   └── notebooks/               # Analysis and visualization scripts
│       └── plot_data.py         # Plotly-based data visualization
├── docs/                        # Documentation (numbered files)
│   ├── 00_developer_guide.md   # Development setup and guidelines
│   ├── 01_architecture.md      # System design and data flow
│   ├── 02_data_fetching.md     # Data fetching module documentation
│   ├── 03_algorithms.md        # Trading algorithms and ML models
│   ├── 04_api_reference.md     # Detailed API documentation
│   └── 05_examples.md          # Practical usage examples
├── main.py                      # Main entry point
├── pyproject.toml               # UV/pip project configuration
├── uv.lock                      # UV lock file
├── ibapi-10.45.1-py3-none-any.whl  # Local IB API wheel
├── README.md                    # Project overview
├── CLAUDE.md                    # This file - development context
└── .gitignore                   # Git ignore rules
```

## Current Branch
- **Branch**: `update-deps-and-remove-devcontainer`
- **Default Branch**: `main`

## Key Dependencies
- `ibapi` - Interactive Brokers API (local wheel file, version 10.45.1)
- `polars>=1.31.0` - High-performance DataFrame library
- `plotly[express]>=6.2.0` - Interactive visualization
- `loguru>=0.7.3` - Logging
- `rich>=14.0.0` - Terminal formatting
- `pydantic-settings>=2.0.0` - Configuration management
- `scikit-learn>=1.8.0` - Machine learning (Decision Trees)
- `torch>=2.11.0` - Deep learning (Neural Networks)
- `jupyterlab>=4.4.4` - Jupyter notebooks
- `ruff>=0.12.3` - Linting and formatting

## Module Overview

### src/data_fetching/
Handles all interactions with Interactive Brokers API for fetching historical market data.

- **ibapi_wrapper.py**: Base wrapper class extending EWrapper and EClient with multi-request support
- **historical_data_fetcher.py**: High-level interface for fetching historical data with automatic connection management
- **date_converter.py**: Utilities for converting datetime formats to integer date representations (YYYYMMDD)

### src/algo/
Trading algorithms, technical indicators, and machine learning models.

- **models.py**: Pydantic models for type-safe data structures (ContractSpec, BarFrequency, SecurityType, etc.)
- **bollinger_bands.py**: Calculate Bollinger Bands technical indicator
- **train_algo.py**: Complete ML pipeline including:
  - Feature engineering (normalized prices, BB ratios, ATR-based targets)
  - Decision Tree classifier for directional prediction
  - Neural Network (PyTorch) for advanced modeling
  - Backtesting and performance visualization

### src/notebooks/
Analysis scripts and visualization tools (can be run as Python scripts or in Jupyter).

- **plot_data.py**: Plotly-based visualizations including candlestick charts, Bollinger Bands, and intraday analysis

## Configuration
- Settings can be managed via environment variables or `.env` file
- IB API connection defaults:
  - Host: `127.0.0.1`
  - Port: `4002` (paper trading) or `4001` (live trading)
  - Client ID: `1`

## Data Flow
1. **Data Fetching**: `HistoricalDataFetcher` → IB Gateway/TWS → Historical bars
2. **Data Processing**: Raw bars → Polars DataFrame → Technical indicators (Bollinger Bands)
3. **Feature Engineering**: OHLCV data → Normalized features → ML-ready dataset
4. **Model Training**: Features → Decision Tree/Neural Network → Predictions
5. **Visualization**: Results → Plotly charts → Analysis

## Important Notes
- IBapi connects to TWS/Gateway on `localhost:4002` (paper trading) or `4001` (live)
- All IB connection code uses configurable host/port parameters
- Use `uv` commands instead of pip/poetry
- Data is stored in Parquet format for efficient storage and retrieval
- ML models use time-series train/test splits to prevent lookahead bias
- ATR-based targets are used instead of fixed percentage targets for better adaptability

## Development Workflow
1. **Setup**: `uv sync` to install dependencies
2. **Data Fetching**: Use `HistoricalDataFetcher` to download market data
3. **Analysis**: Run visualization scripts in `src/notebooks/`
4. **Model Training**: Execute `src/algo/train_algo.py` for ML experiments
5. **Testing**: Ensure IB Gateway/TWS is running before fetching data

## When Making Changes
- Update dependencies in `pyproject.toml`
- Run `uv sync` after updating dependencies
- Add docstrings to all new functions (Google-style format)
- Update relevant documentation in `docs/` folder
- Test with paper trading account before live trading
- Use type hints for all function parameters and returns

## Code Style
- **Formatter**: Ruff (configured in `pyproject.toml`)
- **Docstrings**: Google-style format
- **Type Hints**: Required for all public functions
- **Imports**: Organized (standard library → third-party → local)
- **Line Length**: 88 characters (Black-compatible)

## Documentation
- All documentation files in `docs/` use numbered naming: `<XY>_<topic>.md`
- Start with `00_developer_guide.md` for setup and guidelines
- See `docs/` folder for comprehensive documentation

## Testing
- Manual testing with IB paper trading account
- Verify data fetching with small date ranges first
- Check ML model performance on test set before deployment
- Use `loguru` for debugging and monitoring

## Resources
- [Interactive Brokers API Documentation](https://interactivebrokers.github.io/tws-api/)
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- Project documentation: See `docs/` folder
