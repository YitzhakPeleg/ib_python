# 00 - Developer Guide

## Overview

This guide provides comprehensive instructions for setting up your development environment, understanding the project workflow, and contributing to the ib-python project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

1. **Python 3.12 or higher**
   ```bash
   python --version  # Should be >= 3.12
   ```

2. **UV Package Manager** (Recommended)
   ```bash
   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Verify installation
   uv --version
   ```

3. **Interactive Brokers TWS or IB Gateway**
   - Download from [Interactive Brokers](https://www.interactivebrokers.com/)
   - Required for fetching historical market data
   - Use paper trading account for development

### Optional Tools

- **Git** - Version control
- **VS Code** - Recommended IDE with Python extension
- **Jupyter Lab** - For notebook-based analysis (included in dependencies)

## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ib_python
```

### 2. Install Dependencies

#### Using UV (Recommended)

```bash
# Sync all dependencies from uv.lock
uv sync

# Install project in editable mode
uv pip install -e .
```

#### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Configure IB Gateway/TWS

1. **Start IB Gateway or TWS**
   - Use paper trading account for development
   - Default ports:
     - Paper trading: `4002`
     - Live trading: `4001`

2. **Enable API Connections**
   - In TWS/Gateway: Configure → Settings → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Check "Allow connections from localhost only"
   - Set "Socket port" to `4002` (paper) or `4001` (live)
   - Uncheck "Read-Only API"

3. **Test Connection**
   ```python
   from src.data_fetching import HistoricalDataFetcher
   
   # This should connect without errors
   with HistoricalDataFetcher(host="127.0.0.1", port=4002) as fetcher:
       print("Connected successfully!")
   ```

### 4. Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
uv pip list

# Run a simple test
python -c "import polars; import torch; print('All imports successful!')"
```

## Project Structure

```
ib_python/
├── src/                          # Source code
│   ├── data_fetching/           # IB API wrappers
│   │   ├── __init__.py
│   │   ├── ibapi_wrapper.py
│   │   ├── historical_data_fetcher.py
│   │   └── date_converter.py
│   ├── algo/                    # Algorithms and ML
│   │   ├── models.py
│   │   ├── bollinger_bands.py
│   │   └── train_algo.py
│   └── notebooks/               # Analysis scripts
│       └── plot_data.py
├── docs/                        # Documentation
│   ├── 00_developer_guide.md   # This file
│   ├── 01_architecture.md
│   ├── 02_data_fetching.md
│   ├── 03_algorithms.md
│   ├── 04_api_reference.md
│   └── 05_examples.md
├── data/                        # Data storage (created at runtime)
├── main.py                      # Entry point
├── pyproject.toml               # Project configuration
├── uv.lock                      # Dependency lock file
├── README.md                    # Project overview
└── CLAUDE.md                    # Development context
```

## Development Workflow

### 1. Create a New Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the style guidelines below
- Add docstrings to all functions (Google-style format)
- Include type hints for all parameters and returns
- Add inline comments for complex logic

### 3. Test Your Changes

```bash
# Ensure IB Gateway is running
# Run your code
python src/your_module.py

# Or use Jupyter for interactive testing
jupyter lab
```

### 4. Format and Lint

```bash
# Format code
ruff format src/

# Check for issues
ruff check src/

# Fix auto-fixable issues
ruff check --fix src/
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
# Create PR on GitHub/GitLab
```

## Code Style Guidelines

### Python Style

- **PEP 8 Compliant**: Follow Python's style guide
- **Line Length**: 88 characters (Black-compatible)
- **Imports**: Organize in three groups:
  1. Standard library
  2. Third-party packages
  3. Local modules

Example:
```python
import threading
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from loguru import logger

from src.algo.models import ContractSpec, BarFrequency
```

### Docstrings

Use **Google-style docstrings** for all public functions and classes:

```python
def calculate_bollinger_bands(
    df: pl.DataFrame, window: int, stds: float
) -> pl.DataFrame:
    """Calculate Bollinger Bands for a Polars DataFrame.

    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    that are standard deviations away from the middle band.

    Args:
        df: DataFrame with columns ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        window: Moving average period (e.g., 20 for 20-period SMA)
        stds: Number of standard deviations for the bands (typically 2.0)

    Returns:
        DataFrame with added columns: 'bb_lower', 'bb_mid', 'bb_upper'

    Raises:
        ValueError: If window is less than 2 or stds is negative

    Example:
        >>> df = pl.read_csv("AAPL_1_min.csv")
        >>> df_with_bb = calculate_bollinger_bands(df, window=20, stds=2.0)
        >>> print(df_with_bb.columns)
        ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'bb_lower', 'bb_mid', 'bb_upper']
    """
    # Implementation here
```

### Type Hints

Always include type hints:

```python
from typing import Optional
from datetime import datetime, timedelta
import polars as pl

def fetch_data(
    symbol: str,
    start_date: datetime,
    end_date: Optional[datetime] = None,
    frequency: str = "1 hour"
) -> pl.DataFrame:
    """Fetch historical data."""
    # Implementation
```

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

```python
# Good
class HistoricalDataFetcher:
    DEFAULT_PORT = 4002
    
    def __init__(self):
        self._connected = False
    
    def get_historical_data(self):
        pass
    
    def _connect(self):
        pass
```

### Comments

- Use inline comments sparingly for complex logic
- Prefer self-documenting code with clear variable names
- Add comments for non-obvious business logic

```python
# Calculate ATR-based targets instead of fixed percentages
# This adapts to market volatility
target_price = entry_price + (atr * 0.5)
```

## Testing

### Manual Testing

1. **Start IB Gateway**
   ```bash
   # Ensure paper trading gateway is running on port 4002
   ```

2. **Test Data Fetching**
   ```python
   from datetime import datetime, timedelta
   from src.data_fetching import HistoricalDataFetcher
   from src.algo.models import ContractSpec, BarFrequency
   
   contract = ContractSpec(symbol="AAPL")
   
   with HistoricalDataFetcher(port=4002) as fetcher:
       df = fetcher.get_historical_data(
           contract=contract,
           end_date=datetime.now(),
           duration=timedelta(days=1),
           frequency=BarFrequency.ONE_MIN
       )
       print(f"Fetched {len(df)} bars")
   ```

3. **Test Indicators**
   ```python
   from src.algo.bollinger_bands import calculate_bollinger_bands
   
   df_with_bb = calculate_bollinger_bands(df, window=20, stds=2.0)
   print(df_with_bb.head())
   ```

### Unit Testing (Future)

```bash
# When tests are added
pytest tests/ -v
```

## Common Tasks

### Adding a New Dependency

```bash
# Using UV
uv add package-name

# Using pip
pip install package-name
# Then update pyproject.toml manually
```

### Fetching Data for a New Symbol

```python
from datetime import datetime, timedelta
from src.data_fetching import get_data
from src.algo.models import BarFrequency

get_data(
    symbol="TSLA",
    end_date=datetime(2024, 12, 31, 23, 59, 59),
    frequency=BarFrequency.FIVE_MIN,
    duration=timedelta(days=30)
)
```

### Training a New Model

```python
from src.algo.train_algo import main

# Train with custom parameters
main(
    ticker="AAPL",
    max_depth=5,
    min_samples_leaf=15
)
```

### Creating Visualizations

```python
# Edit src/notebooks/plot_data.py with your ticker
# Then run:
python src/notebooks/plot_data.py
```

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to IB Gateway

**Solutions**:
1. Verify IB Gateway/TWS is running
2. Check port number (4002 for paper, 4001 for live)
3. Ensure API is enabled in Gateway settings
4. Check firewall settings
5. Try restarting IB Gateway

```python
# Test connection
from src.data_fetching import HistoricalDataFetcher

try:
    fetcher = HistoricalDataFetcher(host="127.0.0.1", port=4002)
    print("Connected!")
except RuntimeError as e:
    print(f"Connection failed: {e}")
```

### Data Fetching Timeouts

**Problem**: Historical data requests timeout

**Solutions**:
1. Reduce the duration or increase timeout
2. Check IB API rate limits
3. Verify market hours for the symbol
4. Use smaller bar frequencies for longer durations

```python
# Increase timeout
df = fetcher.get_historical_data(
    contract=contract,
    duration=timedelta(days=1),
    frequency=BarFrequency.ONE_MIN,
    timeout=timedelta(minutes=5)  # Increase from default
)
```

### Import Errors

**Problem**: Module not found errors

**Solutions**:
1. Ensure you're in the project root directory
2. Install in editable mode: `uv pip install -e .`
3. Check Python path: `echo $PYTHONPATH`
4. Verify virtual environment is activated

### Memory Issues with Large Datasets

**Problem**: Out of memory when processing large datasets

**Solutions**:
1. Use Polars lazy evaluation
2. Process data in chunks
3. Use streaming mode for large files
4. Filter data early in the pipeline

```python
# Use lazy evaluation
df = pl.scan_parquet("large_file.parquet")
result = df.filter(pl.col("date") > 20240101).collect()
```

## Best Practices

1. **Always use paper trading** for development and testing
2. **Test with small date ranges** before fetching large datasets
3. **Use type hints** for better IDE support and error catching
4. **Write docstrings** for all public functions
5. **Log important events** using loguru
6. **Handle errors gracefully** with try-except blocks
7. **Use context managers** for resource management (e.g., `with HistoricalDataFetcher()`)
8. **Store data in Parquet format** for efficiency
9. **Use ATR-based targets** instead of fixed percentages
10. **Validate data** before training ML models

## Getting Help

- **Documentation**: Check `docs/` folder for detailed guides
- **IB API Docs**: [https://interactivebrokers.github.io/tws-api/](https://interactivebrokers.github.io/tws-api/)
- **Polars Docs**: [https://pola-rs.github.io/polars/](https://pola-rs.github.io/polars/)
- **PyTorch Docs**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

## Next Steps

- Read [01_architecture.md](01_architecture.md) to understand the system design
- Review [02_data_fetching.md](02_data_fetching.md) for data fetching details
- Explore [05_examples.md](05_examples.md) for practical examples
