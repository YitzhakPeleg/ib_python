# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                        # Install/update dependencies
uv add <package>               # Add a package
uv run python <script>.py      # Run a script in the venv

# Linting / formatting
uv run ruff check src/
uv run ruff format src/
```

## Architecture

The project is a data-fetching and charting toolkit for intraday equity data via Interactive Brokers. There is no test suite.

### Data layer (`src/models/`, `data/`)

- `src/models/models.py` — Pydantic types: `ContractSpec`, `BarFrequency`, `SecurityType`, `Exchange`, `Currency`, `Duration`
- `src/models/paths.py` — `DATA_PATH` constant and `get_file(ticker, frequency)` which resolves Parquet filenames like `SPY_1_min.parquet`
- `data/` — Parquet files fetched from IB API, named `{TICKER}_{frequency}.parquet` (e.g., `SPY_1_min.parquet`)
- All DataFrames use a `DateTime` column (Polars `Datetime`) and `Open/High/Low/Close/Volume`. A `date` column (`pl.Date`) is added by `resample_to_timeframe`.

### Data fetching (`src/data_fetching/`)

- `ibapi_wrapper.py` — extends IB's `EWrapper`/`EClient` with multi-request support and threading
- `historical_data_fetcher.py` — high-level fetcher; connects to TWS/Gateway at `localhost:4002` (paper) or `4001` (live)
- IB Gateway must be running before any fetch; fetched data is saved as Parquet

### Utilities (`src/algo/`, `src/visualization/`)

- `src/algo/resample_bars.py` — `resample_to_timeframe(df, timeframe)`: resample any OHLC DataFrame to a Polars interval string (`"5m"`, `"15m"`, `"1h"`, etc.)
- `src/visualization/plotting.py` — `plot_bars(df, ...)`: interactive Plotly candlestick chart with optional Bollinger Bands overlay, volume subplot, and trade markers

## Key conventions

- **Package manager**: `uv` only — never `pip` directly
- **DataFrame library**: Polars (not pandas)
- **Formatter/linter**: Ruff (88-char line length, Black-compatible)
- **Import paths**: scripts use `from src.algo...` / `from src.models...`; modules inside `src/` use bare module names (e.g., `from models import ...`)
- **IB connection**: paper trading port `4002`, live `4001`; never hardcode live credentials
