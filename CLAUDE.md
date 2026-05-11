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

# Run a strategy backtest
python run_first_bar_breakout.py
python run_grid_search_dra.py
python run_grid_search_rr.py
python run_grid_search_fixed.py
python run_grid_search_mixed_fixed.py
python run_all_grid_searches_5min.py

# ML pipeline
uv run python src/algo/train_algo.py
uv run python src/algo/train_signal_model.py
```

## Architecture

The project is a backtesting and signal-generation toolkit for intraday equity trading, primarily focused on SPY via Interactive Brokers. There is no test suite — validation is done by running strategy scripts and checking `results/`.

### Data layer (`src/models/`, `data/`)

- `src/models/models.py` — shared Pydantic/dataclass types: `ContractSpec`, `BarFrequency`, `SignalType`, `TradeSetup`, `SignalResult`
- `src/models/paths.py` — `DATA_PATH` constant (repo root `/data/`) and `get_file(ticker, frequency)` which resolves Parquet filenames like `SPY_1_min.parquet`
- `data/` — Parquet files fetched from IB API, named `{TICKER}_{frequency}.parquet` (e.g., `SPY_1_min.parquet`)
- All DataFrames use a `DateTime` column (Polars `Datetime`) and `Open/High/Low/Close/Volume`. A `date` column (`pl.Date`) is derived as needed.

### Data fetching (`src/data_fetching/`)

- `ibapi_wrapper.py` — extends IB's `EWrapper`/`EClient` with multi-request support and threading
- `historical_data_fetcher.py` — high-level fetcher; connects to TWS/Gateway at `localhost:4002` (paper) or `4001` (live)
- IB Gateway must be running before any fetch; fetched data is saved as Parquet

### Strategy layer (`src/algo/`)

Each strategy variant is a self-contained class with a `backtest(df)` method that returns a list of `TradeResult` dataclasses. All variants follow the same First Bar Breakout pattern — trade the breakout of the 9:30 AM bar — but differ in how TP/SL are sized:

| File | TP/SL sizing |
|---|---|
| `first_bar_breakout.py` | Risk-reward ratio × first-bar range |
| `first_bar_breakout_dra.py` | Daily Range Average (DRA) × coefficient K |
| `first_bar_breakout_fixed.py` | Fixed dollar amount |
| `first_bar_breakout_mixed_fixed.py` | Mixed: fixed TP, DRA-based SL |

Supporting modules:
- `daily_range.py` — `daily_range_avg(df, n)`: rolling average of daily high-low range, joined back to intraday df
- `resample_bars.py` — resample 1-min data to any timeframe (`resample_to_5min`, `resample_to_timeframe`)
- `backtester.py` — generic backtest engine for `TradeSetup`/`SignalResult` objects (used by ML signal path)
- `signal_generator.py` — `SignalGenerator` loads a `.joblib` sklearn model, runs feature engineering, and emits `TradeSetup` objects
- `signal_detector.py` / `feature_engineering.py` / `labeling.py` — feature engineering for the ML pipeline (morning window 09:00–11:00 ET)

Grid search runners (`run_grid_search_*.py` at repo root) iterate over parameter combinations and write CSVs + summary `.txt` files to `results/`.

### Visualization (`src/visualization/`)

- `plotting.py` — Plotly-based candlestick/indicator charts (`plot_bars`)
- `src/notebooks/` — standalone analysis scripts; can be run directly or in JupyterLab

## Key conventions

- **Package manager**: `uv` only — never `pip` directly
- **DataFrame library**: Polars (not pandas)
- **Formatter/linter**: Ruff (88-char line length, Black-compatible)
- **Docstrings**: Google style on all public functions
- **Import paths**: scripts at repo root use `from src.algo...` / `from src.models...`; modules inside `src/` use relative imports or bare module names (e.g., `from models import ...`)
- **Results**: all output CSVs and summary files go to `results/`
- **IB connection**: paper trading port `4002`, live `4001`; never hardcode live credentials
