# ib-python

A Python toolkit for fetching intraday equity data from Interactive Brokers, resampling OHLC bars, and visualizing candlestick charts.

## Requirements

- Python >= 3.12
- [UV](https://astral.sh/uv) package manager
- Interactive Brokers TWS or IB Gateway (for data fetching)

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and set your IB connection parameters:

```
IB_HOST=127.0.0.1
IB_PORT=4002       # paper trading; 4001 = live
IB_CLIENT_ID=123
```

IB Gateway must be running before any data fetch.

## Project structure

```
src/
├── data_fetching/       # IB API wrapper and historical data fetcher
│   ├── ibapi_wrapper.py
│   ├── historical_data_fetcher.py
│   └── date_converter.py
├── models/              # Pydantic types and file path helpers
│   ├── models.py        # ContractSpec, BarFrequency, SecurityType, …
│   └── paths.py         # DATA_PATH, get_file(ticker, frequency)
├── algo/
│   └── resample_bars.py # resample_to_timeframe(df, "5m" | "15m" | …)
└── visualization/
    └── plotting.py      # plot_bars(df, …) — Plotly candlestick chart

data/                    # Parquet files: {TICKER}_{frequency}.parquet
docs/                    # Architecture and data-fetching docs
```

## Quick start

### 1. Fetch historical data

```python
from datetime import datetime, timedelta
from src.data_fetching.historical_data_fetcher import HistoricalDataFetcher
from src.models.models import ContractSpec, BarFrequency

contract = ContractSpec(symbol="SPY")

with HistoricalDataFetcher(port=4002) as fetcher:
    df = fetcher.get_historical_data(
        contract=contract,
        end_date=datetime.now(),
        duration=timedelta(days=30),
        frequency=BarFrequency.ONE_MIN,
        timezone="US/Eastern",
    )

df.write_parquet("data/SPY_1_min.parquet")
```

### 2. Resample to a higher timeframe

```python
import polars as pl
from src.algo.resample_bars import resample_to_timeframe

df = pl.read_parquet("data/SPY_1_min.parquet")
df_5m = resample_to_timeframe(df, "5m")
```

### 3. Plot

```python
from src.visualization.plotting import plot_bars

plot_bars(df_5m, title="SPY — 5-minute bars")
```

`plot_bars` accepts optional pre-calculated Bollinger Band columns (`bb_upper`, `bb_mid`, `bb_lower`) and a `trades` DataFrame for entry/exit markers.

## IB Gateway setup

1. Start IB Gateway (paper trading account recommended)
2. Enable API: Configure → Settings → API → Settings → "Enable ActiveX and Socket Clients"
3. Set socket port to `4002` (paper) or `4001` (live)

See [docs/02_data_fetching.md](docs/02_data_fetching.md) for full usage details.

## Development

```bash
uv run ruff check src/    # lint
uv run ruff format src/   # format
uv run jupyter lab        # notebooks
```
