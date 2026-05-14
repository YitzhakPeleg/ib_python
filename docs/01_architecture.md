# 01 - Architecture

## Overview

ib-python is a data-fetching and charting toolkit for intraday equity research. Data flows from IB Gateway → Parquet storage → resample/plot.

## Component layers

```
┌─────────────────────────────────────────────────────┐
│                   ib-python                          │
├──────────────┬──────────────┬───────────────────────┤
│ data_fetching│    models    │  algo / visualization  │
│  (IB API)    │ (Pydantic)   │ (resample, plot)       │
└──────┬───────┴──────────────┴───────────┬───────────┘
       │                                  │
       ▼                                  ▼
IB Gateway/TWS                    data/ (Parquet files)
```

### Data fetching (`src/data_fetching/`)

```
ibapi_wrapper.py              # IBapi(EWrapper, EClient) — multi-request tracking
historical_data_fetcher.py    # HistoricalDataFetcher — high-level interface
date_converter.py             # add_date_int_column() — YYYYMMDD int column
```

- `IBapi` tracks concurrent requests via `Dict[int, Request]`; each `Request` holds bar data and a `threading.Event` for completion signaling
- `HistoricalDataFetcher` wraps `IBapi` with connection management and context-manager support
- Fetched data is returned as Polars DataFrames with columns `DateTime, Open, High, Low, Close, Volume`

### Models (`src/models/`)

```
models.py    # ContractSpec, BarFrequency, SecurityType, Exchange, Currency, Duration
paths.py     # DATA_PATH, get_file(ticker, frequency) → Path
```

Pydantic types used across all layers for type-safe contract specification and file path resolution.

### Algo (`src/algo/`)

```
resample_bars.py    # resample_to_timeframe(df, timeframe) → pl.DataFrame
```

Uses Polars `group_by_dynamic` to aggregate OHLCV data to any interval string (`"5m"`, `"15m"`, `"1h"`, etc.). Adds a `date` column (`pl.Date`).

### Visualization (`src/visualization/`)

```
plotting.py    # plot_bars(df, …) → go.Figure | None
```

Interactive Plotly candlestick chart with optional Bollinger Band overlay (pre-calculated columns), volume subplot, and trade-entry/exit markers.

## Data flow

```
HistoricalDataFetcher.get_historical_data()
    │
    ├── reqHistoricalData() → IB Gateway
    │       ↓ callbacks
    │   historicalData()     # accumulate bars in Request.data
    │   historicalDataEnd()  # set Request.ready event
    │
    ├── wait_for_data()      # block until ready or timeout
    └── get_data()           # export as Polars DataFrame
            │
            ▼
    df.write_parquet("data/SPY_1_min.parquet")
            │
            ▼
    resample_to_timeframe(df, "5m")
            │
            ▼
    plot_bars(df_5m, …)
```

## Design patterns

**Context manager** — automatic connection cleanup:
```python
with HistoricalDataFetcher() as fetcher:
    df = fetcher.get_historical_data(...)
```

**Request tracking** — thread-safe, concurrent requests:
```python
class IBapi:
    requests: Dict[int, Request]   # keyed by reqId
    # historicalData() appends; historicalDataEnd() sets .ready
```

**Pydantic models** — validated inputs:
```python
contract = ContractSpec(symbol="SPY")          # defaults: SMART, USD, STK
freq = BarFrequency.ONE_MIN                    # "1 min"
```

## Technology choices

| Technology | Purpose |
|------------|---------|
| Polars | DataFrames — fast, Arrow-native, no pandas |
| IBapi (local wheel) | Official IB API |
| Pydantic | Type-safe contract/config models |
| Plotly | Interactive candlestick charts |
| Loguru | Structured logging |
| PyTorch + scikit-learn | Installed; ML modules TBD |

## Next steps

- [02_data_fetching.md](02_data_fetching.md) — full data fetching reference
