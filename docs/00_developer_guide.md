# 00 - Developer Guide

## Prerequisites

1. **Python 3.12+**
2. **UV** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. **IB Gateway or TWS** — use paper trading account for development

## Setup

```bash
uv sync
cp .env.example .env   # fill in IB_HOST, IB_PORT, IB_CLIENT_ID
```

## Project structure

```
ib_python/
├── src/
│   ├── data_fetching/           # IB API wrappers
│   │   ├── ibapi_wrapper.py
│   │   ├── historical_data_fetcher.py
│   │   └── date_converter.py
│   ├── models/                  # Pydantic types and path helpers
│   │   ├── models.py
│   │   └── paths.py
│   ├── algo/
│   │   └── resample_bars.py
│   └── visualization/
│       └── plotting.py
├── data/                        # Parquet files (created at runtime)
├── docs/
├── pyproject.toml
├── uv.lock
└── CLAUDE.md
```

## Development workflow

```bash
# Lint and format
uv run ruff check src/
uv run ruff format src/

# Run a script
uv run python src/data_fetching/historical_data_fetcher.py

# Interactive analysis
uv run jupyter lab
```

New tasks are tracked as GitHub issues. Use `/task <title>` in Claude Code to scaffold a branch and issue automatically.

## Code style

- **Line length**: 88 chars (Black-compatible, enforced by Ruff)
- **Imports**: stdlib → third-party → local
- **Type hints** on all public functions
- **Logging**: `loguru` (`from loguru import logger`)

Example:

```python
import polars as pl
from loguru import logger
from src.models.models import ContractSpec, BarFrequency
```

## Fetching data

```python
from datetime import datetime, timedelta
from src.data_fetching.historical_data_fetcher import HistoricalDataFetcher
from src.models.models import ContractSpec, BarFrequency

contract = ContractSpec(symbol="AAPL")

with HistoricalDataFetcher(port=4002) as fetcher:
    df = fetcher.get_historical_data(
        contract=contract,
        end_date=datetime.now(),
        duration=timedelta(days=1),
        frequency=BarFrequency.ONE_MIN,
        timeout=timedelta(minutes=5),
    )
print(f"Fetched {len(df)} bars")
```

## Troubleshooting

**Cannot connect to IB Gateway**
- Verify Gateway is running
- Check port (4002 paper / 4001 live)
- Enable API in Gateway: Configure → Settings → API → Settings

**Request times out**
- Increase `timeout` parameter
- Reduce duration or use a larger bar size

**Import errors**
- Run from project root with `uv run python ...`

## Next steps

- [01_architecture.md](01_architecture.md) — system design and data flow
- [02_data_fetching.md](02_data_fetching.md) — full data fetching reference
