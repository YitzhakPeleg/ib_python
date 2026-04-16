# CLAUDE.md - Project Context & Guidelines

## Project Overview
- **Name**: ib-python
- **Purpose**: Interactive Brokers API wrapper and market data utilities
- **Python Version**: >=3.12

## Package Management
- **Tool**: UV (uv) - NOT pip
- **Commands**:
  - Install dependencies: `uv sync`
  - Add package: `uv pip install <package>`
  - Install project in editable mode: `uv pip install -e .`

## Project Structure
```
ib_python/
├── src/
│   ├── settings.py          # Pydantic settings configuration
│   ├── get_ask_price.py     # IBapi wrapper for market data
│   ├── check.py
│   └── hourly_prices.py
├── main.py
├── pyproject.toml           # UV/pip project config
├── EURUSD_Hourly.csv
├── README.md
└── .env.example             # Environment variables template
```

## Current Branch
- **Branch**: `update-deps-and-remove-devcontainer`
- **Default Branch**: `main`

## Key Dependencies
- `pydantic-settings>=2.0.0` - Configuration management
- `ibapi` - Local wheel file (IBapi 10.45.1)
- `loguru` - Logging
- `polars` - Data processing
- `plotly` - Visualization

## Configuration
- Settings are managed via `src/settings.py` using Pydantic Settings
- Configuration can be set via `.env` file (copy from `.env.example`)
- Settings are accessible as: `from settings import settings`

## Important Notes
- IBapi connects to TWS/Gateway on `localhost:7497` (paper trading) or `7496` (live)
- All IB connection code should use settings for configuration
- Use `uv` commands instead of pip/poetry

## When Making Changes
- Update dependencies in `pyproject.toml`
- Use `uv sync` after updating dependencies
- Follow the settings pattern for any new configuration
