# ib-python

This project provides tools and utilities for working with Interactive Brokers (IB) API in Python, including data analysis and visualization.

## Features
- Integration with Interactive Brokers API (ibapi)
- Data analysis with pandas and polars
- Visualization with plotly

## Installation

1. Clone this repository.
2. Install dependencies using your preferred Python package manager (e.g., pip or pipx):
   ```bash
   pip install .
   ```
   Or, to install from source with all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Example usage:
```python
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
# ... your code here ...
```

## Project Structure
- `main.py` — Main entry point
- `src/` — Source code
- `ibapi-10.30.1-py3-none-any.whl` — Local IB API wheel

## License

This project is licensed under the MIT License.
