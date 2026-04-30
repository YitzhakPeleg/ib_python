# 02 - Data Fetching

## Overview

This document provides comprehensive documentation for the data fetching module, which handles all interactions with the Interactive Brokers API for retrieving historical market data.

## Table of Contents

- [Module Structure](#module-structure)
- [Core Components](#core-components)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Module Structure

```
src/data_fetching/
├── __init__.py                    # Public API exports
├── ibapi_wrapper.py              # Base IBapi wrapper class
├── historical_data_fetcher.py    # High-level data fetcher
└── date_converter.py             # Date/time utilities
```

## Core Components

### 1. IBapi Wrapper (`ibapi_wrapper.py`)

The base wrapper class that extends IB's `EWrapper` and `EClient` to provide enhanced functionality.

#### Key Features

- **Multi-Request Support**: Handle multiple concurrent data requests
- **Request Tracking**: Automatic tracking of request state and data
- **Thread-Safe**: Uses threading events for synchronization
- **DataFrame Export**: Convert accumulated data to Polars DataFrames

#### Class: `Request`

A dataclass representing a single historical data request.

```python
@dataclass
class Request:
    data: list = field(default_factory=list)
    ready: threading.Event = field(default_factory=threading.Event)
```

**Attributes:**
- `data`: List of bar data (DateTime, Open, High, Low, Close, Volume)
- `ready`: Threading event signaling request completion

**Methods:**
- `export() -> pl.DataFrame`: Convert accumulated data to Polars DataFrame

#### Class: `IBapi`

Enhanced wrapper combining EWrapper and EClient functionality.

```python
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.requests: Dict[int, Request] = {}
```

**Key Methods:**

- `historicalData(reqId: int, bar)`: Callback for each bar received
- `historicalDataEnd(reqId: int, start: str, end: str)`: Callback when request completes
- `get_data(reqId: int) -> pl.DataFrame`: Retrieve and remove request data
- `wait_for_data(reqId: int, timeout: timedelta | None) -> bool`: Wait for request completion
- `remove_request(reqId: int)`: Clean up request tracking

### 2. Historical Data Fetcher (`historical_data_fetcher.py`)

High-level interface for fetching historical market data with automatic connection management.

#### Class: `HistoricalDataFetcher`

```python
class HistoricalDataFetcher(IBapi):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1
    ):
        """Initialize the fetcher with IB connection parameters."""
```

**Parameters:**
- `host`: IB Gateway/TWS host address (default: "127.0.0.1")
- `port`: IB Gateway/TWS port (default: 4002 for paper trading)
- `client_id`: Unique client identifier (default: 1)

**Key Methods:**

#### `get_historical_data()`

Fetch historical market data for a specified contract.

```python
def get_historical_data(
    self,
    *,
    contract: ContractSpec,
    end_date: Optional[datetime] = None,
    duration: timedelta,
    frequency: BarFrequency = BarFrequency.ONE_HOUR,
    regular_trading_hours: bool = True,
    timeout: timedelta | None = None,
    timezone: Optional[str] = None,
) -> pl.DataFrame:
    """Fetch historical market data.
    
    Args:
        contract: Contract specification (symbol, exchange, etc.)
        end_date: End date for data (None = now)
        duration: How far back to fetch data
        frequency: Bar size (1 min, 1 hour, 1 day, etc.)
        regular_trading_hours: Only include RTH data
        timeout: Maximum wait time for request
        timezone: Timezone for DateTime column (default: UTC)
    
    Returns:
        Polars DataFrame with columns:
        - DateTime: Timestamp with timezone
        - Open: Opening price
        - High: High price
        - Low: Low price
        - Close: Closing price
        - Volume: Trading volume
    
    Raises:
        RuntimeError: If connection fails or request times out
        ValueError: If no data received
    """
```

**Example:**

```python
from datetime import datetime, timedelta
from src.data_fetching import HistoricalDataFetcher
from src.algo.models import ContractSpec, BarFrequency

contract = ContractSpec(symbol="AAPL")

with HistoricalDataFetcher(port=4002) as fetcher:
    df = fetcher.get_historical_data(
        contract=contract,
        end_date=datetime(2024, 12, 31, 23, 59, 59),
        duration=timedelta(days=30),
        frequency=BarFrequency.ONE_HOUR,
        timezone="US/Eastern"
    )
    
print(f"Fetched {len(df)} bars")
print(df.head())
```

#### Context Manager Support

The fetcher supports context manager protocol for automatic connection management:

```python
with HistoricalDataFetcher() as fetcher:
    # Connection automatically established
    df = fetcher.get_historical_data(...)
# Connection automatically closed
```

**Methods:**
- `__enter__()`: Establish connection
- `__exit__()`: Close connection
- `close()`: Manually close connection

### 3. Date Converter (`date_converter.py`)

Utility functions for date/time conversions.

#### Function: `add_date_int_column()`

Add an integer date column in YYYYMMDD format.

```python
def add_date_int_column(df: pl.DataFrame) -> pl.DataFrame:
    """Parse DateTime column and create integer date column.
    
    Handles both String and Datetime types in the DateTime column.
    
    Args:
        df: DataFrame with 'DateTime' column
    
    Returns:
        DataFrame with added 'date' column (Int64) in YYYYMMDD format
    
    Example:
        >>> df = pl.DataFrame({
        ...     "DateTime": ["2024-01-15T09:30:00-05:00"],
        ...     "Close": [150.0]
        ... })
        >>> df = add_date_int_column(df)
        >>> print(df["date"])
        [20240115]
    """
```

**Use Cases:**
- Group data by trading day
- Filter by date ranges
- Join datasets on date
- Create date-based features for ML

## Usage Guide

### Basic Usage

#### 1. Simple Data Fetch

```python
from datetime import datetime, timedelta
from src.data_fetching import HistoricalDataFetcher
from src.algo.models import ContractSpec, BarFrequency

# Create contract
contract = ContractSpec(symbol="AAPL")

# Fetch data
with HistoricalDataFetcher(port=4002) as fetcher:
    df = fetcher.get_historical_data(
        contract=contract,
        end_date=datetime.now(),
        duration=timedelta(days=7),
        frequency=BarFrequency.FIVE_MIN
    )

print(df)
```

#### 2. Multiple Symbols

```python
symbols = ["AAPL", "GOOGL", "MSFT"]

with HistoricalDataFetcher(port=4002) as fetcher:
    for symbol in symbols:
        contract = ContractSpec(symbol=symbol)
        df = fetcher.get_historical_data(
            contract=contract,
            duration=timedelta(days=30),
            frequency=BarFrequency.ONE_HOUR
        )
        
        # Save to file
        df.write_parquet(f"data/{symbol}_1_hour.parquet")
```

#### 3. Different Bar Frequencies

```python
from src.algo.models import BarFrequency

frequencies = [
    BarFrequency.ONE_MIN,
    BarFrequency.FIVE_MIN,
    BarFrequency.ONE_HOUR,
    BarFrequency.ONE_DAY
]

contract = ContractSpec(symbol="SPY")

with HistoricalDataFetcher(port=4002) as fetcher:
    for freq in frequencies:
        df = fetcher.get_historical_data(
            contract=contract,
            duration=timedelta(days=30),
            frequency=freq
        )
        
        filename = f"SPY_{freq.value.replace(' ', '_')}.parquet"
        df.write_parquet(f"data/{filename}")
```

### Advanced Usage

#### 1. Custom Timeout

For large data requests, increase the timeout:

```python
with HistoricalDataFetcher(port=4002) as fetcher:
    df = fetcher.get_historical_data(
        contract=ContractSpec(symbol="AAPL"),
        duration=timedelta(days=365),  # One year
        frequency=BarFrequency.ONE_MIN,
        timeout=timedelta(minutes=10)  # Longer timeout
    )
```

#### 2. Extended Trading Hours

Include pre-market and after-hours data:

```python
df = fetcher.get_historical_data(
    contract=ContractSpec(symbol="AAPL"),
    duration=timedelta(days=1),
    frequency=BarFrequency.ONE_MIN,
    regular_trading_hours=False  # Include extended hours
)
```

#### 3. Different Security Types

```python
from src.algo.models import SecurityType, Exchange

# Futures contract
futures_contract = ContractSpec(
    symbol="ES",
    sec_type=SecurityType.FUTURE,
    exchange=Exchange.SMART,
    currency=Currency.USD
)

# Options contract
options_contract = ContractSpec(
    symbol="AAPL",
    sec_type=SecurityType.OPTION,
    exchange=Exchange.SMART
)
```

#### 4. Timezone Handling

```python
# Get data in specific timezone
df = fetcher.get_historical_data(
    contract=ContractSpec(symbol="AAPL"),
    duration=timedelta(days=7),
    frequency=BarFrequency.ONE_HOUR,
    timezone="US/Eastern"  # Convert to Eastern Time
)

# DateTime column will be in US/Eastern timezone
print(df["DateTime"].dtype)  # Datetime(time_unit='ms', time_zone='US/Eastern')
```

## Configuration

### Connection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | "127.0.0.1" | IB Gateway/TWS host address |
| `port` | 4002 | Port number (4002=paper, 4001=live) |
| `client_id` | 1 | Unique client identifier |

### IB Gateway/TWS Setup

1. **Start IB Gateway or TWS**
   - Use paper trading account for development
   - Live trading requires different port (4001)

2. **Enable API Access**
   - Navigate to: Configure → Settings → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Check "Allow connections from localhost only"
   - Set "Socket port" to 4002 (paper) or 4001 (live)
   - Uncheck "Read-Only API" if you need trading capabilities

3. **Configure Trusted IPs** (if needed)
   - Add 127.0.0.1 to trusted IP addresses

### Bar Frequency Options

Available frequencies from `BarFrequency` enum:

| Frequency | Value | Use Case |
|-----------|-------|----------|
| `ONE_SEC` | "1 sec" | High-frequency analysis |
| `FIVE_SEC` | "5 sec" | Tick-level analysis |
| `ONE_MIN` | "1 min" | Intraday trading |
| `FIVE_MIN` | "5 min" | Short-term patterns |
| `FIFTEEN_MIN` | "15 min" | Intraday trends |
| `THIRTY_MIN` | "30 min" | Swing trading |
| `ONE_HOUR` | "1 hour" | Daily analysis |
| `FOUR_HOUR` | "4 hours" | Multi-day patterns |
| `ONE_DAY` | "1 day" | Long-term analysis |
| `ONE_WEEK` | "1 week" | Weekly trends |
| `ONE_MONTH` | "1 month" | Monthly patterns |

### Duration Limits

IB API has limitations on historical data based on bar size:

| Bar Size | Maximum Duration |
|----------|------------------|
| 1 sec - 30 sec | 1 day |
| 1 min | 30 days |
| 2 min - 30 min | 60 days |
| 1 hour | 1 year |
| 1 day | 20 years |

## Error Handling

### Common Errors

#### 1. Connection Timeout

```python
try:
    with HistoricalDataFetcher(port=4002) as fetcher:
        df = fetcher.get_historical_data(...)
except RuntimeError as e:
    if "Could not connect" in str(e):
        print("IB Gateway is not running or wrong port")
    elif "timed out" in str(e):
        print("Request took too long, try smaller duration")
```

#### 2. No Data Received

```python
try:
    df = fetcher.get_historical_data(...)
except ValueError as e:
    print(f"No data available: {e}")
    # Possible reasons:
    # - Symbol doesn't exist
    # - Market was closed during requested period
    # - Insufficient data history
```

#### 3. Invalid Parameters

```python
from pydantic import ValidationError

try:
    contract = ContractSpec(symbol="")  # Empty symbol
except ValidationError as e:
    print(f"Invalid contract: {e}")
```

### Retry Logic

Implement retry logic for transient failures:

```python
import time
from typing import Optional

def fetch_with_retry(
    fetcher: HistoricalDataFetcher,
    contract: ContractSpec,
    max_retries: int = 3,
    **kwargs
) -> Optional[pl.DataFrame]:
    """Fetch data with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            return fetcher.get_historical_data(
                contract=contract,
                **kwargs
            )
        except (RuntimeError, ValueError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed")
                return None
```

## Best Practices

### 1. Use Context Managers

Always use context managers for automatic connection cleanup:

```python
# Good
with HistoricalDataFetcher() as fetcher:
    df = fetcher.get_historical_data(...)

# Avoid
fetcher = HistoricalDataFetcher()
df = fetcher.get_historical_data(...)
fetcher.close()  # Easy to forget
```

### 2. Respect Rate Limits

IB API has rate limits (~60 requests per 10 minutes):

```python
import time

symbols = ["AAPL", "GOOGL", "MSFT", ...]

with HistoricalDataFetcher() as fetcher:
    for i, symbol in enumerate(symbols):
        df = fetcher.get_historical_data(...)
        
        # Add delay every 50 requests
        if (i + 1) % 50 == 0:
            logger.info("Rate limit pause...")
            time.sleep(600)  # 10 minutes
```

### 3. Save Data Locally

Cache fetched data to avoid repeated API calls:

```python
from pathlib import Path

def get_or_fetch_data(
    symbol: str,
    duration: timedelta,
    frequency: BarFrequency,
    force_refresh: bool = False
) -> pl.DataFrame:
    """Get data from cache or fetch if not available."""
    cache_file = Path(f"data/{symbol}_{frequency.value.replace(' ', '_')}.parquet")
    
    if cache_file.exists() and not force_refresh:
        logger.info(f"Loading cached data for {symbol}")
        return pl.read_parquet(cache_file)
    
    logger.info(f"Fetching fresh data for {symbol}")
    with HistoricalDataFetcher() as fetcher:
        df = fetcher.get_historical_data(
            contract=ContractSpec(symbol=symbol),
            duration=duration,
            frequency=frequency
        )
    
    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    df.write_parquet(cache_file)
    
    return df
```

### 4. Use Appropriate Bar Sizes

Match bar size to your analysis needs:

```python
# For intraday trading (last 7 days)
df = fetcher.get_historical_data(
    duration=timedelta(days=7),
    frequency=BarFrequency.FIVE_MIN
)

# For daily analysis (last year)
df = fetcher.get_historical_data(
    duration=timedelta(days=365),
    frequency=BarFrequency.ONE_DAY
)
```

### 5. Add Date Column

Always add date integer column for grouping:

```python
from src.data_fetching.date_converter import add_date_int_column

df = fetcher.get_historical_data(...)
df = add_date_int_column(df)

# Now you can group by date
daily_stats = df.group_by("date").agg([
    pl.col("Close").first().alias("open"),
    pl.col("Close").last().alias("close"),
    pl.col("High").max().alias("high"),
    pl.col("Low").min().alias("low")
])
```

## Troubleshooting

### Issue: Cannot Connect to IB Gateway

**Symptoms:**
- `RuntimeError: Could not connect to IB`
- Connection timeout

**Solutions:**
1. Verify IB Gateway/TWS is running
2. Check port number (4002 for paper, 4001 for live)
3. Ensure API is enabled in Gateway settings
4. Check firewall settings
5. Try restarting IB Gateway

### Issue: Request Times Out

**Symptoms:**
- `RuntimeError: Historical data request timed out`

**Solutions:**
1. Increase timeout parameter
2. Reduce duration or use larger bar size
3. Check IB API rate limits
4. Verify market hours for the symbol

### Issue: No Data Returned

**Symptoms:**
- Empty DataFrame or `ValueError: No data to export`

**Solutions:**
1. Verify symbol exists and is correct
2. Check if market was open during requested period
3. Try different date range
4. Ensure you have data permissions for the symbol

### Issue: Incorrect Timezone

**Symptoms:**
- DateTime values don't match expected timezone

**Solutions:**
1. Specify timezone parameter explicitly
2. Convert timezone after fetching:
   ```python
   df = df.with_columns(
       pl.col("DateTime").dt.convert_time_zone("US/Eastern")
   )
   ```

## Next Steps

- Review [03_algorithms.md](03_algorithms.md) for data processing and ML
- Check [04_api_reference.md](04_api_reference.md) for complete API documentation
- See [05_examples.md](05_examples.md) for more usage examples
