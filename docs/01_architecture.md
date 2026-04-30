# 01 - Architecture

## Overview

This document describes the system architecture, design patterns, and data flow of the ib-python project. Understanding the architecture will help you navigate the codebase and make informed decisions when extending functionality.

## Table of Contents

- [System Architecture](#system-architecture)
- [Module Design](#module-design)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)
- [Technology Stack](#technology-stack)
- [Performance Considerations](#performance-considerations)

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ib-python System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Data       │      │  Algorithm   │      │ Notebook  │ │
│  │   Fetching   │─────▶│  Processing  │─────▶│ Analysis  │ │
│  │   Module     │      │  Module      │      │ Module    │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         │                      │                     │       │
│         ▼                      ▼                     ▼       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Data Storage (Parquet Files)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         │                                            ▲
         │                                            │
         ▼                                            │
┌─────────────────┐                          ┌───────────────┐
│  IB Gateway/TWS │                          │  User/Client  │
│  (External API) │                          │  Application  │
└─────────────────┘                          └───────────────┘
```

### Component Layers

1. **Data Fetching Layer** (`src/data_fetching/`)
   - Interfaces with Interactive Brokers API
   - Manages connections and requests
   - Converts raw data to structured DataFrames

2. **Algorithm Layer** (`src/algo/`)
   - Technical indicator calculations
   - Feature engineering
   - Machine learning models
   - Data models and type definitions

3. **Analysis Layer** (`src/notebooks/`)
   - Data visualization
   - Exploratory analysis
   - Performance reporting

4. **Storage Layer**
   - Parquet files for efficient data storage
   - Structured file naming conventions

## Module Design

### Data Fetching Module

```
src/data_fetching/
├── __init__.py                    # Public API exports
├── ibapi_wrapper.py              # Base wrapper (EWrapper + EClient)
├── historical_data_fetcher.py    # High-level fetcher interface
└── date_converter.py             # Date/time utilities
```

**Design Principles:**
- **Separation of Concerns**: Low-level API wrapper vs. high-level interface
- **Context Manager Pattern**: Automatic connection management
- **Multi-Request Support**: Handle concurrent data requests
- **Type Safety**: Pydantic models for contracts and parameters

**Key Classes:**

```python
IBapi (EWrapper, EClient)
    ├── Request tracking
    ├── Data accumulation
    └── Event handling

HistoricalDataFetcher (IBapi)
    ├── Connection management
    ├── High-level data fetching
    └── DataFrame conversion
```

### Algorithm Module

```
src/algo/
├── models.py                # Pydantic data models
├── bollinger_bands.py       # Technical indicators
└── train_algo.py           # ML pipeline
```

**Design Principles:**
- **Functional Programming**: Pure functions for indicators
- **Type Safety**: Pydantic models for validation
- **Composability**: Small, reusable functions
- **Performance**: Polars for fast data processing

**Key Components:**

1. **Data Models** (`models.py`)
   - `ContractSpec`: Contract specifications
   - `BarFrequency`: Time intervals
   - `SecurityType`, `Exchange`, `Currency`: Enums

2. **Technical Indicators** (`bollinger_bands.py`)
   - Stateless functions
   - Polars DataFrame input/output
   - Configurable parameters

3. **ML Pipeline** (`train_algo.py`)
   - Feature engineering
   - Model training (Decision Tree, Neural Network)
   - Backtesting and evaluation

### Notebooks Module

```
src/notebooks/
└── plot_data.py            # Visualization scripts
```

**Design Principles:**
- **Interactive Analysis**: Can run as script or in Jupyter
- **Plotly Integration**: Interactive charts
- **Reusable Patterns**: Template for new visualizations

## Data Flow

### 1. Historical Data Fetching Flow

```
User Request
    │
    ▼
HistoricalDataFetcher.get_historical_data()
    │
    ├─▶ Create Contract object
    ├─▶ Format parameters (dates, duration, frequency)
    ├─▶ Generate unique request ID
    │
    ▼
IBapi.reqHistoricalData()
    │
    ▼
IB Gateway/TWS (External)
    │
    ├─▶ historicalData() callbacks (multiple)
    │   └─▶ Accumulate bars in Request.data
    │
    └─▶ historicalDataEnd() callback
        └─▶ Set Request.ready event
    │
    ▼
Wait for completion (with timeout)
    │
    ▼
Convert to Polars DataFrame
    │
    ├─▶ Parse epoch timestamps
    ├─▶ Convert to datetime with timezone
    └─▶ Structure OHLCV columns
    │
    ▼
Return DataFrame to user
```

### 2. Technical Indicator Calculation Flow

```
Raw OHLCV DataFrame
    │
    ▼
calculate_bollinger_bands(df, window, stds)
    │
    ├─▶ Calculate rolling mean (SMA)
    ├─▶ Calculate rolling std deviation
    ├─▶ Compute upper band (SMA + stds * std)
    ├─▶ Compute lower band (SMA - stds * std)
    │
    ▼
DataFrame with BB columns added
    │
    ▼
Ready for analysis or ML
```

### 3. ML Training Pipeline Flow

```
Historical Data (Parquet)
    │
    ▼
Load and preprocess
    │
    ├─▶ Add date integer column
    ├─▶ Calculate Bollinger Bands
    ├─▶ Add row numbering per day
    │
    ▼
Feature Engineering
    │
    ├─▶ Calculate daily ATR
    ├─▶ Calculate overnight gap
    ├─▶ Normalize prices (relative to open)
    ├─▶ Calculate BB/ATR ratio
    ├─▶ Pivot first N bars per day
    │
    ▼
Label Creation (No Leakage)
    │
    ├─▶ Get entry price (bar N+1)
    ├─▶ Calculate future high/low
    ├─▶ Check ATR-based targets
    ├─▶ Assign labels (Long/Short/Neutral)
    │
    ▼
Train/Test Split (Time-Series)
    │
    ├─▶ 80% train, 20% test
    └─▶ No shuffling (preserve time order)
    │
    ▼
Model Training
    │
    ├─▶ Decision Tree Classifier
    │   └─▶ Balanced class weights
    │
    └─▶ Neural Network (PyTorch)
        └─▶ Feature scaling
    │
    ▼
Evaluation
    │
    ├─▶ Classification report
    ├─▶ Feature importance
    └─▶ Cumulative returns comparison
    │
    ▼
Results and Visualizations
```

## Design Patterns

### 1. Context Manager Pattern

Used for automatic resource management:

```python
with HistoricalDataFetcher(host="127.0.0.1", port=4002) as fetcher:
    df = fetcher.get_historical_data(...)
# Connection automatically closed
```

**Benefits:**
- Automatic connection/disconnection
- Exception-safe resource cleanup
- Clean, readable code

### 2. Request Tracking Pattern

Multi-request support with request IDs:

```python
class IBapi:
    def __init__(self):
        self.requests: Dict[int, Request] = {}
    
    def historicalData(self, reqId: int, bar):
        self.requests[reqId].data.append([...])
    
    def historicalDataEnd(self, reqId: int, start, end):
        self.requests[reqId].ready.set()
```

**Benefits:**
- Handle multiple concurrent requests
- Thread-safe data accumulation
- Clean separation of request data

### 3. Dataclass Pattern

Type-safe data structures:

```python
@dataclass
class Request:
    data: list = field(default_factory=list)
    ready: threading.Event = field(default_factory=threading.Event)
```

**Benefits:**
- Automatic `__init__`, `__repr__`, etc.
- Type hints for IDE support
- Immutable with `frozen=True` option

### 4. Pydantic Models Pattern

Validated data models:

```python
class ContractSpec(BaseModel):
    symbol: str = Field(..., description="Stock ticker")
    sec_type: SecurityType = Field(default=SecurityType.STOCK)
    exchange: Exchange = Field(default=Exchange.SMART)
    currency: Currency = Field(default=Currency.USD)
```

**Benefits:**
- Automatic validation
- Type coercion
- JSON serialization
- Clear documentation

### 5. Functional Pipeline Pattern

Composable data transformations:

```python
df = (
    pl.read_parquet("data.parquet")
    .pipe(add_date_int_column)
    .pipe(calculate_bollinger_bands, window=20, stds=2.0)
    .pipe(prepare_ml_dataset, bar_count=10)
)
```

**Benefits:**
- Readable data transformations
- Easy to test individual steps
- Composable and reusable

## Technology Stack

### Core Technologies

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **Python 3.12+** | Language | Modern features, type hints, performance |
| **Polars** | DataFrames | 10-100x faster than pandas, better memory usage |
| **IBapi** | IB Connection | Official Interactive Brokers API |
| **Pydantic** | Validation | Type-safe models, automatic validation |
| **PyTorch** | Deep Learning | GPU support (MPS), flexible architecture |
| **scikit-learn** | ML Models | Decision Trees, preprocessing, metrics |
| **Plotly** | Visualization | Interactive charts, professional quality |
| **Loguru** | Logging | Simple, powerful, colored output |

### Development Tools

| Tool | Purpose |
|------|---------|
| **UV** | Package management (fast, modern) |
| **Ruff** | Linting and formatting (fast, comprehensive) |
| **Jupyter Lab** | Interactive analysis |
| **Git** | Version control |

## Performance Considerations

### 1. Data Processing

**Polars vs Pandas:**
- Polars uses Apache Arrow for memory efficiency
- Lazy evaluation for query optimization
- Parallel processing by default
- 10-100x faster for large datasets

```python
# Lazy evaluation example
df = (
    pl.scan_parquet("large_file.parquet")  # Lazy
    .filter(pl.col("date") > 20240101)
    .select(["DateTime", "Close"])
    .collect()  # Execute
)
```

### 2. IB API Rate Limits

**Considerations:**
- Historical data: ~60 requests per 10 minutes
- Real-time data: Different limits
- Use appropriate timeouts
- Batch requests when possible

**Best Practices:**
```python
# Use longer durations with appropriate bar sizes
df = fetcher.get_historical_data(
    contract=contract,
    duration=timedelta(days=365),  # One year
    frequency=BarFrequency.ONE_HOUR,  # Not 1-minute
    timeout=timedelta(minutes=5)
)
```

### 3. Memory Management

**Strategies:**
- Use Parquet for storage (compressed, columnar)
- Stream large datasets with lazy evaluation
- Process data in chunks when needed
- Clear unused DataFrames

```python
# Efficient data storage
df.write_parquet(
    "data.parquet",
    compression="zstd",  # Good compression ratio
    statistics=True  # Enable predicate pushdown
)
```

### 4. ML Model Performance

**Optimization:**
- Use GPU (MPS on Apple Silicon) for PyTorch
- Batch processing for neural networks
- Feature scaling for better convergence
- Early stopping to prevent overfitting

```python
# GPU acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TradingNN(input_size, num_classes).to(device)
```

## Scalability

### Current Limitations

1. **Single-threaded data fetching**: One connection at a time
2. **In-memory processing**: All data loaded into RAM
3. **Manual model training**: No automated retraining

### Future Enhancements

1. **Parallel data fetching**: Multiple IB connections
2. **Distributed processing**: Dask or Ray for large datasets
3. **Model serving**: REST API for predictions
4. **Automated pipeline**: Airflow or Prefect for orchestration
5. **Database integration**: PostgreSQL/TimescaleDB for storage

## Security Considerations

### Current Implementation

- **Local connections only**: IB API restricted to localhost
- **Paper trading**: Development uses paper account
- **No credentials in code**: Configuration via environment variables

### Best Practices

1. **Never commit credentials** to version control
2. **Use paper trading** for development and testing
3. **Validate all inputs** with Pydantic models
4. **Log security events** (connections, disconnections)
5. **Review IB API permissions** regularly

## Error Handling

### Strategy

1. **Graceful degradation**: Continue on non-critical errors
2. **Informative logging**: Use loguru for detailed logs
3. **Timeout handling**: All IB requests have timeouts
4. **Validation**: Pydantic models catch invalid data early

### Example

```python
try:
    df = fetcher.get_historical_data(...)
except RuntimeError as e:
    logger.error(f"Data fetch failed: {e}")
    # Fallback to cached data or skip
except ValueError as e:
    logger.warning(f"Invalid parameters: {e}")
    # Use default parameters
```

## Testing Strategy

### Current Approach

- **Manual testing**: Interactive testing with IB paper account
- **Integration testing**: End-to-end data fetching and processing
- **Visual validation**: Charts and reports for ML models

### Future Testing

- **Unit tests**: pytest for individual functions
- **Mock IB API**: Test without live connection
- **Backtesting**: Historical validation of strategies
- **CI/CD**: Automated testing on commits

## Next Steps

- Read [02_data_fetching.md](02_data_fetching.md) for detailed data fetching documentation
- Review [03_algorithms.md](03_algorithms.md) for algorithm details
- Check [04_api_reference.md](04_api_reference.md) for complete API documentation
