"""Test script for the plotting module with synthetic data."""

from datetime import datetime, timedelta

import polars as pl
from loguru import logger
from plotting import plot_bars

def create_synthetic_ohlc_data(n_bars: int = 100) -> pl.DataFrame:
    """Create synthetic OHLC data for testing."""
    import random

    random.seed(42)

    # Generate timestamps (1-minute bars)
    start_time = datetime(2026, 4, 17, 9, 30)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_bars)]

    # Generate synthetic price data
    base_price = 150.0
    prices = []
    volumes = []

    current_price = base_price
    for _ in range(n_bars):
        # Random walk
        change = random.uniform(-0.5, 0.5)
        current_price += change

        # Generate OHLC
        open_price = current_price
        high_price = open_price + random.uniform(0, 0.3)
        low_price = open_price - random.uniform(0, 0.3)
        close_price = random.uniform(low_price, high_price)

        prices.append(
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
            }
        )

        # Generate volume
        volumes.append(random.randint(1000, 10000))

        current_price = close_price

    # Create DataFrame
    df = pl.DataFrame(
        {
            "DateTime": timestamps,
            "Open": [p["Open"] for p in prices],
            "High": [p["High"] for p in prices],
            "Low": [p["Low"] for p in prices],
            "Close": [p["Close"] for p in prices],
            "Volume": volumes,
        }
    )

    return df


def add_synthetic_bollinger_bands(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add synthetic Bollinger Bands to the DataFrame."""
    # Simple moving average
    df = df.with_columns(
        [
            pl.col("Close").rolling_mean(window_size=window).alias("bb_mid"),
            pl.col("Close").rolling_std(window_size=window).alias("std"),
        ]
    )

    # Upper and lower bands
    df = df.with_columns(
        [
            (pl.col("bb_mid") + 2 * pl.col("std")).alias("bb_upper"),
            (pl.col("bb_mid") - 2 * pl.col("std")).alias("bb_lower"),
        ]
    ).drop("std")

    return df


def test_basic_chart():
    """Test 1: Basic chart with all features."""
    logger.info("Test 1: Basic chart with Bollinger Bands and Volume")

    df = create_synthetic_ohlc_data(n_bars=100)
    df = add_synthetic_bollinger_bands(df)

    try:
        plot_bars(
            df,
            title="Test Chart - All Features",
            height=800,
            show_fig=False,  # Don't display in test
            return_fig=False,
        )
        logger.success("✓ Test 1 passed: Basic chart created successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Test 1 failed: {e}")
        return False


def test_no_volume():
    """Test 2: Chart without volume subplot."""
    logger.info("Test 2: Chart without volume")

    df = create_synthetic_ohlc_data(n_bars=50)
    df = add_synthetic_bollinger_bands(df)

    try:
        plot_bars(
            df,
            title="Test Chart - No Volume",
            show_volume=False,
            height=600,
            show_fig=False,
            return_fig=False,
        )
        logger.success("✓ Test 2 passed: Chart without volume created successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Test 2 failed: {e}")
        return False


def test_no_bollinger_bands():
    """Test 3: Chart without Bollinger Bands."""
    logger.info("Test 3: Chart without Bollinger Bands")

    df = create_synthetic_ohlc_data(n_bars=50)

    try:
        plot_bars(
            df,
            title="Test Chart - No BB",
            bb_upper_col=None,
            bb_mid_col=None,
            bb_lower_col=None,
            height=700,
            show_fig=False,
            return_fig=False,
        )
        logger.success("✓ Test 3 passed: Chart without BB created successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Test 3 failed: {e}")
        return False


def test_return_figure():
    """Test 4: Return figure object."""
    logger.info("Test 4: Return figure object")

    df = create_synthetic_ohlc_data(n_bars=50)
    df = add_synthetic_bollinger_bands(df)

    try:
        fig = plot_bars(
            df,
            title="Test Chart - Return Figure",
            show_fig=False,
            return_fig=True,
        )

        if fig is None:
            logger.error("✗ Test 4 failed: Figure is None")
            return False

        # Check that figure has traces
        import plotly.graph_objects as go

        if not isinstance(fig, go.Figure):
            logger.error("✗ Test 4 failed: Returned object is not a Figure")
            return False

        # Type narrowing for fig.data
        data = fig.data
        trace_count = len(data)  # type: ignore[arg-type]
        if trace_count == 0:
            logger.error("✗ Test 4 failed: Figure has no traces")
            return False

        logger.success(f"✓ Test 4 passed: Figure returned with {trace_count} traces")
        return True
    except Exception as e:
        logger.error(f"✗ Test 4 failed: {e}")
        return False


def test_light_theme():
    """Test 5: Light theme."""
    logger.info("Test 5: Light theme")

    df = create_synthetic_ohlc_data(n_bars=50)
    df = add_synthetic_bollinger_bands(df)

    try:
        plot_bars(
            df,
            title="Test Chart - Light Theme",
            theme="plotly_white",
            show_fig=False,
            return_fig=False,
        )
        logger.success("✓ Test 5 passed: Light theme chart created successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Test 5 failed: {e}")
        return False


def test_missing_columns():
    """Test 6: Error handling for missing columns."""
    logger.info("Test 6: Error handling for missing columns")

    # Create DataFrame with missing columns
    df = pl.DataFrame(
        {
            "DateTime": [datetime.now()],
            "Open": [150.0],
            # Missing High, Low, Close, Volume
        }
    )

    try:
        plot_bars(df, show_fig=False, return_fig=False)
        logger.error("✗ Test 6 failed: Should have raised ValueError")
        return False
    except ValueError as e:
        logger.success(f"✓ Test 6 passed: Correctly raised ValueError: {e}")
        return True
    except Exception as e:
        logger.error(f"✗ Test 6 failed with unexpected error: {e}")
        return False


def test_empty_dataframe():
    """Test 7: Error handling for empty DataFrame."""
    logger.info("Test 7: Error handling for empty DataFrame")

    df = pl.DataFrame(
        {
            "DateTime": [],
            "Open": [],
            "High": [],
            "Low": [],
            "Close": [],
            "Volume": [],
        }
    )

    try:
        plot_bars(df, show_fig=False, return_fig=False)
        logger.error("✗ Test 7 failed: Should have raised ValueError")
        return False
    except ValueError as e:
        logger.success(f"✓ Test 7 passed: Correctly raised ValueError: {e}")
        return True
    except Exception as e:
        logger.error(f"✗ Test 7 failed with unexpected error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    logger.info("=" * 60)
    logger.info("Running Plotting Module Tests")
    logger.info("=" * 60)

    tests = [
        test_basic_chart,
        test_no_volume,
        test_no_bollinger_bands,
        test_return_figure,
        test_light_theme,
        test_missing_columns,
        test_empty_dataframe,
    ]

    results = []
    for test in tests:
        logger.info("")
        result = test()
        results.append(result)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)
    logger.info(f"Passed: {passed}/{total}")

    if passed == total:
        logger.success("✓ All tests passed!")
    else:
        logger.warning(f"✗ {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)


# Made with Bob
