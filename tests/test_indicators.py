from datetime import date

import polars as pl
import pytest

from src.algo.indicators import BollingerBands, Indicator, MovingAverage

N = 50


@pytest.fixture
def ohlcv():
    close = [float(i) for i in range(1, N + 1)]
    return pl.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": [1000] * N,
        }
    )


@pytest.fixture
def ohlcv_with_date():
    """Two trading days, 25 bars each."""
    close = [float(i) for i in range(1, N + 1)]
    dates = [date(2026, 1, 2)] * 25 + [date(2026, 1, 3)] * 25
    return pl.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": [1000] * N,
            "date": dates,
        }
    )


# --- ABC enforcement ---


def test_indicator_is_abstract():
    with pytest.raises(TypeError):
        Indicator()


def test_concrete_subclass_requires_call():
    class Bad(Indicator):
        pass

    with pytest.raises(TypeError):
        Bad()


# --- BollingerBands ---


def test_bb_appends_columns(ohlcv):
    result = BollingerBands()(ohlcv)
    assert {"bb_upper", "bb_mid", "bb_lower"}.issubset(result.columns)


def test_bb_preserves_original_columns(ohlcv):
    result = BollingerBands()(ohlcv)
    for col in ohlcv.columns:
        assert col in result.columns


def test_bb_row_count_unchanged(ohlcv):
    result = BollingerBands(window=10)(ohlcv)
    assert result.shape[0] == ohlcv.shape[0]


def test_bb_upper_gt_lower(ohlcv):
    result = BollingerBands(window=10)(ohlcv).drop_nulls()
    assert (result["bb_upper"] >= result["bb_lower"]).all()


def test_bb_mid_between_bands(ohlcv):
    result = BollingerBands(window=10)(ohlcv).drop_nulls()
    assert (result["bb_mid"] <= result["bb_upper"]).all()
    assert (result["bb_mid"] >= result["bb_lower"]).all()


# --- MovingAverage ---


def test_ma_default_col_name(ohlcv):
    result = MovingAverage(window=10)(ohlcv)
    assert "ma_10" in result.columns


def test_ma_custom_col_name(ohlcv):
    result = MovingAverage(window=5, out_col="sma_5")(ohlcv)
    assert "sma_5" in result.columns


def test_ma_row_count_unchanged(ohlcv):
    result = MovingAverage(window=10)(ohlcv)
    assert result.shape[0] == ohlcv.shape[0]


def test_ma_preserves_original_columns(ohlcv):
    result = MovingAverage()(ohlcv)
    for col in ohlcv.columns:
        assert col in result.columns


def test_ma_value_correctness(ohlcv):
    # MA of [1..50] over window=5, row index 4 (0-based): mean(1,2,3,4,5) = 3.0
    result = MovingAverage(window=5)(ohlcv)
    assert result["ma_5"][4] == pytest.approx(3.0)


# --- BollingerBands: min_periods ---


def test_bb_min_periods_no_nulls(ohlcv):
    # With min_periods=1 every row should have a non-null bb_mid
    result = BollingerBands(window=20, min_periods=1)(ohlcv)
    assert result["bb_mid"].null_count() == 0


def test_bb_min_periods_no_nan(ohlcv):
    result = BollingerBands(window=20, min_periods=1)(ohlcv)
    assert result["bb_upper"].is_nan().sum() == 0
    assert result["bb_lower"].is_nan().sum() == 0


# --- BollingerBands: reset_per_day ---


def test_bb_reset_per_day_requires_date_column(ohlcv):
    with pytest.raises(ValueError, match="date"):
        BollingerBands(reset_per_day=True)(ohlcv)


def test_bb_reset_per_day_no_nulls(ohlcv_with_date):
    result = BollingerBands(window=10, reset_per_day=True)(ohlcv_with_date)
    assert result["bb_mid"].null_count() == 0


def test_bb_reset_per_day_resets_at_day_boundary(ohlcv_with_date):
    # Day 2 bar 0 (row 25) should use only day-2 data — bb_mid equals its Close
    result = BollingerBands(window=10, min_periods=1, reset_per_day=True)(ohlcv_with_date)
    # The first bar of day 2 (row index 25) has only 1 sample → bb_mid == Close[25]
    assert result["bb_mid"][25] == pytest.approx(result["Close"][25])


def test_bb_reset_per_day_vs_continuous_differ(ohlcv_with_date):
    # Rolling across days vs. reset per day should produce different bb_mid for early bars of day 2
    continuous = BollingerBands(window=10, min_periods=1)(ohlcv_with_date)
    per_day = BollingerBands(window=10, min_periods=1, reset_per_day=True)(ohlcv_with_date)
    # Row 25 = first bar of day 2: continuous uses bars from day 1, per_day resets
    assert continuous["bb_mid"][25] != pytest.approx(per_day["bb_mid"][25])
