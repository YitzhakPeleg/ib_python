import pytest
import polars as pl

from src.algo.indicators import BollingerBands, Indicator, MovingAverage

N = 50


@pytest.fixture
def ohlcv():
    close = [float(i) for i in range(1, N + 1)]
    return pl.DataFrame({
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": [1000] * N,
    })


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
