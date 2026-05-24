"""Tests for daily_browser helper functions."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from src.algo.indicators import BollingerBands, MovingAverage
from src.visualization.daily_browser import _apply_indicators, _get_dates

N = 50


@pytest.fixture
def ohlcv() -> pl.DataFrame:
    """Synthetic 1-min OHLCV DataFrame with a `date` column."""
    closes = [float(i + 1) for i in range(N)]
    base = datetime(2026, 1, 2, 9, 30)
    datetimes = [base + timedelta(minutes=i) for i in range(N)]
    return pl.DataFrame(
        {
            "DateTime": datetimes,
            "Open": closes,
            "High": [c + 0.5 for c in closes],
            "Low": [c - 0.5 for c in closes],
            "Close": closes,
            "Volume": [1000.0] * N,
            "date": [dt.date() for dt in datetimes],
        }
    )


# --- _apply_indicators ---


def test_apply_indicators_bb(ohlcv: pl.DataFrame) -> None:
    df, cols = _apply_indicators(ohlcv, [BollingerBands()])
    assert "BollingerBands" in cols
    assert set(cols["BollingerBands"]) == {"bb_upper", "bb_mid", "bb_lower"}
    assert {"bb_upper", "bb_mid", "bb_lower"}.issubset(df.columns)


def test_apply_indicators_ma(ohlcv: pl.DataFrame) -> None:
    df, cols = _apply_indicators(ohlcv, [MovingAverage(window=10)])
    assert "MovingAverage" in cols
    assert "ma_10" in cols["MovingAverage"]
    assert "ma_10" in df.columns


def test_apply_indicators_multiple(ohlcv: pl.DataFrame) -> None:
    df, cols = _apply_indicators(ohlcv, [BollingerBands(), MovingAverage(window=5)])
    assert set(cols.keys()) == {"BollingerBands", "MovingAverage"}
    bb = set(cols["BollingerBands"])
    ma = set(cols["MovingAverage"])
    assert bb.isdisjoint(ma)
    for col in bb | ma:
        assert col in df.columns


def test_apply_indicators_empty(ohlcv: pl.DataFrame) -> None:
    df, cols = _apply_indicators(ohlcv, [])
    assert cols == {}
    assert df.columns == ohlcv.columns


def test_apply_indicators_preserves_rows(ohlcv: pl.DataFrame) -> None:
    df, _ = _apply_indicators(ohlcv, [BollingerBands(), MovingAverage()])
    assert len(df) == N


# --- _get_dates ---


def test_get_dates_sorted() -> None:
    df = pl.DataFrame(
        {
            "date": [
                date(2026, 1, 3),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 3),
            ]
        }
    )
    result = _get_dates(df)
    assert result == [date(2026, 1, 2), date(2026, 1, 3)]


def test_get_dates_unique() -> None:
    df = pl.DataFrame({"date": [date(2026, 1, 2)] * 10})
    assert _get_dates(df) == [date(2026, 1, 2)]


def test_get_dates_from_ohlcv(ohlcv: pl.DataFrame) -> None:
    result = _get_dates(ohlcv)
    assert len(result) >= 1
    assert result == sorted(result)
