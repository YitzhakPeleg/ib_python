from pathlib import Path

from src.models.models import BarFrequency

DATA_PATH: Path = Path(__file__).absolute().parent.parent.parent / "data"


def get_file(ticker: str, freqency: str | BarFrequency) -> Path:
    return DATA_PATH / (f"{ticker}_{freqency}.parquet".replace(" ", "_"))


if __name__ == "__main__":
    import polars as pl

    ticker = "AAPL"
    freqency = BarFrequency.ONE_MIN
    file_path = get_file(ticker, freqency)
    print(pl.read_parquet(file_path))
