from abc import ABC, abstractmethod

import polars as pl


class Indicator(ABC):
    @abstractmethod
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return df with indicator columns appended via with_columns."""
        ...
