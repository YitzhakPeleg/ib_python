"""Candlestick pattern taxonomy for TA-Lib's 61 `CDL*` pattern-recognition functions.

Groups every pattern by:
- candles: how many bars the pattern's TA-Lib function examines
- family: shape/structure similarity (useful for pooling low-support patterns)
- bias: Bullish / Bearish / Both / Neutral — whether the function's sign is
  fixed, varies by occurrence, or (for Neutral) carries no direction at all
- type: Reversal / Continuation / Indecision
"""

from dataclasses import dataclass
from typing import Literal

import talib

Family = Literal[
    "Doji",
    "Marubozu",
    "Small-body",
    "Hammer-shaped",
    "Inverted-hammer-shaped",
    "Engulfing",
    "Harami",
    "Star",
    "Soldiers/crows",
    "Neck/penetration",
    "Gap-continuation",
    "Gap-reversal crows",
    "Breakaway/ladder",
    "Hikkake",
]
Bias = Literal["Bullish", "Bearish", "Both", "Neutral"]
PatternType = Literal["Reversal", "Continuation", "Indecision"]


@dataclass(frozen=True)
class PatternInfo:
    name: str
    candles: int
    family: Family
    bias: Bias
    type: PatternType


_ROWS: list[PatternInfo] = [
    PatternInfo("CDL2CROWS", 3, "Gap-reversal crows", "Bearish", "Reversal"),
    PatternInfo("CDL3BLACKCROWS", 3, "Soldiers/crows", "Bearish", "Reversal"),
    PatternInfo("CDL3INSIDE", 3, "Harami", "Both", "Reversal"),
    PatternInfo("CDL3LINESTRIKE", 4, "Engulfing", "Both", "Reversal"),
    PatternInfo("CDL3OUTSIDE", 3, "Engulfing", "Both", "Reversal"),
    PatternInfo("CDL3STARSINSOUTH", 3, "Soldiers/crows", "Bullish", "Reversal"),
    PatternInfo("CDL3WHITESOLDIERS", 3, "Soldiers/crows", "Bullish", "Reversal"),
    PatternInfo("CDLABANDONEDBABY", 3, "Star", "Both", "Reversal"),
    PatternInfo("CDLADVANCEBLOCK", 3, "Soldiers/crows", "Bearish", "Reversal"),
    PatternInfo("CDLBELTHOLD", 1, "Marubozu", "Both", "Reversal"),
    PatternInfo("CDLBREAKAWAY", 5, "Breakaway/ladder", "Both", "Reversal"),
    PatternInfo("CDLCLOSINGMARUBOZU", 1, "Marubozu", "Both", "Continuation"),
    PatternInfo("CDLCONCEALBABYSWALL", 4, "Gap-reversal crows", "Bullish", "Reversal"),
    PatternInfo("CDLCOUNTERATTACK", 2, "Neck/penetration", "Both", "Reversal"),
    PatternInfo("CDLDARKCLOUDCOVER", 2, "Neck/penetration", "Bearish", "Reversal"),
    PatternInfo("CDLDOJI", 1, "Doji", "Neutral", "Indecision"),
    PatternInfo("CDLDOJISTAR", 2, "Doji", "Both", "Reversal"),
    PatternInfo("CDLDRAGONFLYDOJI", 1, "Doji", "Bullish", "Reversal"),
    PatternInfo("CDLENGULFING", 2, "Engulfing", "Both", "Reversal"),
    PatternInfo("CDLEVENINGDOJISTAR", 3, "Star", "Bearish", "Reversal"),
    PatternInfo("CDLEVENINGSTAR", 3, "Star", "Bearish", "Reversal"),
    PatternInfo("CDLGAPSIDESIDEWHITE", 3, "Gap-continuation", "Both", "Continuation"),
    PatternInfo("CDLGRAVESTONEDOJI", 1, "Doji", "Bearish", "Reversal"),
    PatternInfo("CDLHAMMER", 1, "Hammer-shaped", "Bullish", "Reversal"),
    PatternInfo("CDLHANGINGMAN", 1, "Hammer-shaped", "Bearish", "Reversal"),
    PatternInfo("CDLHARAMI", 2, "Harami", "Both", "Reversal"),
    PatternInfo("CDLHARAMICROSS", 2, "Harami", "Both", "Reversal"),
    PatternInfo("CDLHIGHWAVE", 1, "Small-body", "Both", "Indecision"),
    PatternInfo("CDLHIKKAKE", 3, "Hikkake", "Both", "Reversal"),
    PatternInfo("CDLHIKKAKEMOD", 3, "Hikkake", "Both", "Reversal"),
    PatternInfo("CDLHOMINGPIGEON", 2, "Harami", "Bullish", "Reversal"),
    PatternInfo("CDLIDENTICAL3CROWS", 3, "Soldiers/crows", "Bearish", "Reversal"),
    PatternInfo("CDLINNECK", 2, "Neck/penetration", "Bearish", "Continuation"),
    PatternInfo("CDLINVERTEDHAMMER", 1, "Inverted-hammer-shaped", "Bullish", "Reversal"),
    PatternInfo("CDLKICKING", 2, "Engulfing", "Both", "Reversal"),
    PatternInfo("CDLKICKINGBYLENGTH", 2, "Engulfing", "Both", "Reversal"),
    PatternInfo("CDLLADDERBOTTOM", 5, "Breakaway/ladder", "Bullish", "Reversal"),
    PatternInfo("CDLLONGLEGGEDDOJI", 1, "Doji", "Neutral", "Indecision"),
    PatternInfo("CDLLONGLINE", 1, "Marubozu", "Both", "Continuation"),
    PatternInfo("CDLMARUBOZU", 1, "Marubozu", "Both", "Continuation"),
    PatternInfo("CDLMATCHINGLOW", 2, "Neck/penetration", "Bullish", "Reversal"),
    PatternInfo("CDLMATHOLD", 5, "Gap-continuation", "Bullish", "Continuation"),
    PatternInfo("CDLMORNINGDOJISTAR", 3, "Star", "Bullish", "Reversal"),
    PatternInfo("CDLMORNINGSTAR", 3, "Star", "Bullish", "Reversal"),
    PatternInfo("CDLONNECK", 2, "Neck/penetration", "Bearish", "Continuation"),
    PatternInfo("CDLPIERCING", 2, "Neck/penetration", "Bullish", "Reversal"),
    PatternInfo("CDLRICKSHAWMAN", 1, "Doji", "Neutral", "Indecision"),
    PatternInfo("CDLRISEFALL3METHODS", 5, "Gap-continuation", "Both", "Continuation"),
    PatternInfo("CDLSEPARATINGLINES", 2, "Gap-continuation", "Both", "Continuation"),
    PatternInfo("CDLSHOOTINGSTAR", 1, "Inverted-hammer-shaped", "Bearish", "Reversal"),
    PatternInfo("CDLSHORTLINE", 1, "Small-body", "Both", "Indecision"),
    PatternInfo("CDLSPINNINGTOP", 1, "Small-body", "Both", "Indecision"),
    PatternInfo("CDLSTALLEDPATTERN", 3, "Soldiers/crows", "Bearish", "Reversal"),
    PatternInfo("CDLSTICKSANDWICH", 3, "Neck/penetration", "Bullish", "Reversal"),
    PatternInfo("CDLTAKURI", 1, "Hammer-shaped", "Bullish", "Reversal"),
    PatternInfo("CDLTASUKIGAP", 3, "Gap-continuation", "Both", "Continuation"),
    PatternInfo("CDLTHRUSTING", 2, "Neck/penetration", "Bearish", "Continuation"),
    PatternInfo("CDLTRISTAR", 3, "Doji", "Both", "Reversal"),
    PatternInfo("CDLUNIQUE3RIVER", 3, "Breakaway/ladder", "Bullish", "Reversal"),
    PatternInfo("CDLUPSIDEGAP2CROWS", 3, "Gap-reversal crows", "Bearish", "Reversal"),
    PatternInfo("CDLXSIDEGAP3METHODS", 5, "Gap-continuation", "Both", "Continuation"),
]

PATTERN_TAXONOMY: dict[str, PatternInfo] = {row.name: row for row in _ROWS}

_talib_names = set(talib.get_function_groups()["Pattern Recognition"])
_taxonomy_names = set(PATTERN_TAXONOMY)
if _talib_names != _taxonomy_names:
    raise RuntimeError(
        "PATTERN_TAXONOMY is out of sync with the installed TA-Lib version: "
        f"missing={_talib_names - _taxonomy_names}, extra={_taxonomy_names - _talib_names}"
    )
