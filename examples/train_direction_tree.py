"""Interpretable direction + TP model for the opening-range breakout,
trained on SPY.

Unlike both the fixed-ATR rule and the day-range GRU's TP-only role, this
model makes the trading decision itself, BEFORE any breakout trigger:

1. DIRECTION (long / short / no_trade): a shallow decision tree (and a
   random forest for comparison), from ~9 hand-engineered features covering
   the prior 14 days (momentum, ATR, volume) and the opening range's own 30
   minutes (width, direction, relative volume) -- deliberately small and
   tabular, unlike the GRU's raw 100-feature sequence, because trees need
   engineered signal rather than raw sequences to avoid overfitting on a
   few hundred training days.
2. TP, as a predicted R-multiple (a decision-tree regressor) of the
   opening range's own width -- how far price is likely to run in the
   chosen direction before either reversing or hitting the opposite edge.

SL is NOT learned: it's mechanically the opposite edge of the opening
range (or_low for a long, or_high for a short), matching the convention
already used throughout algo.range_breakout.

Ground truth, honestly built: for every day, walk the ACTUAL remainder-of-
session bars (5m, after the same 30-minute opening window the model itself
uses) bar-by-bar in both directions, tracking the max favorable excursion
(in R-multiples of the opening range's width) reached BEFORE the opposite
edge would have stopped the trade out -- see _max_favorable_r. This is a
real, first-touch-order-respecting simulation, not a shortcut derived from
session-level high/low alone (which can't tell you whether the stop would
have been hit first).

Execution-style caveat: unlike the trigger-scan strategy in
algo.range_breakout (which waits for a bar to CLOSE beyond the range),
this model commits to entering AT or_high/or_low the instant the opening
window closes -- more responsive, but also more aggressive; it is not
"the same strategy with smarter direction," it is a materially different
entry rule, and the two shouldn't be over-compared on trade count alone.
"""

import numpy as np
import polars as pl
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from algo.hammer_reversal import compute_daily_atr
from algo.resample_bars import resample_to_timeframe
from models.paths import get_file
from range_breakout_realistic_pnl import simulate as pnl_simulate

OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
TICKER = "SPY"
DATA_LABEL = "SPY_full"
DAILY_LOOKBACK = 14
OPENING_BARS = (
    6  # 6 x 5m = 30 min, matching the breakout strategy's own opening_range_minutes
)
MIN_R_FOR_TRADE = (
    1.0  # must reach at least the mechanical 1:1 target before being stopped
)
TRAIN_YEARS = set(range(2018, 2025))
VAL_YEARS = {2025}

FEATURE_NAMES = [
    "ret_1d",
    "ret_5d",
    "atr_prior_pct",
    "vol_ratio",
    "gap_pct",
    "or_width_pct",
    "or_width_over_atr",
    "opening_direction",
    "relative_volume",
]


def _max_favorable_r(
    high: np.ndarray,
    low: np.ndarray,
    entry: float,
    sl: float,
    direction: int,
    risk: float,
) -> float:
    """Max favorable excursion, in R-multiples of `risk`, reached before
    the opposite edge (`sl`) would have stopped the trade out. Walks the
    remainder-of-session bars in entry order, so it respects which level
    was actually touched first -- unlike using the session's high/low
    alone, which can't distinguish "ran to +2R then pulled back" from
    "stopped out at -1R before ever running."
    """
    mfe = 0.0
    for h, lo in zip(high, low, strict=True):
        if direction == 1:
            if lo <= sl:
                break
            mfe = max(mfe, h - entry)
        else:
            if h >= sl:
                break
            mfe = max(mfe, entry - lo)
    return mfe / risk


def build_examples(minute_bars: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Returns (examples_df, remainder_bars_by_date) -- the latter holds
    each day's own remainder-of-session High/Low/Close arrays, needed
    later to walk REAL trades (not just the mfe_r summary) for the
    realistic-P&L integration test.
    """
    daily_df = resample_to_timeframe(minute_bars, "1d")
    daily_df = compute_daily_atr(daily_df)
    intraday_df = resample_to_timeframe(minute_bars, "5m").with_columns(
        pl.col("DateTime").dt.date().alias("date")
    )

    trading_dates = daily_df["date"].to_list()
    closes = daily_df["Close"].to_numpy().astype(np.float64)
    volumes = daily_df["Volume"].to_numpy().astype(np.float64)
    atr_prior_arr = daily_df["atr_prior"].to_numpy().astype(np.float64)

    intraday_by_date = {
        d: g for (d,), g in intraday_df.group_by("date", maintain_order=True)
    }

    rows = []
    remainder_bars_by_date = {}
    for t in range(DAILY_LOOKBACK + 2, len(trading_dates)):
        date = trading_dates[t]
        if date not in intraday_by_date:
            continue
        day_bars = intraday_by_date[date]
        if day_bars.height <= OPENING_BARS:
            continue
        prev_close = closes[t - 1]
        atr_prior = atr_prior_arr[t]
        if np.isnan(prev_close) or np.isnan(atr_prior) or atr_prior <= 0:
            continue

        vol_window = volumes[t - DAILY_LOOKBACK - 1 : t - 1]
        if len(vol_window) < DAILY_LOOKBACK or np.isnan(vol_window).any():
            continue
        vol_mean = np.mean(vol_window)

        ret_1d = closes[t - 1] / closes[t - 2] - 1
        ret_5d = closes[t - 1] / closes[t - 6] - 1
        vol_ratio = volumes[t - 1] / vol_mean
        atr_prior_pct = atr_prior / prev_close

        opening = day_bars.head(OPENING_BARS)
        remainder = day_bars.slice(OPENING_BARS, day_bars.height - OPENING_BARS)
        if remainder.height == 0:
            continue

        or_high = float(opening["High"].max())
        or_low = float(opening["Low"].min())
        or_open = float(opening["Open"][0])
        or_close = float(opening["Close"][-1])
        or_width = or_high - or_low
        if or_width <= 0:
            continue

        gap_pct = (or_open - prev_close) / prev_close
        or_width_pct = or_width / prev_close
        or_width_over_atr = or_width / atr_prior
        opening_direction = 1.0 if or_close > or_open else -1.0
        opening_volume = float(opening["Volume"].sum())
        relative_volume = opening_volume / vol_mean

        high = remainder["High"].to_numpy().astype(np.float64)
        low = remainder["Low"].to_numpy().astype(np.float64)
        close_arr = remainder["Close"].to_numpy().astype(np.float64)

        mfe_long = _max_favorable_r(high, low, or_high, or_low, 1, or_width)
        mfe_short = _max_favorable_r(high, low, or_low, or_high, -1, or_width)

        if mfe_long >= MIN_R_FOR_TRADE and mfe_long >= mfe_short:
            direction = "long"
        elif mfe_short >= MIN_R_FOR_TRADE and mfe_short > mfe_long:
            direction = "short"
        else:
            direction = "no_trade"

        rows.append(
            {
                "date": date,
                "ret_1d": ret_1d,
                "ret_5d": ret_5d,
                "atr_prior_pct": atr_prior_pct,
                "vol_ratio": vol_ratio,
                "gap_pct": gap_pct,
                "or_width_pct": or_width_pct,
                "or_width_over_atr": or_width_over_atr,
                "opening_direction": opening_direction,
                "relative_volume": relative_volume,
                "direction": direction,
                "mfe_long": mfe_long,
                "mfe_short": mfe_short,
                "or_high": or_high,
                "or_low": or_low,
                "or_width": or_width,
                "prev_close": prev_close,
            }
        )
        remainder_bars_by_date[date] = (high, low, close_arr)

    return pl.DataFrame(rows), remainder_bars_by_date


def main() -> None:
    minute_bars = (
        pl.read_parquet(get_file(DATA_LABEL, "1_min")).select(OHLCV).sort("DateTime")
    )
    examples, remainder_bars_by_date = build_examples(minute_bars)
    examples = examples.with_columns(
        pl.col("date")
        .map_elements(lambda d: d.year, return_dtype=pl.Int64)
        .alias("year")
    )
    logger.info(f"{examples.height} labeled days total")
    logger.info(examples["direction"].value_counts())

    train = examples.filter(pl.col("year").is_in(TRAIN_YEARS))
    val = examples.filter(pl.col("year").is_in(VAL_YEARS))
    logger.info(f"train={train.height} days, val={val.height} days")

    x_train = train.select(FEATURE_NAMES).to_numpy()
    y_train = train["direction"].to_numpy()
    x_val = val.select(FEATURE_NAMES).to_numpy()
    y_val = val["direction"].to_numpy()

    # --- direction classifier: shallow tree (interpretable) + forest (comparison) ---
    tree_clf = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=42
    )
    tree_clf.fit(x_train, y_train)
    tree_pred = tree_clf.predict(x_val)

    forest_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    forest_clf.fit(x_train, y_train)
    forest_pred = forest_clf.predict(x_val)

    print("\n=== Decision tree structure (direction) ===")
    print(export_text(tree_clf, feature_names=FEATURE_NAMES))

    print("=== Decision tree: val classification report (SPY 2025) ===")
    print(classification_report(y_val, tree_pred, zero_division=0))
    print(
        "confusion matrix (rows=true, cols=pred), labels:",
        sorted(set(y_val) | set(tree_pred)),
    )
    print(
        confusion_matrix(y_val, tree_pred, labels=sorted(set(y_val) | set(tree_pred)))
    )

    print("\n=== Random forest: val classification report (SPY 2025) ===")
    print(classification_report(y_val, forest_pred, zero_division=0))
    print("feature importances:")
    for name, imp in sorted(
        zip(FEATURE_NAMES, forest_clf.feature_importances_, strict=True),
        key=lambda x: -x[1],
    ):
        print(f"  {name:20s} {imp:.3f}")

    # --- TP regressor: predicted R-multiple, trained only on real trade days ---
    train_traded = train.filter(pl.col("direction") != "no_trade")
    x_train_r = train_traded.select(FEATURE_NAMES).to_numpy()
    y_train_r = np.where(
        train_traded["direction"].to_numpy() == "long",
        train_traded["mfe_long"].to_numpy(),
        train_traded["mfe_short"].to_numpy(),
    )
    tp_reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=15, random_state=42)
    tp_reg.fit(x_train_r, y_train_r)

    val_traded = val.filter(pl.col("direction") != "no_trade")
    if val_traded.height:
        x_val_r = val_traded.select(FEATURE_NAMES).to_numpy()
        y_val_r = np.where(
            val_traded["direction"].to_numpy() == "long",
            val_traded["mfe_long"].to_numpy(),
            val_traded["mfe_short"].to_numpy(),
        )
        pred_r = tp_reg.predict(x_val_r)
        mae_r = float(np.mean(np.abs(pred_r - y_val_r)))
        print(
            f"\n=== TP regressor: MAE on true-direction val days = {mae_r:.3f}R "
            f"(mean actual R = {y_val_r.mean():.3f}) ==="
        )

    # --- realistic $ P&L using the FULL pipeline: predicted direction (tree) + predicted TP (regressor) ---
    val_x = val.select(FEATURE_NAMES).to_numpy()
    pred_direction = tree_clf.predict(val_x)
    pred_r_all = tp_reg.predict(val_x)

    trade_rows = []
    for row, direction, pred_r in zip(
        val.iter_rows(named=True), pred_direction, pred_r_all, strict=True
    ):
        if direction == "no_trade":
            continue
        date = row["date"]
        or_high, or_low, or_width = row["or_high"], row["or_low"], row["or_width"]
        high, low, close_arr = remainder_bars_by_date[date]
        if direction == "long":
            entry, sl = or_high, or_low
            tp = entry + max(pred_r, 0.1) * or_width
            for h, lo in zip(high, low, strict=True):
                if lo <= sl:
                    exit_price = sl
                    break
                if h >= tp:
                    exit_price = tp
                    break
            else:
                exit_price = close_arr[-1]
        else:
            entry, sl = or_low, or_high
            tp = entry - max(pred_r, 0.1) * or_width
            for h, lo in zip(high, low, strict=True):
                if h >= sl:
                    exit_price = sl
                    break
                if lo <= tp:
                    exit_price = tp
                    break
            else:
                exit_price = close_arr[-1]
        trade_rows.append(
            {
                "date": date,
                "direction": 1 if direction == "long" else -1,
                "entry_price": entry,
                "sl": sl,
                "exit_price": exit_price,
            }
        )

    trades_df = (
        pl.DataFrame(trade_rows).sort("date")
        if trade_rows
        else pl.DataFrame(
            schema={
                "date": pl.Date,
                "direction": pl.Int64,
                "entry_price": pl.Float64,
                "sl": pl.Float64,
                "exit_price": pl.Float64,
            }
        )
    )
    logger.info(f"Tree-driven: {trades_df.height} trades on {val.height} SPY 2025 days")

    sim = pnl_simulate(trades_df)
    if sim.height == 0:
        print("\nNo trades simulated (position sizing rounded everything to 0 shares).")
        return
    n = sim.height
    gross = float(sim["gross_pnl"].sum())
    commission = float(sim["commission"].sum())
    net = float(sim["net_pnl"].sum())
    win_pct = float((sim["net_pnl"] > 0).sum()) / n * 100
    max_dd_pct = float(sim["drawdown_pct"].max())
    print("\n=== TREE-DRIVEN direction + TP -- realistic $10k P&L, 2025 only ===")
    print(f"  trades: {n}   win rate: {win_pct:.1f}%")
    print(
        f"  gross P&L: ${gross:,.2f}   commission: ${commission:,.2f}   net P&L: ${net:,.2f}"
    )
    print(f"  final equity: ${10_000.0 + net:,.2f}   max drawdown: {max_dd_pct:.1f}%")


if __name__ == "__main__":
    main()
