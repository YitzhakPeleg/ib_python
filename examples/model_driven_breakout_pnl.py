"""Does the day-range NN's predicted remaining-session high/low make a
better opening-range-breakout TP than the fixed ATR/7 rule?

The model (src/ml) predicts the REMAINDER of the session's high/low after
exactly the same 30-minute opening window the breakout strategy uses to set
or_high/or_low (see the model_pred docstring in src/algo/range_breakout.py)
-- so instead of a TP that's the same ATR fraction every day regardless of
what actually happened in the first 30 minutes, TP becomes "where this
specific day's history + opening behavior suggests price will reach." A
trade is skipped outright if the model predicts no further room beyond
entry, which doubles as a filter against low-expected-value breakouts --
directly targeting the failure mode range_breakout_realistic_pnl.py found
(fixed commissions eating a thin edge on small expected moves).

Fair-comparison caveat: the model (output/day_range_nn_lossgrid/rw1.0_lg1.0)
trained on SPY 2018-2024 plus other tickers; only 2025 is genuinely
out-of-sample for it. So BOTH the model-driven and the fixed-ATR baseline
are restricted to 2025 here, unlike range_breakout_realistic_pnl.py's own
2018-2025 report -- comparing the model against a baseline that also only
sees 2025 is the only way to avoid an apples-to-oranges (in-sample vs
out-of-sample) comparison.
"""

import polars as pl
import torch
from loguru import logger

from algo.range_breakout import run_breakout_backtest
from algo.resample_bars import resample_to_timeframe
from ml import features
from ml.dataset import extract_examples
from ml.model import build_model
from models.paths import get_file
from range_breakout_realistic_pnl import load_trades as load_baseline_trades_all_years
from range_breakout_realistic_pnl import simulate

OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
TICKER = "SPY"
DATA_LABEL = "SPY_full"
VAL_YEARS = [2025]
MODEL_CHECKPOINT = "output/day_range_nn_lossgrid/rw1.0_lg1.0/model.pt"


def build_prediction_lookup() -> dict[tuple[str, object], tuple[float, float]]:
    """{(ticker, date): (pred_high_dollar, pred_low_dollar)} for every SPY
    2025 session, from the accuracy-tuned default model (rw=1.0, lg=1.0).
    """
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    checkpoint = torch.load(MODEL_CHECKPOINT, weights_only=True)
    args = checkpoint["args"]
    model = build_model(
        args["encoder"],
        daily_lookback=args["daily_lookback"],
        intraday_bars=args["intraday_bars"],
        daily_hidden=args["daily_hidden"],
        intraday_hidden=args["intraday_hidden"],
        head_hidden=args["head_hidden"],
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    minute_bars = (
        pl.read_parquet(get_file(DATA_LABEL, "1_min")).select(OHLCV).sort("DateTime")
    )
    daily_df, intraday_df = features.prepare_frames(
        minute_bars, args["intraday_timeframe"]
    )
    daily_df = features.add_daily_return_features(daily_df)
    val_ds = extract_examples(
        daily_df,
        intraday_df,
        VAL_YEARS,
        args["daily_lookback"],
        args["intraday_bars"],
        ticker=TICKER,
    )

    with torch.no_grad():
        pred = model(val_ds.daily.to(device), val_ds.intraday.to(device)).cpu().numpy()

    prev_close = val_ds.meta["prev_close"].to_numpy()
    pred_high_dollar = (pred[:, 0] + 1) * prev_close
    pred_low_dollar = (pred[:, 1] + 1) * prev_close
    dates = val_ds.meta["date"].to_list()

    lookup = {
        (TICKER, date): (float(ph), float(pl_))
        for date, ph, pl_ in zip(dates, pred_high_dollar, pred_low_dollar, strict=True)
    }
    logger.info(f"Built model-prediction lookup for {len(lookup)} SPY 2025 sessions")
    return lookup


def load_model_driven_trades(pred_lookup: dict) -> pl.DataFrame:
    minute_bars_full = (
        pl.read_parquet(get_file(DATA_LABEL, "1_min"))
        .select(OHLCV)
        .with_columns(pl.lit(TICKER).alias("ticker"))
        .sort("DateTime")
    )
    daily_bars = (
        resample_to_timeframe(minute_bars_full.drop("ticker"), "1d")
        .with_columns(pl.lit(TICKER).alias("ticker"))
        .select(OHLCV + ["ticker"])
    )
    trades = run_breakout_backtest(
        minute_bars_full,
        daily_bars,
        signal_timeframe="1m",
        opening_range_minutes=30,
        signal_window_minutes=210,
        range_atr_low_divisor=None,
        range_atr_high_divisor=2.0,
        sl_atr_divisor=5.0,  # SL unchanged from the baseline -- isolates the TP change
        min_relative_volume=0.08,
        allow_multiple_trades_per_day=True,
        tp_model_pred=pred_lookup,
    )
    return trades.filter(pl.col("date").dt.year() == 2025).sort("trigger_time")


def summarize(label: str, sim: pl.DataFrame) -> None:
    if sim.height == 0:
        print(f"{label}: no trades")
        return
    n = sim.height
    gross = float(sim["gross_pnl"].sum())
    commission = float(sim["commission"].sum())
    net = float(sim["net_pnl"].sum())
    max_dd_pct = float(sim["drawdown_pct"].max())
    win_pct = float((sim["net_pnl"] > 0).sum()) / n * 100
    print(f"{label}")
    print(f"  trades: {n}   win rate: {win_pct:.1f}%")
    print(
        f"  gross P&L: ${gross:,.2f}   commission: ${commission:,.2f}   net P&L: ${net:,.2f}"
    )
    print(f"  final equity: ${10_000.0 + net:,.2f}   max drawdown: {max_dd_pct:.1f}%")
    print(f"  avg shares/trade: {sim['shares'].mean():.1f}")


def main() -> None:
    pred_lookup = build_prediction_lookup()

    model_trades = load_model_driven_trades(pred_lookup)
    baseline_trades = load_baseline_trades_all_years().filter(
        pl.col("date").dt.year() == 2025
    )

    logger.info(
        f"2025 trades: model-driven={model_trades.height}, fixed-ATR baseline={baseline_trades.height}"
    )

    print(
        "\n2025-only comparison, identical cost assumptions ($10k start, 1% risk/trade):\n"
    )
    summarize("FIXED-ATR BASELINE (TP=ATR/7, SL=ATR/5)", simulate(baseline_trades))
    print()
    summarize("MODEL-DRIVEN TP (SL=ATR/5, unchanged)", simulate(model_trades))


if __name__ == "__main__":
    main()
