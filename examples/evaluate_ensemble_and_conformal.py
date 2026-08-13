"""Ensemble + conformal prediction evaluation for the day-range predictor.

1. Loads N GRU checkpoints (trained by examples/train_day_range_nn.py with
   different --seed values into output/day_range_nn_ensemble/seed_*/),
   runs all of them on SPY 2025, and reports the ensemble MEAN prediction
   (usually a bit more accurate than any single model — averaging cancels
   out some of each model's own noise) plus the ensemble SPREAD (std
   across models) per day, per side — a per-prediction confidence signal:
   tight agreement across independently-trained models suggests a more
   reliable prediction, wide disagreement suggests less. Validates this by
   checking whether spread actually correlates with realized |error| —
   spread is only a useful signal if it does.

2. Splits SPY 2025 chronologically into a calibration half (first ~125
   days) and a test half (last ~125 days), fits split-conformal intervals
   (src/ml/conformal.py) on the calibration half using the ensemble mean
   prediction, and reports empirical coverage on the test half at two
   confidence levels (80%, 90%) — the real-world check of whether the
   conformal guarantee actually held up here.

   Caveat worth being explicit about: conformal prediction's guarantee
   assumes calibration and test data are exchangeable (roughly: drawn from
   the same distribution, no systematic drift between them). A
   chronological split of ONE calendar year is a weaker setup than an
   i.i.d. shuffle — if SPY's volatility regime shifted between the first
   and second half of 2025, coverage can come out above or below the
   nominal target even though the method is implemented correctly. This
   script reports what actually happened, not a promise that it always
   will.
"""

import glob

import numpy as np
import polars as pl
import torch
from loguru import logger

from ml import conformal, features, metrics
from ml.dataset import extract_examples
from ml.model import build_model
from models.paths import get_file

OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
ENSEMBLE_DIR = "output/day_range_nn_ensemble"
VAL_TICKER = "SPY"
VAL_YEARS = [2025]


def load_ensemble_models(device: torch.device) -> list[torch.nn.Module]:
    checkpoint_paths = sorted(glob.glob(f"{ENSEMBLE_DIR}/seed_*/model.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {ENSEMBLE_DIR}/seed_*/")
    models = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, weights_only=True)
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
        models.append(model)
        logger.info(f"Loaded {path}")
    return models


def build_val_dataset(args: dict):
    file_ticker = f"{VAL_TICKER}_full" if VAL_TICKER == "SPY" else VAL_TICKER
    minute_bars = (
        pl.read_parquet(get_file(file_ticker, "1_min")).select(OHLCV).sort("DateTime")
    )
    daily_df, intraday_df = features.prepare_frames(
        minute_bars, args["intraday_timeframe"]
    )
    daily_df = features.add_daily_return_features(daily_df)
    return extract_examples(
        daily_df,
        intraday_df,
        VAL_YEARS,
        args["daily_lookback"],
        args["intraday_bars"],
        ticker=VAL_TICKER,
    )


def main() -> None:
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    models = load_ensemble_models(device)

    first_checkpoint = torch.load(
        sorted(glob.glob(f"{ENSEMBLE_DIR}/seed_*/model.pt"))[0], weights_only=True
    )
    args = first_checkpoint["args"]
    val_ds = build_val_dataset(args)
    logger.info(
        f"{val_ds.daily.shape[0]} SPY 2025 validation days, {len(models)} ensemble members"
    )

    with torch.no_grad():
        all_preds = np.stack(
            [
                m(val_ds.daily.to(device), val_ds.intraday.to(device)).cpu().numpy()
                for m in models
            ]
        )  # [n_models, n_days, 2]

    mean_pred = all_preds.mean(axis=0)  # [n_days, 2]
    spread = all_preds.std(
        axis=0
    )  # [n_days, 2] -- per-day, per-side ensemble disagreement

    prev_close = val_ds.meta["prev_close"].to_numpy()
    true_high = val_ds.meta["true_high"].to_numpy()
    true_low = val_ds.meta["true_low"].to_numpy()
    pred_high = (mean_pred[:, 0] + 1) * prev_close
    pred_low = (mean_pred[:, 1] + 1) * prev_close

    ensemble_metrics = metrics._summarize(  # noqa: SLF001 -- reusing the shared metric math directly
        pred_high,
        pred_low,
        true_high,
        true_low,
        prev_close,
        lambda_bad=args["lambda_bad"],
        lambda_good=args["lambda_good"],
        p=args["loss_p"],
        range_weight=args["range_weight"],
    )
    logger.info(f"Ensemble-mean metrics: {ensemble_metrics}")

    # --- spread vs. realized error correlation (is spread an informative signal?) ---
    abs_err_high = np.abs(pred_high - true_high)
    abs_err_low = np.abs(pred_low - true_low)
    spread_high_dollar = spread[:, 0] * prev_close  # de-normalize spread to $ terms
    spread_low_dollar = spread[:, 1] * prev_close
    corr_high = float(np.corrcoef(spread_high_dollar, abs_err_high)[0, 1])
    corr_low = float(np.corrcoef(spread_low_dollar, abs_err_low)[0, 1])
    logger.info(
        f"Corr(ensemble spread, |error|): high={corr_high:.3f}  low={corr_low:.3f}"
    )

    # --- chronological calibration/test split for conformal prediction ---
    dates = val_ds.meta["date"].to_list()
    order = np.argsort(dates)
    n = len(order)
    split = n // 2
    calib_idx, test_idx = order[:split], order[split:]

    results_rows = []
    for alpha, label in [(0.2, "80%"), (0.1, "90%")]:
        cal_high = conformal.calibrate(
            pred_high[calib_idx], true_high[calib_idx], alpha
        )
        cal_low = conformal.calibrate(pred_low[calib_idx], true_low[calib_idx], alpha)
        cov_high = conformal.empirical_coverage(
            pred_high[test_idx], true_high[test_idx], cal_high
        )
        cov_low = conformal.empirical_coverage(
            pred_low[test_idx], true_low[test_idx], cal_low
        )
        lo_h, hi_h = cal_high.interval(pred_high[test_idx])
        lo_l, hi_l = cal_low.interval(pred_low[test_idx])
        avg_width_high = float(np.mean(hi_h - lo_h))
        avg_width_low = float(np.mean(hi_l - lo_l))
        logger.info(
            f"[{label} target] coverage: high={cov_high:.1%} low={cov_low:.1%}  "
            f"avg width: high=${avg_width_high:.2f} low=${avg_width_low:.2f}  "
            f"(q_lo/q_hi high={cal_high.q_lo:.3f}/{cal_high.q_hi:.3f}, "
            f"low={cal_low.q_lo:.3f}/{cal_low.q_hi:.3f})"
        )
        results_rows.append(
            {
                "target_coverage": label,
                "alpha": alpha,
                "empirical_coverage_high": cov_high,
                "empirical_coverage_low": cov_low,
                "avg_interval_width_high_dollar": avg_width_high,
                "avg_interval_width_low_dollar": avg_width_low,
                "q_lo_high": cal_high.q_lo,
                "q_hi_high": cal_high.q_hi,
                "q_lo_low": cal_low.q_lo,
                "q_hi_low": cal_low.q_hi,
            }
        )

    conformal_df = pl.DataFrame(results_rows)
    conformal_df.write_csv(f"{ENSEMBLE_DIR}/conformal_results.csv")

    predictions_df = val_ds.meta.with_columns(
        pl.Series("pred_high_mean", pred_high),
        pl.Series("pred_low_mean", pred_low),
        pl.Series("spread_high_dollar", spread_high_dollar),
        pl.Series("spread_low_dollar", spread_low_dollar),
    )
    predictions_df.write_csv(f"{ENSEMBLE_DIR}/ensemble_predictions.csv")

    logger.info(
        f"Wrote {ENSEMBLE_DIR}/conformal_results.csv, {ENSEMBLE_DIR}/ensemble_predictions.csv"
    )


if __name__ == "__main__":
    main()
