"""Evaluation for the day-range predictor: the asymmetric loss value, plain
MAE (return-% and $), and the inside/outside/mixed fraction that directly
measures whether the asymmetric objective is doing its job — plus a trivial
ATR-extension heuristic baseline for context, using the SAME metric set so
the two are directly comparable.
"""

import numpy as np
import polars as pl
import torch

from ml.dataset import DayRangeDataset


def _summarize(
    pred_high: np.ndarray,
    pred_low: np.ndarray,
    true_high: np.ndarray,
    true_low: np.ndarray,
    prev_close: np.ndarray,
    lambda_bad: float,
    lambda_good: float,
    p: int,
    range_weight: float = 0.0,
    loss_type: str = "asymmetric",
) -> dict:
    """All price arrays in raw $ terms; prev_close is each example's own
    reference price used to express things as returns for the loss.

    loss_type selects which of the two src/ml/losses.py loss functions the
    reported "loss" value mirrors: "asymmetric" (AsymmetricRangeLoss — a
    lambda_bad/lambda_good-tilted L1 or L2 on high/low, per p) or "simple"
    (SimpleRangeLoss — plain, non-directional L2 on high/low). Both add
    range_weight * |pred_range - true_range| (simple's range term is
    always L1, not raised to p — matching SimpleRangeLoss's own definition).
    """
    pred_high_ret = pred_high / prev_close - 1
    pred_low_ret = pred_low / prev_close - 1
    true_high_ret = true_high / prev_close - 1
    true_low_ret = true_low / prev_close - 1

    diff_h = pred_high_ret - true_high_ret  # >0: predicted higher than actual (bad)
    diff_l = true_low_ret - pred_low_ret  # >0: predicted lower than actual (bad)

    if loss_type == "simple":
        directional_loss = diff_h**2 + diff_l**2
    else:

        def term(d: np.ndarray) -> np.ndarray:
            return (
                lambda_bad * np.maximum(d, 0) ** p
                + lambda_good * np.maximum(-d, 0) ** p
            )

        directional_loss = term(diff_h) + term(diff_l)

    pred_range_ret = pred_high_ret - pred_low_ret
    true_range_ret = true_high_ret - true_low_ret
    range_diff_ret = pred_range_ret - true_range_ret
    range_term = (
        np.abs(range_diff_ret) if loss_type == "simple" else np.abs(range_diff_ret) ** p
    )

    loss = float(np.mean(directional_loss + range_weight * range_term))

    pred_range_dollar = pred_high - pred_low
    true_range_dollar = true_high - true_low

    inside = (pred_high <= true_high) & (pred_low >= true_low)
    outside = (pred_high > true_high) & (pred_low < true_low)
    mixed = ~(inside | outside)
    n = len(pred_high)

    return {
        "n": n,
        "loss": loss,
        "mae_high_pct": float(np.mean(np.abs(pred_high_ret - true_high_ret))),
        "mae_low_pct": float(np.mean(np.abs(pred_low_ret - true_low_ret))),
        "mae_high_dollar": float(np.mean(np.abs(pred_high - true_high))),
        "mae_low_dollar": float(np.mean(np.abs(pred_low - true_low))),
        "range_mae_pct": float(np.mean(np.abs(range_diff_ret))),
        "range_mae_dollar": float(
            np.mean(np.abs(pred_range_dollar - true_range_dollar))
        ),
        "inside_frac": float(inside.sum() / n),
        "outside_frac": float(outside.sum() / n),
        "mixed_frac": float(mixed.sum() / n),
    }


def evaluate(
    model: torch.nn.Module,
    ds: DayRangeDataset,
    device: torch.device,
    lambda_bad: float = 2.0,
    lambda_good: float = 1.0,
    p: int = 1,
    range_weight: float = 0.0,
    loss_type: str = "asymmetric",
) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(ds.daily.to(device), ds.intraday.to(device)).cpu().numpy()

    prev_close = ds.meta["prev_close"].to_numpy()
    pred_high = (pred[:, 0] + 1) * prev_close
    pred_low = (pred[:, 1] + 1) * prev_close
    true_high = ds.meta["true_high"].to_numpy()
    true_low = ds.meta["true_low"].to_numpy()
    return _summarize(
        pred_high,
        pred_low,
        true_high,
        true_low,
        prev_close,
        lambda_bad,
        lambda_good,
        p,
        range_weight,
        loss_type,
    )


def heuristic_baseline(
    ds: DayRangeDataset,
    atr_multiple: float = 0.25,
    lambda_bad: float = 2.0,
    lambda_good: float = 1.0,
    p: int = 1,
    range_weight: float = 0.0,
    loss_type: str = "asymmetric",
) -> dict:
    """pred_high = input_window_high + atr_multiple * atr_prior, pred_low =
    input_window_low - atr_multiple * atr_prior — a trivial, non-learned
    reference point.
    """
    prev_close = ds.meta["prev_close"].to_numpy()
    input_high = ds.meta["input_window_high"].to_numpy()
    input_low = ds.meta["input_window_low"].to_numpy()
    atr_prior = ds.meta["atr_prior"].to_numpy()
    pred_high = input_high + atr_multiple * atr_prior
    pred_low = input_low - atr_multiple * atr_prior
    true_high = ds.meta["true_high"].to_numpy()
    true_low = ds.meta["true_low"].to_numpy()
    return _summarize(
        pred_high,
        pred_low,
        true_high,
        true_low,
        prev_close,
        lambda_bad,
        lambda_good,
        p,
        range_weight,
        loss_type,
    )


def metrics_to_frame(name_to_metrics: dict[str, dict]) -> pl.DataFrame:
    """name_to_metrics: {"NN": {...}, "heuristic": {...}} -> a wide comparison table."""
    rows = [{"config": name, **metrics} for name, metrics in name_to_metrics.items()]
    return pl.DataFrame(rows)
