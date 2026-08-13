"""Asymmetric directional loss for the day-range predictor: predicting a
narrower/inside range (lower high, higher low than actual) is penalized
less than predicting a wider/outside range (higher high, lower low than
actual) — see this repo's plan doc for the full rationale. Optionally also
penalizes the predicted range WIDTH (high - low) drifting away from the
actual day's range width, independent of direction — added because the
directional term alone doesn't stop the model from converging on a range
that's calibrated in direction but badly mis-sized (e.g. always predicting
a much narrower or much wider band than what actually happens).
"""

import torch
import torch.nn.functional as F
from torch import nn


def _asymmetric_term(
    diff: torch.Tensor, lambda_bad: float, lambda_good: float, p: int
) -> torch.Tensor:
    """diff > 0 is charged at lambda_bad's rate, diff < 0 at lambda_good's
    — the caller controls which sign means "bad" via how `diff` is built.
    """
    return lambda_bad * F.relu(diff).pow(p) + lambda_good * F.relu(-diff).pow(p)


class AsymmetricRangeLoss(nn.Module):
    """pred, target: [batch, 2] = [high_ret, low_ret] (returns relative to
    the same reference price). lambda_bad > lambda_good makes the "wider
    than actual" direction cost more than the "narrower than actual"
    direction, for both the high and the low independently.

    range_weight, if > 0, adds a THIRD term: |pred_range - true_range|^p,
    where pred_range = pred_high - pred_low and true_range = true_high -
    true_low (both still in return-% terms) — a plain, non-directional
    penalty on the predicted range's own width being too far from the
    actual day's realized width, on top of the directional high/low terms.
    """

    def __init__(
        self,
        lambda_bad: float = 2.0,
        lambda_good: float = 1.0,
        p: int = 1,
        range_weight: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lambda_bad = lambda_bad
        self.lambda_good = lambda_good
        self.p = p
        self.range_weight = range_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_h = pred[:, 0] - target[:, 0]  # >0: predicted_high > true_high (WIDE/bad)
        diff_l = target[:, 1] - pred[:, 1]  # >0: predicted_low < true_low (WIDE/bad)
        loss_high = _asymmetric_term(diff_h, self.lambda_bad, self.lambda_good, self.p)
        loss_low = _asymmetric_term(diff_l, self.lambda_bad, self.lambda_good, self.p)
        total = loss_high + loss_low
        if self.range_weight:
            pred_range = pred[:, 0] - pred[:, 1]
            true_range = target[:, 0] - target[:, 1]
            range_term = (pred_range - true_range).abs().pow(self.p)
            total = total + self.range_weight * range_term
        return total.mean() if self.reduction == "mean" else total.sum()


class SimpleRangeLoss(nn.Module):
    """Plain (non-directional) L2 error on the high and low, plus an L1
    penalty on the predicted range's width — a simpler alternative to
    AsymmetricRangeLoss with no directional tilt at all: over- and
    under-prediction cost exactly the same on both the high and the low.
    pred, target: [batch, 2] = [high_ret, low_ret].
    """

    def __init__(self, range_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.range_weight = range_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_high = (pred[:, 0] - target[:, 0]).pow(2)
        loss_low = (pred[:, 1] - target[:, 1]).pow(2)
        total = loss_high + loss_low
        if self.range_weight:
            pred_range = pred[:, 0] - pred[:, 1]
            true_range = target[:, 0] - target[:, 1]
            range_term = (pred_range - true_range).abs()
            total = total + self.range_weight * range_term
        return total.mean() if self.reduction == "mean" else total.sum()


def asymmetric_loss_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_bad: float,
    lambda_good: float,
    p: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same math as AsymmetricRangeLoss.forward, but returns
    (loss_high.mean(), loss_low.mean()) separately — used by metrics.py for
    reporting, not for training.
    """
    diff_h = pred[:, 0] - target[:, 0]
    diff_l = target[:, 1] - pred[:, 1]
    loss_high = _asymmetric_term(diff_h, lambda_bad, lambda_good, p).mean()
    loss_low = _asymmetric_term(diff_l, lambda_bad, lambda_good, p).mean()
    return loss_high, loss_low
