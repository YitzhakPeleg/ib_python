"""Split-conformal prediction intervals for the day-range predictor.

Wraps a point prediction (pred_high, pred_low) in a calibrated interval
with a formal, distribution-free coverage guarantee: given a calibration
set of (prediction, actual) pairs disjoint from whatever set you're
evaluating coverage on, "1 - alpha" of future intervals built the same way
will contain the true value, AS LONG AS the calibration and test data are
exchangeable (here: both drawn from the same regime — see the caveats in
examples/evaluate_ensemble_and_conformal.py about what this does and
doesn't guarantee when the calibration/test split is a chronological
split of one year rather than an i.i.d. shuffle).

Uses SIGNED residuals (asymmetric interval) rather than the more common
symmetric |residual| version, because we already know this model's error
distribution is skewed (the low prediction in particular has a fat
right-side tail) — forcing a symmetric interval would waste width on the
side that's rarely wrong and under-cover the side that's actually risky.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalCalibration:
    alpha: float  # miscoverage rate, e.g. 0.1 for a 90% interval
    q_lo: float  # low quantile of signed residual (pred - true)
    q_hi: float  # high quantile of signed residual (pred - true)

    def interval(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lower_bound, upper_bound) for each prediction in `pred`,
        such that true is expected to fall inside with probability
        1 - alpha (on exchangeable data).
        """
        return pred - self.q_hi, pred - self.q_lo


def calibrate(pred: np.ndarray, true: np.ndarray, alpha: float) -> ConformalCalibration:
    """pred, true: 1D arrays from a CALIBRATION set (held out from both
    training and whatever set coverage will later be measured on).
    Computes the empirical alpha/2 and 1-alpha/2 quantiles of the signed
    residual (pred - true) — see ConformalCalibration.interval for how
    these turn a future point prediction into an interval.
    """
    residuals = pred - true
    q_lo = float(np.quantile(residuals, alpha / 2))
    q_hi = float(np.quantile(residuals, 1 - alpha / 2))
    return ConformalCalibration(alpha=alpha, q_lo=q_lo, q_hi=q_hi)


def empirical_coverage(
    pred: np.ndarray, true: np.ndarray, calibration: ConformalCalibration
) -> float:
    """Fraction of (pred, true) pairs in a TEST set (disjoint from the set
    `calibration` was built on) where true actually falls inside the
    resulting interval — the number to compare against calibration.alpha's
    target (1 - alpha) to check whether the guarantee held up in practice.
    """
    lower, upper = calibration.interval(pred)
    inside = (true >= lower) & (true <= upper)
    return float(inside.mean())
