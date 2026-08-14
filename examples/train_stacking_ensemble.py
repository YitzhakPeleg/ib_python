"""Stacked combination of 3 base day-range models (different range_weight /
lambda_good, same seed/architecture) that span the conservative-to-aggressive
spectrum: (rw=0.5, lg=0.5) leans safely inside, (rw=0.5, lg=1.5) leans
tight/bold, (rw=1.0, lg=1.0) is the accuracy-tuned default in between.

Rather than simply averaging the three, a small MLP learns how to combine
their 6 outputs (pred_high/pred_low from each) into a final [high, low] --
letting the combiner weight the "safe" model more on the high side and the
"bold" model more on the low side (or whatever combination the data
supports), instead of forcing an equal 1/3 blend.

Base models never trained on SPY 2025, so all 250 SPY 2025 days are valid
combiner data -- but training and evaluating the combiner on the same days
would just measure overfitting, so those 250 days are split chronologically:
the first ~60% trains the combiner, the last ~40% evaluates it. The simple
average and the single best base model are evaluated on the identical
held-out slice for a fair three-way comparison.
"""

import os

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch import nn

from ml import features, metrics
from ml.dataset import extract_examples
from ml.losses import AsymmetricRangeLoss
from ml.model import build_model
from models.paths import get_file

OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
VAL_TICKER = "SPY"
VAL_YEARS = [2025]
TRAIN_FRAC = 0.6

BASE_CHECKPOINTS = [
    "output/day_range_nn_lossgrid/rw0.5_lg0.5/model.pt",  # conservative / safe
    "output/day_range_nn_lossgrid/rw1.0_lg1.0/model.pt",  # accuracy-tuned default
    "output/day_range_nn_lossgrid/rw0.5_lg1.5/model.pt",  # tight / bold
]
LAMBDA_BAD, LAMBDA_GOOD, LOSS_P, RANGE_WEIGHT = 1.7, 1.0, 1, 1.0


class StackingCombiner(nn.Module):
    """hidden=0 -> a plain linear combination of the base models' 6 outputs
    (effectively a learned weighted blend); hidden>0 -> adds one small ReLU
    layer for a nonlinear combination, at the cost of more capacity to
    overfit on a ~100-150 day training slice.
    """

    def __init__(self, n_base: int, hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(n_base * 2, 2)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_base * 2, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_base_models(device: torch.device) -> list[nn.Module]:
    models = []
    for path in BASE_CHECKPOINTS:
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
        logger.info(
            f"Loaded {path} (rw={args['range_weight']}, lg={args['lambda_good']})"
        )
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


def dollar_metrics(pred_ret: np.ndarray, meta: pl.DataFrame) -> dict:
    prev_close = meta["prev_close"].to_numpy()
    true_high = meta["true_high"].to_numpy()
    true_low = meta["true_low"].to_numpy()
    pred_high = (pred_ret[:, 0] + 1) * prev_close
    pred_low = (pred_ret[:, 1] + 1) * prev_close
    return metrics._summarize(  # noqa: SLF001 -- reuse the shared metric math directly
        pred_high,
        pred_low,
        true_high,
        true_low,
        prev_close,
        lambda_bad=LAMBDA_BAD,
        lambda_good=LAMBDA_GOOD,
        p=LOSS_P,
        range_weight=RANGE_WEIGHT,
    )


def main() -> None:
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    base_models = load_base_models(device)

    first_args = torch.load(BASE_CHECKPOINTS[0], weights_only=True)["args"]
    val_ds = build_val_dataset(first_args)
    n = val_ds.daily.shape[0]
    logger.info(f"{n} SPY 2025 validation days, {len(base_models)} base models")

    with torch.no_grad():
        base_preds = np.stack(
            [
                m(val_ds.daily.to(device), val_ds.intraday.to(device)).cpu().numpy()
                for m in base_models
            ],
            axis=1,
        )  # [N, n_base, 2]
    stack_input = base_preds.reshape(n, -1)  # [N, n_base*2]
    target = val_ds.target.numpy()  # [N, 2]

    order = np.argsort(val_ds.meta["date"].to_list())
    split = int(n * TRAIN_FRAC)
    train_pool_idx, test_idx = order[:split], order[split:]
    inner_split = int(len(train_pool_idx) * 0.8)
    inner_train_idx = train_pool_idx[:inner_split]
    inner_val_idx = train_pool_idx[inner_split:]
    logger.info(
        f"Combiner: {len(inner_train_idx)} inner-train / {len(inner_val_idx)} "
        f"inner-val (for early stopping) / {len(test_idx)} held-out test days"
    )

    def to_tensor(idx, arr):
        return torch.tensor(arr[idx], dtype=torch.float32, device=device)

    x_inner_train, y_inner_train = (
        to_tensor(inner_train_idx, stack_input),
        to_tensor(inner_train_idx, target),
    )
    x_inner_val, y_inner_val = (
        to_tensor(inner_val_idx, stack_input),
        to_tensor(inner_val_idx, target),
    )
    x_test = to_tensor(test_idx, stack_input)

    loss_fn = AsymmetricRangeLoss(
        lambda_bad=LAMBDA_BAD,
        lambda_good=LAMBDA_GOOD,
        p=LOSS_P,
        range_weight=RANGE_WEIGHT,
    )

    def train_combiner(hidden: int, tag: str) -> tuple[np.ndarray, dict]:
        torch.manual_seed(42)
        combiner = StackingCombiner(n_base=len(base_models), hidden=hidden).to(device)
        optimizer = torch.optim.Adam(combiner.parameters(), lr=5e-3, weight_decay=1e-2)
        best_val_loss, best_state, patience, bad_epochs = float("inf"), None, 30, 0
        for epoch in range(1, 301):
            combiner.train()
            optimizer.zero_grad()
            loss = loss_fn(combiner(x_inner_train), y_inner_train)
            loss.backward()
            optimizer.step()

            combiner.eval()
            with torch.no_grad():
                val_loss = loss_fn(combiner(x_inner_val), y_inner_val).item()
            if val_loss < best_val_loss:
                best_val_loss, best_state, bad_epochs = (
                    val_loss,
                    {k: v.clone() for k, v in combiner.state_dict().items()},
                    0,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        combiner.load_state_dict(best_state)
        logger.info(
            f"[{tag}] best inner-val loss={best_val_loss:.6f} (epoch stopped {epoch})"
        )
        combiner.eval()
        with torch.no_grad():
            return combiner(x_test).cpu().numpy(), best_state

    stacked_linear_pred_test, linear_state = train_combiner(hidden=0, tag="linear")
    stacked_mlp_pred_test, mlp_state = train_combiner(hidden=8, tag="mlp-8")

    avg_pred_test = base_preds[test_idx].mean(axis=1)  # simple average, same test slice
    best_single_idx = 1  # rw=1.0, lg=1.0 -- the accuracy-tuned default
    best_single_pred_test = base_preds[test_idx, best_single_idx, :]

    meta_test = val_ds.meta[test_idx.tolist()]
    results = {
        "stacked_linear": dollar_metrics(stacked_linear_pred_test, meta_test),
        "stacked_mlp": dollar_metrics(stacked_mlp_pred_test, meta_test),
        "simple_average": dollar_metrics(avg_pred_test, meta_test),
        "best_single_(1.0,1.0)": dollar_metrics(best_single_pred_test, meta_test),
    }
    for i, cfg in enumerate(["(rw=0.5,lg=0.5)", "(rw=1.0,lg=1.0)", "(rw=0.5,lg=1.5)"]):
        results[f"base_{cfg}"] = dollar_metrics(base_preds[test_idx, i, :], meta_test)

    summary_rows = [
        {
            "model": name,
            "n": r["n"],
            "mae_high_pct": round(r["mae_high_pct"] * 100, 3),
            "mae_low_pct": round(r["mae_low_pct"] * 100, 3),
            "mae_high_dollar": round(r["mae_high_dollar"], 2),
            "mae_low_dollar": round(r["mae_low_dollar"], 2),
            "combined_mae_dollar": round(r["mae_high_dollar"] + r["mae_low_dollar"], 2),
            "inside_pct": round(r["inside_frac"] * 100, 1),
            "outside_pct": round(r["outside_frac"] * 100, 1),
        }
        for name, r in results.items()
    ]
    summary_df = pl.DataFrame(summary_rows)
    out_dir = "output/day_range_nn_stacking"
    os.makedirs(out_dir, exist_ok=True)
    summary_df.write_csv(f"{out_dir}/comparison_held_out_{len(test_idx)}days.csv")
    torch.save(linear_state, f"{out_dir}/combiner_linear.pt")
    torch.save(mlp_state, f"{out_dir}/combiner_mlp.pt")

    # the linear combiner's weights are directly interpretable: how much
    # each base model contributes to the final high / low prediction
    linear_weight = linear_state["net.weight"].cpu().numpy()  # [2, n_base*2]
    logger.info(f"Linear combiner weights (rows=[out_high, out_low]):\n{linear_weight}")

    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(summary_df)
    logger.info(
        f"Wrote {out_dir}/comparison_held_out_{len(test_idx)}days.csv, "
        f"{out_dir}/combiner_linear.pt, {out_dir}/combiner_mlp.pt"
    )


if __name__ == "__main__":
    main()
