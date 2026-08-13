"""Day-range predictor — training driver.

Trains a small neural net on one or more tickers' data across
`--train-years` to predict `--val-ticker`'s `--val-years` daily high/low,
given (a) the prior 14 trading days' daily OHLCV and (b) the target day's
own first `--intraday-bars` bars of `--intraday-timeframe` size. Loss is
asymmetric: predicting a narrower/inside range is penalized less than
predicting a wider/outside one (see src/ml/losses.py).

Defaults: train on SPY 2018-2024, validate on SPY 2025. Pass e.g.
`--train-tickers SPY AAPL AMZN` to pool multiple tickers' training data —
each ticker gets its own no-lookahead feature/target extraction (prices are
never mixed across tickers; everything is expressed as returns relative to
that ticker's own prior close before the examples are concatenated).

See src/ml/ for the reusable feature/dataset/model/loss/metric code this
script orchestrates.
"""

import argparse
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import torch
from loguru import logger
from plotly.subplots import make_subplots
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from ml import features, metrics
from ml.dataset import DayRangeDataset, combine_datasets, extract_examples
from ml.losses import AsymmetricRangeLoss, SimpleRangeLoss
from ml.model import build_model
from models.paths import get_file

OHLCV = ["DateTime", "Open", "High", "Low", "Close", "Volume"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the day-range predictor NN")
    p.add_argument("--train-tickers", nargs="+", default=["SPY"])
    p.add_argument("--val-ticker", default="SPY")
    p.add_argument(
        "--train-years", nargs="+", type=int, default=list(range(2018, 2025))
    )
    p.add_argument("--val-years", nargs="+", type=int, default=[2025])
    p.add_argument(
        "--extra-train-years-other-tickers",
        nargs="+",
        type=int,
        default=[2025, 2026],
        help=(
            "Extra years appended to --train-years, but ONLY for tickers other than "
            "--val-ticker (so the validation ticker's own held-out year is never "
            "trained on, while other tickers' otherwise-unused later data still gets "
            "used — a different instrument's 2025 doesn't leak anything about SPY's)."
        ),
    )
    p.add_argument("--intraday-bars", type=int, default=features.INTRADAY_INPUT_BARS)
    p.add_argument("--intraday-timeframe", default=features.INTRADAY_TIMEFRAME)
    p.add_argument("--daily-lookback", type=int, default=features.DAILY_LOOKBACK_DAYS)
    p.add_argument("--encoder", choices=["gru", "mlp", "transformer"], default="gru")
    p.add_argument("--daily-hidden", type=int, default=32)
    p.add_argument("--intraday-hidden", type=int, default=16)
    p.add_argument("--head-hidden", type=int, default=32)
    p.add_argument("--d-model", type=int, default=32, help="transformer encoder only")
    p.add_argument("--nhead", type=int, default=4, help="transformer encoder only")
    p.add_argument("--num-layers", type=int, default=2, help="transformer encoder only")
    p.add_argument(
        "--dim-feedforward", type=int, default=128, help="transformer encoder only"
    )
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument(
        "--loss-type", choices=["asymmetric", "simple"], default="asymmetric"
    )
    p.add_argument("--lambda-bad", type=float, default=2.0)
    p.add_argument("--lambda-good", type=float, default=1.0)
    p.add_argument("--loss-p", type=int, default=1, choices=[1, 2])
    p.add_argument("--range-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="linear LR warmup over this many epochs before holding at --lr; "
        "mainly matters for --encoder transformer, which is prone to stalling "
        "without it, but harmless to leave on for gru/mlp too",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="output/day_range_nn")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_minute_bars(ticker: str) -> pl.DataFrame:
    # SPY's merged 2018-2026 history lives under the "SPY_full" file label
    # (see docs/04_range_breakout_strategy.md) — every other ticker's own
    # 1-min file only covers whatever window IB returned for it.
    file_ticker = f"{ticker}_full" if ticker == "SPY" else ticker
    return (
        pl.read_parquet(get_file(file_ticker, "1_min")).select(OHLCV).sort("DateTime")
    )


def build_dataset(
    ticker: str, years: list[int], args: argparse.Namespace
) -> DayRangeDataset:
    minute_bars = load_minute_bars(ticker)
    daily_df, intraday_df = features.prepare_frames(
        minute_bars, args.intraday_timeframe
    )
    daily_df = features.add_daily_return_features(daily_df)
    return extract_examples(
        daily_df,
        intraday_df,
        years,
        args.daily_lookback,
        args.intraday_bars,
        ticker=ticker,
    )


def plot_training_curves(history: pl.DataFrame, output_path: Path) -> None:
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Asymmetric loss (train vs val)",
            "Validation inside / outside / mixed fraction",
            "Validation MAE (% of reference price)",
        ),
        shared_xaxes=True,
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(x=history["epoch"], y=history["train_loss"], name="train_loss"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=history["epoch"], y=history["val_loss"], name="val_loss"),
        row=1,
        col=1,
    )
    for col, color in [
        ("val_inside_frac", "#00cc44"),
        ("val_outside_frac", "#ff3333"),
        ("val_mixed_frac", "#cccc00"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=history["epoch"], y=history[col], name=col, line=dict(color=color)
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=history["epoch"], y=history["val_mae_high_pct"], name="val_mae_high_pct"
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history["epoch"], y=history["val_mae_low_pct"], name="val_mae_low_pct"
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history["val_range_mae_pct"],
            name="val_range_mae_pct",
            line=dict(dash="dot"),
        ),
        row=3,
        col=1,
    )
    fig.update_xaxes(title_text="epoch", row=3, col=1)
    fig.update_layout(height=900, title="Day-range predictor — training curves")
    fig.write_html(str(output_path))
    logger.info(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")

    other_ticker_years = sorted(
        set(args.train_years) | set(args.extra_train_years_other_tickers)
    )
    train_datasets = [
        build_dataset(
            t,
            args.train_years if t == args.val_ticker else other_ticker_years,
            args,
        )
        for t in args.train_tickers
    ]
    train_ds = (
        combine_datasets(train_datasets)
        if len(train_datasets) > 1
        else train_datasets[0]
    )
    val_ds = build_dataset(args.val_ticker, args.val_years, args)
    logger.info(
        f"{train_ds.daily.shape[0]} train days ({args.train_tickers}, "
        f"{args.val_ticker} restricted to {args.train_years}, others also allowed "
        f"{sorted(set(args.extra_train_years_other_tickers) - set(args.train_years))}), "
        f"{val_ds.daily.shape[0]} val days ({args.val_ticker} x {args.val_years})"
    )
    for name, count in zip(
        args.train_tickers, [ds.daily.shape[0] for ds in train_datasets]
    ):
        logger.info(f"  {name}: {count} train days")

    model = build_model(
        args.encoder,
        daily_lookback=args.daily_lookback,
        intraday_bars=args.intraday_bars,
        daily_hidden=args.daily_hidden,
        intraday_hidden=args.intraday_hidden,
        head_hidden=args.head_hidden,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    if args.loss_type == "simple":
        loss_fn = SimpleRangeLoss(args.range_weight)
    else:
        loss_fn = AsymmetricRangeLoss(
            args.lambda_bad, args.lambda_good, args.loss_p, args.range_weight
        )
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup_epochs = max(args.warmup_epochs, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: min((epoch + 1) / warmup_epochs, 1.0)
    )

    train_loader = DataLoader(
        TensorDataset(train_ds.daily, train_ds.intraday, train_ds.target),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_daily = val_ds.daily.to(device)
    val_intraday = val_ds.intraday.to(device)
    val_target = val_ds.target.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for daily_batch, intraday_batch, target_batch in train_loader:
            daily_batch = daily_batch.to(device)
            intraday_batch = intraday_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            pred = model(daily_batch, intraday_batch)
            loss = loss_fn(pred, target_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / n_batches
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_daily, val_intraday)
            val_loss = loss_fn(val_pred, val_target).item()
        val_metrics = metrics.evaluate(
            model,
            val_ds,
            device,
            args.lambda_bad,
            args.lambda_good,
            args.loss_p,
            args.range_weight,
            args.loss_type,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_inside_frac": val_metrics["inside_frac"],
                "val_outside_frac": val_metrics["outside_frac"],
                "val_mixed_frac": val_metrics["mixed_frac"],
                "val_mae_high_pct": val_metrics["mae_high_pct"],
                "val_mae_low_pct": val_metrics["mae_low_pct"],
                "val_range_mae_pct": val_metrics["range_mae_pct"],
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                f"epoch {epoch:4d}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"state_dict": model.state_dict(), "args": vars(args)}, checkpoint_path
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.6f})"
                )
                break

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(f"Reloaded best checkpoint (val_loss={best_val_loss:.6f})")

    history = pl.DataFrame(history_rows)
    history_path = output_dir / "training_history.csv"
    history.write_csv(history_path)
    plot_training_curves(history, output_dir / "training_curves.html")

    nn_metrics = metrics.evaluate(
        model,
        val_ds,
        device,
        args.lambda_bad,
        args.lambda_good,
        args.loss_p,
        args.range_weight,
        args.loss_type,
    )
    heuristic_metrics = metrics.heuristic_baseline(
        val_ds,
        lambda_bad=args.lambda_bad,
        lambda_good=args.lambda_good,
        p=args.loss_p,
        range_weight=args.range_weight,
        loss_type=args.loss_type,
    )
    comparison = metrics.metrics_to_frame(
        {"NN": nn_metrics, "heuristic": heuristic_metrics}
    )

    pl.Config.set_tbl_width_chars(200)
    val_label = f"{args.val_ticker} {args.val_years}"
    print(f"\n{'=' * 10} Validation ({val_label}) — NN vs heuristic {'=' * 10}")
    print(comparison)

    metrics_path = (
        output_dir
        / f"metrics_val_{args.val_ticker}_{min(args.val_years)}-{max(args.val_years)}.csv"
    )
    comparison.write_csv(metrics_path)

    model.eval()
    with torch.no_grad():
        pred = model(val_daily, val_intraday).cpu().numpy()
    prev_close = val_ds.meta["prev_close"].to_numpy()
    pred_high = (pred[:, 0] + 1) * prev_close
    pred_low = (pred[:, 1] + 1) * prev_close
    predictions = val_ds.meta.with_columns(
        pl.Series("pred_high", pred_high),
        pl.Series("pred_low", pred_low),
    ).with_columns(
        pl.when(
            (pl.col("pred_high") <= pl.col("true_high"))
            & (pl.col("pred_low") >= pl.col("true_low"))
        )
        .then(pl.lit("inside"))
        .when(
            (pl.col("pred_high") > pl.col("true_high"))
            & (pl.col("pred_low") < pl.col("true_low"))
        )
        .then(pl.lit("outside"))
        .otherwise(pl.lit("mixed"))
        .alias("outcome")
    )
    predictions_path = (
        output_dir
        / f"predictions_val_{args.val_ticker}_{min(args.val_years)}-{max(args.val_years)}.csv"
    )
    predictions.write_csv(predictions_path)

    logger.info(
        f"Wrote {checkpoint_path}, {history_path}, {metrics_path}, {predictions_path}"
    )


if __name__ == "__main__":
    main()
