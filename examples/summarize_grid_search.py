"""Aggregate output/day_range_nn_grid/b{bars}_rw{rw}_lg{lg}/metrics_val_*.csv
into one sortable table, so the intraday-bars / range-weight / lambda-good
sweep launched from examples/train_day_range_nn.py can be compared at a
glance instead of by opening 36 CSVs by hand.
"""

import re
from pathlib import Path

import polars as pl
from loguru import logger

GRID_DIR = Path("output/day_range_nn_grid")
DIR_PATTERN = re.compile(r"b(?P<bars>\d+)_rw(?P<rw>[\d.]+)_lg(?P<lg>[\d.]+)")


def main() -> None:
    rows = []
    for run_dir in sorted(GRID_DIR.glob("b*_rw*_lg*")):
        match = DIR_PATTERN.match(run_dir.name)
        if not match:
            continue
        metrics_paths = list(run_dir.glob("metrics_val_*.csv"))
        if not metrics_paths:
            logger.warning(
                f"No metrics CSV in {run_dir}, skipping (likely still running)"
            )
            continue
        metrics = pl.read_csv(metrics_paths[0]).filter(pl.col("config") == "NN")
        if metrics.height == 0:
            continue
        row = metrics.row(0, named=True)
        rows.append(
            {
                "intraday_bars": int(match["bars"]),
                "range_weight": float(match["rw"]),
                "lambda_good": float(match["lg"]),
                "mae_high_pct": row["mae_high_pct"],
                "mae_low_pct": row["mae_low_pct"],
                "mae_high_dollar": row["mae_high_dollar"],
                "mae_low_dollar": row["mae_low_dollar"],
                "inside_frac": row["inside_frac"],
                "outside_frac": row["outside_frac"],
                "mixed_frac": row["mixed_frac"],
                "loss": row["loss"],
            }
        )

    if not rows:
        logger.warning(f"No completed runs found under {GRID_DIR}")
        return

    df = pl.DataFrame(rows).sort("loss")
    out_path = GRID_DIR / "grid_search_summary.csv"
    df.write_csv(out_path)
    logger.info(f"{df.height} completed runs summarized -> {out_path}")

    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_float="full"):
        print(df.head(15))

    baseline = df.filter(
        (pl.col("intraday_bars") == 6)
        & (pl.col("range_weight") == 1.0)
        & (pl.col("lambda_good") == 1.0)
    )
    if baseline.height:
        logger.info(
            f"Current default (bars=6, rw=1.0, lg=1.0): {baseline.row(0, named=True)}"
        )


if __name__ == "__main__":
    main()
