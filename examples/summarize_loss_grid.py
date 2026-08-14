"""Aggregate output/day_range_nn_lossgrid/rw{rw}_lg{lg}/metrics_val_*.csv into
one per-model table (all 9 range_weight x lambda_good combinations, fixed
intraday_bars=6, single shared seed), as a precursor to picking a subset of
these models to combine into an ensemble.
"""

import re
from pathlib import Path

import polars as pl
from loguru import logger

GRID_DIR = Path("output/day_range_nn_lossgrid")
DIR_PATTERN = re.compile(r"rw(?P<rw>[\d.]+)_lg(?P<lg>[\d.]+)")


def main() -> None:
    rows = []
    for run_dir in sorted(GRID_DIR.glob("rw*_lg*")):
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
                "range_weight": float(match["rw"]),
                "lambda_good": float(match["lg"]),
                "mae_high_pct": round(row["mae_high_pct"] * 100, 3),
                "mae_low_pct": round(row["mae_low_pct"] * 100, 3),
                "mae_high_dollar": round(row["mae_high_dollar"], 2),
                "mae_low_dollar": round(row["mae_low_dollar"], 2),
                "combined_mae_dollar": round(
                    row["mae_high_dollar"] + row["mae_low_dollar"], 2
                ),
                "inside_pct": round(row["inside_frac"] * 100, 1),
                "outside_pct": round(row["outside_frac"] * 100, 1),
                "mixed_pct": round(row["mixed_frac"] * 100, 1),
                "run_dir": str(run_dir),
            }
        )

    if not rows:
        logger.warning(f"No completed runs found under {GRID_DIR}")
        return

    df = pl.DataFrame(rows).sort(["range_weight", "lambda_good"])
    out_path = GRID_DIR / "loss_grid_summary.csv"
    df.write_csv(out_path)
    logger.info(f"{df.height}/9 completed runs -> {out_path}")

    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df.drop("run_dir"))


if __name__ == "__main__":
    main()
