#!/usr/bin/env python3
"""
Export FP and FN errors from dev-test predictions to CSV.

Usage:
    python experiments/scripts/show_devtest_errors.py
"""

from pathlib import Path
import polars as pl

PREDICTIONS_PATH = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "devtest_predictions.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "devtest_errors.csv"


def main():
    df = pl.read_csv(PREDICTIONS_PATH)

    # False Negatives: actual=Yes, pred=No
    fn = df.filter((pl.col("class_label") == "Yes") & (pl.col("pred") == "No")).with_columns(pl.lit("FN").alias("error_type"))

    # False Positives: actual=No, pred=Yes
    fp = df.filter((pl.col("class_label") == "No") & (pl.col("pred") == "Yes")).with_columns(pl.lit("FP").alias("error_type"))

    # Combine and save
    errors = pl.concat([fn, fp]).sort("error_type", "prob", descending=[False, True])
    errors.write_csv(OUTPUT_PATH)

    print(f"Saved {len(fn)} FN + {len(fp)} FP = {len(errors)} errors to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
