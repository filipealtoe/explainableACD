#!/usr/bin/env python3
"""
Prepare CT24 confidence data for gradient boosting classifier.

Converts parquet files from the checkworthiness pipeline to TSV format
expected by the ct24_classifier_core.py script.

Input: data/processed/CT24_with_confidences/*.parquet
Output: data/processed/CT24_classifier_ready/{train,dev,test}.tsv
"""

import sys
from pathlib import Path

import polars as pl

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = REPO_ROOT / "data" / "processed" / "CT24_with_confidences"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "CT24_classifier_ready"

# Column mapping: current names -> expected by classifier script
COLUMN_RENAME = {
    "checkability_self_conf": "checkability_cc",
    "verifiability_self_conf": "verifiability_cc",
    "harm_self_conf": "harmpot_cc",
    "checkability_logprob_conf": "checkability_logprob",
    "verifiability_logprob_conf": "verifiability_logprob",
    "harm_logprob_conf": "harmpot_logprob",
}

# Feature column order (classifier expects this exact order + label as last column)
FEATURE_ORDER = [
    "verifiability_cc",
    "verifiability_logprob",
    "checkability_cc",
    "checkability_logprob",
    "harmpot_cc",
    "harmpot_logprob",
]


# =============================================================================
# DATA PROCESSING
# =============================================================================


def find_latest_parquet(split: str) -> Path | None:
    """Find the most recent parquet file for a given split."""
    pattern = f"CT24_{split}_*.parquet"
    files = sorted(INPUT_DIR.glob(pattern))
    return files[-1] if files else None


def process_split(input_path: Path, output_path: Path, split_name: str) -> dict:
    """
    Process a single split: rename columns, impute nulls, encode labels.

    Returns statistics about the processing.
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")
    print(f"Input: {input_path}")

    # Load parquet
    df = pl.read_parquet(input_path)
    n_original = len(df)
    print(f"Loaded {n_original:,} rows")

    # Check for required columns
    missing_cols = set(COLUMN_RENAME.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {input_path}: {missing_cols}")

    # Rename columns
    df = df.rename(COLUMN_RENAME)
    print(f"Renamed columns: {list(COLUMN_RENAME.keys())} -> {list(COLUMN_RENAME.values())}")

    # Check for nulls and impute with median
    null_counts = {}
    for col in FEATURE_ORDER:
        null_count = df[col].null_count()
        if null_count > 0:
            median_val = df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median_val))
            null_counts[col] = null_count
            print(f"  Imputed {null_count} nulls in '{col}' with median={median_val:.4f}")

    # Encode labels: Yes=1, No=0
    label_dist = df["label"].value_counts().sort("label")
    print(f"\nLabel distribution:")
    for row in label_dist.iter_rows(named=True):
        print(f"  {row['label']}: {row['count']:,} ({row['count']/n_original*100:.1f}%)")

    df = df.with_columns(pl.when(pl.col("label") == "Yes").then(1).otherwise(0).alias("label_encoded"))

    # Also keep sample_id for later reference (useful for CT24 submission format)
    # Select columns in expected order: features + label
    df_features = df.select(FEATURE_ORDER + ["label_encoded"])

    # Also create a mapping file for sample_id -> row index
    df_mapping = df.select(["sample_id", "label"])

    # Export features as TSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_csv(output_path, separator="\t")
    print(f"\nExported to: {output_path}")
    print(f"Shape: {df_features.shape}")

    # Export mapping file
    mapping_path = output_path.with_suffix(".mapping.tsv")
    df_mapping.write_csv(mapping_path, separator="\t")
    print(f"Mapping file: {mapping_path}")

    return {
        "split": split_name,
        "n_rows": n_original,
        "n_nulls_imputed": sum(null_counts.values()),
        "null_details": null_counts,
        "label_yes": label_dist.filter(pl.col("label") == "Yes")["count"].item(),
        "label_no": label_dist.filter(pl.col("label") == "No")["count"].item(),
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("CT24 Data Preparation for Gradient Boosting Classifier")
    print("=" * 70)

    stats = []

    for split in ["train", "dev", "test"]:
        input_path = find_latest_parquet(split)
        if input_path is None:
            print(f"\n⚠️  WARNING: No parquet file found for {split} split in {INPUT_DIR}")
            continue

        output_path = OUTPUT_DIR / f"{split}.tsv"
        split_stats = process_split(input_path, output_path, split)
        stats.append(split_stats)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for s in stats:
        class_ratio = s["label_no"] / s["label_yes"] if s["label_yes"] > 0 else float("inf")
        print(f"\n{s['split'].upper()}:")
        print(f"  Rows: {s['n_rows']:,}")
        print(f"  Labels: {s['label_yes']:,} Yes ({s['label_yes']/s['n_rows']*100:.1f}%) / "
              f"{s['label_no']:,} No ({s['label_no']/s['n_rows']*100:.1f}%)")
        print(f"  Class ratio (No:Yes): {class_ratio:.2f}:1")
        if s["n_nulls_imputed"] > 0:
            print(f"  Nulls imputed: {s['n_nulls_imputed']}")

    print(f"\n✅ All files exported to: {OUTPUT_DIR}")
    print("\nExpected column order in TSV files:")
    print(f"  {FEATURE_ORDER + ['label_encoded']}")


if __name__ == "__main__":
    main()
