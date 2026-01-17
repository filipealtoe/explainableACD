#!/usr/bin/env python3
"""Prepare v5 dataset: Normalized claims merged with original text fallback.

This script creates a new dataset where:
- text = normalized_claim (if has_claim=True)
- text = original_text (if has_claim=False or no normalized claim)

Output is ready for the v4 LLM feature generation pipeline.

Usage:
    python experiments/scripts/prepare_v5_dataset.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# =============================================================================
# Configuration
# =============================================================================

RAW_DATA_DIR = Path("data/raw/CT24_checkworthy_english")
NORMALIZED_CLAIMS_DIR = Path("data/processed/CT24_normalized_claims")
OUTPUT_DIR = Path("data/processed/CT24_v5_normalized")

RAW_FILES = {
    "train": "CT24_checkworthy_english_train.tsv",
    "dev": "CT24_checkworthy_english_dev.tsv",
    "test": "CT24_checkworthy_english_test_gold.tsv",
}


# =============================================================================
# Functions
# =============================================================================

def load_raw_data(split: str) -> pl.DataFrame:
    """Load raw data with original text and labels."""
    df = pl.read_csv(RAW_DATA_DIR / RAW_FILES[split], separator="\t")
    df = df.rename({
        "Sentence_id": "sentence_id",
        "Text": "original_text",
        "class_label": "label",
    })
    df = df.with_columns(pl.col("sentence_id").cast(pl.Utf8))
    return df


def load_normalized_claims(split: str) -> pl.DataFrame:
    """Load normalized claims for a split."""
    pattern = f"CT24_{split}_normalized_*.parquet"
    files = sorted(NORMALIZED_CLAIMS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No normalized claims found for {split}")

    df = pl.read_parquet(files[-1])
    # Keep only relevant columns
    df = df.select([
        "sentence_id",
        "normalized_claim",
        "has_claim",
    ])
    return df


def merge_and_prepare(split: str) -> pl.DataFrame:
    """Merge normalized claims with original text, creating v5 dataset."""

    raw_df = load_raw_data(split)
    norm_df = load_normalized_claims(split)

    # Join
    merged = raw_df.join(norm_df, on="sentence_id", how="left")

    # Create the text column: use normalized_claim if available, else original_text
    merged = merged.with_columns([
        pl.when(pl.col("has_claim") & pl.col("normalized_claim").is_not_null())
        .then(pl.col("normalized_claim"))
        .otherwise(pl.col("original_text"))
        .alias("text"),

        # Track which source was used
        pl.when(pl.col("has_claim") & pl.col("normalized_claim").is_not_null())
        .then(pl.lit("normalized"))
        .otherwise(pl.lit("original"))
        .alias("text_source"),
    ])

    # Select final columns (matching raw format for compatibility)
    result = merged.select([
        "sentence_id",
        "text",
        "label",
        "original_text",
        "normalized_claim",
        "has_claim",
        "text_source",
    ])

    return result


def main():
    print("\n" + "=" * 60)
    print("PREPARE V5 DATASET")
    print("=" * 60)
    print("Merging normalized claims with original text fallback")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for split in ["train", "dev", "test"]:
        print(f"\n{'#'*60}")
        print(f"# Processing {split.upper()}")
        print(f"{'#'*60}")

        df = merge_and_prepare(split)

        # Stats
        n_total = len(df)
        n_normalized = (df["text_source"] == "normalized").sum()
        n_original = (df["text_source"] == "original").sum()

        # By label
        yes_df = df.filter(pl.col("label") == "Yes")
        no_df = df.filter(pl.col("label") == "No")
        yes_normalized = (yes_df["text_source"] == "normalized").sum()
        no_normalized = (no_df["text_source"] == "normalized").sum()

        print(f"\n  Total samples: {n_total}")
        print(f"  Using normalized claim: {n_normalized} ({100*n_normalized/n_total:.1f}%)")
        print(f"  Using original text: {n_original} ({100*n_original/n_total:.1f}%)")
        print(f"\n  By label:")
        print(f"    Checkworthy (Yes): {yes_normalized}/{len(yes_df)} normalized ({100*yes_normalized/len(yes_df):.1f}%)")
        print(f"    Non-checkworthy (No): {no_normalized}/{len(no_df)} normalized ({100*no_normalized/len(no_df):.1f}%)")

        # Save
        output_path = OUTPUT_DIR / f"{split}_v5_text.parquet"
        df.write_parquet(output_path)
        print(f"\n  Saved: {output_path}")

        # Also save TSV format for compatibility with existing scripts
        tsv_df = df.select([
            pl.col("sentence_id").alias("Sentence_id"),
            pl.col("text").alias("Text"),
            pl.col("label").alias("class_label"),
        ])
        tsv_path = OUTPUT_DIR / f"CT24_checkworthy_english_{split}_v5.tsv"
        tsv_df.write_csv(tsv_path, separator="\t")
        print(f"  Saved: {tsv_path}")

        all_stats.append({
            "split": split,
            "total": n_total,
            "normalized": n_normalized,
            "original": n_original,
            "pct_normalized": 100*n_normalized/n_total,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Split':<10} {'Total':<10} {'Normalized':<15} {'Original':<15} {'% Norm':<10}")
    print("-" * 60)
    for s in all_stats:
        print(f"{s['split']:<10} {s['total']:<10} {s['normalized']:<15} {s['original']:<15} {s['pct_normalized']:<10.1f}%")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Run v4 LLM feature generation on v5 text:

   python experiments/scripts/generate_confidence_features.py \\
       --data-dir data/processed/CT24_v5_normalized \\
       --output-dir data/processed/CT24_llm_features_v5

   (You may need to modify generate_confidence_features.py to accept custom paths)

2. Or copy the TSV files to the raw data location and run with modified paths.
""")


if __name__ == "__main__":
    main()
