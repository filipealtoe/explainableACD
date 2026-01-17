#!/usr/bin/env python3
"""
Create augmented training dataset by adding hard synthetic samples.

Takes:
- Original cleaned training data
- Validated synthetic data (with is_hard column)

Outputs:
- Augmented training dataset (shuffled, deduplicated)

Usage:
    python experiments/scripts/create_augmented_dataset.py \
        experiments/results/synthetic_data/synthetic_deepseek_deepseek-chat_5600_20260111_025623_validated.parquet

    # Only add hard samples (default)
    python experiments/scripts/create_augmented_dataset.py path/to/validated.parquet --hard-only

    # Add all samples
    python experiments/scripts/create_augmented_dataset.py path/to/validated.parquet --all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib

import polars as pl
import numpy as np

# Paths
CLEAN_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_clean"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_augmented"


def text_hash(text: str) -> str:
    """Create hash of text for deduplication."""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]


def check_near_duplicates(texts: list[str], threshold: float = 0.9) -> set[int]:
    """Find indices of near-duplicate texts using simple token overlap."""
    from collections import defaultdict

    # Create token sets
    token_sets = [set(t.lower().split()) for t in texts]

    # Find duplicates (O(n²) but fine for small datasets)
    duplicates = set()
    for i in range(len(texts)):
        if i in duplicates:
            continue
        for j in range(i + 1, len(texts)):
            if j in duplicates:
                continue
            # Jaccard similarity
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            if union > 0 and intersection / union >= threshold:
                duplicates.add(j)  # Keep i, remove j

    return duplicates


def main():
    parser = argparse.ArgumentParser(description="Create augmented training dataset")
    parser.add_argument("synthetic_path", type=Path, help="Path to validated synthetic data")
    parser.add_argument("--hard-only", action="store_true", default=True, help="Only add hard samples (default)")
    parser.add_argument("--all", action="store_true", help="Add all synthetic samples")
    parser.add_argument("--dedup-threshold", type=float, default=0.85, help="Jaccard threshold for deduplication")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    if args.all:
        args.hard_only = False

    print("=" * 70)
    print("CREATE AUGMENTED TRAINING DATASET")
    print("=" * 70)

    # Load original training data
    print("\n📦 Loading original training data...")
    train_path = CLEAN_DATA_DIR / "CT24_train_clean.parquet"
    if train_path.exists():
        original_df = pl.read_parquet(train_path)
    else:
        train_tsv = CLEAN_DATA_DIR / "CT24_train_clean.tsv"
        original_df = pl.read_csv(train_tsv, separator="\t")

    print(f"   Original samples: {len(original_df)}")
    orig_yes = original_df.filter(pl.col("class_label") == "Yes").height
    orig_no = original_df.filter(pl.col("class_label") == "No").height
    print(f"   Original distribution: Yes={orig_yes} ({100*orig_yes/len(original_df):.1f}%), No={orig_no} ({100*orig_no/len(original_df):.1f}%)")

    # Load synthetic data
    print(f"\n📦 Loading synthetic data from: {args.synthetic_path}")
    synthetic_df = pl.read_parquet(args.synthetic_path)
    print(f"   Total synthetic: {len(synthetic_df)}")

    # Filter to hard samples if requested
    if args.hard_only:
        if "is_hard" not in synthetic_df.columns:
            raise ValueError("Synthetic data must have 'is_hard' column. Run validate_synthetic_hardness.py first.")
        synthetic_df = synthetic_df.filter(pl.col("is_hard") == True)
        print(f"   Hard samples only: {len(synthetic_df)}")

    syn_yes = synthetic_df.filter(pl.col("class_label") == "Yes").height
    syn_no = synthetic_df.filter(pl.col("class_label") == "No").height
    print(f"   Synthetic distribution: Yes={syn_yes} ({100*syn_yes/len(synthetic_df):.1f}%), No={syn_no} ({100*syn_no/len(synthetic_df):.1f}%)")

    # Check for exact duplicates with original
    print("\n🔍 Checking for duplicates with original data...")
    original_hashes = set(text_hash(t) for t in original_df["Text"].to_list())
    synthetic_texts = synthetic_df["Text"].to_list()

    duplicate_indices = []
    for i, text in enumerate(synthetic_texts):
        if text_hash(text) in original_hashes:
            duplicate_indices.append(i)

    if duplicate_indices:
        print(f"   ⚠️  Found {len(duplicate_indices)} exact duplicates with original - removing")
        synthetic_df = synthetic_df.with_row_index("_idx").filter(~pl.col("_idx").is_in(duplicate_indices)).drop("_idx")

    # Check for near-duplicates within synthetic
    print(f"\n🔍 Checking for near-duplicates within synthetic (threshold={args.dedup_threshold})...")
    synthetic_texts = synthetic_df["Text"].to_list()
    near_dup_indices = check_near_duplicates(synthetic_texts, args.dedup_threshold)

    if near_dup_indices:
        print(f"   ⚠️  Found {len(near_dup_indices)} near-duplicates - removing")
        synthetic_df = synthetic_df.with_row_index("_idx").filter(~pl.col("_idx").is_in(near_dup_indices)).drop("_idx")

    print(f"   Final synthetic samples: {len(synthetic_df)}")

    # Generate unique Sentence_ids for synthetic samples
    max_id = original_df["Sentence_id"].max()
    if isinstance(max_id, str):
        max_id = int(max_id)

    new_ids = list(range(max_id + 1, max_id + 1 + len(synthetic_df)))

    # Prepare synthetic data with matching schema
    synthetic_for_merge = pl.DataFrame({
        "Sentence_id": new_ids,
        "Text": synthetic_df["Text"].to_list(),
        "class_label": synthetic_df["class_label"].to_list(),
    })

    # Add source column to both
    original_with_source = original_df.select(["Sentence_id", "Text", "class_label"]).with_columns(
        pl.lit("original").alias("source")
    )
    synthetic_with_source = synthetic_for_merge.with_columns(
        pl.lit("synthetic").alias("source")
    )

    # Combine
    print("\n🔀 Combining and shuffling...")
    combined_df = pl.concat([original_with_source, synthetic_with_source])

    # Shuffle
    np.random.seed(args.seed)
    shuffle_indices = np.random.permutation(len(combined_df))
    combined_df = combined_df.with_row_index("_idx").filter(
        pl.col("_idx").is_in(shuffle_indices.tolist())
    ).sort(pl.col("_idx").map_elements(lambda x: shuffle_indices.tolist().index(x), return_dtype=pl.Int64)).drop("_idx")

    # Actually shuffle properly
    combined_df = combined_df.sample(fraction=1.0, shuffle=True, seed=args.seed)

    print(f"   Combined samples: {len(combined_df)}")

    # Final distribution
    final_yes = combined_df.filter(pl.col("class_label") == "Yes").height
    final_no = combined_df.filter(pl.col("class_label") == "No").height
    final_orig = combined_df.filter(pl.col("source") == "original").height
    final_syn = combined_df.filter(pl.col("source") == "synthetic").height

    print(f"\n📊 Final dataset statistics:")
    print(f"   Total: {len(combined_df)}")
    print(f"   Original: {final_orig} ({100*final_orig/len(combined_df):.1f}%)")
    print(f"   Synthetic: {final_syn} ({100*final_syn/len(combined_df):.1f}%)")
    print(f"   Yes: {final_yes} ({100*final_yes/len(combined_df):.1f}%)")
    print(f"   No: {final_no} ({100*final_no/len(combined_df):.1f}%)")

    # Compare to original
    print(f"\n📈 Change from original:")
    print(f"   Samples: {len(original_df)} → {len(combined_df)} (+{len(combined_df)-len(original_df)})")
    orig_yes_pct = 100 * orig_yes / len(original_df)
    final_yes_pct = 100 * final_yes / len(combined_df)
    print(f"   Yes ratio: {orig_yes_pct:.1f}% → {final_yes_pct:.1f}% ({final_yes_pct - orig_yes_pct:+.1f}%)")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full augmented dataset
    output_path = OUTPUT_DIR / "CT24_train_augmented.parquet"
    combined_df.write_parquet(output_path)
    print(f"\n💾 Saved to: {output_path}")

    # Also save TSV version (without source column for compatibility)
    tsv_df = combined_df.select(["Sentence_id", "Text", "class_label"])
    tsv_path = OUTPUT_DIR / "CT24_train_augmented.tsv"
    tsv_df.write_csv(tsv_path, separator="\t")
    print(f"💾 TSV copy: {tsv_path}")

    # Save metadata
    import json
    meta = {
        "original_samples": len(original_df),
        "synthetic_samples_added": final_syn,
        "total_samples": len(combined_df),
        "hard_only": args.hard_only,
        "dedup_threshold": args.dedup_threshold,
        "seed": args.seed,
        "synthetic_source": str(args.synthetic_path),
        "class_distribution": {
            "Yes": final_yes,
            "No": final_no,
        },
    }
    meta_path = OUTPUT_DIR / "augmentation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Metadata: {meta_path}")

    print(f"\n✅ Done! Use {output_path} for training.")


if __name__ == "__main__":
    main()
