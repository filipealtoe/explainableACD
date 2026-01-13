#!/usr/bin/env python3
"""Quick validation of V4 LLM features - check for nulls and parse errors."""

from pathlib import Path
import polars as pl

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_llm_features_v4"

def validate_split(split: str):
    file = DATA_DIR / f"{split}_llm_features.parquet"
    if not file.exists():
        print(f"  {split}: NOT FOUND")
        return

    df = pl.read_parquet(file)
    n = len(df)

    # Check for nulls
    null_counts = {col: df[col].null_count() for col in df.columns}
    cols_with_nulls = {k: v for k, v in null_counts.items() if v > 0}

    # Check parse issues
    parse_cols = [c for c in df.columns if "parse_issue" in c]
    parse_issues = {col: df[col].sum() for col in parse_cols}

    print(f"\n  {split.upper()}: {n} samples")
    print(f"  {'─'*40}")

    if cols_with_nulls:
        print(f"  ❌ Columns with nulls:")
        for col, count in cols_with_nulls.items():
            print(f"      {col}: {count} ({100*count/n:.1f}%)")
    else:
        print(f"  ✓ No null values")

    if any(v > 0 for v in parse_issues.values()):
        print(f"  ⚠️  Parse issues:")
        for col, count in parse_issues.items():
            if count > 0:
                print(f"      {col}: {count} ({100*count/n:.1f}%)")
    else:
        print(f"  ✓ No parse issues")

    # Quick stats on key features
    print(f"\n  Key feature ranges:")
    for feat in ["check_score", "verif_score", "harm_score", "check_p_yes", "check_entropy"]:
        if feat in df.columns:
            col = df[feat]
            print(f"      {feat}: [{col.min():.2f}, {col.max():.2f}] mean={col.mean():.2f}")


if __name__ == "__main__":
    print("=" * 50)
    print("V4 LLM Features Validation")
    print("=" * 50)

    for split in ["train", "dev", "test"]:
        validate_split(split)

    print()
