"""
Create Subsets

Generates random subsets of the final Phase 9 dataset for testing and experimentation.
Sizes: 10, 100, 1k, 10k, 100k, 500k.

Input: data/processed/tweets_v9.parquet
Output: data/processed/subsets/tweets_v9_subset_{N}.parquet
"""

import sys
from pathlib import Path

import polars as pl

# Path setup
repo_root = Path(__file__).resolve().parents[2]
input_path = repo_root / "data" / "processed" / "tweets_v9.parquet"
output_dir = repo_root / "data" / "processed" / "subsets"

SIZES = [10, 100, 1000, 10000, 100000, 500000]


def main():
    print("=" * 60)
    print("CREATING DATASET SUBSETS")
    print("=" * 60)

    # 1. Load Data
    print("\n[1] Loading final dataset...")
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        print("Please run Phase 9 first!")
        sys.exit(1)

    df = pl.read_parquet(input_path)
    total_rows = len(df)
    print(f"    Loaded {total_rows:,} rows")

    # 2. Create Subsets
    print(f"\n[2] Generating subsets in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in SIZES:
        if n > total_rows:
            print(f"    Skipping size {n:,} (larger than total dataset)")
            continue

        print(f"    Sampling {n:,} rows...", end=" ")

        # Sample randomly with fixed seed for reproducibility
        subset = df.sample(n=n, shuffle=True, seed=42)

        outfile = output_dir / f"tweets_v9_subset_{n}.parquet"
        subset.write_parquet(outfile)
        print(f"Saved to {outfile.name}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
