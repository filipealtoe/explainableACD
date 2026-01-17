#!/usr/bin/env python
"""
Create Chronologically-Ordered Dataset Subsets

Creates subsets of the tweet dataset that preserve temporal order,
suitable for simulating production streaming scenarios.
"""

import sys
from pathlib import Path

import mlflow
import polars as pl

# Add project root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))


def create_chronological_subsets(
    input_path: Path,
    output_dir: Path,
    subset_sizes: list[int],
) -> dict:
    """
    Create chronologically-ordered subsets.

    Args:
        input_path: Path to input parquet file
        output_dir: Directory to save subsets
        subset_sizes: List of subset sizes to create

    Returns:
        Dictionary with subset metadata
    """
    # Load and sort chronologically
    print(f"Loading data from {input_path}")
    df = pl.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows")

    # Sort by timestamp
    df = df.sort("created_at_utc")
    print("Sorted chronologically")

    # Get temporal bounds
    min_time = df["created_at_utc"].min()
    max_time = df["created_at_utc"].max()
    print(f"Date range: {min_time} to {max_time}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for size in subset_sizes:
        if size > len(df):
            print(f"Skipping {size:,} (larger than dataset)")
            continue

        # Take first N rows (chronologically)
        subset = df.head(size)

        # Get subset temporal bounds
        subset_min = subset["created_at_utc"].min()
        subset_max = subset["created_at_utc"].max()

        # Calculate temporal coverage
        n_hours = subset.select(pl.col("created_at_utc").dt.truncate("1h")).n_unique()
        n_days = subset.select(pl.col("created_at_utc").dt.truncate("1d")).n_unique()

        # Save subset
        suffix = _format_size(size)
        output_path = output_dir / f"tweets_chrono_{suffix}.parquet"
        subset.write_parquet(output_path)

        results[size] = {
            "file": str(output_path),
            "rows": len(subset),
            "start_time": str(subset_min),
            "end_time": str(subset_max),
            "n_hours": n_hours,
            "n_days": n_days,
        }

        print(f"\n{suffix}:")
        print(f"  Rows: {len(subset):,}")
        print(f"  Period: {subset_min} to {subset_max}")
        print(f"  Spans: {n_days} days, {n_hours} hours")
        print(f"  Saved: {output_path}")

    return results


def _format_size(size: int) -> str:
    """Format size as human-readable string."""
    if size >= 1_000_000:
        return f"{size // 1_000_000}M"
    elif size >= 1_000:
        return f"{size // 1_000}k"
    else:
        return str(size)


def main():
    input_path = repo_root / "data" / "processed" / "tweets_v9.parquet"
    output_dir = repo_root / "data" / "subsets"

    # Subset sizes to create
    subset_sizes = [
        1_000,  # 1k - quick testing
        5_000,  # 5k - small experiments
        10_000,  # 10k - medium experiments
        50_000,  # 50k - larger experiments
        100_000,  # 100k - significant subset
        250_000,  # 250k - ~30% of data
        500_000,  # 500k - ~60% of data
    ]

    print("=" * 60)
    print("CREATING CHRONOLOGICAL SUBSETS")
    print("=" * 60)

    results = create_chronological_subsets(input_path, output_dir, subset_sizes)

    # MLflow logging
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("dataset_processing")

    with mlflow.start_run(run_name="chronological_subsets"):
        mlflow.log_params(
            {
                "input_file": str(input_path),
                "output_dir": str(output_dir),
                "n_subsets": len(results),
            }
        )

        for size, meta in results.items():
            suffix = _format_size(size)
            mlflow.log_metrics(
                {
                    f"subset_{suffix}_rows": meta["rows"],
                    f"subset_{suffix}_days": meta["n_days"],
                    f"subset_{suffix}_hours": meta["n_hours"],
                }
            )

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Size':<10} {'Rows':>10} {'Days':>6} {'Hours':>6} {'Start Date':<12} {'End Date':<12}")
    print("-" * 60)
    for size in sorted(results.keys()):
        meta = results[size]
        suffix = _format_size(size)
        start = meta["start_time"][:10]
        end = meta["end_time"][:10]
        print(f"{suffix:<10} {meta['rows']:>10,} {meta['n_days']:>6} {meta['n_hours']:>6} {start:<12} {end:<12}")


if __name__ == "__main__":
    main()
