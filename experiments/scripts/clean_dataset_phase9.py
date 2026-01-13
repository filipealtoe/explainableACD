"""
Phase 9: Final Filtering

Filters out rows flagged as non-English in Phase 8 to prepare the final dataset for topic modeling.

Input: data/processed/tweets_v8_refined.parquet
Output: data/processed/tweets_v9.parquet
"""

import sys
from pathlib import Path

import mlflow
import polars as pl

# Path setup
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def main():
    print("=" * 60)
    print("PHASE 9: FINAL FILTERING")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Phase 8 Output
    # =========================================================================
    print("\n[1] Loading Phase 8 output...")
    input_path = repo_root / "data" / "processed" / "tweets_v8_refined.parquet"
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        print("Please run Phase 8 first!")
        sys.exit(1)

    df = pl.read_parquet(input_path)
    print(f"    Loaded {len(df):,} rows")

    # =========================================================================
    # Step 2: Filter Non-English
    # =========================================================================
    print("\n[2] Filtering non-English tweets...")

    non_english_count = df.filter(~pl.col("is_english")).shape[0]

    # Keep only English
    df_filtered = df.filter(pl.col("is_english"))

    print(f"    Removed {non_english_count:,} non-English rows.")
    print(f"    Remaining: {len(df_filtered):,} rows.")

    # =========================================================================
    # Step 3: Save Output
    # =========================================================================
    print("\n[3] Saving final dataset for topic modeling...")
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "tweets_v9.parquet"
    df_filtered.write_parquet(output_path)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Step 4: Log to MLflow
    # =========================================================================
    print("\n[4] Logging to MLflow...")
    mlflow.set_experiment("dataset_processing")

    with mlflow.start_run(run_name="tweets_v9_ready"):
        mlflow.log_params(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "initial_rows": len(df),
                "final_rows": len(df_filtered),
                "filtered_count": non_english_count,
            }
        )

        mlflow.log_metrics({"rows_removed": non_english_count, "final_row_count": len(df_filtered)})

        mlflow.log_artifact(str(output_path))
        print(f"    MLflow Run ID: {mlflow.active_run().info.run_id}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("READY FOR TOPIC MODELING")
    print("=" * 60)
    print(f"Final Dataset: {len(df_filtered):,} tweets")
    print(f"File: {output_path}")


if __name__ == "__main__":
    main()
