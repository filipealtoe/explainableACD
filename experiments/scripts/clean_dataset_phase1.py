"""
Phase 1: Dataset Core Cleaning

Performs essential data quality fixes:
1. Remove exact duplicate rows
2. Deduplicate by ID (keep highest engagement version)
3. Convert IDs to string (fix precision loss)
4. Parse timestamps to UTC
5. Drop redundant/broken columns

Input: data/raw/tweets_ai.parquet
Output: data/processed/tweets_v1_clean.parquet
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
    print("PHASE 1: DATASET CORE CLEANING")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[1] Loading data...")
    input_path = repo_root / "data" / "raw" / "tweets_ai.parquet"
    df = pl.read_parquet(input_path)
    initial_rows = len(df)
    print(f"    Loaded {initial_rows:,} rows, {len(df.columns)} columns")

    # =========================================================================
    # Step 2: Remove Exact Duplicate Rows
    # =========================================================================
    print("\n[2] Removing exact duplicate rows...")
    df = df.unique()
    rows_after_exact_dedup = len(df)
    rows_removed_exact_dupes = initial_rows - rows_after_exact_dedup
    print(f"    Removed {rows_removed_exact_dupes:,} exact duplicates")
    print(f"    Rows remaining: {rows_after_exact_dedup:,}")

    # =========================================================================
    # Step 3: Deduplicate by ID (keep highest engagement)
    # =========================================================================
    print("\n[3] Deduplicating by ID (keeping highest engagement)...")
    df = df.with_columns(
        (pl.col("replies_count") + pl.col("retweets_count") + pl.col("likes_count")).alias("_total_engagement")
    )
    df = df.sort("_total_engagement", descending=True)
    df = df.unique(subset=["id"], keep="first")
    df = df.drop("_total_engagement")

    rows_after_id_dedup = len(df)
    rows_removed_dupe_ids = rows_after_exact_dedup - rows_after_id_dedup
    print(f"    Removed {rows_removed_dupe_ids:,} duplicate IDs")
    print(f"    Rows remaining: {rows_after_id_dedup:,}")

    # =========================================================================
    # Step 4: Filter to English Only
    # =========================================================================
    print("\n[4] Filtering to English only...")
    df = df.filter(pl.col("language") == "en")
    rows_after_lang_filter = len(df)
    rows_removed_non_english = rows_after_id_dedup - rows_after_lang_filter
    print(f"    Removed {rows_removed_non_english:,} non-English rows")
    print(f"    Rows remaining: {rows_after_lang_filter:,}")

    # =========================================================================
    # Step 5: Convert IDs to String
    # =========================================================================
    print("\n[5] Converting IDs to string...")
    df = df.with_columns(
        [
            pl.col("id").cast(pl.Int64).cast(pl.Utf8).alias("id"),
            pl.col("conversation_id").cast(pl.Int64).cast(pl.Utf8).alias("conversation_id"),
        ]
    )
    print("    IDs converted to string format")

    # =========================================================================
    # Step 6: Parse Timestamps to UTC
    # =========================================================================
    print("\n[6] Parsing timestamps to UTC...")
    # Format: "YYYY-MM-DD HH:MM:SS EST" or "EDT"
    # EST = UTC-5, EDT = UTC-4

    df = df.with_columns(
        [
            pl.col("created_at").str.slice(-3).alias("_tz_suffix"),
            pl.col("created_at").str.slice(0, 19).alias("_datetime_str"),
        ]
    )

    df = df.with_columns(pl.col("_datetime_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("created_at_utc"))

    # Convert to UTC based on timezone suffix
    df = df.with_columns(
        pl.when(pl.col("_tz_suffix") == "EST")
        .then(pl.col("created_at_utc") + pl.duration(hours=5))
        .when(pl.col("_tz_suffix") == "EDT")
        .then(pl.col("created_at_utc") + pl.duration(hours=4))
        .otherwise(pl.col("created_at_utc"))
        .alias("created_at_utc")
    )

    df = df.drop(["_tz_suffix", "_datetime_str"])

    # Check for parse failures
    null_timestamps = df["created_at_utc"].null_count()
    if null_timestamps > 0:
        print(f"    WARNING: {null_timestamps} timestamps failed to parse")
    else:
        print("    All timestamps parsed successfully")

    min_date = df["created_at_utc"].min()
    max_date = df["created_at_utc"].max()
    print(f"    Date range: {min_date} to {max_date}")

    # =========================================================================
    # Step 7: Drop Redundant/Broken Columns
    # =========================================================================
    print("\n[7] Dropping redundant columns...")

    # Create has_cashtags before dropping cashtags
    df = df.with_columns((pl.col("cashtags") != "[]").alias("has_cashtags"))

    columns_to_drop = [
        "date",  # Redundant with created_at_utc
        "time",  # Redundant with created_at_utc
        "timezone",  # Misleading (scraper TZ, not user TZ)
        "retweet",  # Broken (all False)
        "language",  # Redundant after English filter (all "en")
        "cashtags",  # Replaced by has_cashtags binary flag
    ]
    df = df.drop(columns_to_drop)
    print(f"    Dropped: {columns_to_drop}")

    # =========================================================================
    # Step 8: Rename and Reorder Columns
    # =========================================================================
    print("\n[8] Finalizing schema...")
    # Validate video column is binary before converting
    video_values = set(df["video"].unique().to_list())
    assert video_values <= {0, 1}, f"video column has unexpected values: {video_values}"
    print(f"    video column validated as binary: {video_values}")
    df = df.with_columns(pl.col("video").cast(pl.Boolean).alias("has_video"))
    df = df.drop("video")

    # Log has_cashtags stats
    cashtag_count = df.filter(pl.col("has_cashtags")).shape[0]
    print(f"    has_cashtags: {cashtag_count:,} tweets ({100 * cashtag_count / len(df):.2f}%)")
    final_columns = [
        # Identifiers
        "id",
        "conversation_id",
        # Timestamp
        "created_at_utc",
        # Content
        "tweet",
        # Engagement
        "replies_count",
        "retweets_count",
        "likes_count",
        # Media/Links (as-is for Phase 2)
        "urls",
        "photos",
        "has_video",
        "has_cashtags",
        "hashtags",
        "quote_url",
        "thumbnail",
    ]
    df = df.select(final_columns)
    print(f"    Final columns: {len(final_columns)}")

    # =========================================================================
    # Step 9: Save Output
    # =========================================================================
    print("\n[9] Saving cleaned dataset...")
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "tweets_v1_clean.parquet"
    df.write_parquet(output_path)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Step 10: Log to MLflow
    # =========================================================================
    print("\n[10] Logging to MLflow...")
    mlflow.set_experiment("dataset_processing")

    with mlflow.start_run(run_name="tweets_v1_clean"):
        mlflow.log_params(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "input_rows": initial_rows,
                "output_rows": len(df),
                "columns_dropped": str(columns_to_drop),
                "final_columns": str(final_columns),
            }
        )

        mlflow.log_metrics(
            {
                "rows_removed_exact_dupes": rows_removed_exact_dupes,
                "rows_removed_dupe_ids": rows_removed_dupe_ids,
                "rows_removed_non_english": rows_removed_non_english,
                "total_rows_removed": initial_rows - len(df),
                "final_row_count": len(df),
                "final_column_count": len(final_columns),
            }
        )

        mlflow.log_artifact(str(output_path))

        run_id = mlflow.active_run().info.run_id
        print(f"    MLflow Run ID: {run_id}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Input rows:           {initial_rows:,}")
    print(f"After exact dedup:    {rows_after_exact_dedup:,} (-{rows_removed_exact_dupes:,})")
    print(f"After ID dedup:       {rows_after_id_dedup:,} (-{rows_removed_dupe_ids:,})")
    print(f"After English filter: {rows_after_lang_filter:,} (-{rows_removed_non_english:,})")
    print(f"Final rows:           {len(df):,}")
    print(f"Total removed:        {initial_rows - len(df):,} ({100 * (initial_rows - len(df)) / initial_rows:.1f}%)")
    print(f"\nColumns: {len(df.columns)} (dropped {len(columns_to_drop)})")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
