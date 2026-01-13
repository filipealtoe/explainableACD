"""
Phase 2: Feature Engineering

Reads Phase 1 output and adds:
- Count features: url_count, hashtag_count, photo_count
- Binary features: has_*, is_quote_tweet, is_weekend
- Time features: year, month, day, day_of_week, hour
- Text features: tweet_length, word_count
- Engagement features: total_engagement

Input: data/processed/tweets_v1_clean.parquet (from Phase 1)
Output: data/processed/tweets_v2_features.parquet
"""

import ast
import sys
from pathlib import Path

import mlflow
import polars as pl

# Path setup
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def safe_list_len(val: str) -> int:
    """Safely parse a string representation of a list and return its length."""
    if val is None or val == "[]" or val == "":
        return 0
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return len(parsed)
        return 0
    except:
        return 0


def main():
    print("=" * 60)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Phase 1 Output
    # =========================================================================
    print("\n[1] Loading Phase 1 output...")
    input_path = repo_root / "data" / "processed" / "tweets_v1_clean.parquet"
    df = pl.read_parquet(input_path)
    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")

    # =========================================================================
    # Step 2: Create Count Features
    # =========================================================================
    print("\n[2] Creating count features...")

    # Use map_elements for safe list parsing
    # Note: cashtags was already converted to has_cashtags in Phase 1
    df = df.with_columns(
        [
            pl.col("urls").map_elements(safe_list_len, return_dtype=pl.Int64).alias("url_count"),
            pl.col("photos").map_elements(safe_list_len, return_dtype=pl.Int64).alias("photo_count"),
            pl.col("hashtags").map_elements(safe_list_len, return_dtype=pl.Int64).alias("hashtag_count"),
        ]
    )

    print(
        f"    url_count:     min={df['url_count'].min()}, max={df['url_count'].max()}, mean={df['url_count'].mean():.2f}"
    )
    print(
        f"    photo_count:   min={df['photo_count'].min()}, max={df['photo_count'].max()}, mean={df['photo_count'].mean():.2f}"
    )
    print(
        f"    hashtag_count: min={df['hashtag_count'].min()}, max={df['hashtag_count'].max()}, mean={df['hashtag_count'].mean():.2f}"
    )

    # =========================================================================
    # Step 3: Create Binary Features
    # =========================================================================
    print("\n[3] Creating binary features...")

    # Note: has_video and has_cashtags already exist from Phase 1
    df = df.with_columns(
        [
            # From counts
            (pl.col("url_count") > 0).alias("has_urls"),
            (pl.col("photo_count") > 0).alias("has_photos"),
            (pl.col("hashtag_count") > 0).alias("has_hashtags"),
            # From raw columns
            (pl.col("thumbnail").is_not_null() & (pl.col("thumbnail") != "")).alias("has_thumbnail"),
            (pl.col("quote_url").is_not_null() & (pl.col("quote_url") != "")).alias("is_quote_tweet"),
            # Engagement flags
            (pl.col("replies_count") > 0).alias("has_replies"),
            (pl.col("retweets_count") > 0).alias("has_retweets"),
            (pl.col("likes_count") > 0).alias("has_likes"),
        ]
    )

    # Composite binary features
    # Note: has_media removed as it equals has_video in this dataset (all photos have video)
    df = df.with_columns((pl.col("has_replies") | pl.col("has_retweets") | pl.col("has_likes")).alias("has_engagement"))

    # Print stats
    for col in [
        "has_urls",
        "has_photos",
        "has_video",
        "has_hashtags",
        "has_cashtags",
        "has_thumbnail",
        "is_quote_tweet",
        "has_engagement",
    ]:
        count = df.filter(pl.col(col)).shape[0]
        print(f"    {col}: {count:,} ({100 * count / len(df):.1f}%)")

    # =========================================================================
    # Step 4: Create Time Features
    # =========================================================================
    print("\n[4] Creating time features...")

    df = df.with_columns(
        [
            pl.col("created_at_utc").dt.year().alias("year"),
            pl.col("created_at_utc").dt.month().alias("month"),
            pl.col("created_at_utc").dt.week().alias("week_of_year"),
            ((pl.col("created_at_utc").dt.day() - 1) // 7 + 1).alias("week_of_month"),
            pl.col("created_at_utc").dt.day().alias("day"),
            pl.col("created_at_utc").dt.weekday().alias("day_of_week"),  # 0=Monday, 6=Sunday
            pl.col("created_at_utc").dt.hour().alias("hour"),
            pl.col("created_at_utc").dt.minute().alias("minute"),
        ]
    )

    # is_weekend (Saturday=5, Sunday=6)
    df = df.with_columns((pl.col("day_of_week") >= 5).alias("is_weekend"))

    weekend_count = df.filter(pl.col("is_weekend")).shape[0]
    print(f"    Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"    is_weekend: {weekend_count:,} ({100 * weekend_count / len(df):.1f}%)")

    # =========================================================================
    # Step 5: Create Text Features
    # =========================================================================
    print("\n[5] Creating text features...")

    df = df.with_columns(
        [
            pl.col("tweet").str.len_chars().alias("tweet_length"),
            pl.col("tweet").str.split(" ").list.len().alias("word_count"),
        ]
    )

    print(
        f"    tweet_length: min={df['tweet_length'].min()}, max={df['tweet_length'].max()}, mean={df['tweet_length'].mean():.1f}"
    )
    print(
        f"    word_count:   min={df['word_count'].min()}, max={df['word_count'].max()}, mean={df['word_count'].mean():.1f}"
    )

    # =========================================================================
    # Step 6: Create Engagement Features
    # =========================================================================
    print("\n[6] Creating engagement features...")

    df = df.with_columns(
        (pl.col("replies_count") + pl.col("retweets_count") + pl.col("likes_count")).alias("total_engagement")
    )

    print(
        f"    total_engagement: min={df['total_engagement'].min()}, max={df['total_engagement'].max()}, mean={df['total_engagement'].mean():.1f}"
    )

    # =========================================================================
    # Step 7: Select Final Columns
    # =========================================================================
    print("\n[7] Finalizing schema...")

    final_columns = [
        # Identifiers
        "id",
        "conversation_id",
        # Timestamp
        "created_at_utc",
        # Time features
        "year",
        "month",
        "week_of_year",
        "week_of_month",
        "day",
        "day_of_week",
        "hour",
        "minute",
        "is_weekend",
        # Content
        "tweet",
        "tweet_length",
        "word_count",
        # Engagement (raw)
        "replies_count",
        "retweets_count",
        "likes_count",
        "total_engagement",
        # Engagement (binary)
        "has_replies",
        "has_retweets",
        "has_likes",
        "has_engagement",
        # Media/Links (counts)
        "url_count",
        "photo_count",
        "hashtag_count",
        # Media/Links (binary)
        "has_urls",
        "has_photos",
        "has_video",
        "has_hashtags",
        "has_cashtags",
        "has_thumbnail",
        "is_quote_tweet",
    ]

    df = df.select(final_columns)
    print(f"    Final columns: {len(final_columns)}")

    # =========================================================================
    # Step 8: Save Output
    # =========================================================================
    print("\n[8] Saving cleaned dataset...")
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "tweets_v2_features.parquet"
    df.write_parquet(output_path)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Step 9: Log to MLflow
    # =========================================================================
    print("\n[9] Logging to MLflow...")
    mlflow.set_experiment("dataset_processing")

    with mlflow.start_run(run_name="tweets_v2_features"):
        mlflow.log_params(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "row_count": len(df),
                "final_columns": len(final_columns),
            }
        )

        mlflow.log_metrics(
            {
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
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Input (from Phase 1): {len(df):,} rows")
    print(f"Output columns:       {len(df.columns)}")
    print(f"Output: {output_path}")

    # Print schema
    print("\n" + "=" * 60)
    print("FINAL SCHEMA")
    print("=" * 60)
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")


if __name__ == "__main__":
    main()
