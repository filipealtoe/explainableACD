"""
Phase 3: Engagement Normalization and Ratios

Reads Phase 2 output and adds:
- Log transforms: replies_log, retweets_log, likes_log, engagement_log, engagement_log_sum
- Percentile ranks: replies_pct, retweets_pct, likes_pct, engagement_pct, engagement_pct_mean
- Engagement ratios: reply_ratio, retweet_ratio, like_ratio
- Virality ratios: retweet_to_like_ratio, reply_to_like_ratio
- Content density: hashtag_density, url_density

Input: data/processed/tweets_v2_features.parquet (from Phase 2)
Output: data/processed/tweets_v3_normalized.parquet
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
    print("PHASE 3: ENGAGEMENT NORMALIZATION AND RATIOS")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Phase 2 Output
    # =========================================================================
    print("\n[1] Loading Phase 2 output...")
    input_path = repo_root / "data" / "processed" / "tweets_v2_features.parquet"
    df = pl.read_parquet(input_path)
    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")

    # =========================================================================
    # Step 2: Log Transforms
    # =========================================================================
    print("\n[2] Creating log transforms...")

    df = df.with_columns(
        [
            (pl.col("replies_count") + 1).log().alias("replies_log"),
            (pl.col("retweets_count") + 1).log().alias("retweets_log"),
            (pl.col("likes_count") + 1).log().alias("likes_log"),
            (pl.col("total_engagement") + 1).log().alias("engagement_log"),
        ]
    )

    # Sum of logs (rewards balanced engagement)
    df = df.with_columns(
        (pl.col("replies_log") + pl.col("retweets_log") + pl.col("likes_log")).alias("engagement_log_sum")
    )

    print(f"    replies_log:        min={df['replies_log'].min():.2f}, max={df['replies_log'].max():.2f}")
    print(f"    retweets_log:       min={df['retweets_log'].min():.2f}, max={df['retweets_log'].max():.2f}")
    print(f"    likes_log:          min={df['likes_log'].min():.2f}, max={df['likes_log'].max():.2f}")
    print(f"    engagement_log:     min={df['engagement_log'].min():.2f}, max={df['engagement_log'].max():.2f}")
    print(f"    engagement_log_sum: min={df['engagement_log_sum'].min():.2f}, max={df['engagement_log_sum'].max():.2f}")

    # =========================================================================
    # Step 3: Percentile Ranks
    # =========================================================================
    print("\n[3] Creating percentile ranks...")

    # Calculate percentile rank for each engagement column
    total_rows = len(df)

    df = df.with_columns(
        [
            (pl.col("replies_count").rank() / total_rows).alias("replies_pct"),
            (pl.col("retweets_count").rank() / total_rows).alias("retweets_pct"),
            (pl.col("likes_count").rank() / total_rows).alias("likes_pct"),
            (pl.col("total_engagement").rank() / total_rows).alias("engagement_pct"),
        ]
    )

    # Mean percentile (average of the three)
    df = df.with_columns(
        ((pl.col("replies_pct") + pl.col("retweets_pct") + pl.col("likes_pct")) / 3).alias("engagement_pct_mean")
    )

    print(f"    replies_pct:        min={df['replies_pct'].min():.4f}, max={df['replies_pct'].max():.4f}")
    print(f"    retweets_pct:       min={df['retweets_pct'].min():.4f}, max={df['retweets_pct'].max():.4f}")
    print(f"    likes_pct:          min={df['likes_pct'].min():.4f}, max={df['likes_pct'].max():.4f}")
    print(f"    engagement_pct:     min={df['engagement_pct'].min():.4f}, max={df['engagement_pct'].max():.4f}")
    print(
        f"    engagement_pct_mean: min={df['engagement_pct_mean'].min():.4f}, max={df['engagement_pct_mean'].max():.4f}"
    )

    # =========================================================================
    # Step 4: Engagement Ratios
    # =========================================================================
    print("\n[4] Creating engagement ratios...")

    # Avoid division by zero - use total_engagement + 1
    df = df.with_columns(
        [
            (pl.col("replies_count") / (pl.col("total_engagement") + 1)).alias("reply_ratio"),
            (pl.col("retweets_count") / (pl.col("total_engagement") + 1)).alias("retweet_ratio"),
            (pl.col("likes_count") / (pl.col("total_engagement") + 1)).alias("like_ratio"),
        ]
    )

    print(
        f"    reply_ratio:   min={df['reply_ratio'].min():.4f}, max={df['reply_ratio'].max():.4f}, mean={df['reply_ratio'].mean():.4f}"
    )
    print(
        f"    retweet_ratio: min={df['retweet_ratio'].min():.4f}, max={df['retweet_ratio'].max():.4f}, mean={df['retweet_ratio'].mean():.4f}"
    )
    print(
        f"    like_ratio:    min={df['like_ratio'].min():.4f}, max={df['like_ratio'].max():.4f}, mean={df['like_ratio'].mean():.4f}"
    )

    # =========================================================================
    # Step 5: Virality Ratios
    # =========================================================================
    print("\n[5] Creating virality ratios...")

    df = df.with_columns(
        [
            (pl.col("retweets_count") / (pl.col("likes_count") + 1)).alias("retweet_to_like_ratio"),
            (pl.col("replies_count") / (pl.col("likes_count") + 1)).alias("reply_to_like_ratio"),
        ]
    )

    print(
        f"    retweet_to_like_ratio: min={df['retweet_to_like_ratio'].min():.4f}, max={df['retweet_to_like_ratio'].max():.4f}, mean={df['retweet_to_like_ratio'].mean():.4f}"
    )
    print(
        f"    reply_to_like_ratio:   min={df['reply_to_like_ratio'].min():.4f}, max={df['reply_to_like_ratio'].max():.4f}, mean={df['reply_to_like_ratio'].mean():.4f}"
    )

    # =========================================================================
    # Step 6: Content Density Ratios
    # =========================================================================
    print("\n[6] Creating content density ratios...")

    df = df.with_columns(
        [
            (pl.col("hashtag_count") / (pl.col("word_count") + 1)).alias("hashtag_density"),
            (pl.col("url_count") / (pl.col("word_count") + 1)).alias("url_density"),
        ]
    )

    print(
        f"    hashtag_density: min={df['hashtag_density'].min():.4f}, max={df['hashtag_density'].max():.4f}, mean={df['hashtag_density'].mean():.4f}"
    )
    print(
        f"    url_density:     min={df['url_density'].min():.4f}, max={df['url_density'].max():.4f}, mean={df['url_density'].mean():.4f}"
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
        # Engagement (log transformed)
        "replies_log",
        "retweets_log",
        "likes_log",
        "engagement_log",
        "engagement_log_sum",
        # Engagement (percentile rank)
        "replies_pct",
        "retweets_pct",
        "likes_pct",
        "engagement_pct",
        "engagement_pct_mean",
        # Engagement ratios
        "reply_ratio",
        "retweet_ratio",
        "like_ratio",
        # Virality ratios
        "retweet_to_like_ratio",
        "reply_to_like_ratio",
        # Media/Links (counts)
        "url_count",
        "photo_count",
        "hashtag_count",
        # Content density
        "hashtag_density",
        "url_density",
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
    print("\n[8] Saving dataset...")
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "tweets_v3_normalized.parquet"
    df.write_parquet(output_path)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Step 9: Log to MLflow
    # =========================================================================
    print("\n[9] Logging to MLflow...")
    mlflow.set_experiment("dataset_processing")

    with mlflow.start_run(run_name="tweets_v3_normalized"):
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
    print("NORMALIZATION SUMMARY")
    print("=" * 60)
    print(f"Input (from Phase 2): {len(df):,} rows")
    print(f"Output columns:       {len(df.columns)}")
    print(f"Output: {output_path}")

    # Print new features
    print("\n" + "=" * 60)
    print("NEW FEATURES ADDED")
    print("=" * 60)
    new_features = [
        "replies_log",
        "retweets_log",
        "likes_log",
        "engagement_log",
        "engagement_log_sum",
        "replies_pct",
        "retweets_pct",
        "likes_pct",
        "engagement_pct",
        "engagement_pct_mean",
        "reply_ratio",
        "retweet_ratio",
        "like_ratio",
        "retweet_to_like_ratio",
        "reply_to_like_ratio",
        "hashtag_density",
        "url_density",
    ]
    for feat in new_features:
        print(f"  {feat}: {df[feat].dtype}")


if __name__ == "__main__":
    main()
