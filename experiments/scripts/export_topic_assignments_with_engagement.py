import sys
from pathlib import Path

import polars as pl

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.models.streaming_bertopic import StreamingBERTopicPipeline


def main() -> None:
    print("Loading pipeline state...")
    output_dir = repo_root / "experiments" / ".cache"

    pipeline = StreamingBERTopicPipeline(output_dir=str(output_dir))
    pipeline.load_state()

    print(f"Loaded {len(pipeline.doc_index)} document assignments")

    # Create tweet-to-topic mapping with integer IDs
    print("\nCreating tweet-topic mapping...")
    assignments_df = pl.DataFrame(
        {
            "tweet_id": [int(tid) for tid in pipeline.doc_index],
            "topic": pipeline.topic_assignments,
            "timestamp": pipeline.timestamps,
        }
    )

    print(f"Mapping created: {len(assignments_df)} tweets")

    # Load raw tweets with engagement data
    print("\nLoading raw tweets with engagement metrics...")
    tweets_path = repo_root / "data" / "raw" / "tweets_ai.parquet"
    tweets = pl.read_parquet(tweets_path)

    # Convert ID to int for matching
    tweets_eng = tweets.select(
        [pl.col("id").cast(pl.Int64).alias("tweet_id"), "replies_count", "retweets_count", "likes_count"]
    )

    # Join assignments with engagement data
    print("\nJoining topic assignments with engagement data...")
    result = assignments_df.join(tweets_eng, on="tweet_id", how="left")

    # Show join statistics
    matched = result.filter(pl.col("replies_count").is_not_null()).height
    print(f"  Matched {matched}/{len(result)} tweets with engagement data")

    # Fill nulls with 0 for engagement metrics
    result = result.with_columns(
        [
            pl.col("replies_count").fill_null(0),
            pl.col("retweets_count").fill_null(0),
            pl.col("likes_count").fill_null(0),
        ]
    )

    # Calculate total engagement
    result = result.with_columns(
        (pl.col("replies_count") + pl.col("retweets_count") + pl.col("likes_count")).alias("total_engagement")
    )

    # Save full dataset
    print("\nSaving results...")
    output_path = repo_root / "data" / "tweet_topic_assignments_with_engagement.csv"
    result.write_csv(output_path)

    print(f"\n✓ Saved to: {output_path}")
    print(f"  Total tweets: {len(result)}")
    print(f"  Columns: {result.columns}")

    # Show summary stats
    print("\nEngagement statistics:")
    print(
        result.select(
            [
                pl.col("replies_count").mean().alias("avg_replies"),
                pl.col("retweets_count").mean().alias("avg_retweets"),
                pl.col("likes_count").mean().alias("avg_likes"),
                pl.col("total_engagement").mean().alias("avg_total_engagement"),
            ]
        )
    )

    print("\nTop 5 topics by total engagement:")
    topic_engagement = (
        result.group_by("topic")
        .agg([pl.col("total_engagement").sum().alias("total_eng"), pl.count().alias("tweet_count")])
        .sort("total_eng", descending=True)
        .head(5)
    )
    print(topic_engagement)


if __name__ == "__main__":
    main()
