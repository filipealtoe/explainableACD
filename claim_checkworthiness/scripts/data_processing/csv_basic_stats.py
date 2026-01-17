"""Get basic statistics from large CSV using Polars lazy evaluation."""

from pathlib import Path

import polars as pl

INPUT_CSV = Path("data/raw/ioa_tweets.csv")


def get_basic_stats() -> None:
    """Extract basic statistics using streaming/lazy evaluation."""
    print(f"Analyzing {INPUT_CSV}...")
    print(f"File size: {INPUT_CSV.stat().st_size / (1024**3):.2f} GB\n")

    # Lazy scan - doesn't load into memory
    lf = pl.scan_csv(
        INPUT_CSV,
        try_parse_dates=True,
        infer_schema_length=10000,
        low_memory=True,
    )

    # Get column names
    print("=" * 50)
    print("COLUMNS")
    print("=" * 50)
    schema = lf.collect_schema()
    for col, dtype in schema.items():
        print(f"  {col}: {dtype}")

    # Basic counts and date range
    print("\n" + "=" * 50)
    print("BASIC METRICS")
    print("=" * 50)

    basic_stats = lf.select(
        pl.len().alias("total_rows"),
        pl.col("tweet_time").min().alias("earliest_tweet"),
        pl.col("tweet_time").max().alias("latest_tweet"),
        pl.col("userid").n_unique().alias("unique_users"),
        pl.col("is_retweet").sum().alias("retweet_count"),
    ).collect(streaming=True)

    print(basic_stats)

    # Engagement statistics
    print("\n" + "=" * 50)
    print("ENGAGEMENT STATS")
    print("=" * 50)

    engagement_stats = lf.select(
        pl.col("like_count").sum().alias("total_likes"),
        pl.col("like_count").mean().alias("avg_likes"),
        pl.col("like_count").max().alias("max_likes"),
        pl.col("retweet_count").sum().alias("total_retweets"),
        pl.col("retweet_count").mean().alias("avg_retweets"),
        pl.col("retweet_count").max().alias("max_retweets"),
        pl.col("reply_count").sum().alias("total_replies"),
        pl.col("reply_count").mean().alias("avg_replies"),
        pl.col("quote_count").sum().alias("total_quotes"),
        pl.col("quote_count").mean().alias("avg_quotes"),
    ).collect(streaming=True)

    print(engagement_stats)

    # Language distribution (top 10)
    print("\n" + "=" * 50)
    print("TOP 10 TWEET LANGUAGES")
    print("=" * 50)

    lang_dist = (
        lf.group_by("tweet_language")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
        .collect(streaming=True)
    )

    print(lang_dist)

    # Tweets per year
    print("\n" + "=" * 50)
    print("TWEETS PER YEAR")
    print("=" * 50)

    yearly = (
        lf.with_columns(pl.col("tweet_time").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.len().alias("count"))
        .sort("year")
        .collect(streaming=True)
    )

    print(yearly)

    # Top tweet clients
    print("\n" + "=" * 50)
    print("TOP 10 TWEET CLIENTS")
    print("=" * 50)

    clients = (
        lf.group_by("tweet_client_name")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
        .collect(streaming=True)
    )

    print(clients)

    # Retweet vs Original ratio
    print("\n" + "=" * 50)
    print("RETWEET VS ORIGINAL")
    print("=" * 50)

    rt_ratio = (
        lf.group_by("is_retweet")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum() * 100).alias("percentage"))
        .collect(streaming=True)
    )

    print(rt_ratio)


if __name__ == "__main__":
    get_basic_stats()
