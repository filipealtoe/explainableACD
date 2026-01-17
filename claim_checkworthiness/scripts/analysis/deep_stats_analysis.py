"""Deep statistical analysis of the IOA tweets dataset - memory optimized."""

from pathlib import Path

import polars as pl

PARQUET_PATH = Path("data/raw/ioa_tweets.parquet")


def run_query(name: str, lf: pl.LazyFrame) -> pl.DataFrame:
    """Run a lazy query and print result."""
    print(f"\n{name}")
    result = lf.collect()
    print(result)
    return result


def main() -> None:
    """Run comprehensive statistical analysis."""
    print("=" * 60)
    print("DEEP STATISTICAL ANALYSIS - IOA TWEETS DATASET")
    print("=" * 60)

    # 1. RECORD COUNT VERIFICATION
    print("\n" + "=" * 60)
    print("1. RECORD COUNT VERIFICATION")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    count_result = lf.select(pl.len().alias("total_records")).collect()
    count = count_result.item()
    print(f"Total records in parquet: {count:,}")

    # 2. SCHEMA
    print("\n" + "=" * 60)
    print("2. SCHEMA")
    print("=" * 60)
    schema = lf.collect_schema()
    for col, dtype in schema.items():
        print(f"  {col}: {dtype}")

    # 3. MISSING VALUES
    print("\n" + "=" * 60)
    print("3. MISSING VALUES ANALYSIS")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    null_counts = lf.select([pl.col(col).null_count().alias(col) for col in schema.names()]).collect()

    print(f"{'Column':<30} {'Nulls':>15} {'% Missing':>12}")
    print("-" * 57)
    for col in schema.names():
        null_count = null_counts[col][0]
        pct = (null_count / count) * 100
        if null_count > 0:
            print(f"{col:<30} {null_count:>15,} {pct:>11.2f}%")

    # 4. TIME RANGE
    print("\n" + "=" * 60)
    print("4. TIME RANGE")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    time_stats = lf.select(
        pl.col("tweet_time").min().alias("earliest"),
        pl.col("tweet_time").max().alias("latest"),
    ).collect()
    print(f"Earliest tweet: {time_stats['earliest'][0]}")
    print(f"Latest tweet: {time_stats['latest'][0]}")

    # 5. YEARLY DISTRIBUTION
    print("\n" + "=" * 60)
    print("5. YEARLY DISTRIBUTION")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    yearly = (
        lf.with_columns(pl.col("tweet_time").dt.year().alias("year"))
        .group_by("year")
        .agg(
            [
                pl.len().alias("tweets"),
                pl.col("userid").n_unique().alias("active_users"),
                pl.col("is_retweet").mean().alias("rt_ratio"),
            ]
        )
        .sort("year")
        .collect()
    )
    print(yearly)

    # 6. TOP MONTHS
    print("\n" + "=" * 60)
    print("6. TOP 20 MONTHS BY VOLUME")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    monthly = (
        lf.with_columns(pl.col("tweet_time").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(20)
        .collect()
    )
    print(monthly)

    # 7. USER STATISTICS
    print("\n" + "=" * 60)
    print("7. USER STATISTICS")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    user_count = lf.select(pl.col("userid").n_unique()).collect().item()
    print(f"Unique users: {user_count:,}")
    print(f"Avg tweets/user: {count / user_count:,.1f}")

    # Top users
    print("\nTop 20 most active users:")
    lf = pl.scan_parquet(PARQUET_PATH)
    top_users = (
        lf.group_by("userid")
        .agg(
            [
                pl.len().alias("tweets"),
                pl.col("is_retweet").sum().alias("retweets"),
                pl.col("follower_count").max().alias("followers"),
            ]
        )
        .sort("tweets", descending=True)
        .head(20)
        .collect()
    )
    print(top_users)

    # User activity quantiles
    print("\nUser activity distribution:")
    lf = pl.scan_parquet(PARQUET_PATH)
    user_dist = (
        lf.group_by("userid")
        .agg(pl.len().alias("tweets"))
        .select(
            [
                pl.col("tweets").min().alias("min"),
                pl.col("tweets").quantile(0.25).alias("p25"),
                pl.col("tweets").median().alias("median"),
                pl.col("tweets").quantile(0.75).alias("p75"),
                pl.col("tweets").quantile(0.95).alias("p95"),
                pl.col("tweets").quantile(0.99).alias("p99"),
                pl.col("tweets").max().alias("max"),
            ]
        )
        .collect()
    )
    print(user_dist)

    # 8. ENGAGEMENT
    print("\n" + "=" * 60)
    print("8. ENGAGEMENT STATISTICS")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    engagement = lf.select(
        [
            pl.col("like_count").sum().alias("total_likes"),
            pl.col("like_count").mean().alias("avg_likes"),
            pl.col("like_count").median().alias("med_likes"),
            pl.col("like_count").max().alias("max_likes"),
            (pl.col("like_count") > 0).sum().alias("with_likes"),
            pl.col("retweet_count").sum().alias("total_rt"),
            pl.col("retweet_count").mean().alias("avg_rt"),
            pl.col("retweet_count").max().alias("max_rt"),
            pl.col("reply_count").sum().alias("total_replies"),
            pl.col("reply_count").mean().alias("avg_replies"),
            pl.col("quote_count").sum().alias("total_quotes"),
        ]
    ).collect()

    print(
        f"Likes - Total: {engagement['total_likes'][0]:,.0f}, Avg: {engagement['avg_likes'][0]:.2f}, Med: {engagement['med_likes'][0]:.0f}, Max: {engagement['max_likes'][0]:,.0f}"
    )
    print(
        f"Retweets - Total: {engagement['total_rt'][0]:,.0f}, Avg: {engagement['avg_rt'][0]:.2f}, Max: {engagement['max_rt'][0]:,.0f}"
    )
    print(f"Replies - Total: {engagement['total_replies'][0]:,.0f}, Avg: {engagement['avg_replies'][0]:.2f}")
    print(f"Quotes - Total: {engagement['total_quotes'][0]:,.0f}")
    print(f"Tweets with likes: {engagement['with_likes'][0]:,} ({engagement['with_likes'][0] / count * 100:.1f}%)")

    # 9. LANGUAGE
    print("\n" + "=" * 60)
    print("9. LANGUAGE DISTRIBUTION")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    langs = (
        lf.group_by("tweet_language")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("like_count").mean().alias("avg_likes"),
            ]
        )
        .with_columns((pl.col("count") / count * 100).alias("pct"))
        .sort("count", descending=True)
        .head(15)
        .collect()
    )
    print(langs)

    # 10. RETWEET ANALYSIS
    print("\n" + "=" * 60)
    print("10. RETWEET VS ORIGINAL")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    rt_stats = (
        lf.group_by("is_retweet")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("like_count").mean().alias("avg_likes"),
            ]
        )
        .with_columns((pl.col("count") / count * 100).alias("pct"))
        .collect()
    )
    print(rt_stats)

    # Top RT sources
    print("\nTop 20 most retweeted accounts:")
    lf = pl.scan_parquet(PARQUET_PATH)
    rt_sources = (
        lf.filter(pl.col("is_retweet") == True)
        .group_by("retweet_userid")
        .agg(pl.len().alias("times_rt"))
        .sort("times_rt", descending=True)
        .head(20)
        .collect()
    )
    print(rt_sources)

    # 11. CLIENT ANALYSIS
    print("\n" + "=" * 60)
    print("11. TWEET CLIENT SOURCES")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    clients = (
        lf.group_by("tweet_client_name")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("is_retweet").mean().alias("rt_ratio"),
            ]
        )
        .with_columns((pl.col("count") / count * 100).alias("pct"))
        .sort("count", descending=True)
        .head(20)
        .collect()
    )
    print(clients)

    # 12. ACCOUNT CREATION
    print("\n" + "=" * 60)
    print("12. ACCOUNT CREATION YEARS")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    creation = (
        lf.with_columns(pl.col("account_creation_date").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("userid").n_unique().alias("accounts"))
        .sort("year")
        .collect()
    )
    print(creation)

    # 13. CONTENT FEATURES
    print("\n" + "=" * 60)
    print("13. CONTENT FEATURES")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    content = lf.select(
        [
            (pl.col("hashtags") != "[]").sum().alias("with_hashtags"),
            (pl.col("urls") != "[]").sum().alias("with_urls"),
            (pl.col("user_mentions") != "[]").sum().alias("with_mentions"),
            pl.col("in_reply_to_userid").is_not_null().sum().alias("replies"),
        ]
    ).collect()

    print(f"Tweets with hashtags: {content['with_hashtags'][0]:,} ({content['with_hashtags'][0] / count * 100:.1f}%)")
    print(f"Tweets with URLs: {content['with_urls'][0]:,} ({content['with_urls'][0] / count * 100:.1f}%)")
    print(f"Tweets with mentions: {content['with_mentions'][0]:,} ({content['with_mentions'][0] / count * 100:.1f}%)")
    print(f"Reply tweets: {content['replies'][0]:,} ({content['replies'][0] / count * 100:.1f}%)")

    # 14. TEMPORAL PATTERNS
    print("\n" + "=" * 60)
    print("14. HOUR OF DAY (UTC)")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    hourly = (
        lf.with_columns(pl.col("tweet_time").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(pl.len().alias("count"))
        .sort("hour")
        .collect()
    )
    print(hourly)

    # 15. DAY OF WEEK
    print("\n" + "=" * 60)
    print("15. DAY OF WEEK")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    daily = (
        lf.with_columns(pl.col("tweet_time").dt.weekday().alias("weekday"))
        .group_by("weekday")
        .agg(pl.len().alias("count"))
        .sort("weekday")
        .collect()
    )
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for row in daily.iter_rows():
        print(f"  {day_names[row[0]]}: {row[1]:,}")

    # 16. FOLLOWER STATS
    print("\n" + "=" * 60)
    print("16. FOLLOWER/FOLLOWING STATS (per user)")
    print("=" * 60)

    lf = pl.scan_parquet(PARQUET_PATH)
    follower_stats = (
        lf.group_by("userid")
        .agg(
            [
                pl.col("follower_count").max().alias("followers"),
                pl.col("following_count").max().alias("following"),
            ]
        )
        .select(
            [
                pl.col("followers").mean().alias("avg_followers"),
                pl.col("followers").median().alias("med_followers"),
                pl.col("followers").max().alias("max_followers"),
                pl.col("following").mean().alias("avg_following"),
                pl.col("following").median().alias("med_following"),
            ]
        )
        .collect()
    )
    print(follower_stats)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
