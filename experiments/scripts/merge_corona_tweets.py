"""
Merge coronavirus tweet JSON files into a single parquet file.
Extracts all relevant fields including nested engagement metrics.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import polars as pl

INPUT_DIR = Path("data/raw/corona_virus")
OUTPUT_PARQUET = Path("data/raw/corona_tweets.parquet")


def parse_twitter_date(date_str: str) -> datetime | None:
    """Parse Twitter's date format."""
    try:
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
    except (ValueError, TypeError):
        return None


def extract_tweet_data(tweet: dict) -> dict:
    """Extract all relevant fields from a tweet JSON object."""

    # Get full text (from extended_tweet if available, otherwise text)
    if tweet.get("extended_tweet"):
        full_text = tweet["extended_tweet"].get("full_text", tweet.get("text", ""))
    else:
        full_text = tweet.get("text", "")

    # Get user info
    user = tweet.get("user", {})

    # Get hashtags
    entities = tweet.get("entities", {})
    if tweet.get("extended_tweet", {}).get("entities"):
        entities = tweet["extended_tweet"]["entities"]
    hashtags = [h["text"] for h in entities.get("hashtags", [])]
    urls = [u["expanded_url"] for u in entities.get("urls", []) if u.get("expanded_url")]
    mentions = [m["screen_name"] for m in entities.get("user_mentions", [])]

    # Check if retweet
    is_retweet = "retweeted_status" in tweet

    # Get retweet info and ALL engagement from original tweet
    rt_user_id = None
    rt_tweet_id = None
    rt_retweet_count = None
    rt_favorite_count = None
    rt_reply_count = None
    rt_quote_count = None
    rt_text = None
    if is_retweet:
        rt_status = tweet.get("retweeted_status", {})
        rt_user = rt_status.get("user", {})
        rt_user_id = rt_user.get("id_str")
        rt_tweet_id = rt_status.get("id_str")
        rt_retweet_count = rt_status.get("retweet_count")
        rt_favorite_count = rt_status.get("favorite_count")
        rt_reply_count = rt_status.get("reply_count")
        rt_quote_count = rt_status.get("quote_count")
        # Get full text from original
        if rt_status.get("extended_tweet"):
            rt_text = rt_status["extended_tweet"].get("full_text", rt_status.get("text", ""))
        else:
            rt_text = rt_status.get("text", "")

    # Get quote info and ALL engagement from quoted tweet
    is_quote = tweet.get("is_quote_status", False)
    quoted_tweet_id = tweet.get("quoted_status_id_str")
    quoted_retweet_count = None
    quoted_favorite_count = None
    quoted_reply_count = None
    quoted_quote_count = None
    quoted_text = None
    if tweet.get("quoted_status"):
        qs = tweet.get("quoted_status", {})
        quoted_retweet_count = qs.get("retweet_count")
        quoted_favorite_count = qs.get("favorite_count")
        quoted_reply_count = qs.get("reply_count")
        quoted_quote_count = qs.get("quote_count")
        if qs.get("extended_tweet"):
            quoted_text = qs["extended_tweet"].get("full_text", qs.get("text", ""))
        else:
            quoted_text = qs.get("text", "")

    return {
        # Tweet identifiers
        "tweet_id": tweet.get("id_str"),
        "tweet_created_at": parse_twitter_date(tweet.get("created_at")),
        "timestamp_ms": tweet.get("timestamp_ms"),
        # User info
        "user_id": user.get("id_str"),
        "user_name": user.get("name"),
        "user_screen_name": user.get("screen_name"),
        "user_location": user.get("location"),
        "user_description": user.get("description"),
        "user_url": user.get("url"),
        "user_followers_count": user.get("followers_count"),
        "user_friends_count": user.get("friends_count"),
        "user_listed_count": user.get("listed_count"),
        "user_favourites_count": user.get("favourites_count"),
        "user_statuses_count": user.get("statuses_count"),
        "user_verified": user.get("verified"),
        "user_protected": user.get("protected"),
        "user_created_at": parse_twitter_date(user.get("created_at")),
        # Tweet content
        "tweet_text": full_text,
        "tweet_source": tweet.get("source"),
        "tweet_lang": tweet.get("lang"),
        "tweet_truncated": tweet.get("truncated"),
        # Tweet engagement (always 0 for streamed tweets)
        "retweet_count": tweet.get("retweet_count"),
        "favorite_count": tweet.get("favorite_count"),
        "reply_count": tweet.get("reply_count"),
        "quote_count": tweet.get("quote_count"),
        # Retweet info
        "is_retweet": is_retweet,
        "rt_tweet_id": rt_tweet_id,
        "rt_user_id": rt_user_id,
        "rt_text": rt_text,
        "rt_retweet_count": rt_retweet_count,
        "rt_favorite_count": rt_favorite_count,
        "rt_reply_count": rt_reply_count,
        "rt_quote_count": rt_quote_count,
        # Quote tweet info
        "is_quote": is_quote,
        "quoted_tweet_id": quoted_tweet_id,
        "quoted_text": quoted_text,
        "quoted_retweet_count": quoted_retweet_count,
        "quoted_favorite_count": quoted_favorite_count,
        "quoted_reply_count": quoted_reply_count,
        "quoted_quote_count": quoted_quote_count,
        # Reply info
        "in_reply_to_status_id": tweet.get("in_reply_to_status_id_str"),
        "in_reply_to_user_id": tweet.get("in_reply_to_user_id_str"),
        "in_reply_to_screen_name": tweet.get("in_reply_to_screen_name"),
        # Entities
        "hashtags": "|".join(hashtags) if hashtags else None,
        "urls": "|".join(urls) if urls else None,
        "user_mentions": "|".join(mentions) if mentions else None,
        # Location
        "coordinates": str(tweet.get("coordinates")) if tweet.get("coordinates") else None,
        "place": str(tweet.get("place")) if tweet.get("place") else None,
        "geo": str(tweet.get("geo")) if tweet.get("geo") else None,
    }


def merge_files() -> None:
    """Merge all JSON files into a single parquet."""

    start_time = time.time()
    files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"Found {len(files)} files to process")

    all_tweets = []
    errors = 0

    for file_path in files:
        file_start = time.time()
        file_tweets = 0
        file_errors = 0

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tweet = json.loads(line)
                    extracted = extract_tweet_data(tweet)
                    all_tweets.append(extracted)
                    file_tweets += 1
                except (json.JSONDecodeError, Exception):
                    file_errors += 1
                    errors += 1

        file_elapsed = time.time() - file_start
        print(f"  {file_path.name}: {file_tweets:,} tweets, {file_errors} errors ({file_elapsed:.1f}s)")

    print(f"\nTotal tweets extracted: {len(all_tweets):,}")
    print(f"Total errors: {errors:,}")

    print("\nConverting to DataFrame...")
    schema = {
        "tweet_id": pl.String,
        "tweet_created_at": pl.Datetime,
        "timestamp_ms": pl.String,
        "user_id": pl.String,
        "user_name": pl.String,
        "user_screen_name": pl.String,
        "user_location": pl.String,
        "user_description": pl.String,
        "user_url": pl.String,
        "user_followers_count": pl.Int64,
        "user_friends_count": pl.Int64,
        "user_listed_count": pl.Int64,
        "user_favourites_count": pl.Int64,
        "user_statuses_count": pl.Int64,
        "user_verified": pl.Boolean,
        "user_protected": pl.Boolean,
        "user_created_at": pl.Datetime,
        "tweet_text": pl.String,
        "tweet_source": pl.String,
        "tweet_lang": pl.String,
        "tweet_truncated": pl.Boolean,
        "retweet_count": pl.Int64,
        "favorite_count": pl.Int64,
        "reply_count": pl.Int64,
        "quote_count": pl.Int64,
        "is_retweet": pl.Boolean,
        "rt_tweet_id": pl.String,
        "rt_user_id": pl.String,
        "rt_text": pl.String,
        "rt_retweet_count": pl.Int64,
        "rt_favorite_count": pl.Int64,
        "rt_reply_count": pl.Int64,
        "rt_quote_count": pl.Int64,
        "is_quote": pl.Boolean,
        "quoted_tweet_id": pl.String,
        "quoted_text": pl.String,
        "quoted_retweet_count": pl.Int64,
        "quoted_favorite_count": pl.Int64,
        "quoted_reply_count": pl.Int64,
        "quoted_quote_count": pl.Int64,
        "in_reply_to_status_id": pl.String,
        "in_reply_to_user_id": pl.String,
        "in_reply_to_screen_name": pl.String,
        "hashtags": pl.String,
        "urls": pl.String,
        "user_mentions": pl.String,
        "coordinates": pl.String,
        "place": pl.String,
        "geo": pl.String,
    }
    df = pl.DataFrame(all_tweets, schema=schema)

    print(f"\nSchema ({len(df.schema)} columns):")
    for col, dtype in df.schema.items():
        print(f"  {col}: {dtype}")

    print(f"\nWriting to {OUTPUT_PARQUET}...")
    df.write_parquet(OUTPUT_PARQUET, compression="zstd", compression_level=3)

    output_size = OUTPUT_PARQUET.stat().st_size / (1024**2)
    elapsed = time.time() - start_time

    print("\nDone!")
    print(f"Output size: {output_size:.1f} MB")
    print(f"Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    merge_files()
