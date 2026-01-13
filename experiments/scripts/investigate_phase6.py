import re
from pathlib import Path

import polars as pl

# Path setup
repo_root = Path(__file__).resolve().parents[2]
input_path = repo_root / "data" / "processed" / "tweets_v5_cleaned.parquet"


def check_is_emoji_only(text: str) -> bool:
    """Check if text contains NO alphanumeric characters."""
    if not isinstance(text, str):
        return True  # Empty or null is treated as "no alphanumeric"
    return not bool(re.search(r"[a-zA-Z0-9]", text))


def main():
    print("Loading data...")
    df = pl.read_parquet(input_path)

    # Create temporary analysis columns
    df = df.with_columns(
        [
            pl.col("tweet").map_elements(check_is_emoji_only, return_dtype=pl.Boolean).alias("is_emoji_only"),
            pl.col("tweet").str.len_chars().alias("len_cleaned"),
        ]
    )

    # 1. Emoji Only / No Alphanumeric
    emoji_only = df.filter(pl.col("is_emoji_only"))
    print("\n" + "=" * 60)
    print(f"NO ALPHANUMERIC CHARS (Emoji only / Symbols): {len(emoji_only):,} rows")
    print("=" * 60)
    print(emoji_only.select(["tweet_original", "tweet"]).head(20))

    # 2. Extremely Short Tweets (< 5 chars) but NOT emoji only
    short_tweets = df.filter((pl.col("len_cleaned") < 5) & (~pl.col("is_emoji_only")))
    print("\n" + "=" * 60)
    print(f"EXTREMELY SHORT TWEETS (< 5 chars, with text): {len(short_tweets):,} rows")
    print("=" * 60)
    print(short_tweets.select(["tweet_original", "tweet"]).head(20))

    # 3. Empty Tweets
    empty_tweets = df.filter(pl.col("len_cleaned") == 0)
    print("\n" + "=" * 60)
    print(f"EMPTY TWEETS (after cleaning): {len(empty_tweets):,} rows")
    print("=" * 60)
    if len(empty_tweets) > 0:
        print(empty_tweets.select(["tweet_original", "tweet"]).head(20))
    else:
        print("None found.")


if __name__ == "__main__":
    main()
