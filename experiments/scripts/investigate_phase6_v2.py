import re
from pathlib import Path

import polars as pl

# Path setup
repo_root = Path(__file__).resolve().parents[2]
input_path = repo_root / "data" / "processed" / "tweets_v5_cleaned.parquet"


def check_is_number_only(text: str) -> bool:
    """Check if text contains ONLY digits and whitespace/punctuation."""
    if not isinstance(text, str):
        return False
    # Remove whitespace and punctuation
    cleaned = re.sub(r"[\s\W]", "", text)
    return cleaned.isdigit() and len(cleaned) > 0


def check_is_punct_only(text: str) -> bool:
    """Check if text contains ONLY punctuation."""
    if not isinstance(text, str):
        return False
    # Remove whitespace
    cleaned = re.sub(r"\s", "", text)
    return len(cleaned) > 0 and all(not c.isalnum() for c in cleaned)


def check_is_mention_only(text: str) -> bool:
    """Check if text contains ONLY @USER tokens (and whitespace)."""
    if not isinstance(text, str):
        return False
    # Remove @USER and whitespace
    cleaned = re.sub(r"@USER", "", text)
    cleaned = re.sub(r"\s", "", cleaned)
    return len(cleaned) == 0 and "@USER" in text


def main():
    print("Loading data...")
    df = pl.read_parquet(input_path)

    # Create temporary analysis columns
    df = df.with_columns(
        [
            pl.col("tweet").map_elements(check_is_number_only, return_dtype=pl.Boolean).alias("is_number_only"),
            pl.col("tweet").map_elements(check_is_punct_only, return_dtype=pl.Boolean).alias("is_punct_only"),
            pl.col("tweet").map_elements(check_is_mention_only, return_dtype=pl.Boolean).alias("is_mention_only"),
            pl.col("tweet").str.split(" ").list.len().alias("word_count_temp"),
        ]
    )

    # 1. Numbers Only
    numbers_only = df.filter(pl.col("is_number_only"))
    print("\n" + "=" * 60)
    print(f"NUMBERS ONLY: {len(numbers_only):,} rows")
    print("=" * 60)
    if len(numbers_only) > 0:
        print(numbers_only.select(["tweet_original", "tweet"]).head(20))

    # 2. Punctuation Only
    punct_only = df.filter(pl.col("is_punct_only"))
    print("\n" + "=" * 60)
    print(f"PUNCTUATION ONLY: {len(punct_only):,} rows")
    print("=" * 60)
    if len(punct_only) > 0:
        print(punct_only.select(["tweet_original", "tweet"]).head(20))

    # 3. Mentions Only (e.g., just tagging people)
    mentions_only = df.filter(pl.col("is_mention_only"))
    print("\n" + "=" * 60)
    print(f"MENTIONS ONLY (@USER tokens only): {len(mentions_only):,} rows")
    print("=" * 60)
    if len(mentions_only) > 0:
        print(mentions_only.select(["tweet_original", "tweet"]).head(20))

    # 4. Single Word Tweets
    single_word = df.filter(pl.col("word_count_temp") == 1)
    print("\n" + "=" * 60)
    print(f"SINGLE WORD TWEETS: {len(single_word):,} rows")
    print("=" * 60)
    if len(single_word) > 0:
        print(single_word.select(["tweet_original", "tweet"]).head(20))


if __name__ == "__main__":
    main()
