from pathlib import Path

import polars as pl

# Path setup
repo_root = Path(__file__).resolve().parents[2]
input_path = repo_root / "data" / "processed" / "tweets_v5_cleaned.parquet"


def main():
    print("Loading data...")
    df = pl.read_parquet(input_path)

    # Identify single word tweets
    df = df.with_columns(pl.col("tweet").str.split(" ").list.len().alias("word_count_temp"))
    single_word_tweets = df.filter(pl.col("word_count_temp") == 1)

    print(f"\nFound {len(single_word_tweets)} single-word tweets.")
    print("Displaying all cases:")
    print("-" * 80)

    # Using iter_rows to print full strings without truncation
    for row in single_word_tweets.select(["tweet_original", "tweet"]).iter_rows(named=True):
        print(f"ORIG:  {row['tweet_original']}")
        print(f"CLEAN: {row['tweet']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
