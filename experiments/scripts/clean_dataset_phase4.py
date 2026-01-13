"""
Phase 4: Text Pattern Feature Extraction

Reads Phase 3 output and adds text pattern features extracted from raw tweet text:
- Mention features: mention_count, is_reply, is_manual_retweet, has_via_attribution
- Punctuation features: has_question, has_exclamation
- Content features: has_emoji, has_all_caps, has_numbers, has_stats
- Quality signals: is_truncated, excessive_hashtags, has_many_urls

Input: data/processed/tweets_v3_normalized.parquet (from Phase 3)
Output: data/processed/tweets_v4_features.parquet
"""

import re
import sys
from pathlib import Path

import emoji
import mlflow
import polars as pl

# Path setup
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def count_emojis(text: str) -> int:
    """Count emojis in text using the emoji library."""
    if not isinstance(text, str):
        return 0
    return emoji.emoji_count(text)


def has_all_caps_check(text: str) -> bool:
    """Check if text contains ALL CAPS words (3+ chars, not common abbreviations)."""
    if not isinstance(text, str):
        return False
    # Find words that are 3+ uppercase letters
    # This excludes common 2-letter abbreviations like AI, ML, US, UK
    caps_words = re.findall(r"\b[A-Z]{3,}\b", text)
    # Filter out common acceptable abbreviations
    common_abbrevs = {"API", "URL", "CEO", "USA", "PDF", "RSS", "IBM", "AWS", "GPU", "CPU", "RAM", "SSD", "NLP", "LLM"}
    for word in caps_words:
        if word not in common_abbrevs:
            return True
    return False


def main():
    print("=" * 60)
    print("PHASE 4: TEXT PATTERN FEATURE EXTRACTION")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Phase 3 Output
    # =========================================================================
    print("\n[1] Loading Phase 3 output...")
    input_path = repo_root / "data" / "processed" / "tweets_v3_normalized.parquet"
    df = pl.read_parquet(input_path)
    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")

    # =========================================================================
    # Step 2: Preserve Original Tweet
    # =========================================================================
    print("\n[2] Preserving original tweet column...")
    df = df.rename({"tweet": "tweet_original"})
    print("    Renamed 'tweet' -> 'tweet_original'")

    # =========================================================================
    # Step 3: Mention Features
    # =========================================================================
    print("\n[3] Creating mention features...")

    df = df.with_columns(
        [
            # Count @mentions
            pl.col("tweet_original").str.count_matches(r"@\w+").alias("mention_count"),
            # Is reply tweet (starts with @)
            pl.col("tweet_original").str.starts_with("@").alias("is_reply_tweet"),
            # Is manual retweet (starts with RT @)
            pl.col("tweet_original").str.contains(r"^RT @").alias("is_manual_retweet"),
            # Has via attribution (contains "via @" - credits source like "via @techcrunch")
            pl.col("tweet_original").str.contains(r"(?i)via @").alias("has_via_attribution"),
        ]
    )

    reply_count = df.filter(pl.col("is_reply_tweet")).shape[0]
    rt_count = df.filter(pl.col("is_manual_retweet")).shape[0]
    via_count = df.filter(pl.col("has_via_attribution")).shape[0]

    print(
        f"    mention_count: min={df['mention_count'].min()}, max={df['mention_count'].max()}, mean={df['mention_count'].mean():.2f}"
    )
    print(f"    is_reply_tweet: {reply_count:,} ({100 * reply_count / len(df):.1f}%)")
    print(f"    is_manual_retweet: {rt_count:,} ({100 * rt_count / len(df):.1f}%)")
    print(f"    has_via_attribution: {via_count:,} ({100 * via_count / len(df):.1f}%)")

    # =========================================================================
    # Step 4: Punctuation Features
    # =========================================================================
    print("\n[4] Creating punctuation features...")

    df = df.with_columns(
        [
            # Has question mark
            pl.col("tweet_original").str.contains(r"\?").alias("has_question"),
            # Has exclamation mark
            pl.col("tweet_original").str.contains(r"!").alias("has_exclamation"),
        ]
    )

    question_count = df.filter(pl.col("has_question")).shape[0]
    exclaim_count = df.filter(pl.col("has_exclamation")).shape[0]

    print(f"    has_question: {question_count:,} ({100 * question_count / len(df):.1f}%)")
    print(f"    has_exclamation: {exclaim_count:,} ({100 * exclaim_count / len(df):.1f}%)")

    # =========================================================================
    # Step 5: Content Features
    # =========================================================================
    print("\n[5] Creating content features...")

    # Emoji count using the emoji library (most comprehensive)
    df = df.with_columns(
        pl.col("tweet_original").map_elements(count_emojis, return_dtype=pl.Int64).alias("emoji_count")
    )

    # Derive has_emoji from emoji_count
    df = df.with_columns((pl.col("emoji_count") > 0).alias("has_emoji"))

    # ALL CAPS detection (3+ chars, excludes common abbreviations)
    df = df.with_columns(
        pl.col("tweet_original").map_elements(has_all_caps_check, return_dtype=pl.Boolean).alias("has_all_caps")
    )

    df = df.with_columns(
        [
            # Has any numbers
            pl.col("tweet_original").str.contains(r"\d").alias("has_numbers"),
            # Has statistics (money symbols followed by digit, or percentage)
            pl.col("tweet_original").str.contains(r"[\$\u20AC\u00A3]\d|\d%").alias("has_stats"),
        ]
    )

    has_emoji_count = df.filter(pl.col("has_emoji")).shape[0]
    caps_count = df.filter(pl.col("has_all_caps")).shape[0]
    numbers_count = df.filter(pl.col("has_numbers")).shape[0]
    stats_count = df.filter(pl.col("has_stats")).shape[0]

    print(
        f"    emoji_count: min={df['emoji_count'].min()}, max={df['emoji_count'].max()}, mean={df['emoji_count'].mean():.2f}"
    )
    print(f"    has_emoji: {has_emoji_count:,} ({100 * has_emoji_count / len(df):.1f}%)")
    print(f"    has_all_caps: {caps_count:,} ({100 * caps_count / len(df):.1f}%)")
    print(f"    has_numbers: {numbers_count:,} ({100 * numbers_count / len(df):.1f}%)")
    print(f"    has_stats: {stats_count:,} ({100 * stats_count / len(df):.1f}%)")

    # =========================================================================
    # Step 6: Quality/Spam Signal Features
    # =========================================================================
    print("\n[6] Creating quality signal features...")

    df = df.with_columns(
        [
            # Is truncated (ends with ellipsis ... or unicode ellipsis)
            pl.col("tweet_original").str.contains(r"\.{3}$|\u2026$").alias("is_truncated"),
            # Excessive hashtags (>5)
            (pl.col("hashtag_count") > 5).alias("excessive_hashtags"),
            # Has many URLs (>2)
            (pl.col("url_count") > 2).alias("has_many_urls"),
        ]
    )

    truncated_count = df.filter(pl.col("is_truncated")).shape[0]
    excess_hash_count = df.filter(pl.col("excessive_hashtags")).shape[0]
    many_urls_count = df.filter(pl.col("has_many_urls")).shape[0]

    print(f"    is_truncated: {truncated_count:,} ({100 * truncated_count / len(df):.1f}%)")
    print(f"    excessive_hashtags: {excess_hash_count:,} ({100 * excess_hash_count / len(df):.1f}%)")
    print(f"    has_many_urls: {many_urls_count:,} ({100 * many_urls_count / len(df):.1f}%)")

    # =========================================================================
    # Step 7: Save Output
    # =========================================================================
    print("\n[7] Saving dataset...")
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "tweets_v4_features.parquet"
    df.write_parquet(output_path)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Step 8: Log to MLflow
    # =========================================================================
    print("\n[8] Logging to MLflow...")
    mlflow.set_experiment("dataset_processing")

    new_features = [
        "mention_count",
        "is_reply_tweet",
        "is_manual_retweet",
        "has_via_attribution",
        "has_question",
        "has_exclamation",
        "emoji_count",
        "has_emoji",
        "has_all_caps",
        "has_numbers",
        "has_stats",
        "is_truncated",
        "excessive_hashtags",
        "has_many_urls",
    ]

    with mlflow.start_run(run_name="tweets_v4_features"):
        mlflow.log_params(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "row_count": len(df),
                "input_columns": len(df.columns) - len(new_features),
                "new_features": len(new_features),
                "final_columns": len(df.columns),
            }
        )

        # Log feature percentages as metrics
        mlflow.log_metrics(
            {
                "final_row_count": len(df),
                "final_column_count": len(df.columns),
                "pct_is_reply_tweet": 100 * reply_count / len(df),
                "pct_is_manual_retweet": 100 * rt_count / len(df),
                "pct_has_via_attribution": 100 * via_count / len(df),
                "pct_has_question": 100 * question_count / len(df),
                "pct_has_exclamation": 100 * exclaim_count / len(df),
                "pct_has_emoji": 100 * has_emoji_count / len(df),
                "pct_has_all_caps": 100 * caps_count / len(df),
                "pct_has_numbers": 100 * numbers_count / len(df),
                "pct_has_stats": 100 * stats_count / len(df),
                "pct_is_truncated": 100 * truncated_count / len(df),
                "pct_excessive_hashtags": 100 * excess_hash_count / len(df),
                "pct_has_many_urls": 100 * many_urls_count / len(df),
                "avg_mention_count": df["mention_count"].mean(),
                "avg_emoji_count": df["emoji_count"].mean(),
                "max_emoji_count": df["emoji_count"].max(),
            }
        )

        mlflow.log_artifact(str(output_path))

        run_id = mlflow.active_run().info.run_id
        print(f"    MLflow Run ID: {run_id}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Input (from Phase 3): {len(df):,} rows")
    print(f"Input columns:        {len(df.columns) - len(new_features)}")
    print(f"New features added:   {len(new_features)}")
    print(f"Output columns:       {len(df.columns)}")
    print(f"Output: {output_path}")

    # Print new features
    print("\n" + "=" * 60)
    print("NEW FEATURES ADDED")
    print("=" * 60)
    for feat in new_features:
        dtype = df[feat].dtype
        if dtype == pl.Boolean:
            count = df.filter(pl.col(feat)).shape[0]
            print(f"  {feat}: {dtype} ({count:,} = {100 * count / len(df):.1f}%)")
        else:
            print(f"  {feat}: {dtype} (mean={df[feat].mean():.2f}, max={df[feat].max()})")


if __name__ == "__main__":
    main()
