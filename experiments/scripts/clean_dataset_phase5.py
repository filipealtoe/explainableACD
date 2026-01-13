"""
Phase 5: Text Cleaning

Clean tweet text while preserving the original. Create a new tweet column with cleaned text from tweet_original.

Input: data/processed/tweets_v4_features.parquet (from Phase 4)
Output: data/processed/tweets_v5_cleaned.parquet

Operations:
1. Create new tweet column from tweet_original
2. HTML entity decode: html.unescape()
3. Fix mojibake: ftfy.fix_text()
4. NFKC normalization: unicodedata.normalize("NFKC", text)
5. Remove URLs: re.sub(r"https?://\\S+", "", text)
6. Replace @mentions: re.sub(r"(^|\\s)@\\w+", r"\1@USER", text)
7. Clean repeated chars: re.sub(r"(.)\1{2,}", r"\1\1", text)
8. Clean repeated punct: re.sub(r"([!?.])\1+", r"\1", text)
9. Remove # from hashtags: re.sub(r"#(\\w+)", r"\1", text)
10. Normalize whitespace: re.sub(r"\\s+", " ", text).strip()
"""

import html
import re
import sys
import unicodedata
from pathlib import Path

import ftfy
import mlflow
import polars as pl

# Path setup
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def clean_tweet(text: str) -> str:
    """Apply all cleaning steps to a tweet."""
    if not isinstance(text, str):
        return ""

    # Step 2: HTML decode
    text = html.unescape(text)

    # Step 3: Fix mojibake
    text = ftfy.fix_text(text)

    # Step 4: NFKC normalize
    text = unicodedata.normalize("NFKC", text)

    # Step 5: Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Step 6: Replace @mentions
    # Using (^|\s) to capture start of string or preceding space
    text = re.sub(r"(^|\s)@\w+", r"\1@USER", text)

    # Step 7: Clean repeated chars (keep max 2)
    # e.g., loooove -> loove
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Step 8: Clean repeated punctuation
    # e.g., !!! -> !
    text = re.sub(r"([!?.])\1+", r"\1", text)

    # Step 9: Remove # from hashtags (keep the word)
    # e.g., #AI -> AI
    text = re.sub(r"#(\w+)", r"\1", text)

    # Step 10: Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print("=" * 60)
    print("PHASE 5: TEXT CLEANING")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Phase 4 Output
    # =========================================================================
    print("\n[1] Loading Phase 4 output...")
    input_path = repo_root / "data" / "processed" / "tweets_v4_features.parquet"
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    df = pl.read_parquet(input_path)
    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")

    # =========================================================================
    # Step 2: Apply Text Cleaning
    # =========================================================================
    print("\n[2] Applying text cleaning pipeline...")

    # Apply cleaning to create 'tweet' column
    df = df.with_columns(pl.col("tweet_original").map_elements(clean_tweet, return_dtype=pl.String).alias("tweet"))

    # =========================================================================
    # Step 3: Calculate Metrics
    # =========================================================================
    print("\n[3] Calculating cleaning metrics...")

    # Calculate lengths
    df = df.with_columns(
        [
            pl.col("tweet_original").str.len_chars().alias("len_original"),
            pl.col("tweet").str.len_chars().alias("len_cleaned"),
        ]
    )

    avg_len_before = df["len_original"].mean()
    avg_len_after = df["len_cleaned"].mean()
    avg_reduction = avg_len_before - avg_len_after

    # Calculate percentage of tweets modified
    modified_count = df.filter(pl.col("tweet_original") != pl.col("tweet")).shape[0]
    pct_modified = 100 * modified_count / len(df)

    print(f"    Avg length before: {avg_len_before:.2f}")
    print(f"    Avg length after:  {avg_len_after:.2f}")
    print(f"    Avg reduction:     {avg_reduction:.2f} chars")
    print(f"    Tweets modified:   {modified_count:,} ({pct_modified:.1f}%)")

    # Drop temporary length columns if not needed in final output,
    # but the plan didn't say to drop them. The plan said "Output Columns Added: tweet".
    # It also listed "Metrics to Log" but not "Columns to keep".
    # Usually we want to keep the cleaned dataset clean.
    # I'll drop the temp length columns to match "Total columns: 65 (64 + 1)".
    df = df.drop(["len_original", "len_cleaned"])

    # =========================================================================
    # Step 4: Save Output
    # =========================================================================
    print("\n[4] Saving dataset...")
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "tweets_v5_cleaned.parquet"
    df.write_parquet(output_path)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Step 5: Log to MLflow
    # =========================================================================
    print("\n[5] Logging to MLflow...")
    mlflow.set_experiment("dataset_processing")

    with mlflow.start_run(run_name="tweets_v5_cleaned"):
        mlflow.log_params(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "row_count": len(df),
                "input_columns": len(df.columns) - 1,  # -1 for 'tweet'
                "final_columns": len(df.columns),
                "cleaning_steps": "html_decode,fix_mojibake,nfkc,remove_urls,anonymize_mentions,clean_chars,clean_punct,remove_hashtags,normalize_whitespace",
            }
        )

        mlflow.log_metrics(
            {
                "avg_length_before": avg_len_before,
                "avg_length_after": avg_len_after,
                "avg_length_reduction": avg_reduction,
                "pct_tweets_modified": pct_modified,
                "final_row_count": len(df),
            }
        )

        mlflow.log_artifact(str(output_path))

        run_id = mlflow.active_run().info.run_id
        print(f"    MLflow Run ID: {run_id}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Input:  {len(df):,} rows")
    print(f"Output: {output_path}")
    print(f"Modified: {pct_modified:.1f}% of tweets")
    print("-" * 60)

    # Show a few examples of changes
    print("Examples of changes:")
    changed_df = df.filter(pl.col("tweet_original") != pl.col("tweet")).head(5)
    for row in changed_df.iter_rows(named=True):
        print(f"ORIG: {row['tweet_original']}")
        print(f"CLEAN: {row['tweet']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
