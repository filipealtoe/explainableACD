"""
Phase 6: Advanced Text Cleaning

Final text processing steps: contractions, demojization, filtering artifacts, and final metadata.

Input: data/processed/tweets_v5_cleaned.parquet (from Phase 5)
Output: data/processed/tweets_v6_text_processed.parquet

Operations:
1. Rename columns: tweet_length -> original_tweet_length, word_count -> original_word_count
2. Expand contractions: contractions.fix() (e.g., can't -> cannot)
3. Demojize emojis: emoji.demojize() (e.g., 😊 -> :smiling_face:)
4. Filter noise: Remove tweets with <= 1 word (after cleaning)
5. Add metadata: cleaned_length, is_emoji_only
"""

import sys
from pathlib import Path

import contractions
import emoji
import mlflow
import polars as pl

# Path setup
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def expand_contractions_text(text: str) -> str:
    """Expand English contractions."""
    if not isinstance(text, str):
        return ""
    try:
        return contractions.fix(text)
    except Exception:
        return text


def demojize_text(text: str) -> str:
    """Convert emojis to text format."""
    if not isinstance(text, str):
        return ""
    return emoji.demojize(text)


def main():
    print("=" * 60)
    print("PHASE 6: ADVANCED TEXT CLEANING")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Phase 5 Output
    # =========================================================================
    print("\n[1] Loading Phase 5 output...")
    input_path = repo_root / "data" / "processed" / "tweets_v5_cleaned.parquet"
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    df = pl.read_parquet(input_path)
    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")

    # =========================================================================
    # Step 2: Rename Columns
    # =========================================================================
    print("\n[2] Renaming original metadata columns...")
    rename_map = {}
    if "tweet_length" in df.columns:
        rename_map["tweet_length"] = "original_tweet_length"
    if "word_count" in df.columns:
        rename_map["word_count"] = "original_word_count"

    if rename_map:
        df = df.rename(rename_map)
        print(f"    Renamed: {rename_map}")
    else:
        print("    No columns to rename found (already renamed?)")

    # =========================================================================
    # Step 3: Expand Contractions
    # =========================================================================
    print("\n[3] Expanding contractions...")
    df = df.with_columns(pl.col("tweet").map_elements(expand_contractions_text, return_dtype=pl.String).alias("tweet"))

    # =========================================================================
    # Step 5: Demojize Emojis
    # =========================================================================
    print("\n[5] Demojizing emojis...")
    df = df.with_columns(pl.col("tweet").map_elements(demojize_text, return_dtype=pl.String).alias("tweet"))

    # =========================================================================
    # Step 6: Filter Noise (Single Word / Empty Tweets)
    # =========================================================================
    print("\n[6] Filtering noise...")

    # Calculate word count of the CLEANED text
    # We use space splitting. Empty strings result in [""], which has len 1, so we handle empty separately.
    df = df.with_columns(
        pl.when(pl.col("tweet").str.len_chars() == 0)
        .then(0)
        .otherwise(pl.col("tweet").str.split(" ").list.len())
        .alias("cleaned_word_count")
    )

    # Identify rows to drop (<= 1 word)
    noise_df = df.filter(pl.col("cleaned_word_count") <= 1)
    noise_count = len(noise_df)

    print(f"    Found {noise_count:,} rows with <= 1 word (noise/artifacts).")

    # Filter them out
    df_filtered = df.filter(pl.col("cleaned_word_count") > 1)
    print(f"    Filtered dataset: {len(df):,} -> {len(df_filtered):,} rows (-{noise_count})")
    df = df_filtered

    # =========================================================================
    # Step 7: Final Metadata
    # =========================================================================
    print("\n[7] Calculating final metadata...")
    df = df.with_columns(pl.col("tweet").str.len_chars().alias("cleaned_length"))

    avg_len = df["cleaned_length"].mean()
    print(f"    Avg cleaned length: {avg_len:.2f}")

    # =========================================================================
    # Step 8: Save Output
    # =========================================================================
    print("\n[8] Saving final dataset...")
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "tweets_v6_text_processed.parquet"
    df.write_parquet(output_path)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Step 9: Log to MLflow
    # =========================================================================
    print("\n[9] Logging to MLflow...")
    mlflow.set_experiment("dataset_processing")

    with mlflow.start_run(run_name="tweets_v6_text_processed"):
        mlflow.log_params(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "initial_rows": len(df) + noise_count,
                "final_rows": len(df),
                "noise_filtered": noise_count,
                "columns": len(df.columns),
                "operations": "rename_cols,expand_contractions,is_emoji_only,demojize,filter_single_word,cleaned_length",
            }
        )

        mlflow.log_metrics(
            {
                "pct_noise_filtered": 100 * noise_count / (len(df) + noise_count),
                "avg_cleaned_length": avg_len,
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
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Final rows:    {len(df):,}")
    print(f"Columns:       {len(df.columns)}")
    print(f"Location:      {output_path}")
    print("-" * 60)

    print("Examples of final text:")
    for row in df.head(5).iter_rows(named=True):
        print(f"FINAL: {row['tweet']}")
        print(f"LEN:   {row['cleaned_length']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
