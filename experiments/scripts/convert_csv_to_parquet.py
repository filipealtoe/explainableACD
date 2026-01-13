"""Convert large CSV to Parquet using Polars streaming."""

import time
from pathlib import Path

import polars as pl

INPUT_CSV = Path("data/raw/ioa_tweets.csv")
OUTPUT_PARQUET = Path("data/raw/ioa_tweets.parquet")


def convert_csv_to_parquet() -> None:
    """Convert CSV to Parquet using lazy evaluation and streaming."""
    print(f"Converting {INPUT_CSV} to Parquet...")
    print(f"Input size: {INPUT_CSV.stat().st_size / (1024**3):.2f} GB")

    start_time = time.time()

    # Define schema for proper type inference
    schema_overrides = {
        "tweetid": pl.Int64,
        "follower_count": pl.Int64,
        "following_count": pl.Int64,
        "quote_count": pl.Float64,
        "reply_count": pl.Float64,
        "like_count": pl.Float64,
        "retweet_count": pl.Float64,
        "is_retweet": pl.Boolean,
        "latitude": pl.Utf8,  # Keep as string since it has 'absent' values
        "longitude": pl.Utf8,
    }

    # Use streaming to process without loading all into memory
    lf = pl.scan_csv(
        INPUT_CSV,
        schema_overrides=schema_overrides,
        try_parse_dates=True,
        infer_schema_length=10000,
        low_memory=True,
    )

    # Sink to parquet with compression
    lf.sink_parquet(
        OUTPUT_PARQUET,
        compression="zstd",
        compression_level=3,
        row_group_size=100_000,
    )

    elapsed = time.time() - start_time
    output_size = OUTPUT_PARQUET.stat().st_size / (1024**3)

    print(f"Conversion complete in {elapsed / 60:.1f} minutes")
    print(f"Output size: {output_size:.2f} GB")
    print(f"Compression ratio: {INPUT_CSV.stat().st_size / OUTPUT_PARQUET.stat().st_size:.1f}x")


if __name__ == "__main__":
    convert_csv_to_parquet()
