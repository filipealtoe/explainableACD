"""
Efficient basic statistics for large parquet files using Polars.
Uses lazy evaluation and processes columns individually to avoid memory issues.
"""

import sys
import time
from pathlib import Path

import polars as pl


def get_basic_stats(parquet_path: str) -> None:
    """Compute and print basic statistics for all columns in a parquet file."""

    start_time = time.time()
    path = Path(parquet_path)

    if not path.exists():
        print(f"Error: File not found: {parquet_path}")
        sys.exit(1)

    print(f"{'=' * 60}")
    print("PARQUET FILE STATISTICS")
    print(f"File: {parquet_path}")
    print(f"{'=' * 60}\n")

    # Get file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # Use scan for lazy evaluation
    lf = pl.scan_parquet(parquet_path)
    schema = lf.collect_schema()

    print(f"Number of columns: {len(schema)}")

    # Get row count efficiently
    row_count = lf.select(pl.len()).collect().item()
    print(f"Number of rows: {row_count:,}")

    print(f"\n{'=' * 60}")
    print("COLUMN-BY-COLUMN STATISTICS")
    print(f"{'=' * 60}")

    for col_name, dtype in schema.items():
        col_start = time.time()
        print(f"\n{'─' * 60}")
        print(f"  {col_name}")
        print(f"  Type: {dtype}")

        try:
            # Basic counts for all types
            basic = lf.select(
                [
                    pl.col(col_name).count().alias("count"),
                    pl.col(col_name).null_count().alias("nulls"),
                ]
            ).collect()

            count = basic["count"][0]
            nulls = basic["nulls"][0]
            print(f"  Non-null: {count:,}")
            print(f"  Null: {nulls:,} ({100 * nulls / row_count:.2f}%)")

            # Type-specific stats
            if dtype in (
                pl.Int64,
                pl.Int32,
                pl.Int16,
                pl.Int8,
                pl.UInt64,
                pl.UInt32,
                pl.UInt16,
                pl.UInt8,
                pl.Float64,
                pl.Float32,
            ):
                # Numeric stats
                stats = lf.select(
                    [
                        pl.col(col_name).mean().alias("mean"),
                        pl.col(col_name).std().alias("std"),
                        pl.col(col_name).min().alias("min"),
                        pl.col(col_name).max().alias("max"),
                        pl.col(col_name).median().alias("median"),
                    ]
                ).collect()

                mean = stats["mean"][0]
                std = stats["std"][0]
                min_val = stats["min"][0]
                max_val = stats["max"][0]
                median = stats["median"][0]

                if mean is not None:
                    print(f"  Mean: {mean:,.4f}")
                    print(f"  Std: {std:,.4f}" if std else "  Std: N/A")
                    print(f"  Min: {min_val:,}")
                    print(f"  Max: {max_val:,}")
                    print(f"  Median: {median:,.4f}")

            elif dtype == pl.Boolean:
                stats = lf.select(
                    [
                        pl.col(col_name).sum().alias("true_count"),
                    ]
                ).collect()

                true_count = stats["true_count"][0] or 0
                false_count = count - true_count
                print(f"  True: {true_count:,} ({100 * true_count / count:.2f}%)" if count > 0 else "  True: 0")
                print(f"  False: {false_count:,} ({100 * false_count / count:.2f}%)" if count > 0 else "  False: 0")

            elif dtype == pl.Date or dtype.is_temporal():
                stats = lf.select(
                    [
                        pl.col(col_name).min().alias("min"),
                        pl.col(col_name).max().alias("max"),
                    ]
                ).collect()

                print(f"  Min: {stats['min'][0]}")
                print(f"  Max: {stats['max'][0]}")

            else:
                # String columns - basic length stats only (unique is expensive)
                stats = lf.select(
                    [
                        pl.col(col_name).str.len_bytes().mean().alias("avg_len"),
                        pl.col(col_name).str.len_bytes().max().alias("max_len"),
                    ]
                ).collect()

                avg_len = stats["avg_len"][0]
                max_len = stats["max_len"][0]
                print(f"  Avg length: {avg_len:.1f}" if avg_len else "  Avg length: N/A")
                print(f"  Max length: {max_len:,}" if max_len else "  Max length: N/A")

            col_elapsed = time.time() - col_start
            print(f"  (computed in {col_elapsed:.2f}s)")

        except Exception as e:
            print(f"  Error computing stats: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ioa_tweets.parquet"
    get_basic_stats(parquet_path)
