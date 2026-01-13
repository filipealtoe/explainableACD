import json
import sys
from pathlib import Path

import polars as pl

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def create_combined_timeseries(
    count_path: Path,
    engagement_path: Path,
    top_topics: list[int],
    count_weight: float = 0.4,
    engagement_weight: float = 0.6,
) -> pl.DataFrame:
    """
    Create combined timeseries using normalized count and engagement.

    Formula: composite = count_weight × z_count + engagement_weight × z_engagement
    """
    print("Loading count timeseries...")
    count_ts = pl.read_csv(count_path)

    print("Loading engagement timeseries...")
    engagement_ts = pl.read_csv(engagement_path)

    # Filter to top topics
    count_ts = count_ts.filter(pl.col("topic").is_in(top_topics))
    engagement_ts = engagement_ts.filter(pl.col("topic").is_in(top_topics))

    print(f"Processing {len(top_topics)} topics")
    print(f"Weights: {count_weight:.1f} count + {engagement_weight:.1f} engagement")

    # Get common time columns (intersection of both timeseries)
    count_cols = set(count_ts.columns)
    engagement_cols = set(engagement_ts.columns)
    common_cols = count_cols & engagement_cols
    common_cols.discard("topic")  # Remove 'topic' from time columns
    time_cols = sorted(list(common_cols))

    print(f"Time points: {len(time_cols)} (common between count and engagement)")

    # Keep only common columns in both dataframes
    count_ts = count_ts.select(["topic"] + time_cols)
    engagement_ts = engagement_ts.select(["topic"] + time_cols)

    # Convert to long format for normalization
    print("\nConverting to long format...")
    count_long = count_ts.melt(id_vars=["topic"], value_vars=time_cols, variable_name="timestamp", value_name="count")

    engagement_long = engagement_ts.melt(
        id_vars=["topic"], value_vars=time_cols, variable_name="timestamp", value_name="engagement"
    )

    # Join count and engagement
    print("Joining count and engagement data...")
    combined = count_long.join(engagement_long, on=["topic", "timestamp"], how="inner")

    print(f"Combined data: {len(combined)} rows")

    # Calculate z-scores for each metric PER TOPIC
    print("\nCalculating z-scores per topic...")
    combined = combined.with_columns(
        [
            # Z-score for count
            ((pl.col("count") - pl.col("count").mean().over("topic")) / pl.col("count").std().over("topic")).alias(
                "z_count"
            ),
            # Z-score for engagement
            (
                (pl.col("engagement") - pl.col("engagement").mean().over("topic"))
                / pl.col("engagement").std().over("topic")
            ).alias("z_engagement"),
        ]
    )

    # Handle NaN/inf from zero std (topics with constant values)
    combined = combined.with_columns(
        [pl.col("z_count").fill_nan(0).fill_null(0), pl.col("z_engagement").fill_nan(0).fill_null(0)]
    )

    # Calculate composite score
    print(f"Creating composite score: {count_weight} × z_count + {engagement_weight} × z_engagement")
    combined = combined.with_columns(
        (pl.col("z_count") * count_weight + pl.col("z_engagement") * engagement_weight).alias("composite")
    )

    # Show some statistics
    print("\nComposite score statistics:")
    stats = combined.select(
        [
            pl.col("composite").mean().alias("mean"),
            pl.col("composite").std().alias("std"),
            pl.col("composite").min().alias("min"),
            pl.col("composite").max().alias("max"),
            pl.col("composite").median().alias("median"),
        ]
    )
    print(stats)

    # Convert back to wide format
    print("\nPivoting back to wide format...")
    composite_ts = combined.select(["topic", "timestamp", "composite"]).pivot(
        index="topic", columns="timestamp", values="composite", aggregate_function="first"
    )

    composite_ts = composite_ts.fill_null(0)

    print(f"Created composite timeseries: {composite_ts.shape[0]} topics × {composite_ts.shape[1] - 1} time points")

    return composite_ts


def main() -> None:
    # Configuration
    count_path = repo_root / "mlruns" / "4b7e95a391254fdaa79a1fb34c6a4a55" / "artifacts" / "topic_timeseries_H.csv"
    engagement_path = repo_root / "data" / "topic_timeseries_engagement_H.csv"
    keywords_path = repo_root / "mlruns" / "4b7e95a391254fdaa79a1fb34c6a4a55" / "artifacts" / "topic_keywords.json"

    # Top 10 topics
    top_topics = [74, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Load topic keywords
    print("Loading topic keywords...")
    with open(keywords_path, encoding="utf-8") as f:
        topic_keywords = {int(k): v for k, v in json.load(f).items()}

    # Create combined timeseries
    # 40% weight on count, 60% weight on engagement
    combined_ts = create_combined_timeseries(
        count_path=count_path,
        engagement_path=engagement_path,
        top_topics=top_topics,
        count_weight=0.4,
        engagement_weight=0.6,
    )

    # Save combined timeseries
    output_path = repo_root / "data" / "topic_timeseries_combined_H.csv"
    combined_ts.write_csv(output_path)
    print(f"\n✓ Saved combined timeseries to: {output_path}")

    # Show preview
    print("\nCombined timeseries preview (first 5 columns):")
    preview_cols = ["topic"] + combined_ts.columns[1:6]
    print(combined_ts.select(preview_cols))

    print("\n" + "=" * 60)
    print("Running anomaly detection on combined metric...")
    print("=" * 60)
    print()

    # Import and run anomaly detection
    from src.models.anomaly_detection import run_anomaly_detection_from_files

    run_id = run_anomaly_detection_from_files(
        timeseries_path=str(output_path), top_topics=top_topics, topic_keywords=topic_keywords, z_threshold=3.0
    )

    print(f"\n{'=' * 60}")
    print("Combined anomaly detection complete!")
    print(f"MLflow Run ID: {run_id}")
    print(f"{'=' * 60}")
    print("\nThis detects anomalies where:")
    print("  - 40% weight: Unusual tweet volume")
    print("  - 60% weight: Unusual engagement")
    print("  → Finds major events with BOTH high activity AND high engagement")
    print()


if __name__ == "__main__":
    main()
