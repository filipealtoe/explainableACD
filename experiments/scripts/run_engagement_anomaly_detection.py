import json
import sys
from pathlib import Path

import polars as pl

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def create_engagement_timeseries(data_path: Path, top_topics: list[int], time_freq: str = "1h") -> pl.DataFrame:
    """Create engagement timeseries aggregated by topic and time."""
    print(f"Loading engagement data from {data_path}...")
    df = pl.read_csv(data_path)

    print(f"Loaded {len(df)} tweets")

    # De-duplicate tweets (pipeline may have processed some tweets multiple times)
    original_count = len(df)
    df = df.unique(subset=["tweet_id"])
    if len(df) < original_count:
        print(f"Removed {original_count - len(df)} duplicate tweets")

    # Filter to top topics only
    df = df.filter(pl.col("topic").is_in(top_topics))
    print(f"Filtered to {len(df)} tweets in top {len(top_topics)} topics")

    # Parse timestamp (ISO 8601 format with microseconds)
    df = df.with_columns(pl.col("timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f", strict=False))

    # Truncate to hourly bins
    df = df.with_columns(pl.col("timestamp").dt.truncate(time_freq).alias("time_bin"))

    # Aggregate engagement by topic and time bin
    print("Aggregating engagement by topic and time...")
    ts = (
        df.group_by(["topic", "time_bin"])
        .agg(pl.col("total_engagement").sum().alias("engagement"))
        .sort(["topic", "time_bin"])
    )

    # Pivot to wide format (topic × timestamps)
    print("Pivoting to wide format...")
    ts_pivot = ts.pivot(index="topic", columns="time_bin", values="engagement", aggregate_function="first")

    # Fill nulls with 0
    ts_pivot = ts_pivot.fill_null(0)

    print(f"Created timeseries: {ts_pivot.shape[0]} topics × {ts_pivot.shape[1] - 1} time points")

    return ts_pivot


def main() -> None:
    # Configuration
    data_path = repo_root / "data" / "tweet_topic_assignments_with_engagement.csv"
    artifacts_dir = repo_root / "mlruns" / "4b7e95a391254fdaa79a1fb34c6a4a55" / "artifacts"
    keywords_path = artifacts_dir / "topic_keywords.json"

    # Top 10 topics (same as count-based analysis)
    top_topics = [74, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Load topic keywords
    print("Loading topic keywords...")
    with open(keywords_path, encoding="utf-8") as f:
        topic_keywords = {int(k): v for k, v in json.load(f).items()}

    # Create engagement timeseries
    engagement_ts = create_engagement_timeseries(data_path=data_path, top_topics=top_topics, time_freq="1h")

    # Save engagement timeseries
    output_path = repo_root / "data" / "topic_timeseries_engagement_H.csv"
    engagement_ts.write_csv(output_path)
    print(f"\n✓ Saved engagement timeseries to: {output_path}")

    # Show preview
    print("\nEngagement timeseries preview:")
    print(engagement_ts.head())

    # Show some statistics
    print("\nEngagement statistics per topic:")
    time_cols = [col for col in engagement_ts.columns if col != "topic"]
    for topic in top_topics:
        topic_data = engagement_ts.filter(pl.col("topic") == topic)
        if topic_data.height > 0:
            topic_row = topic_data.to_dicts()[0]
            values = [topic_row[col] for col in time_cols]
            total_eng = sum(values)
            max_eng = max(values)
            mean_eng = total_eng / len(values) if values else 0

            keywords = ", ".join(topic_keywords.get(topic, [])[:3])
            print(f"  Topic {topic} ({keywords}): total={total_eng:,.0f}, max={max_eng:,.0f}, mean={mean_eng:.1f}")

    print("\n" + "=" * 60)
    print("Next step: Run anomaly detection on engagement data")
    print("=" * 60)
    print("\nNow importing and running engagement anomaly detection...")

    # Import and run anomaly detection
    from src.models.anomaly_detection import run_anomaly_detection_from_files

    run_id = run_anomaly_detection_from_files(
        timeseries_path=str(output_path), top_topics=top_topics, topic_keywords=topic_keywords, z_threshold=3.0
    )

    print(f"\n{'=' * 60}")
    print("Engagement anomaly detection complete!")
    print(f"MLflow Run ID: {run_id}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
