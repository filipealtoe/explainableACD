#!/usr/bin/env python
"""
Cluster Analysis Utility

Analyzes clustering results to understand cluster quality and content.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def load_results(path: Path) -> pl.DataFrame:
    """Load clustered results."""
    df = pl.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def cluster_size_distribution(df: pl.DataFrame) -> pl.DataFrame:
    """Get cluster size distribution."""
    # Filter to clustered tweets only
    clustered = df.filter(pl.col("cluster_id") >= 0)

    dist = (
        clustered.group_by("cluster_id")
        .agg(pl.len().alias("size"))
        .group_by("size")
        .agg(pl.len().alias("count"))
        .sort("size")
    )
    return dist


def top_clusters(df: pl.DataFrame, n: int = 10) -> pl.DataFrame:
    """Get top N clusters by size."""
    clustered = df.filter(pl.col("cluster_id") >= 0)

    top = (
        clustered.group_by("cluster_id")
        .agg(
            [
                pl.len().alias("size"),
                pl.col("cluster_similarity").mean().alias("avg_similarity"),
            ]
        )
        .sort("size", descending=True)
        .head(n)
    )
    return top


def cluster_samples(
    df: pl.DataFrame,
    cluster_id: int,
    text_column: str = "tweet_enriched",
    n_samples: int = 5,
) -> list[str]:
    """Get sample tweets from a cluster."""
    samples = df.filter(pl.col("cluster_id") == cluster_id).select(text_column).head(n_samples)[text_column].to_list()
    return samples


def compute_cluster_stats(df: pl.DataFrame) -> dict:
    """Compute overall clustering statistics."""
    total = len(df)
    clustered = df.filter(pl.col("cluster_id") >= 0)
    n_clustered = len(clustered)

    if n_clustered == 0:
        return {"total": total, "clustered": 0, "clusters": 0}

    cluster_sizes = clustered.group_by("cluster_id").agg(pl.len().alias("size"))["size"].to_numpy()

    return {
        "total_tweets": total,
        "clustered_tweets": n_clustered,
        "filtered_tweets": total - n_clustered,
        "n_clusters": len(cluster_sizes),
        "avg_cluster_size": float(cluster_sizes.mean()),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "max_cluster_size": int(cluster_sizes.max()),
        "min_cluster_size": int(cluster_sizes.min()),
        "clusters_size_1": int((cluster_sizes == 1).sum()),
        "clusters_size_2_5": int(((cluster_sizes >= 2) & (cluster_sizes <= 5)).sum()),
        "clusters_size_6_10": int(((cluster_sizes >= 6) & (cluster_sizes <= 10)).sum()),
        "clusters_size_11_50": int(((cluster_sizes >= 11) & (cluster_sizes <= 50)).sum()),
        "clusters_size_50_plus": int((cluster_sizes > 50).sum()),
    }


def analyze_cluster_content(
    df: pl.DataFrame,
    cluster_id: int,
    text_column: str = "tweet_enriched",
) -> dict:
    """Analyze content of a specific cluster."""
    cluster_df = df.filter(pl.col("cluster_id") == cluster_id)

    if len(cluster_df) == 0:
        return {"error": f"Cluster {cluster_id} not found"}

    # Get engagement stats
    engagement_cols = ["replies_count", "retweets_count", "likes_count"]
    available_cols = [c for c in engagement_cols if c in cluster_df.columns]

    stats = {
        "cluster_id": cluster_id,
        "size": len(cluster_df),
        "avg_similarity": float(cluster_df["cluster_similarity"].mean()),
    }

    for col in available_cols:
        stats[f"{col}_mean"] = float(cluster_df[col].mean())
        stats[f"{col}_sum"] = int(cluster_df[col].sum())

    # Get sample tweets
    stats["sample_tweets"] = cluster_df[text_column].head(5).to_list()

    return stats


def print_report(df: pl.DataFrame, text_column: str = "tweet_enriched"):
    """Print a full analysis report."""
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS REPORT")
    print("=" * 70)

    # Overall stats
    stats = compute_cluster_stats(df)
    print("\n## Overall Statistics\n")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:,}")

    # Size distribution
    print("\n## Cluster Size Distribution\n")
    dist = cluster_size_distribution(df)
    for row in dist.head(20).to_dicts():
        print(f"  Size {row['size']:3d}: {row['count']:,} clusters")
    if len(dist) > 20:
        print(f"  ... and {len(dist) - 20} more size categories")

    # Top clusters
    print("\n## Top 10 Clusters by Size\n")
    top = top_clusters(df, n=10)
    for row in top.to_dicts():
        print(f"  Cluster {row['cluster_id']:4d}: {row['size']:4d} tweets (avg sim: {row['avg_similarity']:.3f})")

    # Sample content from top 3 clusters
    print("\n## Sample Content from Top 3 Clusters\n")
    for row in top.head(3).to_dicts():
        cid = row["cluster_id"]
        print(f"\n### Cluster {cid} ({row['size']} tweets)\n")
        samples = cluster_samples(df, cid, text_column=text_column, n_samples=3)
        for i, sample in enumerate(samples, 1):
            # Truncate long tweets
            if len(sample) > 200:
                sample = sample[:200] + "..."
            print(f"  {i}. {sample}\n")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze clustering results")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/pipeline_output/tweets_clustered.parquet"),
        help="Path to clustered results",
    )
    parser.add_argument(
        "--text-column",
        default="tweet_enriched",
        help="Text column name",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        help="Analyze specific cluster ID",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        return

    df = load_results(args.input)

    if args.cluster is not None:
        # Analyze specific cluster
        analysis = analyze_cluster_content(df, args.cluster, args.text_column)
        print(f"\n## Cluster {args.cluster} Analysis\n")
        for k, v in analysis.items():
            if k == "sample_tweets":
                print("\n  Sample tweets:")
                for i, t in enumerate(v, 1):
                    if len(t) > 200:
                        t = t[:200] + "..."
                    print(f"    {i}. {t}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
    else:
        # Full report
        print_report(df, text_column=args.text_column)


if __name__ == "__main__":
    main()
