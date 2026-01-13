"""
Quick validation script to estimate claim volume from OnlineCosineClustering.

Tests on 1 day of US elections data to validate:
1. Number of clusters generated
2. Cluster size distribution
3. Estimated anomalous clusters (potential claims)
4. Time to process

Run: python experiments/scripts/validate_clustering_volume.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

import numpy as np
import polars as pl

from src.pipeline.modules.clusterer import Clusterer, ClustererConfig
from src.pipeline.modules.embedder import Embedder, EmbedderConfig


def load_one_day(parquet_path: str, target_date: str) -> pl.DataFrame:
    """Load tweets for a single day."""
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(pl.col("created_at").dt.date().alias("date"))
    df_day = df.filter(pl.col("date") == pl.lit(target_date).str.to_date())
    return df_day


def clean_tweet(text: str) -> str:
    """Basic tweet cleaning."""
    import re
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#", "", text)  # Remove hashtag symbol
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("=" * 70)
    print("CLUSTERING VOLUME VALIDATION")
    print("=" * 70)

    # Configuration
    parquet_path = root / "data/raw/us_elections_tweets.parquet"
    target_date = "2020-11-04"  # Election day - high volume

    # Load data
    print(f"\n[1/5] Loading tweets for {target_date}...")
    start = time.time()
    df = load_one_day(str(parquet_path), target_date)
    print(f"  Loaded {len(df):,} tweets in {time.time() - start:.1f}s")

    # Clean tweets
    print("\n[2/5] Cleaning tweets...")
    start = time.time()
    df = df.with_columns(
        pl.col("tweet").map_elements(clean_tweet, return_dtype=pl.Utf8).alias("tweet_clean")
    )
    # Filter empty tweets
    df = df.filter(pl.col("tweet_clean").str.len_chars() > 10)
    print(f"  {len(df):,} tweets after cleaning in {time.time() - start:.1f}s")

    # Sample if too large (for quick validation)
    max_tweets = 50000  # Process 50K for quick test
    if len(df) > max_tweets:
        print(f"\n  Sampling {max_tweets:,} tweets for quick validation...")
        df = df.sample(n=max_tweets, seed=42)

    # Generate embeddings
    print("\n[3/5] Generating embeddings...")
    embedder_config = EmbedderConfig(
        model_name="all-MiniLM-L6-v2",  # Faster, smaller model for validation
        batch_size=128,
        device="mps",
        normalize=True,
    )
    embedder = Embedder(embedder_config)

    start = time.time()
    texts = df["tweet_clean"].to_list()
    embeddings = embedder.embed(texts)
    embed_time = time.time() - start
    print(f"  Generated {len(embeddings):,} embeddings in {embed_time:.1f}s")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    # Run clustering with different thresholds
    print("\n[4/5] Testing clustering with different thresholds...")

    thresholds = [0.70, 0.75, 0.80, 0.85]
    results = []

    for threshold in thresholds:
        config = ClustererConfig(
            similarity_threshold=threshold,
            min_cluster_size=1,
            max_clusters=50000,
            max_representatives_per_cluster=5,
        )
        clusterer = Clusterer(config, embedding_dim=embeddings.shape[1])

        # Create fake tweet IDs
        tweet_ids = [f"tweet_{i}" for i in range(len(embeddings))]

        start = time.time()
        cluster_ids, similarities = clusterer.algorithm.assign_batch(
            embeddings, tweet_ids, show_progress=False
        )
        cluster_time = time.time() - start

        # Get stats
        stats = clusterer.get_stats()
        sizes = np.array(clusterer.algorithm.state.sizes)

        # Compute distribution
        n_size_1 = (sizes == 1).sum()
        n_size_gte_5 = (sizes >= 5).sum()
        n_size_gte_10 = (sizes >= 10).sum()
        n_size_gte_50 = (sizes >= 50).sum()
        n_size_gte_100 = (sizes >= 100).sum()

        results.append({
            "threshold": threshold,
            "n_clusters": stats["n_clusters"],
            "avg_size": stats["avg_cluster_size"],
            "max_size": stats["max_cluster_size"],
            "size_1": n_size_1,
            "size_gte_5": n_size_gte_5,
            "size_gte_10": n_size_gte_10,
            "size_gte_50": n_size_gte_50,
            "size_gte_100": n_size_gte_100,
            "time_s": cluster_time,
        })

        print(f"\n  Threshold {threshold}:")
        print(f"    Clusters: {stats['n_clusters']:,}")
        print(f"    Avg size: {stats['avg_cluster_size']:.1f}")
        print(f"    Max size: {stats['max_cluster_size']:,}")
        print(f"    Size >= 10: {n_size_gte_10:,} clusters")
        print(f"    Size >= 50: {n_size_gte_50:,} clusters")
        print(f"    Time: {cluster_time:.1f}s")

    # Summary and projections
    print("\n" + "=" * 70)
    print("[5/5] PROJECTIONS FOR FULL DATASET")
    print("=" * 70)

    # Use threshold 0.75 for projections (your existing default)
    best = [r for r in results if r["threshold"] == 0.75][0]

    tweets_processed = len(df)
    total_tweets = 1_522_909
    scale_factor = total_tweets / tweets_processed

    print(f"\n  Based on {tweets_processed:,} tweets (threshold=0.75):")
    print(f"  - Clusters created: {best['n_clusters']:,}")
    print(f"  - Significant clusters (size>=10): {best['size_gte_10']:,}")
    print(f"  - Large clusters (size>=50): {best['size_gte_50']:,}")

    print(f"\n  Projected for full dataset ({total_tweets:,} tweets):")
    print(f"  - Est. clusters: ~{int(best['n_clusters'] * scale_factor * 0.7):,}")
    print(f"    (0.7x factor: clusters merge across days)")
    print(f"  - Est. significant clusters: ~{int(best['size_gte_10'] * scale_factor * 0.5):,}")
    print(f"  - Est. large clusters: ~{int(best['size_gte_50'] * scale_factor * 0.4):,}")

    # Estimate anomalous clusters (claims)
    # Assume ~10-20% of significant clusters trigger anomaly
    est_significant = int(best['size_gte_10'] * scale_factor * 0.5)
    est_anomalous_low = int(est_significant * 0.10)
    est_anomalous_high = int(est_significant * 0.20)

    print(f"\n  Estimated CLAIMS (anomalous clusters):")
    print(f"  - Conservative: ~{est_anomalous_low:,} claims")
    print(f"  - Optimistic: ~{est_anomalous_high:,} claims")

    # Train/test split estimate
    print(f"\n  Train/Test split estimate:")
    print(f"  - Train (Oct 22-Nov 3): ~{int(est_anomalous_low * 0.4)}-{int(est_anomalous_high * 0.4)} claims")
    print(f"  - Test (Nov 4-8): ~{int(est_anomalous_low * 0.6)}-{int(est_anomalous_high * 0.6)} claims")
    print(f"    (More in test due to election spike)")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    # Save results
    results_df = pl.DataFrame(results)
    output_path = root / "experiments/results/clustering_validation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
