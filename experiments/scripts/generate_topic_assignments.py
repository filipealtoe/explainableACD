import pickle
import sys
from pathlib import Path

import polars as pl

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def main() -> None:
    print("Loading BERTopic model...")
    # Load the saved BERTopic model
    model_path = repo_root / "mlruns" / "models" / "m-2effd448842a4ffb9a89520477f03615" / "artifacts" / "model.pkl"

    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)

    print(f"Model loaded. Number of topics: {len(topic_model.get_topic_info())}")

    # Load tweets
    print("\nLoading tweets...")
    tweets_path = repo_root / "data" / "raw" / "tweets_ai.parquet"

    # Use polars for faster loading
    df = pl.read_parquet(tweets_path)
    print(f"Loaded {len(df)} tweets")

    # Get tweet IDs and text
    tweet_ids = df["id"].to_list()
    texts = df["tweet"].to_list()

    print("\nPredicting topics for all tweets...")
    # Get topic assignments
    topics, _ = topic_model.transform(texts)

    # Create mapping dataframe
    print("\nCreating topic assignments file...")
    assignments = pl.DataFrame({"tweet_id": tweet_ids, "topic": topics})

    # Save to CSV
    output_path = repo_root / "data" / "tweet_topic_assignments.csv"
    assignments.write_csv(output_path)

    print(f"\n✓ Saved topic assignments to: {output_path}")
    print(f"  Total tweets: {len(assignments)}")
    print(f"  Unique topics: {assignments['topic'].n_unique()}")

    # Show topic distribution
    print("\nTopic distribution:")
    topic_counts = assignments.group_by("topic").agg(pl.count().alias("count")).sort("count", descending=True).head(10)
    print(topic_counts)


if __name__ == "__main__":
    main()
