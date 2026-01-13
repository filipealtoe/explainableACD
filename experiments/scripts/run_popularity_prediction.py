"""
Popularity Prediction: Predict whether an anomaly will go Viral or Fizzle.

This script:
1. Loads detected anomalies (168h granularity)
2. Merges consecutive anomalies into events
3. Extracts features from a 6-hour peek window
4. Predicts 7-day horizon engagement (binary: Viral vs Fizzle)
5. Trains an interpretable Decision Tree classifier
6. Generates validation visualizations
"""

import ast
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

matplotlib.use("Agg")  # Non-interactive backend for saving figures

# Path setup
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


# =============================================================================
# CONFIGURATION
# =============================================================================
TOP_TOPICS = [74, 0, 1, 2, 3, 4, 5, 6, 7, 8]
PEEK_HOURS = 6
HORIZON_DAYS = 7
MERGE_GAP_HOURS = 6
MIN_PEEK_TWEETS = 5
TREE_MAX_DEPTH = 4
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Feature names for the model
FEATURE_NAMES = [
    "z_score_at_detection",
    "z_max_peek",
    "momentum",
    "rt_ratio",
    "reply_ratio",
    "engagement_efficiency",
    "tweet_volume",
    "url_density",
    "media_density",
    "video_density",
    "hashtag_avg",
    "quote_ratio",
    "retweet_ratio",
    "hour_of_day",
    "is_weekend",
]


# =============================================================================
# DATA LOADING
# =============================================================================
def find_latest_anomaly_file(experiment_id: str = "360627897138559911") -> Path:
    """Find the anomalies_hourly_168h.csv with the most anomalies from MLflow runs."""
    mlruns_dir = repo_root / "mlruns" / experiment_id
    if not mlruns_dir.exists():
        raise FileNotFoundError(f"MLflow experiment directory not found: {mlruns_dir}")

    # Find all run directories
    run_dirs = [d for d in mlruns_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    # Find the run with the most anomalies
    best_file = None
    best_count = 0

    for run_dir in run_dirs:
        anomaly_file = run_dir / "artifacts" / "anomalies_hourly_168h.csv"
        if anomaly_file.exists():
            # Count lines (subtract 1 for header)
            with open(anomaly_file) as f:
                line_count = sum(1 for _ in f) - 1
            if line_count > best_count:
                best_count = line_count
                best_file = anomaly_file

    if best_file is None:
        raise FileNotFoundError("No anomalies_hourly_168h.csv found in any MLflow run")

    print(f"Found {best_count} anomalies in best run")
    return best_file


def load_anomalies(anomaly_path: Path) -> pl.DataFrame:
    """Load and filter anomalies to top topics."""
    df = pl.read_csv(anomaly_path)

    # Parse timestamp
    df = df.with_columns(pl.col("timestamp").str.to_datetime().alias("timestamp"))

    # Filter to top topics
    df = df.filter(pl.col("topic").is_in(TOP_TOPICS))

    print(f"Loaded {len(df)} anomalies for {len(TOP_TOPICS)} topics")
    return df


def load_tweets_with_content() -> pl.DataFrame:
    """Load tweet assignments joined with raw tweet content features."""
    # Load assignments
    assignments_path = repo_root / "data" / "tweet_topic_assignments_with_engagement.csv"
    assignments = pl.read_csv(assignments_path)

    # Parse timestamp
    assignments = assignments.with_columns(pl.col("timestamp").str.to_datetime().alias("timestamp"))

    # Load raw tweets for content features
    raw_path = repo_root / "data" / "raw" / "tweets_ai.parquet"
    raw_tweets = pl.read_parquet(raw_path)

    # Convert ID to int64 for join
    raw_tweets = raw_tweets.with_columns(pl.col("id").cast(pl.Int64).alias("tweet_id"))

    # Select only needed columns from raw tweets
    raw_tweets = raw_tweets.select(["tweet_id", "urls", "photos", "video", "hashtags", "quote_url", "retweet"])

    # Join
    tweets = assignments.join(raw_tweets, on="tweet_id", how="left")

    # Filter to top topics
    tweets = tweets.filter(pl.col("topic").is_in(TOP_TOPICS))

    print(f"Loaded {len(tweets)} tweets with content features")
    return tweets


# =============================================================================
# ANOMALY MERGING
# =============================================================================
def merge_anomalies(df: pl.DataFrame, gap_hours: int = MERGE_GAP_HOURS) -> pl.DataFrame:
    """Merge consecutive anomalies within gap_hours into single events."""
    df = df.sort(["topic", "timestamp"])

    # Calculate time diff from previous row (per topic)
    df = df.with_columns((pl.col("timestamp") - pl.col("timestamp").shift(1).over("topic")).alias("time_diff"))

    # New event if gap > threshold or first row of topic
    df = df.with_columns(
        ((pl.col("time_diff") > timedelta(hours=gap_hours)) | pl.col("time_diff").is_null())
        .cum_sum()
        .over("topic")
        .alias("event_id")
    )

    # Aggregate to event level
    events = df.group_by(["topic", "event_id"]).agg(
        [
            pl.col("timestamp").min().alias("event_start"),
            pl.col("z_score").max().alias("z_max_at_detection"),
            pl.col("z_score").first().alias("z_score_at_detection"),
            pl.len().alias("anomaly_count"),
        ]
    )

    print(f"Merged {len(df)} anomalies into {len(events)} events")
    return events


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def has_content(val) -> bool:
    """Check if a string-list column has content."""
    if val is None:
        return False
    if isinstance(val, str):
        return val != "" and val != "[]"
    return False


def count_list_items(val) -> int:
    """Count items in a string representation of a list."""
    if val is None:
        return 0
    if isinstance(val, str):
        if val == "" or val == "[]":
            return 0
        try:
            return len(ast.literal_eval(val))
        except (ValueError, SyntaxError):
            return 0
    return 0


def extract_features(
    event: dict,
    tweets: pl.DataFrame,
    all_events: pl.DataFrame,
) -> dict | None:
    """Extract all features for a single anomaly event."""
    topic = event["topic"]
    event_start = event["event_start"]
    peek_end = event_start + timedelta(hours=PEEK_HOURS)
    horizon_start = peek_end
    horizon_end = horizon_start + timedelta(days=HORIZON_DAYS)

    # Get tweets for this topic
    topic_tweets = tweets.filter(pl.col("topic") == topic)

    # Peek window tweets
    peek_tweets = topic_tweets.filter((pl.col("timestamp") >= event_start) & (pl.col("timestamp") < peek_end))

    # Horizon window tweets
    horizon_tweets = topic_tweets.filter((pl.col("timestamp") >= horizon_start) & (pl.col("timestamp") < horizon_end))

    # Skip if insufficient data
    if len(peek_tweets) < MIN_PEEK_TWEETS:
        return None

    # Calculate horizon engagement for target
    horizon_engagement = horizon_tweets["total_engagement"].sum()
    if horizon_engagement is None:
        horizon_engagement = 0

    # -------------------------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    tweet_count = len(peek_tweets)

    # --- Intensity Features ---
    z_score_at_detection = event["z_score_at_detection"]
    z_max_peek = event["z_max_at_detection"]  # From merged anomalies

    # Momentum: approximate from engagement trend in peek window
    # Split peek into first and second half
    peek_mid = event_start + timedelta(hours=PEEK_HOURS / 2)
    first_half = peek_tweets.filter(pl.col("timestamp") < peek_mid)
    second_half = peek_tweets.filter(pl.col("timestamp") >= peek_mid)

    first_eng = first_half["total_engagement"].sum() if len(first_half) > 0 else 0
    second_eng = second_half["total_engagement"].sum() if len(second_half) > 0 else 0

    # Normalize by count to get rate
    first_rate = first_eng / max(len(first_half), 1)
    second_rate = second_eng / max(len(second_half), 1)
    momentum = second_rate - first_rate

    # --- Engagement Composition Features ---
    total_rt = peek_tweets["retweets_count"].sum()
    total_replies = peek_tweets["replies_count"].sum()
    total_likes = peek_tweets["likes_count"].sum()
    total_engagement = peek_tweets["total_engagement"].sum()

    # Avoid division by zero
    total_engagement_safe = max(total_engagement, 1)

    rt_ratio = total_rt / total_engagement_safe
    reply_ratio = total_replies / total_engagement_safe
    engagement_efficiency = total_engagement / tweet_count
    tweet_volume = tweet_count

    # --- Content/Coordination Features ---
    # Convert to Python for easier processing
    peek_data = peek_tweets.to_dicts()

    url_count = sum(1 for row in peek_data if has_content(row.get("urls")))
    photo_count = sum(1 for row in peek_data if has_content(row.get("photos")))
    video_count = sum(1 for row in peek_data if row.get("video") == 1)
    quote_count = sum(1 for row in peek_data if row.get("quote_url") is not None)
    retweet_count = sum(1 for row in peek_data if row.get("retweet") is True)

    hashtag_counts = [count_list_items(row.get("hashtags")) for row in peek_data]
    hashtag_avg = np.mean(hashtag_counts) if hashtag_counts else 0

    url_density = url_count / tweet_count
    media_density = (photo_count + video_count) / tweet_count
    video_density = video_count / tweet_count
    quote_ratio = quote_count / tweet_count
    retweet_ratio = retweet_count / tweet_count

    # --- Temporal Features ---
    hour_of_day = event_start.hour
    day_of_week = event_start.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    return {
        "topic": topic,
        "event_start": event_start,
        "z_score_at_detection": z_score_at_detection,
        "z_max_peek": z_max_peek,
        "momentum": momentum,
        "rt_ratio": rt_ratio,
        "reply_ratio": reply_ratio,
        "engagement_efficiency": engagement_efficiency,
        "tweet_volume": tweet_volume,
        "url_density": url_density,
        "media_density": media_density,
        "video_density": video_density,
        "hashtag_avg": hashtag_avg,
        "quote_ratio": quote_ratio,
        "retweet_ratio": retweet_ratio,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "horizon_engagement": horizon_engagement,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualizations(
    clf: DecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    dataset: pl.DataFrame,
    output_dir: str,
) -> dict:
    """Create all 6 visualization panels."""
    paths = {}

    # Panel 1: Decision Tree Plot
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        clf,
        feature_names=FEATURE_NAMES,
        class_names=["Fizzle", "Viral"],
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=10,
    )
    ax.set_title("Decision Tree: Predicting Viral vs Fizzle Anomalies", fontsize=14)
    tree_path = os.path.join(output_dir, "panel1_decision_tree.png")
    plt.savefig(tree_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["decision_tree"] = tree_path

    # Panel 2: Feature Importance Bar Chart
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(FEATURE_NAMES)))
    ax.barh(
        range(len(FEATURE_NAMES)),
        importances[indices[::-1]],
        color=colors,
    )
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in indices[::-1]])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Decision Tree)")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()
    importance_path = os.path.join(output_dir, "panel2_feature_importance.png")
    plt.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["feature_importance"] = importance_path

    # Panel 3: Feature Distributions by Class (Box Plots)
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURE_NAMES):
        ax = axes[i]
        viral_data = dataset.filter(pl.col("target") == 1)[feature].to_numpy()
        fizzle_data = dataset.filter(pl.col("target") == 0)[feature].to_numpy()

        bp = ax.boxplot(
            [fizzle_data, viral_data],
            labels=["Fizzle", "Viral"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("lightcoral")
        bp["boxes"][1].set_facecolor("lightgreen")
        ax.set_title(feature, fontsize=10)
        ax.tick_params(axis="x", labelsize=9)

    # Remove empty subplots if any
    for j in range(len(FEATURE_NAMES), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Feature Distributions: Fizzle vs Viral", fontsize=14)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "panel3_feature_distributions.png")
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["feature_distributions"] = boxplot_path

    # Panel 4: Confusion Matrix + Metrics
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Fizzle", "Viral"],
        yticklabels=["Fizzle", "Viral"],
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )

    # Add metrics as text
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics_text = f"Accuracy: {acc:.3f}\nPrecision: {prec:.3f}\nRecall: {rec:.3f}\nF1: {f1:.3f}"
    ax.text(
        1.35,
        0.5,
        metrics_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_title("Confusion Matrix", fontsize=14)
    cm_path = os.path.join(output_dir, "panel4_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["confusion_matrix"] = cm_path

    # Panel 5: Target & Sample Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Class distribution
    class_counts = dataset.group_by("target").len().sort("target")
    labels = ["Fizzle", "Viral"]
    counts = class_counts["len"].to_list()
    colors = ["lightcoral", "lightgreen"]
    ax1.bar(labels, counts, color=colors)
    ax1.set_ylabel("Count")
    ax1.set_title("Class Distribution")
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax1.text(i, count + 1, str(count), ha="center", fontsize=11)

    # Samples per topic
    topic_counts = dataset.group_by("topic").len().sort("topic")
    ax2.bar(
        [str(t) for t in topic_counts["topic"].to_list()],
        topic_counts["len"].to_list(),
        color="steelblue",
    )
    ax2.set_xlabel("Topic")
    ax2.set_ylabel("Count")
    ax2.set_title("Samples per Topic")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    dist_path = os.path.join(output_dir, "panel5_distributions.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["distributions"] = dist_path

    # Panel 6: Example Cases Table
    # Add predictions to dataset for analysis
    X_all = dataset.select(FEATURE_NAMES).to_numpy()
    y_all = dataset["target"].to_numpy()
    y_pred_all = clf.predict(X_all)

    dataset_with_pred = dataset.with_columns(
        pl.Series("predicted", y_pred_all),
        pl.Series("correct", y_pred_all == y_all),
    )

    # Get examples
    correct_viral = dataset_with_pred.filter((pl.col("target") == 1) & (pl.col("correct"))).head(3)
    correct_fizzle = dataset_with_pred.filter((pl.col("target") == 0) & (pl.col("correct"))).head(3)
    misclassified = dataset_with_pred.filter(~pl.col("correct")).head(3)

    # Create text summary
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    text_content = "EXAMPLE CASES\n" + "=" * 80 + "\n\n"

    text_content += "CORRECT VIRAL PREDICTIONS:\n" + "-" * 40 + "\n"
    for row in correct_viral.to_dicts():
        text_content += f"  Topic {row['topic']}, Start: {row['event_start']}\n"
        text_content += f"    engagement_eff={row['engagement_efficiency']:.1f}, media_dens={row['media_density']:.2f}, rt_ratio={row['rt_ratio']:.2f}\n"

    text_content += "\nCORRECT FIZZLE PREDICTIONS:\n" + "-" * 40 + "\n"
    for row in correct_fizzle.to_dicts():
        text_content += f"  Topic {row['topic']}, Start: {row['event_start']}\n"
        text_content += f"    engagement_eff={row['engagement_efficiency']:.1f}, media_dens={row['media_density']:.2f}, rt_ratio={row['rt_ratio']:.2f}\n"

    text_content += "\nMISCLASSIFIED EXAMPLES:\n" + "-" * 40 + "\n"
    for row in misclassified.to_dicts():
        actual = "Viral" if row["target"] == 1 else "Fizzle"
        predicted = "Viral" if row["predicted"] == 1 else "Fizzle"
        text_content += f"  Topic {row['topic']}, Start: {row['event_start']}\n"
        text_content += f"    Actual: {actual}, Predicted: {predicted}\n"
        text_content += f"    engagement_eff={row['engagement_efficiency']:.1f}, media_dens={row['media_density']:.2f}, rt_ratio={row['rt_ratio']:.2f}\n"

    ax.text(
        0.05,
        0.95,
        text_content,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_title("Panel 6: Example Cases Analysis", fontsize=14)

    examples_path = os.path.join(output_dir, "panel6_example_cases.png")
    plt.savefig(examples_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["example_cases"] = examples_path

    return paths


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("=" * 60)
    print("POPULARITY PREDICTION: Viral vs Fizzle Classification")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load anomalies
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading anomalies...")
    anomaly_path = find_latest_anomaly_file()
    print(f"Using anomaly file: {anomaly_path}")
    anomalies = load_anomalies(anomaly_path)

    # -------------------------------------------------------------------------
    # Step 2: Merge consecutive anomalies into events
    # -------------------------------------------------------------------------
    print("\n[Step 2] Merging consecutive anomalies...")
    events = merge_anomalies(anomalies)

    # -------------------------------------------------------------------------
    # Step 3: Load tweets with content features
    # -------------------------------------------------------------------------
    print("\n[Step 3] Loading tweets with content features...")
    tweets = load_tweets_with_content()

    # -------------------------------------------------------------------------
    # Step 4: Extract features for each event
    # -------------------------------------------------------------------------
    print("\n[Step 4] Extracting features...")
    dataset_rows = []
    events_list = events.to_dicts()

    for i, event in enumerate(events_list):
        if (i + 1) % 50 == 0:
            print(f"  Processing event {i + 1}/{len(events_list)}...")

        features = extract_features(event, tweets, events)
        if features is not None:
            dataset_rows.append(features)

    print(
        f"Extracted features for {len(dataset_rows)} events (skipped {len(events_list) - len(dataset_rows)} with insufficient data)"
    )

    if len(dataset_rows) < 20:
        print("ERROR: Not enough samples to train a model. Need at least 20.")
        return

    # -------------------------------------------------------------------------
    # Step 5: Build dataset and calculate target
    # -------------------------------------------------------------------------
    print("\n[Step 5] Building dataset and calculating target...")
    dataset = pl.DataFrame(dataset_rows)

    # Calculate median engagement for binary split
    median_engagement = dataset["horizon_engagement"].median()
    print(f"Median horizon engagement: {median_engagement:.1f}")

    # Create binary target
    dataset = dataset.with_columns((pl.col("horizon_engagement") > median_engagement).cast(pl.Int32).alias("target"))

    # Show class distribution
    class_dist = dataset.group_by("target").len()
    print("Class distribution:")
    print(class_dist)

    # -------------------------------------------------------------------------
    # Step 6: Train Decision Tree
    # -------------------------------------------------------------------------
    print("\n[Step 6] Training Decision Tree...")

    X = dataset.select(FEATURE_NAMES).to_numpy()
    y = dataset["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    # Feature importances
    print("\nTop 5 Feature Importances:")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(5, len(FEATURE_NAMES))):
        print(f"  {FEATURE_NAMES[indices[i]]}: {importances[indices[i]]:.3f}")

    # -------------------------------------------------------------------------
    # Step 7: Create visualizations and log to MLflow
    # -------------------------------------------------------------------------
    print("\n[Step 7] Creating visualizations and logging to MLflow...")

    mlflow.set_tracking_uri(f"file://{repo_root}/mlruns")
    mlflow.set_experiment("popularity_prediction")

    with mlflow.start_run(run_name="decision_tree_classifier"):
        # Log parameters
        mlflow.log_params(
            {
                "topics": str(TOP_TOPICS),
                "peek_hours": PEEK_HOURS,
                "horizon_days": HORIZON_DAYS,
                "merge_gap_hours": MERGE_GAP_HOURS,
                "min_peek_tweets": MIN_PEEK_TWEETS,
                "tree_max_depth": TREE_MAX_DEPTH,
                "test_size": TEST_SIZE,
                "num_samples": len(dataset),
                "num_features": len(FEATURE_NAMES),
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "median_engagement": median_engagement,
            }
        )

        # Create visualizations and log artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create all visualization panels
            viz_paths = create_visualizations(clf, X_test, y_test, y_pred, dataset, tmpdir)

            # Save dataset
            dataset_path = os.path.join(tmpdir, "prediction_dataset.csv")
            dataset.write_csv(dataset_path)

            # Log all artifacts
            for name, path in viz_paths.items():
                mlflow.log_artifact(path)
            mlflow.log_artifact(dataset_path)

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total anomaly events processed: {len(events_list)}")
    print(f"Events with sufficient data: {len(dataset)}")
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    print(f"Accuracy: {acc:.1%} ({'PASS' if acc > 0.55 else 'FAIL'} - threshold 55%)")
    print(f"Top feature: {FEATURE_NAMES[indices[0]]} ({importances[indices[0]]:.1%})")

    # Sanity checks
    print("\nSanity Checks:")
    max_importance = importances.max()
    print(f"  Max feature importance < 50%: {'PASS' if max_importance < 0.5 else 'WARN'} ({max_importance:.1%})")

    viral_count = dataset.filter(pl.col("target") == 1).height
    fizzle_count = dataset.filter(pl.col("target") == 0).height
    balance_ratio = min(viral_count, fizzle_count) / max(viral_count, fizzle_count)
    print(f"  Class balance (min/max): {'PASS' if balance_ratio > 0.8 else 'WARN'} ({balance_ratio:.2f})")

    print(f"\nVisualization panels saved to MLflow run: {run_id}")
    print("Done!")


if __name__ == "__main__":
    main()
