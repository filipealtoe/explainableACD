"""
Visualize Topic Predictions: Multi-panel view showing time series, anomalies, and predictions.

For a specific topic, this creates a 4-panel visualization:
1. Tweet volume over time
2. Engagement over time
3. Composite z-score with anomaly threshold (red dashed line at z=3)
4. Prediction timeline showing Viral/Fizzle predictions for each anomaly event
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

matplotlib.use("Agg")

# Path setup
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# Import the prediction model components
from sklearn.tree import DecisionTreeClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================
TOP_TOPICS = [74, 0, 1, 2, 3, 4, 5, 6, 7, 8]
PEEK_HOURS = 6
HORIZON_DAYS = 7
MERGE_GAP_HOURS = 6
MIN_PEEK_TWEETS = 5
Z_THRESHOLD = 3.0

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
# DATA LOADING (reuse from run_popularity_prediction.py)
# =============================================================================
def find_best_anomaly_file(experiment_id: str = "360627897138559911") -> Path:
    """Find the anomalies_hourly_168h.csv with the most anomalies."""
    mlruns_dir = repo_root / "mlruns" / experiment_id
    best_file = None
    best_count = 0

    for run_dir in mlruns_dir.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("."):
            anomaly_file = run_dir / "artifacts" / "anomalies_hourly_168h.csv"
            if anomaly_file.exists():
                with open(anomaly_file) as f:
                    line_count = sum(1 for _ in f) - 1
                if line_count > best_count:
                    best_count = line_count
                    best_file = anomaly_file

    return best_file


def load_prediction_dataset() -> pl.DataFrame:
    """Load the prediction dataset from the latest popularity_prediction run."""
    # Find the popularity_prediction experiment
    mlruns_dir = repo_root / "mlruns"

    # Search for the prediction_dataset.csv
    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith("."):
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir():
                    dataset_file = run_dir / "artifacts" / "prediction_dataset.csv"
                    if dataset_file.exists():
                        return pl.read_csv(dataset_file)

    raise FileNotFoundError("prediction_dataset.csv not found in any MLflow run")


def load_timeseries_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the count and engagement timeseries."""
    # Load from the topic modeling run
    count_path = repo_root / "mlruns" / "4b7e95a391254fdaa79a1fb34c6a4a55" / "artifacts" / "topic_timeseries_H.csv"
    engagement_path = repo_root / "data" / "topic_timeseries_engagement_H.csv"

    count_ts = pl.read_csv(count_path)
    engagement_ts = pl.read_csv(engagement_path)

    return count_ts, engagement_ts


def train_model_and_predict(dataset: pl.DataFrame) -> tuple[DecisionTreeClassifier, np.ndarray]:
    """Train the decision tree and get predictions for all samples."""
    X = dataset.select(FEATURE_NAMES).to_numpy()
    y = dataset["target"].to_numpy()

    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)

    predictions = clf.predict(X)
    return clf, predictions


def calculate_composite_zscore(
    count_ts: pl.DataFrame, engagement_ts: pl.DataFrame, topic: int
) -> tuple[list, list, list]:
    """Calculate composite z-score for a topic over time."""
    # Get time columns (exclude 'topic' column)
    count_cols = [c for c in count_ts.columns if c != "topic"]
    eng_cols = [c for c in engagement_ts.columns if c != "topic"]

    # Find common columns
    common_cols = sorted(set(count_cols) & set(eng_cols))

    # Get data for this topic
    topic_count = count_ts.filter(pl.col("topic") == topic)
    topic_eng = engagement_ts.filter(pl.col("topic") == topic)

    if len(topic_count) == 0 or len(topic_eng) == 0:
        return [], [], []

    count_row = topic_count.to_dicts()[0]
    eng_row = topic_eng.to_dicts()[0]

    timestamps = []
    count_values = []
    eng_values = []
    z_composite = []

    # Calculate z-scores with 168h rolling window
    window_size = 168  # 7 days in hours

    for i, col in enumerate(common_cols):
        timestamps.append(col)
        count_val = count_row.get(col, 0) or 0
        eng_val = eng_row.get(col, 0) or 0

        count_values.append(count_val)
        eng_values.append(eng_val)

        # Calculate z-score using rolling window
        if i < window_size:
            # Not enough history, use global baseline
            count_baseline = count_values[: i + 1]
            eng_baseline = eng_values[: i + 1]
        else:
            count_baseline = count_values[i - window_size : i]
            eng_baseline = eng_values[i - window_size : i]

        # Z-score for count
        count_mean = np.mean(count_baseline)
        count_std = np.std(count_baseline)
        z_count = (count_val - count_mean) / max(count_std, 1e-6)

        # Z-score for engagement
        eng_mean = np.mean(eng_baseline)
        eng_std = np.std(eng_baseline)
        z_eng = (eng_val - eng_mean) / max(eng_std, 1e-6)

        # Composite
        z_comp = 0.4 * z_count + 0.6 * z_eng
        z_composite.append(z_comp)

    return timestamps, count_values, eng_values, z_composite


def create_topic_visualization(
    topic: int,
    timestamps: list,
    count_values: list,
    eng_values: list,
    z_composite: list,
    dataset: pl.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    """Create 4-panel visualization for a topic."""

    # Filter dataset for this topic
    topic_mask = dataset["topic"].to_numpy() == topic
    topic_dataset = dataset.filter(pl.col("topic") == topic)
    topic_predictions = predictions[topic_mask]
    topic_targets = topic_dataset["target"].to_numpy()

    # Parse event timestamps
    event_starts = topic_dataset["event_start"].to_list()

    # Convert timestamp strings to indices
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    # Create figure with 4 panels
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"Topic {topic}: Time Series, Anomalies, and Predictions", fontsize=14, fontweight="bold")

    # Convert timestamps to numeric for plotting
    x = range(len(timestamps))

    # -------------------------------------------------------------------------
    # Panel 1: Tweet Volume
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.fill_between(x, count_values, alpha=0.3, color="steelblue")
    ax1.plot(x, count_values, color="steelblue", linewidth=0.8)
    ax1.set_ylabel("Tweet Count", fontsize=10)
    ax1.set_title("Panel 1: Tweet Volume Over Time", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 2: Engagement
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.fill_between(x, eng_values, alpha=0.3, color="green")
    ax2.plot(x, eng_values, color="green", linewidth=0.8)
    ax2.set_ylabel("Total Engagement", fontsize=10)
    ax2.set_title("Panel 2: Engagement Over Time", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 3: Composite Z-Score with Threshold
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.fill_between(x, z_composite, alpha=0.3, color="purple")
    ax3.plot(x, z_composite, color="purple", linewidth=0.8, label="Composite Z-Score")

    # Red dashed threshold line
    ax3.axhline(y=Z_THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"Threshold (z={Z_THRESHOLD})")

    # Mark anomaly points (where z > threshold)
    anomaly_x = [i for i, z in enumerate(z_composite) if z > Z_THRESHOLD]
    anomaly_y = [z_composite[i] for i in anomaly_x]
    ax3.scatter(anomaly_x, anomaly_y, color="red", s=20, zorder=5, alpha=0.7)

    ax3.set_ylabel("Composite Z-Score", fontsize=10)
    ax3.set_title("Panel 3: Composite Z-Score (0.4×volume + 0.6×engagement) with Anomaly Threshold", fontsize=11)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 4: Prediction Timeline
    # -------------------------------------------------------------------------
    ax4 = axes[3]

    # Draw a baseline
    ax4.axhline(y=0.5, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Plot each anomaly event with prediction
    for i, (event_start, pred, actual) in enumerate(zip(event_starts, topic_predictions, topic_targets)):
        # Normalize event_start format: "2017-05-28T00:00:00.000000" -> "2017-05-28 00"
        event_str = str(event_start).replace("T", " ")[:13]  # "2017-05-28 00"

        # Find matching timestamp
        event_idx = None
        for ts_idx, ts in enumerate(timestamps):
            ts_normalized = ts[:13]  # "2017-01-26 11"
            if event_str == ts_normalized:
                event_idx = ts_idx
                break

        if event_idx is None:
            continue

        # Determine color and marker based on prediction and correctness
        correct = pred == actual

        if pred == 1:  # Predicted Viral
            color = "green" if correct else "lightgreen"
            marker = "^"  # Triangle up
            y_pos = 0.75
        else:  # Predicted Fizzle
            color = "red" if correct else "lightcoral"
            marker = "v"  # Triangle down
            y_pos = 0.25

        # Plot the prediction marker
        ax4.scatter(
            event_idx,
            y_pos,
            color=color,
            marker=marker,
            s=150,
            edgecolors="black" if correct else "gray",
            linewidths=1.5,
            zorder=5,
        )

        # Add vertical line to show event timing
        ax4.axvline(x=event_idx, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

    # Create legend for Panel 4
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="green",
            markersize=12,
            markeredgecolor="black",
            label="Predicted Viral (Correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="lightgreen",
            markersize=12,
            markeredgecolor="gray",
            label="Predicted Viral (Wrong)",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="red",
            markersize=12,
            markeredgecolor="black",
            label="Predicted Fizzle (Correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="lightcoral",
            markersize=12,
            markeredgecolor="gray",
            label="Predicted Fizzle (Wrong)",
        ),
    ]
    ax4.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax4.set_ylim(0, 1)
    ax4.set_yticks([0.25, 0.75])
    ax4.set_yticklabels(["Fizzle", "Viral"])
    ax4.set_ylabel("Prediction", fontsize=10)
    ax4.set_title("Panel 4: Anomaly Event Predictions (▲=Viral, ▼=Fizzle, dark=correct, light=wrong)", fontsize=11)
    ax4.grid(True, alpha=0.3, axis="x")

    # X-axis labels (show every Nth timestamp)
    n_ticks = 12
    tick_step = max(1, len(timestamps) // n_ticks)
    tick_positions = range(0, len(timestamps), tick_step)
    tick_labels = [timestamps[i][:10] for i in tick_positions]  # Show date only
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax4.set_xlabel("Date", fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to: {output_path}")


def main(topic_to_visualize: int = 0):
    print("=" * 60)
    print(f"TOPIC {topic_to_visualize} VISUALIZATION")
    print("=" * 60)

    if topic_to_visualize not in TOP_TOPICS:
        print(f"WARNING: Topic {topic_to_visualize} not in TOP_TOPICS: {TOP_TOPICS}")
        print("Continuing anyway...")

    # Load data
    print("\n[1] Loading timeseries data...")
    count_ts, engagement_ts = load_timeseries_data()

    print("\n[2] Loading prediction dataset...")
    dataset = load_prediction_dataset()
    print(f"    Loaded {len(dataset)} samples")

    print("\n[3] Training model and getting predictions...")
    clf, predictions = train_model_and_predict(dataset)

    print("\n[4] Calculating composite z-scores...")
    timestamps, count_values, eng_values, z_composite = calculate_composite_zscore(
        count_ts, engagement_ts, topic_to_visualize
    )
    print(f"    {len(timestamps)} time points")

    print("\n[5] Creating visualization...")

    # Create output directory
    output_dir = repo_root / "experiments" / "results" / "topic_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"topic_{topic_to_visualize}_predictions.png"

    create_topic_visualization(
        topic=topic_to_visualize,
        timestamps=timestamps,
        count_values=count_values,
        eng_values=eng_values,
        z_composite=z_composite,
        dataset=dataset,
        predictions=predictions,
        output_path=output_path,
    )

    # Print summary for this topic
    topic_data = dataset.filter(pl.col("topic") == topic_to_visualize)
    topic_preds = predictions[dataset["topic"].to_numpy() == topic_to_visualize]
    topic_targets = topic_data["target"].to_numpy()

    n_events = len(topic_data)
    n_viral_pred = sum(topic_preds == 1)
    n_fizzle_pred = sum(topic_preds == 0)
    n_correct = sum(topic_preds == topic_targets)
    accuracy = n_correct / n_events if n_events > 0 else 0

    print("\n" + "=" * 60)
    print(f"TOPIC {topic_to_visualize} SUMMARY")
    print("=" * 60)
    print(f"Total anomaly events: {n_events}")
    print(f"Predicted Viral: {n_viral_pred}")
    print(f"Predicted Fizzle: {n_fizzle_pred}")
    print(f"Correct predictions: {n_correct}/{n_events} ({accuracy:.1%})")
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize topic predictions")
    parser.add_argument(
        "--topic", "-t", type=int, default=0, help=f"Topic ID to visualize (default: 0). Available: {TOP_TOPICS}"
    )
    parser.add_argument("--all", "-a", action="store_true", help="Generate visualizations for all top topics")

    args = parser.parse_args()

    if args.all:
        for topic in TOP_TOPICS:
            print(f"\n{'#' * 70}")
            print(f"# GENERATING VISUALIZATION FOR TOPIC {topic}")
            print(f"{'#' * 70}")
            main(topic)
    else:
        main(args.topic)
