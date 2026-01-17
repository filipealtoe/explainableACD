import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def create_validation_plot(
    topic: int,
    topic_keywords: dict,
    count_ts: pl.DataFrame,
    engagement_ts: pl.DataFrame,
    composite_ts: pl.DataFrame,
    anomalies: pl.DataFrame,
    common_time_cols: list[str],
) -> go.Figure:
    """Create 3-panel validation plot for a single topic."""

    keywords = ", ".join(topic_keywords.get(topic, [])[:3])

    # Get data for this topic
    topic_count = count_ts.filter(pl.col("topic") == topic)
    topic_engagement = engagement_ts.filter(pl.col("topic") == topic)
    topic_composite = composite_ts.filter(pl.col("topic") == topic)

    if len(topic_count) == 0:
        return None

    # Use only common time columns
    time_cols = common_time_cols

    # Extract values
    count_row = topic_count.to_dicts()[0]
    engagement_row = topic_engagement.to_dicts()[0]
    composite_row = topic_composite.to_dicts()[0]

    timestamps = time_cols
    count_values = [count_row[col] for col in time_cols]
    engagement_values = [engagement_row[col] for col in time_cols]
    composite_values = [composite_row[col] for col in time_cols]

    # Filter anomalies for this topic
    topic_anomalies = anomalies.filter(pl.col("topic") == topic) if len(anomalies) > 0 else pl.DataFrame()

    # Create subplot figure
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Tweet Count over Time",
            "Total Engagement over Time",
            "Composite Z-Score (0.4×count + 0.6×engagement)",
        ),
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.34],
    )

    # Panel 1: Tweet Count
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=count_values,
            mode="lines",
            name="Tweet Count",
            line=dict(color="#1f77b4", width=1.5),
            hovertemplate="Time: %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Engagement
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=engagement_values,
            mode="lines",
            name="Total Engagement",
            line=dict(color="#2ca02c", width=1.5),
            hovertemplate="Time: %{x}<br>Engagement: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Panel 3: Composite Z-Score
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=composite_values,
            mode="lines",
            name="Composite Score",
            line=dict(color="#9467bd", width=1.5),
            hovertemplate="Time: %{x}<br>Z-Score: %{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Add threshold line at z=3.0 in panel 3
    fig.add_hline(
        y=3.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Anomaly Threshold (z=3.0)",
        annotation_position="right",
        row=3,
        col=1,
    )

    # Add anomaly markers to all three panels
    if len(topic_anomalies) > 0:
        anomaly_timestamps = topic_anomalies["timestamp"].to_list()

        # Find corresponding values for each anomaly timestamp
        for ts in anomaly_timestamps:
            if ts in timestamps:
                idx = timestamps.index(ts)

                # Add red X to panel 1 (count)
                fig.add_trace(
                    go.Scatter(
                        x=[ts],
                        y=[count_values[idx]],
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="x", line=dict(width=2)),
                        showlegend=False,
                        hovertemplate=f"ANOMALY<br>Time: {ts}<br>Count: {count_values[idx]}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                # Add red X to panel 2 (engagement)
                fig.add_trace(
                    go.Scatter(
                        x=[ts],
                        y=[engagement_values[idx]],
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="x", line=dict(width=2)),
                        showlegend=False,
                        hovertemplate=f"ANOMALY<br>Time: {ts}<br>Engagement: {engagement_values[idx]}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

                # Add red X to panel 3 (composite)
                fig.add_trace(
                    go.Scatter(
                        x=[ts],
                        y=[composite_values[idx]],
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="x", line=dict(width=2)),
                        showlegend=False,
                        hovertemplate=f"ANOMALY<br>Time: {ts}<br>Z-Score: {composite_values[idx]:.2f}<extra></extra>",
                    ),
                    row=3,
                    col=1,
                )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Topic {topic}: {keywords}</b><br><sub>Validation Plot - Anomaly Detection Analysis</sub>",
            x=0.5,
            xanchor="center",
        ),
        height=900,
        width=1400,
        template="plotly_white",
        showlegend=False,
        hovermode="x",
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Tweet Count", row=1, col=1)
    fig.update_yaxes(title_text="Total Engagement", row=2, col=1)
    fig.update_yaxes(title_text="Composite Z-Score", row=3, col=1)

    return fig


def main() -> None:
    print("=" * 60)
    print("Creating Validation Plots for Anomaly Detection")
    print("=" * 60)
    print()

    # Load data
    print("Loading timeseries data...")
    count_ts = pl.read_csv(
        repo_root / "mlruns" / "4b7e95a391254fdaa79a1fb34c6a4a55" / "artifacts" / "topic_timeseries_H.csv"
    )
    engagement_ts = pl.read_csv(repo_root / "data" / "topic_timeseries_engagement_H.csv")
    composite_ts = pl.read_csv(repo_root / "data" / "topic_timeseries_combined_H.csv")

    # Find the most recent combined anomaly detection run
    print("Finding most recent combined anomaly detection run...")
    mlruns_dir = repo_root / "mlruns" / "360627897138559911"

    # Get all run directories
    run_dirs = [d for d in mlruns_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not run_dirs:
        print("ERROR: No anomaly detection runs found!")
        return

    # Sort by modification time to get most recent
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_run = run_dirs[0]

    print(f"Using run: {latest_run.name}")

    # Load anomalies - try all granularity levels
    anomalies_all = []
    for granularity in ["hourly_24h", "hourly_168h", "daily_30d", "weekly_global"]:
        anomaly_path = latest_run / "artifacts" / f"anomalies_{granularity}.csv"
        if anomaly_path.exists():
            df = pl.read_csv(anomaly_path)
            print(f"  Loaded {len(df)} anomalies from {granularity}")
            anomalies_all.append(df)

    # Combine all anomalies (we'll use hourly_168h as it's comprehensive)
    if anomalies_all:
        anomalies = anomalies_all[1] if len(anomalies_all) > 1 else anomalies_all[0]  # Use hourly_168h
        print(f"\nUsing {len(anomalies)} anomalies for validation plots")
    else:
        print("ERROR: No anomaly files found!")
        return

    # Load topic keywords
    print("Loading topic keywords...")
    keywords_path = repo_root / "mlruns" / "4b7e95a391254fdaa79a1fb34c6a4a55" / "artifacts" / "topic_keywords.json"
    with open(keywords_path, encoding="utf-8") as f:
        topic_keywords = {int(k): v for k, v in json.load(f).items()}

    # Top 10 topics
    top_topics = [74, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Find common time columns across all three timeseries
    print("Finding common timestamps across all timeseries...")
    count_cols = set(count_ts.columns)
    engagement_cols = set(engagement_ts.columns)
    composite_cols = set(composite_ts.columns)

    common_cols = count_cols & engagement_cols & composite_cols
    common_cols.discard("topic")
    common_time_cols = sorted(list(common_cols))

    print(f"Common time points: {len(common_time_cols)}")

    # Create output directory
    output_dir = repo_root / "experiments" / "results" / "validation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating validation plots for {len(top_topics)} topics...")
    print(f"Output directory: {output_dir}")
    print()

    # Create plots for each topic
    for topic in top_topics:
        keywords = ", ".join(topic_keywords.get(topic, [])[:3])
        print(f"  Creating plot for Topic {topic}: {keywords}")

        fig = create_validation_plot(
            topic=topic,
            topic_keywords=topic_keywords,
            count_ts=count_ts,
            engagement_ts=engagement_ts,
            composite_ts=composite_ts,
            anomalies=anomalies,
            common_time_cols=common_time_cols,
        )

        if fig is not None:
            output_path = output_dir / f"validation_topic_{topic}.html"
            fig.write_html(output_path)
            print(f"    ✓ Saved to {output_path.name}")

    print()
    print("=" * 60)
    print("✓ All validation plots created!")
    print(f"📁 Location: {output_dir}")
    print("=" * 60)
    print("\nOpen these HTML files in your browser to validate anomalies:")
    print("  - Top panel: Tweet count (shows if volume spiked)")
    print("  - Middle panel: Engagement (shows if interactions spiked)")
    print("  - Bottom panel: Composite z-score (shows combined metric)")
    print("  - Red X: Detected anomalies")
    print()


if __name__ == "__main__":
    main()
