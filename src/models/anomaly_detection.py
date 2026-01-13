import logging
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import plotly.graph_objects as go
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_timeseries_from_file(file_path: str) -> pl.DataFrame:
    logger.info(f"Loading timeseries from {file_path}")
    df = pl.read_csv(file_path)
    logger.info(f"Loaded timeseries with shape {df.shape}")
    return df


def pivot_timeseries_to_long(df: pl.DataFrame, top_topics: list[int]) -> pl.DataFrame:
    logger.info("Converting wide format to long format")
    topic_col = df.columns[0]
    time_cols = [col for col in df.columns if col != topic_col]

    df_filtered = df.filter(pl.col(topic_col).is_in(top_topics))

    df_long = df_filtered.melt(id_vars=[topic_col], value_vars=time_cols, variable_name="timestamp", value_name="count")

    df_long = df_long.with_columns(pl.col("timestamp").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False))

    df_long = df_long.sort(["topic", "timestamp"])
    logger.info(f"Long format shape: {df_long.shape}")
    return df_long


def resample_to_daily(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Resampling to daily granularity")
    df_daily = (
        df.with_columns(pl.col("timestamp").dt.truncate("1d").alias("date"))
        .group_by(["topic", "date"])
        .agg(pl.col("count").sum().alias("count"))
        .sort(["topic", "date"])
        .rename({"date": "timestamp"})
    )

    logger.info(f"Daily data shape: {df_daily.shape}")
    return df_daily


def resample_to_weekly(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Resampling to weekly granularity")
    df_weekly = (
        df.with_columns(pl.col("timestamp").dt.truncate("1w").alias("week"))
        .group_by(["topic", "week"])
        .agg(pl.col("count").sum().alias("count"))
        .sort(["topic", "week"])
        .rename({"week": "timestamp"})
    )

    logger.info(f"Weekly data shape: {df_weekly.shape}")
    return df_weekly


def detect_anomalies_zscore(
    df: pl.DataFrame,
    window_size: int,
    threshold: float = 3.0,
    is_global: bool = False,
    min_count: int = 10,
    min_std: float = 2.0,
) -> pl.DataFrame:
    logger.info(f"Detecting anomalies with window_size={window_size}, threshold={threshold}, global={is_global}")
    logger.info(f"Filters: min_count={min_count}, min_std={min_std}")

    anomalies = []
    topics = df["topic"].unique().sort().to_list()

    for topic in topics:
        topic_data = df.filter(pl.col("topic") == topic).sort("timestamp")
        timestamps = topic_data["timestamp"].to_list()
        counts = topic_data["count"].to_numpy()

        for i in range(window_size, len(counts)):
            current_count = counts[i]
            current_time = timestamps[i]

            # Filter 1: Skip if current count is too low (likely noise)
            if current_count < min_count:
                continue

            if is_global:
                baseline = counts[:i]
            else:
                baseline = counts[i - window_size : i]

            if len(baseline) == 0:
                continue

            mean_baseline = np.mean(baseline)
            std_baseline = np.std(baseline)

            # Filter 2: Skip if baseline has no variance or very low variance
            if std_baseline < min_std:
                continue

            z_score = (current_count - mean_baseline) / std_baseline

            # Filter 3: Check z-score threshold
            if z_score > threshold:
                anomalies.append(
                    {
                        "topic": topic,
                        "timestamp": current_time,
                        "count": current_count,
                        "z_score": z_score,
                        "baseline_mean": mean_baseline,
                        "baseline_std": std_baseline,
                        "is_anomaly": True,
                    }
                )

    anomaly_df = pl.DataFrame(anomalies)
    logger.info(f"Found {len(anomalies)} anomalies (after filtering)")
    return anomaly_df


def create_timeseries_plot(
    df: pl.DataFrame, anomalies: pl.DataFrame, title: str, top_topics: list[int], topic_keywords: dict[int, list[str]]
) -> go.Figure:
    logger.info(f"Creating timeseries plot: {title}")

    fig = go.Figure()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, topic in enumerate(top_topics):
        topic_data = df.filter(pl.col("topic") == topic).sort("timestamp")
        keywords = topic_keywords.get(topic, [])[:3]
        label = f"Topic {topic}: {', '.join(keywords)}"

        fig.add_trace(
            go.Scatter(
                x=topic_data["timestamp"],
                y=topic_data["count"],
                mode="lines",
                name=label,
                line=dict(color=colors[idx % len(colors)], width=1.5),
                hovertemplate=f"{label}<br>Time: %{{x}}<br>Count: %{{y}}<extra></extra>",
            )
        )

        if len(anomalies) > 0:
            topic_anomalies = anomalies.filter(pl.col("topic") == topic)
        else:
            topic_anomalies = pl.DataFrame()

        if len(topic_anomalies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=topic_anomalies["timestamp"],
                    y=topic_anomalies["count"],
                    mode="markers",
                    name=f"Anomalies (Topic {topic})",
                    marker=dict(color="red", size=8, symbol="x"),
                    hovertemplate=f"{label}<br>Time: %{{x}}<br>Count: %{{y}}<br>Z-Score: %{{customdata:.2f}}<extra></extra>",
                    customdata=topic_anomalies["z_score"],
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Count",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    return fig


def create_heatmap(df: pl.DataFrame, anomalies: pl.DataFrame, title: str, top_topics: list[int]) -> go.Figure:
    logger.info(f"Creating heatmap: {title}")

    all_z_scores = []
    all_timestamps = []

    for topic in top_topics:
        topic_data = df.filter(pl.col("topic") == topic).sort("timestamp")
        timestamps = topic_data["timestamp"].to_list()
        counts = topic_data["count"].to_numpy()

        topic_z_scores = []
        for j in range(len(counts)):
            if j < 24:
                topic_z_scores.append(0)
            else:
                baseline = counts[max(0, j - 168) : j]
                if len(baseline) > 0 and np.std(baseline) > 0:
                    z = (counts[j] - np.mean(baseline)) / np.std(baseline)
                    topic_z_scores.append(z)
                else:
                    topic_z_scores.append(0)

        all_z_scores.append(topic_z_scores)
        if len(all_timestamps) == 0:
            all_timestamps = timestamps

    z_values = np.array(all_z_scores)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=[str(t) for t in all_timestamps],
            y=[f"Topic {t}" for t in top_topics],
            colorscale="RdYlGn_r",
            zmid=0,
            zmin=-3,
            zmax=6,
            colorbar=dict(title="Z-Score"),
            hovertemplate="Topic: %{y}<br>Time: %{x}<br>Z-Score: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title="Topic", height=500, width=1400, template="plotly_white"
    )

    return fig


def create_summary_chart(results: dict[str, pl.DataFrame], top_topics: list[int]) -> go.Figure:
    logger.info("Creating summary comparison chart")

    fig = go.Figure()

    granularities = list(results.keys())
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, topic in enumerate(top_topics):
        counts = []
        for gran in granularities:
            anomalies = results[gran]
            if len(anomalies) > 0:
                count = len(anomalies.filter(pl.col("topic") == topic))
            else:
                count = 0
            counts.append(count)

        fig.add_trace(go.Bar(name=f"Topic {topic}", x=granularities, y=counts, marker_color=colors[idx % len(colors)]))

    fig.update_layout(
        title="Anomaly Count by Granularity Level and Topic",
        xaxis_title="Granularity Level",
        yaxis_title="Number of Anomalies Detected",
        barmode="group",
        template="plotly_white",
        height=600,
        width=1200,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def run_anomaly_detection(
    run_id: str, top_topics: list[int], topic_keywords: dict[int, list[str]], z_threshold: float = 3.0
) -> str:
    logger.info("Starting multi-granularity anomaly detection")

    # Set MLflow tracking URI to repo root
    repo_root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"file://{repo_root}/mlruns")

    mlflow.set_experiment("anomaly_detection")

    with mlflow.start_run(run_name="multi_granularity_detection"):
        mlflow.log_params(
            {
                "source_run_id": run_id,
                "top_topics": str(top_topics),
                "z_threshold": z_threshold,
                "granularities": "hourly_24h, hourly_168h, daily_30d, weekly_global",
            }
        )

        df_hourly = load_timeseries_from_mlflow(run_id)
        df_long = pivot_timeseries_to_long(df_hourly, top_topics)

        df_daily = resample_to_daily(df_long)
        df_weekly = resample_to_weekly(df_daily)

        results = {}

        logger.info("=== Level 1: Hourly with 24h window ===")
        anomalies_h24 = detect_anomalies_zscore(df_long, window_size=24, threshold=z_threshold)
        results["hourly_24h"] = anomalies_h24
        mlflow.log_metric("anomalies_hourly_24h", len(anomalies_h24))

        logger.info("=== Level 2: Hourly with 168h (7d) window ===")
        anomalies_h168 = detect_anomalies_zscore(df_long, window_size=168, threshold=z_threshold)
        results["hourly_168h"] = anomalies_h168
        mlflow.log_metric("anomalies_hourly_168h", len(anomalies_h168))

        logger.info("=== Level 3: Daily with 30d window ===")
        anomalies_d30 = detect_anomalies_zscore(df_daily, window_size=30, threshold=z_threshold)
        results["daily_30d"] = anomalies_d30
        mlflow.log_metric("anomalies_daily_30d", len(anomalies_d30))

        logger.info("=== Level 4: Weekly with Global window ===")
        anomalies_w_global = detect_anomalies_zscore(df_weekly, window_size=4, threshold=z_threshold, is_global=True)
        results["weekly_global"] = anomalies_w_global
        mlflow.log_metric("anomalies_weekly_global", len(anomalies_w_global))

        with tempfile.TemporaryDirectory() as tmpdir:
            for level_name, anomalies in results.items():
                csv_path = os.path.join(tmpdir, f"anomalies_{level_name}.csv")
                anomalies.write_csv(csv_path)
                mlflow.log_artifact(csv_path)
                logger.info(f"Saved {level_name} anomalies CSV")

            fig1 = create_timeseries_plot(
                df_long, anomalies_h24, "Anomalies: Hourly (24h window)", top_topics, topic_keywords
            )
            html1 = os.path.join(tmpdir, "timeseries_hourly_24h.html")
            fig1.write_html(html1)
            mlflow.log_artifact(html1)

            fig2 = create_timeseries_plot(
                df_long, anomalies_h168, "Anomalies: Hourly (168h window)", top_topics, topic_keywords
            )
            html2 = os.path.join(tmpdir, "timeseries_hourly_168h.html")
            fig2.write_html(html2)
            mlflow.log_artifact(html2)

            fig3 = create_timeseries_plot(
                df_daily, anomalies_d30, "Anomalies: Daily (30d window)", top_topics, topic_keywords
            )
            html3 = os.path.join(tmpdir, "timeseries_daily_30d.html")
            fig3.write_html(html3)
            mlflow.log_artifact(html3)

            fig4 = create_timeseries_plot(
                df_weekly, anomalies_w_global, "Anomalies: Weekly (Global window)", top_topics, topic_keywords
            )
            html4 = os.path.join(tmpdir, "timeseries_weekly_global.html")
            fig4.write_html(html4)
            mlflow.log_artifact(html4)

            fig5 = create_heatmap(df_long, anomalies_h24, "Heatmap: Hourly (24h window)", top_topics)
            html5 = os.path.join(tmpdir, "heatmap_hourly_24h.html")
            fig5.write_html(html5)
            mlflow.log_artifact(html5)

            fig6 = create_heatmap(df_long, anomalies_h168, "Heatmap: Hourly (168h window)", top_topics)
            html6 = os.path.join(tmpdir, "heatmap_hourly_168h.html")
            fig6.write_html(html6)
            mlflow.log_artifact(html6)

            fig7 = create_summary_chart(results, top_topics)
            html7 = os.path.join(tmpdir, "summary_comparison.html")
            fig7.write_html(html7)
            mlflow.log_artifact(html7)

            logger.info("All visualizations saved to MLflow")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Multi-granularity anomaly detection complete. Run ID: {run_id}")
        return run_id


def run_anomaly_detection_from_files(
    timeseries_path: str, top_topics: list[int], topic_keywords: dict[int, list[str]], z_threshold: float = 3.0
) -> str:
    logger.info("Starting multi-granularity anomaly detection from files")

    # Set MLflow tracking URI to repo root
    repo_root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"file://{repo_root}/mlruns")

    mlflow.set_experiment("anomaly_detection")

    with mlflow.start_run(run_name="multi_granularity_detection"):
        mlflow.log_params(
            {
                "timeseries_path": timeseries_path,
                "top_topics": str(top_topics),
                "z_threshold": z_threshold,
                "min_count": 10,
                "min_std": 2.0,
                "granularities": "hourly_24h, hourly_168h, daily_30d, weekly_global",
            }
        )

        df_hourly = load_timeseries_from_file(timeseries_path)
        df_long = pivot_timeseries_to_long(df_hourly, top_topics)

        df_daily = resample_to_daily(df_long)
        df_weekly = resample_to_weekly(df_daily)

        results = {}

        logger.info("=== Level 1: Hourly with 24h window ===")
        anomalies_h24 = detect_anomalies_zscore(df_long, window_size=24, threshold=z_threshold)
        results["hourly_24h"] = anomalies_h24
        mlflow.log_metric("anomalies_hourly_24h", len(anomalies_h24))

        logger.info("=== Level 2: Hourly with 168h (7d) window ===")
        anomalies_h168 = detect_anomalies_zscore(df_long, window_size=168, threshold=z_threshold)
        results["hourly_168h"] = anomalies_h168
        mlflow.log_metric("anomalies_hourly_168h", len(anomalies_h168))

        logger.info("=== Level 3: Daily with 30d window ===")
        anomalies_d30 = detect_anomalies_zscore(df_daily, window_size=30, threshold=z_threshold)
        results["daily_30d"] = anomalies_d30
        mlflow.log_metric("anomalies_daily_30d", len(anomalies_d30))

        logger.info("=== Level 4: Weekly with Global window ===")
        anomalies_w_global = detect_anomalies_zscore(df_weekly, window_size=4, threshold=z_threshold, is_global=True)
        results["weekly_global"] = anomalies_w_global
        mlflow.log_metric("anomalies_weekly_global", len(anomalies_w_global))

        with tempfile.TemporaryDirectory() as tmpdir:
            for level_name, anomalies in results.items():
                csv_path = os.path.join(tmpdir, f"anomalies_{level_name}.csv")
                anomalies.write_csv(csv_path)
                mlflow.log_artifact(csv_path)
                logger.info(f"Saved {level_name} anomalies CSV")

            fig1 = create_timeseries_plot(
                df_long, anomalies_h24, "Anomalies: Hourly (24h window)", top_topics, topic_keywords
            )
            html1 = os.path.join(tmpdir, "timeseries_hourly_24h.html")
            fig1.write_html(html1)
            mlflow.log_artifact(html1)

            fig2 = create_timeseries_plot(
                df_long, anomalies_h168, "Anomalies: Hourly (168h window)", top_topics, topic_keywords
            )
            html2 = os.path.join(tmpdir, "timeseries_hourly_168h.html")
            fig2.write_html(html2)
            mlflow.log_artifact(html2)

            fig3 = create_timeseries_plot(
                df_daily, anomalies_d30, "Anomalies: Daily (30d window)", top_topics, topic_keywords
            )
            html3 = os.path.join(tmpdir, "timeseries_daily_30d.html")
            fig3.write_html(html3)
            mlflow.log_artifact(html3)

            fig4 = create_timeseries_plot(
                df_weekly, anomalies_w_global, "Anomalies: Weekly (Global window)", top_topics, topic_keywords
            )
            html4 = os.path.join(tmpdir, "timeseries_weekly_global.html")
            fig4.write_html(html4)
            mlflow.log_artifact(html4)

            fig5 = create_heatmap(df_long, anomalies_h24, "Heatmap: Hourly (24h window)", top_topics)
            html5 = os.path.join(tmpdir, "heatmap_hourly_24h.html")
            fig5.write_html(html5)
            mlflow.log_artifact(html5)

            fig6 = create_heatmap(df_long, anomalies_h168, "Heatmap: Hourly (168h window)", top_topics)
            html6 = os.path.join(tmpdir, "heatmap_hourly_168h.html")
            fig6.write_html(html6)
            mlflow.log_artifact(html6)

            fig7 = create_summary_chart(results, top_topics)
            html7 = os.path.join(tmpdir, "summary_comparison.html")
            fig7.write_html(html7)
            mlflow.log_artifact(html7)

            logger.info("All visualizations saved to MLflow")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Multi-granularity anomaly detection complete. Run ID: {run_id}")
        return run_id
