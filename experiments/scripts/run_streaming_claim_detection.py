#!/usr/bin/env python3
"""
Streaming Real-Time Claim Detection Pipeline Runner

This script runs the full streaming claim detection pipeline on the
US Elections tweets dataset, processing tweets in temporal order
and outputting prioritized claims.

Usage:
    python experiments/scripts/run_streaming_claim_detection.py

    # With custom config:
    python experiments/scripts/run_streaming_claim_detection.py --config experiments/configs/custom.yaml

    # Resume from checkpoint:
    python experiments/scripts/run_streaming_claim_detection.py --resume data/pipeline_output/streaming

Output:
    - data/pipeline_output/streaming/claims.parquet
    - data/pipeline_output/streaming/claim_timeseries.parquet
    - data/pipeline_output/streaming/summary.json

Reference: See /Users/sergiopinto/.claude/plans/snug-tickling-melody.md for architecture details.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(root / ".env")

import polars as pl
import yaml
from tqdm import tqdm

from src.streaming.pipeline import ClaimPipeline, PipelineConfig
from src.streaming.simulator import StreamingSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_mlflow(config: dict) -> None:
    """Initialize MLflow tracking if enabled."""
    mlflow_config = config.get("mlflow", {})
    if not mlflow_config.get("enabled", False):
        return

    try:
        import mlflow

        tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = mlflow_config.get("experiment_name", "streaming_claim_detection")
        mlflow.set_experiment(experiment_name)

        run_name = f"{mlflow_config.get('run_name_prefix', 'run')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)

        # Log config
        mlflow.log_params({
            "window_size": config.get("simulator", {}).get("window_size", "1h"),
            "z_threshold": config.get("anomaly", {}).get("z_threshold", 3.0),
            "dedup_threshold": config.get("registry", {}).get("dedup_threshold", 0.85),
            "groq_model": config.get("groq", {}).get("model", "llama-3.1-8b-instant"),
        })

        logger.info(f"MLflow tracking enabled: {experiment_name}/{run_name}")
    except ImportError:
        logger.warning("MLflow not installed, tracking disabled")


async def run_pipeline(
    config: dict,
    resume_from: str | None = None,
) -> None:
    """
    Run the streaming claim detection pipeline.

    Args:
        config: Configuration dictionary
        resume_from: Optional path to resume from checkpoint
    """
    # Extract settings
    data_config = config.get("data", {})
    simulator_config = config.get("simulator", {})
    output_config = config.get("output", {})

    output_dir = Path(output_config.get("output_dir", "data/pipeline_output/streaming"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize simulator
    logger.info("Initializing streaming simulator...")
    simulator = StreamingSimulator(
        data_path=data_config.get("file_path", "data/raw/us_elections_tweets.parquet"),
        time_column=data_config.get("time_column", "created_at"),
        text_column=data_config.get("text_column", "tweet"),
    )

    # Log dataset stats
    stats = simulator.get_date_range_stats()
    logger.info(f"Dataset: {stats['total_tweets']:,} tweets, {stats['total_days']} days")
    logger.info(f"Date range: {stats['start_date']} to {stats['end_date']}")

    # Initialize or load pipeline
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        pipeline = ClaimPipeline.load_state(resume_from, config)
    else:
        logger.info("Initializing pipeline...")
        pipeline = ClaimPipeline(config)

    # Get window parameters
    window_size = simulator_config.get("window_size", "1h")
    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")
    save_every = output_config.get("save_every_n_windows", 24)

    # Count total windows for progress bar (filtered by date range)
    window_counts = simulator.get_window_counts(window_size)
    if start_date or end_date:
        # Filter window counts to date range
        from datetime import datetime as dt
        start_dt = dt.fromisoformat(start_date) if start_date else None
        end_dt = dt.fromisoformat(end_date) if end_date else None

        filtered_counts = window_counts
        if start_dt:
            filtered_counts = filtered_counts.filter(pl.col("window_start") >= start_dt)
        if end_dt:
            filtered_counts = filtered_counts.filter(pl.col("window_start") < end_dt)
        total_windows = len(filtered_counts)
    else:
        total_windows = len(window_counts)

    logger.info(f"Processing {total_windows} windows (size={window_size})...")
    logger.info(f"Checkpointing every {save_every} windows to {output_dir}")

    # Process windows
    window_count = 0
    total_anomalies = 0
    total_claims = 0

    with tqdm(total=total_windows, desc="Processing", unit="window") as pbar:
        async for timestamp, df_window in simulator.iterate_windows_async(
            window_size=window_size,
            start_date=start_date,
            end_date=end_date,
        ):
            # Process window
            result = await pipeline.process_window(df_window, timestamp)

            # Update progress
            window_count += 1
            total_anomalies += result.anomalies_detected
            total_claims = len(pipeline.claim_registry.claims)

            pbar.set_postfix({
                "tweets": result.tweets_processed,
                "anomalies": total_anomalies,
                "claims": total_claims,
            })
            pbar.update(1)

            # Checkpoint
            if window_count % save_every == 0:
                logger.info(f"Checkpointing at window {window_count}...")
                pipeline.save_state(output_dir)

                # Log to MLflow
                try:
                    import mlflow

                    mlflow.log_metrics({
                        "windows_processed": window_count,
                        "total_anomalies": total_anomalies,
                        "total_claims": total_claims,
                    }, step=window_count)
                except (ImportError, Exception):
                    pass

    # Final save
    logger.info("Saving final outputs...")
    pipeline.save_outputs(output_dir)

    # Print summary
    stats = pipeline.get_stats()
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Windows processed: {stats['windows_processed']}")
    logger.info(f"Tweets processed: {stats['total_tweets_processed']:,}")
    logger.info(f"Tweets passed gate: {stats['total_tweets_passed_gate']:,} ({stats['total_tweets_passed_gate']/max(1,stats['total_tweets_processed'])*100:.1f}%)")
    logger.info(f"Anomalies detected: {stats['total_anomalies']}")
    logger.info(f"Claims created: {stats['total_claims']}")
    logger.info(f"Predictions made: {stats['total_predictions']}")

    # Claim registry stats
    registry_stats = stats.get("claim_registry", {})
    logger.info(f"Deduplication rate: {registry_stats.get('dedup_rate', 0)*100:.1f}%")
    logger.info(f"Avg clusters per claim: {registry_stats.get('avg_clusters_per_claim', 0):.2f}")

    # Groq stats
    groq_stats = stats.get("groq_adapter", {})
    logger.info(f"Groq API calls: {groq_stats.get('api_calls', 0)}")
    logger.info(f"Groq cache hits: {groq_stats.get('cache_hits', 0)}")

    logger.info("=" * 70)
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("  - claims.parquet")
    logger.info("  - claim_timeseries.parquet")
    logger.info("  - summary.json")
    logger.info("=" * 70)

    # Log final metrics to MLflow
    try:
        import mlflow

        mlflow.log_metrics({
            "final_windows_processed": stats["windows_processed"],
            "final_tweets_processed": stats["total_tweets_processed"],
            "final_anomalies": stats["total_anomalies"],
            "final_claims": stats["total_claims"],
            "final_predictions": stats["total_predictions"],
            "dedup_rate": registry_stats.get("dedup_rate", 0),
        })
        mlflow.log_artifact(str(output_dir / "summary.json"))
        mlflow.end_run()
    except (ImportError, Exception):
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run streaming claim detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config:
    python experiments/scripts/run_streaming_claim_detection.py

    # With custom config:
    python experiments/scripts/run_streaming_claim_detection.py --config experiments/configs/custom.yaml

    # Resume from checkpoint:
    python experiments/scripts/run_streaming_claim_detection.py --resume data/pipeline_output/streaming
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/streaming_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without processing",
    )

    args = parser.parse_args()

    # Load config
    config_path = root / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Validate config
    try:
        pipeline_config = PipelineConfig.from_dict(config)
        logger.info("Config validated successfully")
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run complete, exiting")
        return

    # Setup MLflow
    setup_mlflow(config)

    # Run pipeline
    asyncio.run(run_pipeline(config, args.resume))


if __name__ == "__main__":
    main()
