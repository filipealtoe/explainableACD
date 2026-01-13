#!/usr/bin/env python
"""
Claim Clustering Pipeline Orchestrator

Runs the full pipeline:
1. Load data
2. Claim Gate filtering
3. Embedding generation
4. Incremental clustering
5. Save results and log to MLflow
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import polars as pl
import yaml

from src.pipeline.modules import (
    ClaimExtractor,
    ClaimExtractorConfig,
    ClaimGate,
    ClaimGateConfig,
    Clusterer,
    ClustererConfig,
    Embedder,
    EmbedderConfig,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def setup_mlflow(config: dict) -> str:
    """Setup MLflow tracking."""
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "mlruns")
    experiment_name = mlflow_config.get("experiment_name", "claim_clustering")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return experiment_name


def load_data(config: dict) -> pl.DataFrame:
    """Load and optionally subset the data."""
    data_config = config["data"]
    input_file = Path(data_config["input_file"])

    logger.info(f"Loading data from {input_file}")
    df = pl.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} rows")

    # Apply subset if specified
    subset_size = config.get("subset_size")
    if subset_size:
        df = df.head(subset_size)
        logger.info(f"Subset to {len(df):,} rows")

    return df


def run_pipeline(config_path: Path, dry_run: bool = False) -> dict:
    """
    Run the full claim clustering pipeline.

    Args:
        config_path: Path to YAML configuration file
        dry_run: If True, only validate config without processing

    Returns:
        Dictionary with pipeline results and statistics
    """
    # Load config
    config = load_config(config_path)
    data_config = config["data"]

    if dry_run:
        logger.info("Dry run - validating config only")
        return {"status": "dry_run", "config": config}

    # Setup output directory
    output_dir = Path(data_config.get("output_dir", "data/pipeline_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    experiment_name = setup_mlflow(config)

    with mlflow.start_run(run_name="pipeline_run"):
        # Log config
        mlflow.log_params(
            {
                "input_file": data_config["input_file"],
                "subset_size": config.get("subset_size", "full"),
                "claim_gate_enabled": config["claim_gate"]["enabled"],
                "embedder_model": config["embedder"]["model_name"],
                "clusterer_threshold": config["clusterer"]["similarity_threshold"],
            }
        )

        results = {}

        # 1. Load data
        logger.info("=" * 60)
        logger.info("STEP 1: Loading data")
        logger.info("=" * 60)
        df = load_data(config)
        results["input_rows"] = len(df)

        # 2. Claim Gate
        logger.info("=" * 60)
        logger.info("STEP 2: Claim Gate filtering")
        logger.info("=" * 60)
        gate_config = ClaimGateConfig.from_dict(config["claim_gate"])
        gate = ClaimGate(gate_config)
        df = gate.apply(df, text_column=data_config["text_column"])

        gate_stats = gate.get_stats(df)
        results["claim_gate"] = gate_stats
        mlflow.log_metrics(
            {
                "claim_gate_passed": gate_stats["passed"],
                "claim_gate_filtered": gate_stats["filtered"],
                "claim_gate_pass_rate": gate_stats["pass_rate"],
            }
        )

        # 3. Embedder
        logger.info("=" * 60)
        logger.info("STEP 3: Generating embeddings")
        logger.info("=" * 60)
        embed_config = EmbedderConfig.from_dict(config["embedder"])
        embedder = Embedder(embed_config)
        df, embeddings = embedder.apply(
            df,
            text_column=data_config["text_column"],
            filter_column="passes_claim_gate",
            output_dir=output_dir,
        )

        embed_stats = embedder.get_stats(embeddings)
        results["embeddings"] = embed_stats
        mlflow.log_metrics(
            {
                "embedding_count": embed_stats["count"],
                "embedding_dim": embed_stats.get("dimension", 0),
            }
        )

        # 4. Clusterer
        logger.info("=" * 60)
        logger.info("STEP 4: Clustering")
        logger.info("=" * 60)
        cluster_config = ClustererConfig.from_dict(config["clusterer"])
        clusterer = Clusterer(cluster_config, embedding_dim=embedder.embedding_dim)
        df = clusterer.apply(
            df,
            embeddings,
            id_column=data_config["id_column"],
            embedding_idx_column="embedding_idx",
            output_dir=output_dir,
        )

        cluster_stats = clusterer.get_stats()
        results["clustering"] = cluster_stats
        mlflow.log_metrics(
            {
                "n_clusters": cluster_stats["n_clusters"],
                "clusters_size_gte_min": cluster_stats.get("clusters_size_gte_min", 0),
                "avg_cluster_size": cluster_stats.get("avg_cluster_size", 0),
            }
        )

        # 5. Claim Extractor (optional, requires API key)
        logger.info("=" * 60)
        logger.info("STEP 5: Claim extraction (LLM)")
        logger.info("=" * 60)
        extractor_config = ClaimExtractorConfig.from_dict(config.get("claim_extractor", {}))
        extractor = ClaimExtractor(extractor_config)

        if extractor_config.enabled and ClaimExtractor.is_available():
            df, claims = extractor.apply(df, text_column=data_config["text_column"])
            extractor_stats = extractor.get_stats(claims)
            results["claim_extractor"] = extractor_stats
            mlflow.log_metrics(
                {
                    "n_claims_extracted": extractor_stats.get("n_claims", 0),
                }
            )

            # Save claims separately
            if claims:
                claims_file = output_dir / "cluster_claims.json"
                import json

                with open(claims_file, "w") as f:
                    json.dump({str(k): v for k, v in claims.items()}, f, indent=2)
                mlflow.log_artifact(str(claims_file))
                logger.info(f"Saved {len(claims)} claims to {claims_file}")
        else:
            if not extractor_config.enabled:
                logger.info("Claim extractor disabled in config")
            else:
                logger.info("Claim extractor skipped (ANTHROPIC_API_KEY not set)")
            results["claim_extractor"] = {"n_claims": 0, "status": "skipped"}

        # 6. Save results
        logger.info("=" * 60)
        logger.info("STEP 6: Saving results")
        logger.info("=" * 60)
        output_file = output_dir / "tweets_clustered.parquet"
        df.write_parquet(output_file)
        logger.info(f"Saved clustered data to {output_file}")

        mlflow.log_artifact(str(output_file))
        results["output_file"] = str(output_file)
        results["output_rows"] = len(df)

        # Final summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Input rows:     {results['input_rows']:,}")
        logger.info(f"Gate passed:    {gate_stats['passed']:,} ({100 * gate_stats['pass_rate']:.1f}%)")
        logger.info(f"Embeddings:     {embed_stats['count']:,}")
        logger.info(f"Clusters:       {cluster_stats['n_clusters']:,}")
        logger.info(f"Output file:    {output_file}")

        run_id = mlflow.active_run().info.run_id
        results["mlflow_run_id"] = run_id
        logger.info(f"MLflow run ID:  {run_id}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run claim clustering pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/pipeline/config/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without processing",
    )
    args = parser.parse_args()

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    results = run_pipeline(args.config, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("Dry run complete - config is valid")
    else:
        logger.info(f"Pipeline complete. Results: {results}")


if __name__ == "__main__":
    main()
