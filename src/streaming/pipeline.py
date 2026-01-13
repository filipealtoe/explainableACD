"""
ClaimPipeline: Async pipeline orchestrator for streaming claim detection.

This module ties together all pipeline components:
1. ClaimGate - Linguistic pre-filtering
2. Embedder - Sentence embeddings
3. Clusterer - OnlineCosineClustering
4. AnomalyDetector - Z-score + Kleinberg ensemble
5. ClaimRegistry - LLM normalization + deduplication
6. ViralityPredictor - XGBoost with user authority features

Usage:
    config = load_yaml("experiments/configs/streaming_config.yaml")
    pipeline = ClaimPipeline(config)

    async for timestamp, df_window in simulator.iterate_windows_async("1h"):
        result = await pipeline.process_window(df_window, timestamp)
        print(f"{timestamp}: {result.anomalies_detected} anomalies, {result.claims_normalized} claims")

    # Save outputs
    pipeline.save_outputs("data/pipeline_output")
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.pipeline.modules.claim_extractor import GroqAsyncAdapter, GroqAsyncAdapterConfig
from src.pipeline.modules.clusterer import Clusterer, ClustererConfig
from src.pipeline.modules.embedder import Embedder, EmbedderConfig
from src.streaming.anomaly_detector import (
    EnsembleAnomalyDetector,
    EnsembleAnomalyDetectorConfig,
)
from src.streaming.claim_gate import ClaimGate, ClaimGateConfig
from src.streaming.claim_registry import ClaimRegistry, ClaimRegistryConfig
from src.streaming.schemas import AnomalyEvent, TimeseriesRecord, WindowResult
from src.streaming.virality_predictor import ViralityPredictor, ViralityPredictorConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    # Module configs
    claim_gate: ClaimGateConfig | None = None
    embedder: EmbedderConfig | None = None
    clusterer: ClustererConfig | None = None
    anomaly: EnsembleAnomalyDetectorConfig | None = None
    groq: GroqAsyncAdapterConfig | None = None
    registry: ClaimRegistryConfig | None = None
    virality: ViralityPredictorConfig | None = None

    # Pipeline settings
    text_column: str = "tweet"
    time_column: str = "created_at"
    save_every_n_windows: int = 24  # Checkpoint every 24 hours

    @classmethod
    def from_dict(cls, config: dict) -> "PipelineConfig":
        """Create pipeline config from nested dictionary."""
        return cls(
            claim_gate=ClaimGateConfig.from_dict(config.get("claim_gate", {}))
            if "claim_gate" in config
            else None,
            embedder=EmbedderConfig.from_dict(config.get("embedder", {}))
            if "embedder" in config
            else None,
            clusterer=ClustererConfig.from_dict(config.get("clusterer", {}))
            if "clusterer" in config
            else None,
            anomaly=EnsembleAnomalyDetectorConfig.from_dict(config.get("anomaly", {}))
            if "anomaly" in config
            else None,
            groq=GroqAsyncAdapterConfig.from_dict(config.get("groq", {}))
            if "groq" in config
            else None,
            registry=ClaimRegistryConfig.from_dict(config.get("registry", {}))
            if "registry" in config
            else None,
            virality=ViralityPredictorConfig.from_dict(config.get("virality", {}))
            if "virality" in config
            else None,
            text_column=config.get("text_column", "tweet"),
            time_column=config.get("time_column", "created_at"),
            save_every_n_windows=config.get("save_every_n_windows", 24),
        )


class ClaimPipeline:
    """
    Async pipeline orchestrator for streaming claim detection.

    Processes tweets in temporal windows, detecting anomalies,
    normalizing claims, and predicting virality.
    """

    def __init__(self, config: PipelineConfig | dict):
        """
        Initialize the pipeline with all components.

        Args:
            config: PipelineConfig or dict with configuration
        """
        if isinstance(config, dict):
            config = PipelineConfig.from_dict(config)

        self.config = config

        # Initialize modules
        logger.info("Initializing pipeline modules...")

        self.claim_gate = ClaimGate(config.claim_gate or ClaimGateConfig())

        self.embedder = Embedder(config.embedder or EmbedderConfig())

        self.clusterer = Clusterer(
            config.clusterer or ClustererConfig(),
            embedding_dim=self.embedder.embedding_dim,
        )

        self.anomaly_detector = EnsembleAnomalyDetector(
            config.anomaly or EnsembleAnomalyDetectorConfig()
        )

        self.groq_adapter = GroqAsyncAdapter(config.groq or GroqAsyncAdapterConfig())

        self.claim_registry = ClaimRegistry(
            self.groq_adapter,
            self.embedder,
            config.registry or ClaimRegistryConfig(),
        )

        self.virality_predictor = ViralityPredictor(
            config.virality or ViralityPredictorConfig()
        )

        # Tracking
        self._window_results: list[WindowResult] = []
        self._timeseries_records: list[TimeseriesRecord] = []
        self._cluster_stats: dict[int, dict] = {}  # cluster_id -> cumulative stats

        logger.info("Pipeline initialized")

    def _get_cluster_tweets(self, df: pl.DataFrame, cluster_id: int) -> list[str]:
        """Get most representative tweets from a cluster for normalization.

        Selects tweets closest to the cluster centroid (highest similarity),
        ensuring the LLM sees the most semantically representative content.
        """
        cluster_df = df.filter(pl.col("cluster_id") == cluster_id)
        max_tweets = self.config.groq.max_tweets_per_cluster if self.config.groq else 5

        # Sort by similarity to centroid (most representative first)
        sorted_df = cluster_df.sort("cluster_similarity", descending=True)

        return sorted_df.head(max_tweets)[self.config.text_column].to_list()

    def _compute_cluster_stats(
        self, df: pl.DataFrame, cluster_id: int
    ) -> tuple[int, int]:
        """Compute count and engagement for a cluster in this window."""
        cluster_df = df.filter(pl.col("cluster_id") == cluster_id)
        count = len(cluster_df)

        engagement = 0
        if "retweet_count" in cluster_df.columns:
            engagement += int(cluster_df["retweet_count"].sum() or 0)
        if "likes" in cluster_df.columns:
            engagement += int(cluster_df["likes"].sum() or 0)

        return count, engagement

    async def process_window(
        self,
        df: pl.DataFrame,
        timestamp: datetime,
    ) -> WindowResult:
        """
        Process a single time window.

        This is the main pipeline method that:
        1. Filters tweets with ClaimGate
        2. Generates embeddings
        3. Assigns to clusters
        4. Detects anomalies
        5. Normalizes claims for anomalous clusters
        6. Predicts virality

        Args:
            df: DataFrame with tweets from this window
            timestamp: Window start timestamp

        Returns:
            WindowResult with processing statistics
        """
        start_time = time.time()

        result = WindowResult(
            timestamp=timestamp,
            tweets_processed=len(df),
        )

        if len(df) == 0:
            return result

        # ===== 1. CLAIM GATE =====
        df = self.claim_gate.apply(df, text_column=self.config.text_column)
        passing_count = df["passes_claim_gate"].sum()
        result.tweets_passed_gate = int(passing_count)

        if passing_count == 0:
            result.processing_time_seconds = time.time() - start_time
            return result

        # ===== 2. EMBEDDING =====
        df, embeddings = self.embedder.apply(
            df,
            text_column=self.config.text_column,
            filter_column="passes_claim_gate",
        )
        result.tweets_embedded = len(embeddings)

        if len(embeddings) == 0:
            result.processing_time_seconds = time.time() - start_time
            return result

        # ===== 3. CLUSTERING =====
        # Get tweet IDs for clustering
        passing_mask = df["passes_claim_gate"].to_numpy()
        tweet_ids = [f"{timestamp.isoformat()}_{i}" for i in range(len(df))]
        passing_tweet_ids = [tid for tid, m in zip(tweet_ids, passing_mask) if m]

        # Assign to clusters
        cluster_ids, similarities = self.clusterer.algorithm.assign_batch(
            embeddings, passing_tweet_ids, show_progress=False
        )

        # Add cluster info to dataframe
        cluster_id_full = np.full(len(df), -1, dtype=np.int32)
        similarity_full = np.zeros(len(df), dtype=np.float32)

        passing_indices = np.where(passing_mask)[0]
        for i, (cid, sim) in enumerate(zip(cluster_ids, similarities)):
            if i < len(passing_indices):
                cluster_id_full[passing_indices[i]] = cid
                similarity_full[passing_indices[i]] = sim

        df = df.with_columns(
            pl.Series(name="cluster_id", values=cluster_id_full),
            pl.Series(name="cluster_similarity", values=similarity_full),
        )

        # Count clusters
        unique_clusters = set(c for c in cluster_ids if c >= 0)
        result.clusters_updated = len(unique_clusters)

        # ===== 4. ANOMALY DETECTION =====
        anomalies: list[AnomalyEvent] = []

        for cluster_id in unique_clusters:
            count, engagement = self._compute_cluster_stats(df, cluster_id)

            # Update cumulative stats
            if cluster_id not in self._cluster_stats:
                self._cluster_stats[cluster_id] = {"total_count": 0, "total_engagement": 0}
            self._cluster_stats[cluster_id]["total_count"] += count
            self._cluster_stats[cluster_id]["total_engagement"] += engagement

            # Detect anomaly
            event = self.anomaly_detector.detect(
                cluster_id=cluster_id,
                timestamp=timestamp,
                count=count,
                engagement=engagement,
            )

            if event:
                anomalies.append(event)

        result.anomalies_detected = len(anomalies)
        result.anomaly_events = anomalies

        # ===== 5. CLAIM NORMALIZATION =====
        if anomalies:
            # Normalize claims for anomalous clusters
            clusters_to_normalize = [
                (a.cluster_id, self._get_cluster_tweets(df, a.cluster_id))
                for a in anomalies
            ]

            # Filter out clusters with no tweets
            clusters_to_normalize = [(cid, tweets) for cid, tweets in clusters_to_normalize if tweets]

            if clusters_to_normalize:
                claim_ids = await asyncio.gather(*[
                    self.claim_registry.get_or_create_claim(cid, tweets, timestamp)
                    for cid, tweets in clusters_to_normalize
                ])

                result.claims_normalized = len(claim_ids)

                # Check for deduplications
                registry_stats = self.claim_registry.get_stats()
                result.claims_deduplicated = registry_stats.get("deduplications", 0)

                # ===== 6. UPDATE CLAIM STATS & PREDICT VIRALITY =====
                for anomaly, claim_id in zip(anomalies, claim_ids):
                    # Update claim stats with anomaly event data
                    self.claim_registry.update_claim_stats(
                        claim_id,
                        timestamp,
                        tweet_count=anomaly.count,
                        engagement=anomaly.engagement,
                    )

                    # Set detection-time stats (only on first detection)
                    self.claim_registry.set_detection_stats(
                        claim_id,
                        z_score=anomaly.z_score,
                        kleinberg_state=anomaly.kleinberg_state,
                    )

                    # Get cluster dataframe for feature extraction
                    cluster_df = df.filter(pl.col("cluster_id") == anomaly.cluster_id)

                    # Predict virality
                    prediction = self.virality_predictor.predict_from_df(
                        cluster_df,
                        z_score_at_detection=anomaly.z_score,
                        detection_time=timestamp,
                    )
                    prediction.claim_id = claim_id

                    result.predictions.append(prediction)

                    # Update claim registry with prediction
                    self.claim_registry.set_virality_prediction(
                        claim_id,
                        prediction.is_viral,
                        prediction.confidence,
                    )

                result.predictions_made = len(result.predictions)

        # ===== 7. TIMESERIES TRACKING =====
        # Record per-claim timeseries for all active claims
        for claim_id, claim in self.claim_registry.claims.items():
            # Get all clusters for this claim
            total_count = 0
            total_engagement = 0

            for cluster_id in claim.cluster_ids:
                if cluster_id in self._cluster_stats:
                    stats = self._cluster_stats[cluster_id]
                    # Get this window's contribution (would need per-window tracking)
                    pass  # Simplified for now

            # For anomalous claims in this window, record timeseries
            for anomaly in anomalies:
                if self.claim_registry.cluster_to_claim.get(anomaly.cluster_id) == claim_id:
                    record = TimeseriesRecord(
                        claim_id=claim_id,
                        timestamp=timestamp,
                        tweet_count=anomaly.count,
                        engagement=anomaly.engagement,
                        z_score=anomaly.z_score,
                        z_score_count=anomaly.z_score_count,
                        z_score_engagement=anomaly.z_score_engagement,
                        kleinberg_state=anomaly.kleinberg_state,
                    )
                    self._timeseries_records.append(record)

        result.processing_time_seconds = time.time() - start_time
        self._window_results.append(result)

        return result

    def get_claims(self) -> list[dict]:
        """Get all claims as dictionaries."""
        return [claim.model_dump() for claim in self.claim_registry.get_all_claims()]

    def get_timeseries(self) -> list[dict]:
        """Get all timeseries records as dictionaries."""
        return [record.model_dump() for record in self._timeseries_records]

    def get_window_results(self) -> list[dict]:
        """Get all window results as dictionaries."""
        return [result.model_dump() for result in self._window_results]

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            "windows_processed": len(self._window_results),
            "total_tweets_processed": sum(r.tweets_processed for r in self._window_results),
            "total_tweets_passed_gate": sum(r.tweets_passed_gate for r in self._window_results),
            "total_anomalies": sum(r.anomalies_detected for r in self._window_results),
            "total_claims": len(self.claim_registry.claims),
            "total_predictions": sum(r.predictions_made for r in self._window_results),
            "claim_gate": self.claim_gate.get_stats(pl.DataFrame()) if hasattr(self.claim_gate, "get_stats") else {},
            "clusterer": self.clusterer.get_stats(),
            "anomaly_detector": self.anomaly_detector.get_stats(),
            "claim_registry": self.claim_registry.get_stats(),
            "groq_adapter": self.groq_adapter.get_stats(),
            "virality_predictor": self.virality_predictor.get_stats(),
        }

    def save_state(self, output_dir: Path | str) -> None:
        """
        Save pipeline state for checkpointing.

        Saves:
        - Clusterer state
        - Claim registry
        - Window results
        - Timeseries records
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save clusterer state
        self.clusterer.algorithm.state.save(output_dir / "clusterer_state.npz")

        # Save claim registry
        self.claim_registry.save(output_dir / "registry")

        # Save window results as parquet
        if self._window_results:
            results_df = pl.DataFrame([r.model_dump() for r in self._window_results])
            results_df.write_parquet(output_dir / "window_results.parquet")

        # Save timeseries as parquet
        if self._timeseries_records:
            ts_df = pl.DataFrame([r.model_dump() for r in self._timeseries_records])
            ts_df.write_parquet(output_dir / "claim_timeseries.parquet")

        logger.info(f"Saved pipeline state to {output_dir}")

    def save_outputs(self, output_dir: Path | str) -> None:
        """
        Save final outputs.

        Creates:
        - claims.parquet: All claims with metadata
        - claim_timeseries.parquet: Temporal evolution of claims
        - summary.json: Pipeline statistics
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save claims
        claims = self.get_claims()
        if claims:
            claims_df = pl.DataFrame(claims)
            claims_df.write_parquet(output_dir / "claims.parquet")
            logger.info(f"Saved {len(claims)} claims to {output_dir / 'claims.parquet'}")

        # Save timeseries
        timeseries = self.get_timeseries()
        if timeseries:
            ts_df = pl.DataFrame(timeseries)
            ts_df.write_parquet(output_dir / "claim_timeseries.parquet")
            logger.info(f"Saved {len(timeseries)} timeseries records to {output_dir / 'claim_timeseries.parquet'}")

        # Save summary
        stats = self.get_stats()
        with open(output_dir / "summary.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Saved summary to {output_dir / 'summary.json'}")

    @classmethod
    def load_state(
        cls,
        output_dir: Path | str,
        config: PipelineConfig | dict,
    ) -> "ClaimPipeline":
        """
        Load pipeline from saved state.

        Args:
            output_dir: Directory containing saved state
            config: Pipeline configuration

        Returns:
            ClaimPipeline with restored state
        """
        output_dir = Path(output_dir)

        # Create pipeline
        pipeline = cls(config)

        # Load clusterer state
        state_file = output_dir / "clusterer_state.npz"
        if state_file.exists():
            from src.pipeline.modules.clusterer import ClusterState

            pipeline.clusterer.algorithm.state = ClusterState.load(state_file)
            logger.info("Loaded clusterer state")

        # Load claim registry
        registry_dir = output_dir / "registry"
        if registry_dir.exists():
            pipeline.claim_registry = ClaimRegistry.load(
                registry_dir,
                pipeline.groq_adapter,
                pipeline.embedder,
                pipeline.config.registry,
            )
            logger.info("Loaded claim registry")

        # Load window results
        results_file = output_dir / "window_results.parquet"
        if results_file.exists():
            results_df = pl.read_parquet(results_file)
            pipeline._window_results = [
                WindowResult(**row) for row in results_df.to_dicts()
            ]
            logger.info(f"Loaded {len(pipeline._window_results)} window results")

        # Load timeseries
        ts_file = output_dir / "claim_timeseries.parquet"
        if ts_file.exists():
            ts_df = pl.read_parquet(ts_file)
            pipeline._timeseries_records = [
                TimeseriesRecord(**row) for row in ts_df.to_dicts()
            ]
            logger.info(f"Loaded {len(pipeline._timeseries_records)} timeseries records")

        return pipeline
