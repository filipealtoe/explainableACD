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
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def _compute_geographic_entropy(countries: set[str], country_counts: Counter) -> float:
    """
    Compute Shannon entropy over country distribution.

    Higher entropy = more geographically spread out.
    Entropy = -sum(p * log2(p)) for each country

    Args:
        countries: Set of unique countries
        country_counts: Counter of country occurrences

    Returns:
        Shannon entropy (0 = single country, higher = more diverse)
    """
    if not countries or len(countries) <= 1:
        return 0.0

    total = sum(country_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for country in countries:
        count = country_counts.get(country, 0)
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _compute_one_user_one_vote_sentiment(
    user_sentiments: dict[str, list[float]]
) -> dict[str, float | None]:
    """
    Compute sentiment metrics with "one user, one vote" correction.

    Instead of averaging all tweets (which lets vocal users dominate),
    we first average per-user, then average across users.

    Args:
        user_sentiments: dict of user_id -> list of sentiment scores

    Returns:
        Dict with:
        - sentiment_mean: Naive tweet-weighted mean
        - sentiment_one_vote: User-weighted mean (one user, one vote)
        - sentiment_std: Std deviation across user means
        - tweets_per_user: Average tweets per user (coordination signal)
    """
    if not user_sentiments:
        return {
            "sentiment_mean": None,
            "sentiment_one_vote": None,
            "sentiment_std": None,
            "tweets_per_user": None,
        }

    # Compute per-user mean sentiment
    user_means = []
    all_sentiments = []

    for user_id, sentiments in user_sentiments.items():
        if sentiments:
            user_mean = sum(sentiments) / len(sentiments)
            user_means.append(user_mean)
            all_sentiments.extend(sentiments)

    if not user_means:
        return {
            "sentiment_mean": None,
            "sentiment_one_vote": None,
            "sentiment_std": None,
            "tweets_per_user": None,
        }

    # Naive mean (tweet-weighted)
    sentiment_mean = sum(all_sentiments) / len(all_sentiments)

    # One user, one vote (user-weighted)
    sentiment_one_vote = sum(user_means) / len(user_means)

    # Standard deviation across users
    if len(user_means) > 1:
        mean_of_means = sentiment_one_vote
        variance = sum((m - mean_of_means) ** 2 for m in user_means) / len(user_means)
        sentiment_std = math.sqrt(variance)
    else:
        sentiment_std = 0.0

    # Tweets per user (coordination/spam signal)
    tweets_per_user = len(all_sentiments) / len(user_means)

    return {
        "sentiment_mean": sentiment_mean,
        "sentiment_one_vote": sentiment_one_vote,
        "sentiment_std": sentiment_std,
        "tweets_per_user": tweets_per_user,
    }

from src.streaming.llm_normalizer import LLMNormalizerAdapter, LLMNormalizerConfig
from src.pipeline.modules.clusterer import Clusterer, ClustererConfig
from src.pipeline.modules.embedder import Embedder, EmbedderConfig
from src.streaming.anomaly_detector import (
    EnsembleAnomalyDetector,
    EnsembleAnomalyDetectorConfig,
)
from src.streaming.claim_gate import ClaimGate, ClaimGateConfig
from src.streaming.claim_registry import ClaimRegistry, ClaimRegistryConfig
from src.streaming.data_ingestion import DataIngestion, DataIngestionConfig
from src.streaming.text_preprocessor import TextPreprocessor, TextPreprocessorConfig
from src.streaming.embedding_storage import EmbeddingStorage
from src.streaming.schemas import (
    AnomalyEvent,
    ClusterInfo,
    ClusterTimeseriesRecord,
    TweetRecord,
    UserInfo,
    WindowResult,
)
from src.streaming.virality_predictor import ViralityPredictor, ViralityPredictorConfig

# Lazy import for checkworthiness (optional component)
_CheckworthinessPredictor = None
_CheckworthinessConfig = None


def _get_checkworthiness_classes():
    """Lazy load checkworthiness classes to avoid import errors when not needed."""
    global _CheckworthinessPredictor, _CheckworthinessConfig
    if _CheckworthinessPredictor is None:
        import sys
        from pathlib import Path
        # Add claim_checkworthiness package to path
        package_root = Path(__file__).resolve().parents[1] / "claim_checkworthiness"
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from src.checkworthiness.predictor import CheckworthinessPredictor, CheckworthinessConfig
        _CheckworthinessPredictor = CheckworthinessPredictor
        _CheckworthinessConfig = CheckworthinessConfig
    return _CheckworthinessPredictor, _CheckworthinessConfig


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    # Module configs
    data_ingestion: DataIngestionConfig | None = None
    text_preprocessor: TextPreprocessorConfig | None = None
    claim_gate: ClaimGateConfig | None = None
    embedder: EmbedderConfig | None = None
    clusterer: ClustererConfig | None = None
    anomaly: EnsembleAnomalyDetectorConfig | None = None
    normalization: LLMNormalizerConfig | None = None
    registry: ClaimRegistryConfig | None = None
    virality: ViralityPredictorConfig | None = None
    checkworthiness: dict | None = None  # CheckworthinessConfig as dict (optional)

    # Pipeline settings
    text_column: str = "tweet"
    time_column: str = "created_at"
    save_every_n_windows: int = 24  # Checkpoint every 24 hours

    # Timeseries settings
    dense_timeseries: bool = True  # If True, include all clusters in every window (with 0s for inactive)

    # Checkworthiness settings
    enable_checkworthiness: bool = False  # Whether to score checkworthiness (requires DeBERTa + LLM)
    checkworthiness_mode: str = "deberta"  # "full" (ensemble), "deberta" (fast), "llm" (no GPU)

    @classmethod
    def from_dict(cls, config: dict) -> "PipelineConfig":
        """Create pipeline config from nested dictionary."""
        return cls(
            data_ingestion=DataIngestionConfig.from_dict(config.get("data_ingestion", {}))
            if "data_ingestion" in config
            else None,
            text_preprocessor=TextPreprocessorConfig.from_dict(config.get("text_preprocessor", {}))
            if "text_preprocessor" in config
            else None,
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
            # Support both "normalization" (new) and "groq" (legacy) config keys
            normalization=LLMNormalizerConfig.from_dict(
                config.get("normalization", config.get("groq", {}))
            )
            if "normalization" in config or "groq" in config
            else None,
            registry=ClaimRegistryConfig.from_dict(config.get("registry", {}))
            if "registry" in config
            else None,
            virality=ViralityPredictorConfig.from_dict(config.get("virality", {}))
            if "virality" in config
            else None,
            checkworthiness=config.get("checkworthiness"),  # Pass as dict, handled in __init__
            text_column=config.get("text_column", "tweet"),
            time_column=config.get("time_column", "created_at"),
            save_every_n_windows=config.get("save_every_n_windows", 24),
            dense_timeseries=config.get("dense_timeseries", True),
            enable_checkworthiness=config.get("enable_checkworthiness", False),
            checkworthiness_mode=config.get("checkworthiness_mode", "deberta"),
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

        self.data_ingestion = DataIngestion(config.data_ingestion or DataIngestionConfig())

        self.text_preprocessor = TextPreprocessor(config.text_preprocessor or TextPreprocessorConfig())

        self.claim_gate = ClaimGate(config.claim_gate or ClaimGateConfig())

        self.embedder = Embedder(config.embedder or EmbedderConfig())

        self.clusterer = Clusterer(
            config.clusterer or ClustererConfig(),
            embedding_dim=self.embedder.embedding_dim,
        )

        self.anomaly_detector = EnsembleAnomalyDetector(
            config.anomaly or EnsembleAnomalyDetectorConfig()
        )

        self.normalizer = LLMNormalizerAdapter(config.normalization or LLMNormalizerConfig())

        self.claim_registry = ClaimRegistry(
            self.normalizer,
            self.embedder,
            config.registry or ClaimRegistryConfig(),
        )

        self.virality_predictor = ViralityPredictor(
            config.virality or ViralityPredictorConfig()
        )

        # Checkworthiness predictor (optional, disabled by default)
        self.checkworthiness_predictor = None
        self._checkworthiness_enabled = config.enable_checkworthiness
        self._checkworthiness_mode = config.checkworthiness_mode

        if config.enable_checkworthiness:
            CheckworthinessPredictor, CheckworthinessConfig = _get_checkworthiness_classes()
            cw_config = CheckworthinessConfig.from_dict(config.checkworthiness or {})
            self.checkworthiness_predictor = CheckworthinessPredictor(cw_config)
            logger.info(f"Checkworthiness predictor initialized (mode={config.checkworthiness_mode})")

        # Tracking
        self._window_results: list[WindowResult] = []
        self._timeseries_records: list[ClusterTimeseriesRecord] = []
        self._cluster_stats: dict[int, dict] = {}  # cluster_id -> cumulative stats

        # NEW: User and tweet tracking for normalized schema
        self._users: dict[str, UserInfo] = {}  # user_id -> UserInfo (deduped)
        self._tweets: list[TweetRecord] = []  # All tweets with cluster assignments

        # NEW: Cluster metadata tracking (for correct timestamps)
        self._cluster_first_seen: dict[int, datetime] = {}  # cluster_id -> first seen timestamp
        self._cluster_last_seen: dict[int, datetime] = {}  # cluster_id -> last activity timestamp
        self._cluster_users: dict[int, set[str]] = {}  # cluster_id -> set of user_ids

        # Geographic spread tracking (cumulative)
        self._cluster_countries: dict[int, set[str]] = {}  # cluster_id -> set of countries
        self._cluster_country_counts: dict[int, Counter] = {}  # cluster_id -> country count distribution
        self._cluster_us_states: dict[int, set[str]] = {}  # cluster_id -> set of US states
        self._cluster_geolocated_count: dict[int, int] = {}  # cluster_id -> count with location
        self._cluster_usa_count: dict[int, int] = {}  # cluster_id -> count from USA

        # Per-user sentiment tracking for "one user, one vote" aggregation
        # cluster_id -> user_id -> list of sentiment_compound values
        self._cluster_user_sentiments: dict[int, dict[str, list[float]]] = {}

        # Track all known cluster IDs (for dense timeseries)
        self._all_known_clusters: set[int] = set()

        logger.info("Pipeline initialized")

    def _get_cluster_tweets(self, df: pl.DataFrame, cluster_id: int) -> list[str]:
        """Get most representative tweets from a cluster for normalization.

        Selects tweets closest to the cluster centroid (highest similarity),
        ensuring the LLM sees the most semantically representative content.
        """
        cluster_df = df.filter(pl.col("cluster_id") == cluster_id)
        max_tweets = self.config.normalization.max_tweets_per_cluster if self.config.normalization else 5

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

    async def _score_checkworthiness(
        self,
        claim_ids: list[str],
        anomalies: list[AnomalyEvent],
    ) -> None:
        """
        Score checkworthiness for newly normalized claims.

        Uses the checkworthiness predictor in the configured mode:
        - "full": DeBERTa ensemble + LLM features + XGBoost (most accurate)
        - "deberta": DeBERTa only (fastest, no API calls)
        - "llm": LLM features + XGBoost only (no GPU needed)

        Updates ClaimInfo with checkworthiness scores.
        """
        if not self.checkworthiness_predictor:
            return

        # Get claim texts for scoring
        claim_texts = []
        claim_id_map = []

        for claim_id in claim_ids:
            claim = self.claim_registry.get_claim(claim_id)
            if claim and claim.is_checkworthy is None:  # Only score if not already scored
                claim_texts.append(claim.claim_text)
                claim_id_map.append(claim_id)

        if not claim_texts:
            return

        logger.debug(f"Scoring checkworthiness for {len(claim_texts)} claims (mode={self._checkworthiness_mode})")

        try:
            if self._checkworthiness_mode == "deberta":
                # Fast DeBERTa-only mode
                probs = self.checkworthiness_predictor.predict_deberta_only(claim_texts)

                for claim_id, prob in zip(claim_id_map, probs):
                    claim = self.claim_registry.claims[claim_id]
                    claim.deberta_prob = float(prob)
                    claim.checkworthiness_prob = float(prob)  # DeBERTa is final in this mode
                    claim.is_checkworthy = prob >= self.checkworthiness_predictor.config.threshold

            elif self._checkworthiness_mode == "full":
                # Full ensemble (DeBERTa + LLM + XGBoost)
                results = await self.checkworthiness_predictor.predict_batch(claim_texts, show_progress=False)

                for claim_id, result in zip(claim_id_map, results):
                    claim = self.claim_registry.claims[claim_id]
                    claim.deberta_prob = result.deberta_prob
                    claim.llm_xgboost_prob = result.llm_prob
                    claim.checkworthiness_prob = result.fused_prob
                    claim.is_checkworthy = result.is_checkworthy
                    claim.checkability_score = result.checkability_score
                    claim.verifiability_score = result.verifiability_score
                    claim.harm_score = result.harm_score

            logger.debug(f"Scored {len(claim_texts)} claims for checkworthiness")

        except Exception as e:
            logger.error(f"Checkworthiness scoring failed: {e}")

    def _track_users_and_tweets(self, df: pl.DataFrame, timestamp: datetime) -> None:
        """
        Track users, tweets, and cluster metadata for normalized schema output.

        Updates:
        - self._users: Unique users with authority metrics
        - self._tweets: All clustered tweets with FKs
        - self._cluster_first_seen / _last_seen: Temporal bounds
        - self._cluster_users: Unique users per cluster
        """
        # Only process tweets that passed ClaimGate and have valid cluster assignments
        clustered_df = df.filter(
            (pl.col("passes_claim_gate") == True) & (pl.col("cluster_id") >= 0)
        )

        if len(clustered_df) == 0:
            return

        # Convert to dict for row-by-row processing
        rows = clustered_df.to_dicts()

        for row in rows:
            cluster_id = int(row["cluster_id"])

            # --- Track cluster timestamps ---
            if cluster_id not in self._cluster_first_seen:
                self._cluster_first_seen[cluster_id] = timestamp
            self._cluster_last_seen[cluster_id] = timestamp

            # --- Track users (deduplicated) ---
            user_id = row.get("user_id") or row.get("author_id")
            if user_id:
                user_id = str(user_id)

                # Track unique users per cluster
                if cluster_id not in self._cluster_users:
                    self._cluster_users[cluster_id] = set()
                self._cluster_users[cluster_id].add(user_id)

            # --- Track geographic data ---
            country = row.get("country")
            if country and isinstance(country, str):
                # Initialize tracking structures if needed
                if cluster_id not in self._cluster_countries:
                    self._cluster_countries[cluster_id] = set()
                    self._cluster_country_counts[cluster_id] = Counter()
                    self._cluster_us_states[cluster_id] = set()
                    self._cluster_geolocated_count[cluster_id] = 0
                    self._cluster_usa_count[cluster_id] = 0

                self._cluster_countries[cluster_id].add(country)
                self._cluster_country_counts[cluster_id][country] += 1
                self._cluster_geolocated_count[cluster_id] += 1

                # Track USA-specific data
                if country == "USA":
                    self._cluster_usa_count[cluster_id] += 1
                    state = row.get("state")
                    if state and isinstance(state, str):
                        self._cluster_us_states[cluster_id].add(state)

            if user_id:
                # Add user if not seen before
                if user_id not in self._users:
                    self._users[user_id] = UserInfo(
                        user_id=user_id,
                        username=str(row.get("username", row.get("user_screen_name", ""))),
                        followers_at_collection=int(row.get("user_followers_count", 0)),
                        verified=bool(row.get("user_verified", False)),
                    )

                # --- Track per-user sentiment for "one user, one vote" ---
                sentiment = row.get("sentiment_compound")
                if sentiment is not None:
                    if cluster_id not in self._cluster_user_sentiments:
                        self._cluster_user_sentiments[cluster_id] = {}
                    if user_id not in self._cluster_user_sentiments[cluster_id]:
                        self._cluster_user_sentiments[cluster_id][user_id] = []
                    self._cluster_user_sentiments[cluster_id][user_id].append(float(sentiment))

            # --- Track tweets (ALL raw columns preserved) ---
            tweet_id = row.get("tweet_id") or row.get("id")
            if tweet_id:
                # Helper to safely convert to int, handling None and floats
                def safe_int(val, default=0):
                    if val is None:
                        return default
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return default

                # Helper to safely convert to float
                def safe_float(val, default=None):
                    if val is None:
                        return default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default

                tweet_record = TweetRecord(
                    # Core identifiers
                    tweet_id=str(tweet_id),
                    user_id=user_id or "",
                    cluster_id=cluster_id,

                    # Tweet content
                    text=str(row.get(self.config.text_column, "")),
                    created_at=row.get(self.config.time_column) or timestamp,

                    # Engagement metrics
                    retweet_count_at_collection=safe_int(row.get("retweet_count", 0)),
                    likes_at_collection=safe_int(row.get("likes", row.get("favorite_count", 0))),

                    # Cluster assignment
                    cluster_similarity=safe_float(row.get("cluster_similarity"), 0.0),
                    passes_claim_gate=bool(row.get("passes_claim_gate", False)),

                    # === RAW DATA COLUMNS ===
                    source=row.get("source"),

                    # User metadata
                    user_name=row.get("user_name"),
                    user_screen_name=row.get("user_screen_name"),
                    user_description=row.get("user_description"),
                    user_join_date=row.get("user_join_date"),
                    user_followers_count=safe_int(row.get("user_followers_count", 0)),
                    user_location=row.get("user_location"),
                    user_verified=bool(row.get("user_verified", False)),

                    # Geolocation
                    lat=safe_float(row.get("lat")),
                    long=safe_float(row.get("long")),
                    city=row.get("city"),
                    country=row.get("country"),
                    continent=row.get("continent"),
                    state=row.get("state"),
                    state_code=row.get("state_code"),

                    # Collection metadata
                    collected_at=row.get("collected_at"),
                    hashtag_source=row.get("hashtag_source"),
                )
                self._tweets.append(tweet_record)

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

        # ===== 0. DATA INGESTION (schema coercion, no filtering) =====
        df = self.data_ingestion.apply(df)

        # ===== 0.5. TEXT PREPROCESSING (creates tweet_clean for embedding) =====
        df = self.text_preprocessor.apply(df)

        # ===== 1. CLAIM GATE (uses original tweet text) =====
        df = self.claim_gate.apply(df, text_column=self.config.text_column)
        passing_count = df["passes_claim_gate"].sum()
        result.tweets_passed_gate = int(passing_count)

        if passing_count == 0:
            result.processing_time_seconds = time.time() - start_time
            return result

        # ===== 2. EMBEDDING (uses cleaned text for better clustering) =====
        # Use tweet_clean if available (from TextPreprocessor), otherwise fall back to raw text
        embed_text_col = "tweet_clean" if "tweet_clean" in df.columns else self.config.text_column
        df, embeddings = self.embedder.apply(
            df,
            text_column=embed_text_col,
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

        # ===== TRACK USERS, TWEETS, AND CLUSTER METADATA =====
        self._track_users_and_tweets(df, timestamp)

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
                        trigger_type=anomaly.trigger,
                        trigger_cluster_id=anomaly.cluster_id,
                    )

                    # Get cluster dataframe for feature extraction
                    cluster_df = df.filter(pl.col("cluster_id") == anomaly.cluster_id)

                    # Get timeseries history for this cluster (for enhanced 42-feature extraction)
                    # Convert Pydantic records to dicts for feature extraction
                    cluster_ts_history = [
                        r.model_dump() if hasattr(r, "model_dump") else r
                        for r in self._timeseries_records
                        if (r.cluster_id if hasattr(r, "cluster_id") else r.get("cluster_id")) == anomaly.cluster_id
                    ]

                    # Predict virality with enhanced features
                    prediction = self.virality_predictor.predict_from_df(
                        cluster_df,
                        z_score_at_detection=anomaly.z_score,
                        detection_time=timestamp,
                        timeseries_history=cluster_ts_history,
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

                # ===== 6.5. CHECKWORTHINESS SCORING (optional) =====
                if self._checkworthiness_enabled and self.checkworthiness_predictor:
                    await self._score_checkworthiness(claim_ids, anomalies)

        # ===== 7. CLUSTER TIMESERIES TRACKING =====
        # Record per-CLUSTER timeseries for evaluation
        # If dense_timeseries=True, include ALL known clusters (with 0s for inactive)
        # If dense_timeseries=False, only include clusters with activity in this window
        anomaly_cluster_ids = {a.cluster_id: a for a in anomalies}

        # Add newly seen clusters to the known set
        self._all_known_clusters.update(unique_clusters)

        # Determine which clusters to include in timeseries
        if self.config.dense_timeseries:
            # DENSE: All known clusters get a row (inactive ones get 0s)
            clusters_to_record = self._all_known_clusters
        else:
            # SPARSE: Only clusters with activity in this window
            clusters_to_record = unique_clusters

        for cluster_id in clusters_to_record:
            # Check if this cluster has activity in this window
            is_active = cluster_id in unique_clusters

            if is_active:
                cluster_df = df.filter(pl.col("cluster_id") == cluster_id)
                count, engagement = self._compute_cluster_stats(df, cluster_id)

                # Compute user authority metrics for this window
                unique_users = 0
                avg_followers = 0.0
                max_followers = 0
                verified_count = 0

                if "user_id" in cluster_df.columns:
                    unique_users = cluster_df["user_id"].n_unique()

                if "user_followers_count" in cluster_df.columns:
                    followers = cluster_df["user_followers_count"]
                    avg_followers = float(followers.mean() or 0.0)
                    max_followers = int(followers.max() or 0)

                if "user_verified" in cluster_df.columns:
                    verified_count = int(cluster_df["user_verified"].sum() or 0)
            else:
                # INACTIVE cluster in this window - set all metrics to 0
                count = 0
                engagement = 0
                unique_users = 0
                avg_followers = 0.0
                max_followers = 0
                verified_count = 0

            # Get claim_id if this cluster has one (denormalized for fast aggregation)
            claim_id = self.claim_registry.cluster_to_claim.get(cluster_id)

            # Check if this cluster triggered an anomaly
            is_anomaly = cluster_id in anomaly_cluster_ids
            anomaly = anomaly_cluster_ids.get(cluster_id)

            # Get detection scores (from anomaly if triggered, else from detector state)
            if anomaly:
                z_score = anomaly.z_score
                z_score_count = anomaly.z_score_count
                z_score_engagement = anomaly.z_score_engagement
                kleinberg_state = anomaly.kleinberg_state
            else:
                # Get current scores from detector (even if not anomalous)
                detector_stats = self.anomaly_detector.get_cluster_stats(cluster_id)
                z_score = detector_stats.get("z_score")
                z_score_count = detector_stats.get("z_score_count")
                z_score_engagement = detector_stats.get("z_score_engagement")
                kleinberg_state = detector_stats.get("kleinberg_state")

            # Create cluster timeseries record
            record = ClusterTimeseriesRecord(
                cluster_id=cluster_id,
                timestamp=timestamp,
                claim_id=claim_id,
                window_size_minutes=60,  # Default 1h windows
                tweet_count=count,
                engagement=engagement,
                unique_users=unique_users,
                avg_followers=avg_followers,
                max_followers=max_followers,
                verified_count=verified_count,
                z_score=z_score,
                z_score_count=z_score_count,
                z_score_engagement=z_score_engagement,
                kleinberg_state=kleinberg_state,
                is_anomaly_trigger=is_anomaly,
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
            "data_ingestion": self.data_ingestion.get_stats(),
            "text_preprocessor": self.text_preprocessor.get_stats(),
            "claim_gate": self.claim_gate.get_stats(pl.DataFrame()) if hasattr(self.claim_gate, "get_stats") else {},
            "clusterer": self.clusterer.get_stats(),
            "anomaly_detector": self.anomaly_detector.get_stats(),
            "claim_registry": self.claim_registry.get_stats(),
            "normalizer": self.normalizer.get_stats(),
            "virality_predictor": self.virality_predictor.get_stats(),
            "checkworthiness": {
                "enabled": self._checkworthiness_enabled,
                "mode": self._checkworthiness_mode if self._checkworthiness_enabled else None,
            },
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

        # NOTE: Timeseries is saved in save_outputs() as cluster_timeseries.parquet
        # We don't save it here to avoid duplication

        logger.info(f"Saved pipeline state to {output_dir}")

    def save_outputs(self, output_dir: Path | str) -> None:
        """
        Save final outputs with normalized schema.

        Creates:
        - users.parquet: Unique users with authority metrics
        - tweets.parquet: All tweets with FKs (cluster_id, user_id)
        - claims.parquet: Normalized claims (deduplicated)
        - clusters.parquet: Cluster metadata with FK to claims
        - cluster_timeseries.parquet: Temporal evolution per cluster
        - cluster_embeddings.npy: Centroid embeddings (separate file)
        - cluster_id_to_idx.json: Mapping for embeddings
        - summary.json: Pipeline statistics
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ===== 1. SAVE USERS =====
        if self._users:
            users_df = pl.DataFrame([u.model_dump() for u in self._users.values()])
            users_df.write_parquet(output_dir / "users.parquet")
            logger.info(f"Saved {len(self._users)} users to {output_dir / 'users.parquet'}")

        # ===== 2. SAVE TWEETS =====
        if self._tweets:
            tweets_df = pl.DataFrame([t.model_dump() for t in self._tweets])
            tweets_df.write_parquet(output_dir / "tweets.parquet")
            logger.info(f"Saved {len(self._tweets)} tweets to {output_dir / 'tweets.parquet'}")

        # ===== 3. SAVE CLAIMS =====
        claims = self.get_claims()
        if claims:
            claims_df = pl.DataFrame(claims)
            claims_df.write_parquet(output_dir / "claims.parquet")
            logger.info(f"Saved {len(claims)} claims to {output_dir / 'claims.parquet'}")

        # ===== 4. SAVE CLUSTERS =====
        # Build cluster info from clusterer state with CORRECT timestamps
        clusters = []
        cluster_state = self.clusterer.algorithm.state

        for cluster_id in range(cluster_state.n_clusters):
            # Get claim_id if this cluster has one
            claim_id = self.claim_registry.cluster_to_claim.get(cluster_id)

            # Get cluster stats
            stats = self._cluster_stats.get(cluster_id, {})

            # Use tracked timestamps (actual tweet timestamps, not current time!)
            first_seen = self._cluster_first_seen.get(cluster_id)
            last_seen = self._cluster_last_seen.get(cluster_id)

            # Get unique users count from tracked set
            unique_users = len(self._cluster_users.get(cluster_id, set()))

            # Compute geographic metrics (if any location data exists for this cluster)
            countries = self._cluster_countries.get(cluster_id, set())
            country_counts = self._cluster_country_counts.get(cluster_id, Counter())
            us_states = self._cluster_us_states.get(cluster_id, set())
            geolocated_count = self._cluster_geolocated_count.get(cluster_id, 0)
            usa_count = self._cluster_usa_count.get(cluster_id, 0)

            # Only set geographic fields if we have location data
            geo_kwargs = {}
            if geolocated_count > 0:
                geo_kwargs["unique_countries"] = len(countries)
                geo_kwargs["unique_us_states"] = len(us_states) if us_states else None
                geo_kwargs["usa_ratio"] = usa_count / geolocated_count
                geo_kwargs["has_international_spread"] = len(countries) > 1
                geo_kwargs["geographic_entropy"] = _compute_geographic_entropy(countries, country_counts)

                # Find dominant country
                if country_counts:
                    dominant = country_counts.most_common(1)[0][0]
                    geo_kwargs["dominant_country"] = dominant

            # Compute sentiment metrics with "one user, one vote" correction
            user_sentiments = self._cluster_user_sentiments.get(cluster_id, {})
            sentiment_metrics = _compute_one_user_one_vote_sentiment(user_sentiments)

            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                claim_id=claim_id,
                first_seen=first_seen or datetime.now(),  # Fallback only if never tracked
                last_seen=last_seen,
                tweet_count_total=stats.get("total_count", 0),
                unique_users_total=unique_users,
                **geo_kwargs,
                **sentiment_metrics,
            )
            clusters.append(cluster_info.model_dump())

        if clusters:
            clusters_df = pl.DataFrame(clusters)
            clusters_df.write_parquet(output_dir / "clusters.parquet")
            logger.info(f"Saved {len(clusters)} clusters to {output_dir / 'clusters.parquet'}")

        # ===== 5. SAVE CLUSTER TIMESERIES =====
        timeseries = self.get_timeseries()
        if timeseries:
            # Use high infer_schema_length to handle mixed None/float values
            ts_df = pl.DataFrame(timeseries, infer_schema_length=len(timeseries))
            ts_df.write_parquet(output_dir / "cluster_timeseries.parquet")
            logger.info(f"Saved {len(timeseries)} timeseries records to {output_dir / 'cluster_timeseries.parquet'}")

        # ===== 4. SAVE CLUSTER EMBEDDINGS =====
        # Extract centroids from clusterer and save separately
        embedding_storage = EmbeddingStorage(
            output_dir=output_dir,
            embedding_dim=self.embedder.embedding_dim,
        )

        # Add all cluster centroids
        if hasattr(cluster_state, 'centroids') and cluster_state.centroids is not None:
            for cluster_id in range(cluster_state.n_clusters):
                if cluster_id < len(cluster_state.centroids):
                    embedding_storage.add_embedding(cluster_id, cluster_state.centroids[cluster_id])

            embedding_storage.save(output_dir)
            logger.info(f"Saved {len(embedding_storage)} cluster embeddings to {output_dir}")

        # ===== 5. SAVE SUMMARY =====
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
                pipeline.normalizer,
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
