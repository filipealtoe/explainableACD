"""
Pydantic schemas for the streaming claim detection pipeline.

These schemas define the data structures used throughout the pipeline,
ensuring type safety and enabling serialization to parquet.

Output files:
- users.parquet: Unique users with authority metrics
- tweets.parquet: All tweets with FKs (partitioned by date)
- clusters.parquet: Cluster metadata with FK to claims
- claims.parquet: Normalized claims (deduplicated)
- cluster_timeseries.parquet: Temporal evolution per cluster
- cluster_embeddings.npy: Centroid embeddings (separate file)
- cluster_id_to_idx.json: Mapping for embeddings
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# USER SCHEMA
# =============================================================================


class UserInfo(BaseModel):
    """
    User record for users.parquet.

    Stores user metadata at time of collection for authority features.
    """

    user_id: str = Field(description="Unique user identifier")
    username: str = Field(description="Twitter handle")
    followers_at_collection: int = Field(default=0, description="Follower count when collected")
    verified: bool = Field(default=False, description="Verified status at collection")


# =============================================================================
# TWEET SCHEMA
# =============================================================================


class TweetRecord(BaseModel):
    """
    Tweet record for tweets.parquet.

    Contains tweet data with FKs to users and clusters.
    Partitioned by date for efficient range queries.

    ALL columns from the raw data are preserved for downstream analysis.
    """

    # Core identifiers
    tweet_id: str = Field(description="Unique tweet identifier")
    user_id: str = Field(description="FK to users.parquet")
    cluster_id: int | None = Field(default=None, description="FK to clusters.parquet (null if not clustered)")

    # Tweet content
    text: str = Field(description="Tweet text content")
    created_at: datetime = Field(description="Tweet timestamp")

    # Engagement metrics
    retweet_count_at_collection: int = Field(default=0, description="RT count when collected")
    likes_at_collection: int = Field(default=0, description="Like count when collected")

    # Cluster assignment
    cluster_similarity: float | None = Field(default=None, description="Similarity to cluster centroid")
    passes_claim_gate: bool = Field(default=False, description="Whether tweet passed linguistic filter")

    # === RAW DATA COLUMNS (preserved from source) ===
    source: str | None = Field(default=None, description="Tweet source (e.g., Twitter for iPhone)")

    # User metadata
    user_name: str | None = Field(default=None, description="User display name")
    user_screen_name: str | None = Field(default=None, description="Twitter handle without @")
    user_description: str | None = Field(default=None, description="User bio")
    user_join_date: datetime | None = Field(default=None, description="When user joined Twitter")
    user_followers_count: int = Field(default=0, description="Follower count at collection")
    user_location: str | None = Field(default=None, description="User-specified location string")
    user_verified: bool = Field(default=False, description="Verified status at collection")

    # Geolocation (derived from user_location or tweet coordinates)
    lat: float | None = Field(default=None, description="Latitude")
    long: float | None = Field(default=None, description="Longitude")
    city: str | None = Field(default=None, description="City name")
    country: str | None = Field(default=None, description="Country name")
    continent: str | None = Field(default=None, description="Continent name")
    state: str | None = Field(default=None, description="State/province name")
    state_code: str | None = Field(default=None, description="State/province code")

    # Collection metadata
    collected_at: datetime | None = Field(default=None, description="When data was collected")
    hashtag_source: str | None = Field(default=None, description="Which hashtag stream this came from")


# =============================================================================
# CLUSTER SCHEMA
# =============================================================================


class ClusterInfo(BaseModel):
    """
    Cluster metadata for clusters.parquet.

    Clusters are groups of semantically similar tweets.
    Many clusters can map to one claim via deduplication.
    """

    cluster_id: int = Field(description="Unique cluster identifier")
    claim_id: str | None = Field(default=None, description="FK to claims.parquet (null if never anomalous)")
    first_seen: datetime = Field(description="When cluster was created")
    last_seen: datetime | None = Field(default=None, description="Last activity timestamp")
    tweet_count_total: int = Field(default=0, description="Total tweets assigned to cluster")
    unique_users_total: int = Field(default=0, description="Distinct users in cluster")

    # Geographic spread metrics (optional, may have high null rate due to sparse location data)
    unique_countries: int | None = Field(default=None, description="Number of distinct countries in cluster")
    unique_us_states: int | None = Field(default=None, description="Number of distinct US states")
    usa_ratio: float | None = Field(default=None, description="Proportion of geolocated tweets from USA")
    has_international_spread: bool | None = Field(default=None, description="True if tweets from >1 country")
    dominant_country: str | None = Field(default=None, description="Most common country in cluster")
    geographic_entropy: float | None = Field(default=None, description="Shannon entropy over country distribution")

    # Sentiment metrics with "one user, one vote" correction
    sentiment_mean: float | None = Field(default=None, description="Naive mean sentiment (tweet-weighted)")
    sentiment_one_vote: float | None = Field(default=None, description="User-weighted sentiment (one user, one vote)")
    sentiment_std: float | None = Field(default=None, description="Sentiment standard deviation across users")
    tweets_per_user: float | None = Field(default=None, description="Avg tweets per user (spam/coordination signal)")


# =============================================================================
# CLAIM SCHEMA
# =============================================================================


class ClaimInfo(BaseModel):
    """
    Represents a normalized claim derived from one or more tweet clusters.

    A claim is the semantic centroid of a cluster, synthesized by the LLM
    into a human-readable factual statement. Multiple clusters can map to
    the same claim via deduplication.
    """

    claim_id: str = Field(description="UUID for the claim")
    claim_text: str = Field(description="Normalized claim text from Groq LLM")
    first_seen: datetime = Field(description="Timestamp of first detection")
    last_seen: datetime | None = Field(default=None, description="Timestamp of last activity")

    # Trigger information (for explainability)
    trigger_type: Literal["zscore", "kleinberg", "both", "bayesian_cold_start", "bayesian_warm"] | None = Field(
        default=None, description="What triggered the anomaly detection (bayesian_* = cold-start enabled)"
    )
    trigger_cluster_id: int | None = Field(
        default=None, description="Which cluster triggered first detection"
    )
    detection_z_score: float | None = Field(default=None, description="Z-score at first detection")
    kleinberg_state: int | None = Field(
        default=None, description="Kleinberg state at detection (0=normal, 1=elevated, 2=burst)"
    )

    # Aggregated stats (computed from all mapped clusters)
    total_clusters: int = Field(default=1, description="Number of clusters mapped to this claim")
    total_tweets: int = Field(default=0, description="Cumulative tweet count across all clusters")
    total_engagement: int = Field(default=0, description="Cumulative engagement (RT + likes)")

    # Content enrichment
    keywords: list[str] = Field(default_factory=list, description="Top keywords extracted from cluster tweets")

    # Virality prediction
    is_viral: bool | None = Field(default=None, description="Ground truth: did it reach viral threshold?")
    viral_confidence: float | None = Field(default=None, description="Model prediction confidence [0-1]")

    # Post-hoc computed fields (filled after full run)
    peak_engagement: int | None = Field(default=None, description="Maximum engagement observed")
    peak_time: datetime | None = Field(default=None, description="Timestamp of peak engagement")
    lead_time_hours: float | None = Field(
        default=None, description="Hours from detection to peak (PRIMARY METRIC)"
    )

    # Legacy field (for backward compatibility, derivable via cluster FK)
    cluster_ids: list[int] = Field(default_factory=list, description="All cluster IDs mapped to this claim")

    # === CHECKWORTHINESS ASSESSMENT ===
    # Final prediction from ensemble model
    is_checkworthy: bool | None = Field(default=None, description="Final checkworthiness prediction")
    checkworthiness_prob: float | None = Field(
        default=None, description="Fused probability (0.6×deberta + 0.4×llm)"
    )

    # Component probabilities (for explainability)
    deberta_prob: float | None = Field(default=None, description="DeBERTa ensemble probability")
    llm_xgboost_prob: float | None = Field(default=None, description="XGBoost on LLM features probability")

    # Module-level assessments (0-100 confidence scores from LLM)
    checkability_score: float | None = Field(
        default=None, description="Is this a factual claim? (0-100)"
    )
    verifiability_score: float | None = Field(
        default=None, description="Can this be verified with public data? (0-100)"
    )
    harm_score: float | None = Field(
        default=None, description="Could this cause societal harm if spread? (0-100)"
    )


class AnomalyEvent(BaseModel):
    """
    Represents an anomaly detection event for a cluster.

    Anomalies are detected using an ensemble of Z-score (sudden spikes)
    and Kleinberg burst detection (sustained bursts).
    """

    cluster_id: int = Field(description="Cluster that triggered the anomaly")
    timestamp: datetime = Field(description="When the anomaly was detected")
    count: int = Field(description="Tweet count at detection")
    engagement: int = Field(description="Engagement at detection")
    z_score: float = Field(description="Composite Z-score (0.4*count + 0.6*engagement)")
    z_score_count: float = Field(description="Z-score for count only")
    z_score_engagement: float = Field(description="Z-score for engagement only")
    kleinberg_state: int = Field(description="Kleinberg state (0=normal, 1=elevated, 2=burst)")
    trigger: Literal["zscore", "kleinberg", "both", "bayesian_cold_start", "bayesian_warm"] = Field(
        description="Which detector triggered (bayesian_* = cold-start detector)"
    )


class ViralityPrediction(BaseModel):
    """
    Represents a virality prediction for a claim.

    Target: reaches 1000+ RTs within 4h of detection (time-bounded definition)
    """

    claim_id: str = Field(description="Claim this prediction is for")
    is_viral: bool = Field(description="Binary prediction: will go viral")
    confidence: float = Field(description="Model confidence [0-1]")
    lead_time_hours: float | None = Field(default=None, description="Hours from detection to peak")
    detection_time: datetime = Field(description="When prediction was made")
    peak_time: datetime | None = Field(default=None, description="Actual peak time (ground truth)")
    features: dict | None = Field(default=None, description="Feature values used for prediction")


class ClusterStats(BaseModel):
    """
    Aggregated statistics for a cluster within a time window.

    Used for tracking cluster evolution over time.
    """

    cluster_id: int
    timestamp: datetime
    tweet_count: int = Field(default=0)
    engagement: int = Field(default=0)
    unique_users: int = Field(default=0)
    avg_followers: float = Field(default=0.0)
    max_followers: int = Field(default=0)
    verified_count: int = Field(default=0)
    rt_count: int = Field(default=0)
    like_count: int = Field(default=0)


class WindowResult(BaseModel):
    """
    Result of processing a single time window.

    Contains aggregated statistics and events from the pipeline.
    """

    timestamp: datetime = Field(description="Window start timestamp")
    window_size: str = Field(default="1h", description="Window size (e.g., '1h', '1d')")
    tweets_processed: int = Field(default=0, description="Total tweets in window")
    tweets_passed_gate: int = Field(default=0, description="Tweets passing ClaimGate filter")
    tweets_embedded: int = Field(default=0, description="Tweets successfully embedded")
    clusters_updated: int = Field(default=0, description="Clusters that received new tweets")
    clusters_created: int = Field(default=0, description="New clusters created")
    anomalies_detected: int = Field(default=0, description="Anomaly events triggered")
    claims_normalized: int = Field(default=0, description="Claims normalized via LLM")
    claims_deduplicated: int = Field(default=0, description="Claims merged due to similarity")
    predictions_made: int = Field(default=0, description="Virality predictions made")
    anomaly_events: list[AnomalyEvent] = Field(default_factory=list)
    predictions: list[ViralityPrediction] = Field(default_factory=list)
    processing_time_seconds: float = Field(default=0.0, description="Wall-clock time to process window")


class ClusterTimeseriesRecord(BaseModel):
    """
    Single record tracking cluster evolution over time.

    Used for cluster_timeseries.parquet. Tracks at CLUSTER level (not claim level)
    because clusters evolve (receive tweets), claims are static (LLM output).

    Composite PK: (cluster_id, timestamp) must be unique.
    """

    # Keys
    cluster_id: int = Field(description="FK to clusters.parquet")
    timestamp: datetime = Field(description="Window start timestamp")
    claim_id: str | None = Field(default=None, description="FK to claims.parquet (denormalized for fast aggregation)")
    window_size_minutes: int = Field(default=60, description="Window size for reproducibility")

    # Activity metrics (this window)
    tweet_count: int = Field(default=0, description="Tweets in this window")
    engagement: int = Field(default=0, description="RT + likes in this window")
    unique_users: int = Field(default=0, description="Distinct users this window")

    # User authority features (pre-computed for ML)
    avg_followers: float = Field(default=0.0, description="Average follower count of users this window")
    max_followers: int = Field(default=0, description="Most influential user this window")
    verified_count: int = Field(default=0, description="Verified users this window")

    # Anomaly detection metrics
    z_score: float | None = Field(default=None, description="Composite z-score")
    z_score_count: float | None = Field(default=None, description="Z-score for count only")
    z_score_engagement: float | None = Field(default=None, description="Z-score for engagement only")
    kleinberg_state: int | None = Field(default=None, description="Burst state (0=normal, 1=elevated, 2=burst)")
    is_anomaly_trigger: bool = Field(default=False, description="Did this window trigger detection?")


# Legacy alias for backward compatibility
TimeseriesRecord = ClusterTimeseriesRecord
