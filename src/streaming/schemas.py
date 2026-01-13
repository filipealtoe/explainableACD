"""
Pydantic schemas for the streaming claim detection pipeline.

These schemas define the data structures used throughout the pipeline,
ensuring type safety and enabling serialization to parquet.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ClaimInfo(BaseModel):
    """
    Represents a normalized claim derived from one or more tweet clusters.

    A claim is the semantic centroid of a cluster, synthesized by the LLM
    into a human-readable factual statement.
    """

    claim_id: str = Field(description="UUID for the claim")
    claim_text: str = Field(description="Normalized claim text from Groq LLM")
    first_seen: datetime = Field(description="Timestamp of first detection")
    last_seen: datetime | None = Field(default=None, description="Timestamp of last observation")
    cluster_ids: list[int] = Field(default_factory=list, description="All cluster IDs mapped to this claim")
    total_tweets: int = Field(default=0, description="Cumulative tweet count across all clusters")
    total_engagement: int = Field(default=0, description="Cumulative engagement (RT + likes)")
    is_viral: bool | None = Field(default=None, description="Virality prediction")
    viral_confidence: float | None = Field(default=None, description="Prediction confidence [0-1]")
    lead_time_hours: float | None = Field(default=None, description="Hours from detection to peak (PRIMARY METRIC)")
    detection_z_score: float | None = Field(default=None, description="Z-score at first detection")
    kleinberg_state: int | None = Field(default=None, description="Kleinberg state at detection (0=normal, 1=elevated, 2=burst)")
    peak_engagement: int | None = Field(default=None, description="Maximum engagement in horizon window")
    peak_time: datetime | None = Field(default=None, description="Timestamp of peak engagement")


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
    trigger: Literal["zscore", "kleinberg", "both"] = Field(description="Which detector triggered")


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


class TimeseriesRecord(BaseModel):
    """
    Single record in the claim timeseries output.

    Used for the claim_timeseries.parquet output file.
    """

    claim_id: str
    timestamp: datetime
    tweet_count: int = Field(default=0)
    engagement: int = Field(default=0)
    z_score: float | None = Field(default=None)
    z_score_count: float | None = Field(default=None)
    z_score_engagement: float | None = Field(default=None)
    kleinberg_state: int | None = Field(default=None)
    cumulative_tweets: int = Field(default=0, description="Running total of tweets")
    cumulative_engagement: int = Field(default=0, description="Running total of engagement")
