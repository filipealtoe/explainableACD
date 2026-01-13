"""
Streaming Real-Time Claim Detection System

This module implements a streaming pipeline for detecting and prioritizing
checkworthy claims from social media at scale.

Architecture:
    TWEETS → ClaimGate → Embedder → Clusterer → AnomalyDetector → ClaimRegistry → ViralityPredictor → CLAIMS

Components:
    - StreamingSimulator: Walk-forward temporal processing
    - ClaimGate: Linguistic pre-filter for claim-like tweets
    - EnsembleAnomalyDetector: Z-score + Kleinberg burst detection
    - ClaimRegistry: LLM normalization + deduplication via Groq
    - ViralityPredictor: XGBoost with user authority features

Primary Metric: Lead time (hours before peak when claim is detected)

Usage:
    from src.streaming import ClaimPipeline, StreamingSimulator

    simulator = StreamingSimulator("data/raw/us_elections_tweets.parquet")
    pipeline = ClaimPipeline(config)

    async for timestamp, df_window in simulator.iterate_windows_async("1h"):
        result = await pipeline.process_window(df_window, timestamp)

    pipeline.save_outputs("data/pipeline_output")
"""

from src.streaming.schemas import (
    AnomalyEvent,
    ClaimInfo,
    ClusterStats,
    TimeseriesRecord,
    ViralityPrediction,
    WindowResult,
)
from src.streaming.claim_gate import ClaimGate, ClaimGateConfig
from src.streaming.simulator import StreamingSimulator, create_train_test_split
from src.streaming.anomaly_detector import (
    EnsembleAnomalyDetector,
    EnsembleAnomalyDetectorConfig,
    KleinbergBurstDetector,
)
from src.streaming.claim_registry import ClaimRegistry, ClaimRegistryConfig
from src.streaming.virality_predictor import (
    ViralityPredictor,
    ViralityPredictorConfig,
    ViralityFeatureExtractor,
    FEATURE_NAMES,
)
from src.streaming.pipeline import ClaimPipeline, PipelineConfig

__all__ = [
    # Schemas
    "AnomalyEvent",
    "ClaimInfo",
    "ClusterStats",
    "TimeseriesRecord",
    "ViralityPrediction",
    "WindowResult",
    # ClaimGate
    "ClaimGate",
    "ClaimGateConfig",
    # Simulator
    "StreamingSimulator",
    "create_train_test_split",
    # Anomaly Detection
    "EnsembleAnomalyDetector",
    "EnsembleAnomalyDetectorConfig",
    "KleinbergBurstDetector",
    # Claim Registry
    "ClaimRegistry",
    "ClaimRegistryConfig",
    # Virality Prediction
    "ViralityPredictor",
    "ViralityPredictorConfig",
    "ViralityFeatureExtractor",
    "FEATURE_NAMES",
    # Pipeline
    "ClaimPipeline",
    "PipelineConfig",
]
