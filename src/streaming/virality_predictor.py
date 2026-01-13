"""
ViralityPredictor: Feature extraction and virality prediction.

This module implements:
1. Feature extraction from the peek window (6h after detection)
2. XGBoost-based virality prediction
3. Lead time calculation (PRIMARY METRIC)

Features (18 total):
- Basic: z_score_at_detection, tweet_count, engagement_sum
- Velocity: tweets_per_hour, engagement_velocity, momentum
- Engagement: rt_ratio, reply_ratio, engagement_efficiency
- Content: url_density, media_density, hashtag_avg
- Temporal: hour_of_day, is_weekend
- USER AUTHORITY (from review synthesis):
  - avg_followers_first_50: avg follower count of first 50 participants
  - max_followers_first_50: single most influential user
  - verified_ratio: % verified users in first hour
  - rt_like_ratio: controversy signal (high RT/like = divisive)
  - unique_user_velocity: new users per hour

Target: reaches 1000+ RTs within 4h of detection (time-bounded definition)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.streaming.schemas import ViralityPrediction

logger = logging.getLogger(__name__)


# Feature names in order (must match training)
FEATURE_NAMES = [
    # Basic (3)
    "z_score_at_detection",
    "tweet_count",
    "engagement_sum",
    # Velocity (3)
    "tweets_per_hour",
    "engagement_velocity",
    "momentum",
    # Engagement composition (3)
    "rt_ratio",
    "like_ratio",
    "engagement_efficiency",
    # Content (3)
    "url_density",
    "media_density",
    "hashtag_avg",
    # Temporal (2)
    "hour_of_day",
    "is_weekend",
    # User authority (4) - from review synthesis
    "avg_followers_first_50",
    "max_followers_first_50",
    "verified_ratio",
    "rt_like_ratio",
]


@dataclass
class ViralityPredictorConfig:
    """Configuration for the virality predictor."""

    enabled: bool = True
    model_path: str = "model/virality_xgboost.pkl"
    peek_hours: int = 6  # Hours after detection to extract features
    horizon_hours: int = 168  # 7 days for ground truth labeling
    virality_threshold: int = 1000  # RTs within 4h to be considered viral
    virality_window_hours: int = 4  # Time window for virality threshold
    min_tweets_for_prediction: int = 5  # Minimum tweets in peek window

    @classmethod
    def from_dict(cls, config: dict) -> "ViralityPredictorConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class ViralityFeatureExtractor:
    """
    Extract features from a cluster's peek window.

    The peek window is the first N hours after anomaly detection,
    used to gather early signals for virality prediction.
    """

    def __init__(self, peek_hours: int = 6):
        self.peek_hours = peek_hours

    def extract(
        self,
        df: pl.DataFrame,
        z_score_at_detection: float = 0.0,
        detection_time: datetime | None = None,
    ) -> dict[str, float]:
        """
        Extract features from cluster tweets in the peek window.

        Args:
            df: DataFrame with tweets from the cluster
                Expected columns: created_at, tweet, retweet_count, likes,
                user_followers_count, user_verified (optional)
            z_score_at_detection: Z-score when anomaly was first detected
            detection_time: When the anomaly was detected

        Returns:
            Dictionary of feature name -> value
        """
        features: dict[str, float] = {}

        if len(df) == 0:
            # Return zeros for all features
            return {name: 0.0 for name in FEATURE_NAMES}

        # Filter to peek window if detection_time provided
        if detection_time is not None and "created_at" in df.columns:
            peek_end = detection_time + timedelta(hours=self.peek_hours)
            df = df.filter(
                (pl.col("created_at") >= detection_time) & (pl.col("created_at") < peek_end)
            )

        if len(df) == 0:
            return {name: 0.0 for name in FEATURE_NAMES}

        # =========== BASIC FEATURES ===========
        features["z_score_at_detection"] = z_score_at_detection
        features["tweet_count"] = float(len(df))

        # Engagement sum
        rt_sum = df["retweet_count"].sum() if "retweet_count" in df.columns else 0
        like_sum = df["likes"].sum() if "likes" in df.columns else 0
        features["engagement_sum"] = float(rt_sum + like_sum)

        # =========== VELOCITY FEATURES ===========
        # Time span in hours
        if "created_at" in df.columns and len(df) > 1:
            time_span = (df["created_at"].max() - df["created_at"].min()).total_seconds() / 3600
            time_span = max(0.1, time_span)  # Avoid division by zero
        else:
            time_span = float(self.peek_hours)

        features["tweets_per_hour"] = features["tweet_count"] / time_span
        features["engagement_velocity"] = features["engagement_sum"] / time_span

        # Momentum: compare second half vs first half engagement
        if len(df) >= 4 and "created_at" in df.columns:
            mid_time = df["created_at"].min() + timedelta(hours=time_span / 2)
            first_half = df.filter(pl.col("created_at") < mid_time)
            second_half = df.filter(pl.col("created_at") >= mid_time)

            first_eng = (
                first_half["retweet_count"].sum() + first_half["likes"].sum()
                if len(first_half) > 0
                else 0
            )
            second_eng = (
                second_half["retweet_count"].sum() + second_half["likes"].sum()
                if len(second_half) > 0
                else 0
            )

            # Momentum: ratio of second half to first half (>1 = accelerating)
            features["momentum"] = float(second_eng / max(1, first_eng))
        else:
            features["momentum"] = 1.0

        # =========== ENGAGEMENT COMPOSITION ===========
        total_engagement = max(1.0, features["engagement_sum"])

        features["rt_ratio"] = float(rt_sum) / total_engagement
        features["like_ratio"] = float(like_sum) / total_engagement
        features["engagement_efficiency"] = total_engagement / max(1.0, features["tweet_count"])

        # =========== CONTENT FEATURES ===========
        if "tweet" in df.columns:
            tweets = df["tweet"].to_list()

            # URL density
            url_count = sum(1 for t in tweets if "http" in str(t).lower())
            features["url_density"] = url_count / max(1, len(tweets))

            # Media density (check for media URLs)
            media_count = sum(
                1 for t in tweets if any(m in str(t).lower() for m in ["pic.twitter", "video", "photo", "media"])
            )
            features["media_density"] = media_count / max(1, len(tweets))

            # Hashtag average
            hashtag_count = sum(str(t).count("#") for t in tweets)
            features["hashtag_avg"] = hashtag_count / max(1, len(tweets))
        else:
            features["url_density"] = 0.0
            features["media_density"] = 0.0
            features["hashtag_avg"] = 0.0

        # =========== TEMPORAL FEATURES ===========
        if detection_time:
            features["hour_of_day"] = float(detection_time.hour)
            features["is_weekend"] = 1.0 if detection_time.weekday() >= 5 else 0.0
        else:
            features["hour_of_day"] = 12.0
            features["is_weekend"] = 0.0

        # =========== USER AUTHORITY FEATURES ===========
        # These are critical for virality prediction (from review synthesis)

        if "user_followers_count" in df.columns:
            # First 50 users (early adopters)
            first_50 = df.head(50)
            followers = first_50["user_followers_count"].drop_nulls()

            if len(followers) > 0:
                features["avg_followers_first_50"] = float(followers.mean())
                features["max_followers_first_50"] = float(followers.max())
            else:
                features["avg_followers_first_50"] = 0.0
                features["max_followers_first_50"] = 0.0
        else:
            features["avg_followers_first_50"] = 0.0
            features["max_followers_first_50"] = 0.0

        if "user_verified" in df.columns:
            first_50 = df.head(50)
            verified = first_50["user_verified"].drop_nulls()
            if len(verified) > 0:
                features["verified_ratio"] = float(verified.sum()) / len(verified)
            else:
                features["verified_ratio"] = 0.0
        else:
            features["verified_ratio"] = 0.0

        # RT/Like ratio (controversy signal)
        features["rt_like_ratio"] = float(rt_sum) / max(1.0, float(like_sum))

        return features

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        return FEATURE_NAMES.copy()


class ViralityPredictor:
    """
    Predict virality using extracted features.

    Uses XGBoost classifier trained on historical data.
    """

    def __init__(self, config: ViralityPredictorConfig | None = None):
        self.config = config or ViralityPredictorConfig()
        self.feature_extractor = ViralityFeatureExtractor(peek_hours=self.config.peek_hours)
        self._model = None
        self._feature_names = FEATURE_NAMES

    @property
    def model(self):
        """Lazy load the trained model."""
        if self._model is None:
            model_path = Path(self.config.model_path)
            if model_path.exists():
                try:
                    import joblib

                    self._model = joblib.load(model_path)
                    logger.info(f"Loaded virality model from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load model: {e}")
                    self._model = None
            else:
                logger.warning(f"Model not found at {model_path}, predictions will use heuristics")
        return self._model

    def _features_to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert feature dict to array in correct order."""
        return np.array([[features.get(name, 0.0) for name in self._feature_names]])

    def predict(
        self,
        features: dict[str, float],
    ) -> tuple[bool, float]:
        """
        Predict virality from features.

        Args:
            features: Feature dictionary from ViralityFeatureExtractor

        Returns:
            Tuple of (is_viral: bool, confidence: float)
        """
        if not self.config.enabled:
            return False, 0.5

        X = self._features_to_array(features)

        if self.model is not None:
            try:
                proba = self.model.predict_proba(X)[0, 1]
                is_viral = proba > 0.5
                return is_viral, float(proba)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}, using heuristics")

        # Fallback: simple heuristics if model not available
        # Use engagement velocity and user authority
        velocity = features.get("engagement_velocity", 0)
        max_followers = features.get("max_followers_first_50", 0)
        z_score = features.get("z_score_at_detection", 0)

        # Simple scoring heuristic
        score = 0.3  # Base probability
        if velocity > 100:
            score += 0.2
        if max_followers > 10000:
            score += 0.2
        if z_score > 5:
            score += 0.15
        if features.get("verified_ratio", 0) > 0.1:
            score += 0.1

        score = min(0.95, max(0.05, score))  # Clamp to [0.05, 0.95]
        return score > 0.5, score

    def predict_from_df(
        self,
        df: pl.DataFrame,
        z_score_at_detection: float,
        detection_time: datetime,
    ) -> ViralityPrediction:
        """
        Full prediction pipeline from DataFrame.

        Args:
            df: DataFrame with cluster tweets
            z_score_at_detection: Z-score when detected
            detection_time: When anomaly was detected

        Returns:
            ViralityPrediction object
        """
        # Extract features
        features = self.feature_extractor.extract(df, z_score_at_detection, detection_time)

        # Predict
        is_viral, confidence = self.predict(features)

        return ViralityPrediction(
            claim_id="",  # Will be set by caller
            is_viral=is_viral,
            confidence=confidence,
            detection_time=detection_time,
            features=features,
        )

    @staticmethod
    def calculate_lead_time(detection_time: datetime, peak_time: datetime) -> float:
        """
        Calculate lead time (PRIMARY METRIC).

        Lead time is how many hours before the peak the claim was detected.
        Higher is better (earlier detection).

        Args:
            detection_time: When the claim was first detected
            peak_time: When the claim reached peak engagement

        Returns:
            Lead time in hours (positive = detected before peak)
        """
        delta = peak_time - detection_time
        return delta.total_seconds() / 3600

    def get_stats(self) -> dict:
        """Get predictor statistics."""
        return {
            "model_loaded": self.model is not None,
            "model_path": self.config.model_path,
            "peek_hours": self.config.peek_hours,
            "horizon_hours": self.config.horizon_hours,
            "virality_threshold": self.config.virality_threshold,
            "feature_count": len(self._feature_names),
            "feature_names": self._feature_names,
        }


def train_virality_model(
    train_df: pl.DataFrame,
    feature_extractor: ViralityFeatureExtractor,
    output_path: str = "model/virality_xgboost.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train a virality prediction model.

    This function should be called separately on historical data
    where ground truth (viral/not viral) is known.

    Args:
        train_df: Training data with features and labels
        feature_extractor: Feature extractor instance
        output_path: Where to save the trained model
        test_size: Fraction for validation
        random_state: Random seed

    Returns:
        Dictionary with training metrics
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import classification_report, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("xgboost and scikit-learn required. Install with: uv add xgboost scikit-learn")

    # This is a placeholder - actual training would extract features from historical data
    # and use actual viral/not viral labels

    logger.info("Training virality model...")

    # Example training flow (would need actual data):
    # X = train_df.select(FEATURE_NAMES).to_numpy()
    # y = train_df["is_viral"].to_numpy()
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # model = xgb.XGBClassifier(
    #     n_estimators=100,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     objective="binary:logistic",
    #     random_state=random_state,
    # )
    # model.fit(X_train, y_train)

    # Save model
    # import joblib
    # Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # joblib.dump(model, output_path)

    logger.warning("train_virality_model is a placeholder - implement with actual training data")

    return {
        "status": "placeholder",
        "message": "Implement with actual training data",
    }
