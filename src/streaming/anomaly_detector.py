"""
Ensemble Anomaly Detection: Z-score + Kleinberg Burst

This module implements two complementary anomaly detection methods:

1. **Z-score**: Detects sudden spikes (>3 std from rolling mean)
   - Good for: sharp, unexpected increases
   - Weakness: misses slow-burn, sustained growth

2. **Kleinberg Burst Detection**: State-machine model for bursts
   - Reference: Kleinberg, "Bursty and Hierarchical Structure in Streams" (KDD 2002)
   - https://www.cs.cornell.edu/home/kleinber/kdd02.html
   - Good for: sustained bursts, state transitions
   - Models: normal → elevated → burst states with transition costs

Ensemble logic: Flag anomaly if EITHER detector triggers (union, not intersection).
This catches both sudden spikes AND slow-burn patterns.

Usage:
    detector = EnsembleAnomalyDetector(z_threshold=3.0)
    event = detector.detect(cluster_id=5, timestamp=now, count=150, engagement=2000)
    if event:
        print(f"Anomaly! Trigger: {event.trigger}, Z-score: {event.z_score}")
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

from src.streaming.schemas import AnomalyEvent

logger = logging.getLogger(__name__)


@dataclass
class RollingStats:
    """
    Rolling statistics for a single cluster/topic.

    Maintains a sliding window of observations for computing
    mean and std without storing all historical data.
    """

    window_size: int = 168  # 7 days of hourly observations
    observations: deque = field(default_factory=lambda: deque(maxlen=168))
    count_observations: deque = field(default_factory=lambda: deque(maxlen=168))
    engagement_observations: deque = field(default_factory=lambda: deque(maxlen=168))

    # Running statistics
    count_sum: float = 0.0
    count_sq_sum: float = 0.0
    engagement_sum: float = 0.0
    engagement_sq_sum: float = 0.0

    def add(self, count: int, engagement: int, timestamp: datetime) -> None:
        """Add a new observation to the rolling window."""
        # If window is full, remove oldest
        if len(self.count_observations) >= self.window_size:
            old_count = self.count_observations[0]
            old_engagement = self.engagement_observations[0]
            self.count_sum -= old_count
            self.count_sq_sum -= old_count * old_count
            self.engagement_sum -= old_engagement
            self.engagement_sq_sum -= old_engagement * old_engagement

        # Add new observation
        self.count_observations.append(count)
        self.engagement_observations.append(engagement)
        self.observations.append(timestamp)

        self.count_sum += count
        self.count_sq_sum += count * count
        self.engagement_sum += engagement
        self.engagement_sq_sum += engagement * engagement

    @property
    def n_observations(self) -> int:
        """Number of observations in the window."""
        return len(self.count_observations)

    def get_count_stats(self) -> tuple[float, float]:
        """Get mean and std of count observations."""
        n = self.n_observations
        if n < 2:
            return 0.0, 0.0

        mean = self.count_sum / n
        variance = (self.count_sq_sum / n) - (mean * mean)
        std = math.sqrt(max(0, variance))  # Protect against negative due to float precision
        return mean, std

    def get_engagement_stats(self) -> tuple[float, float]:
        """Get mean and std of engagement observations."""
        n = self.n_observations
        if n < 2:
            return 0.0, 0.0

        mean = self.engagement_sum / n
        variance = (self.engagement_sq_sum / n) - (mean * mean)
        std = math.sqrt(max(0, variance))
        return mean, std


@dataclass
class KleinbergState:
    """
    State for Kleinberg burst detection on a single cluster.

    The model has n_states levels (default 3):
    - State 0: Normal activity
    - State 1: Elevated activity
    - State 2: Burst/Hot

    Higher states have higher expected rates. State transitions
    have costs that prevent flickering between states.
    """

    current_state: int = 0
    state_history: list = field(default_factory=list)

    # Parameters (can be tuned)
    n_states: int = 3
    gamma: float = 1.0  # State transition cost
    base_rate: float = 1.0  # Expected rate in state 0
    rate_multiplier: float = 2.0  # Rate doubles per state level


@dataclass
class EnsembleAnomalyDetectorConfig:
    """Configuration for the ensemble anomaly detector."""

    enabled: bool = True

    # Z-score parameters
    z_threshold: float = 3.0
    z_window_hours: int = 168  # 7 days
    min_observations: int = 24  # Need at least 1 day of data
    min_std_count: float = 2.0  # Minimum std to avoid flat-line detection
    min_std_engagement: float = 5.0

    # Composite score weights
    count_weight: float = 0.4
    engagement_weight: float = 0.6

    # Minimum activity to consider
    min_count: int = 10
    min_engagement: int = 50

    # Kleinberg parameters
    kleinberg_enabled: bool = True
    kleinberg_n_states: int = 3
    kleinberg_gamma: float = 1.0  # State transition cost
    kleinberg_trigger_state: int = 2  # State that triggers anomaly

    @classmethod
    def from_dict(cls, config: dict) -> "EnsembleAnomalyDetectorConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class KleinbergBurstDetector:
    """
    Kleinberg burst detection using an infinite automaton model.

    This is a simplified online version of the Kleinberg model that
    tracks state transitions based on observed rates vs expected rates.

    Reference: Kleinberg, J. "Bursty and Hierarchical Structure in Streams" (KDD 2002)
    """

    def __init__(self, n_states: int = 3, gamma: float = 1.0, rate_multiplier: float = 2.0):
        """
        Initialize the Kleinberg detector.

        Args:
            n_states: Number of burst states (0=normal, 1=elevated, 2+=burst)
            gamma: Cost of state transitions (higher = more stable)
            rate_multiplier: How much rate increases per state level
        """
        self.n_states = n_states
        self.gamma = gamma
        self.rate_multiplier = rate_multiplier
        self.states: dict[int, KleinbergState] = {}

    def _get_expected_rate(self, state: int, base_rate: float) -> float:
        """Get expected rate for a given state level."""
        return base_rate * (self.rate_multiplier ** state)

    def _log_likelihood(self, count: int, rate: float) -> float:
        """Log-likelihood of observing count given Poisson rate."""
        if rate <= 0:
            return -float("inf") if count > 0 else 0.0
        # Poisson log-likelihood (up to constant)
        return count * math.log(rate) - rate

    def _transition_cost(self, from_state: int, to_state: int) -> float:
        """Cost of transitioning between states."""
        if to_state > from_state:
            # Cost to increase state
            return self.gamma * (to_state - from_state)
        else:
            # Free to decrease state
            return 0.0

    def detect(self, cluster_id: int, count: int, base_rate: float) -> int:
        """
        Update state for a cluster and return current state.

        Uses Viterbi-like algorithm to find optimal state given
        the observation and transition costs.

        Args:
            cluster_id: Cluster identifier
            count: Observed count in this time period
            base_rate: Expected base rate (from historical data)

        Returns:
            Current state (0=normal, 1=elevated, 2+=burst)
        """
        if cluster_id not in self.states:
            self.states[cluster_id] = KleinbergState(n_states=self.n_states)

        state = self.states[cluster_id]

        if base_rate <= 0:
            base_rate = 1.0  # Default base rate

        # Find optimal next state using Viterbi-like selection
        best_state = 0
        best_score = -float("inf")

        for candidate_state in range(self.n_states):
            expected_rate = self._get_expected_rate(candidate_state, base_rate)
            ll = self._log_likelihood(count, expected_rate)
            trans_cost = self._transition_cost(state.current_state, candidate_state)
            score = ll - trans_cost

            if score > best_score:
                best_score = score
                best_state = candidate_state

        # Update state
        state.current_state = best_state
        state.state_history.append(best_state)

        return best_state

    def get_state(self, cluster_id: int) -> int:
        """Get current state for a cluster."""
        if cluster_id in self.states:
            return self.states[cluster_id].current_state
        return 0


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining Z-score and Kleinberg burst detection.

    Flags an anomaly if EITHER detector triggers, providing robustness
    against different types of anomalous patterns.
    """

    def __init__(self, config: EnsembleAnomalyDetectorConfig | None = None):
        """
        Initialize the ensemble detector.

        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or EnsembleAnomalyDetectorConfig()

        # Per-cluster rolling statistics for Z-score
        self.cluster_stats: dict[int, RollingStats] = {}

        # Kleinberg detector
        self.kleinberg = KleinbergBurstDetector(
            n_states=self.config.kleinberg_n_states,
            gamma=self.config.kleinberg_gamma,
        )

        # Statistics
        self.total_detections = 0
        self.zscore_triggers = 0
        self.kleinberg_triggers = 0
        self.both_triggers = 0

    def _get_or_create_stats(self, cluster_id: int) -> RollingStats:
        """Get or create rolling statistics for a cluster."""
        if cluster_id not in self.cluster_stats:
            self.cluster_stats[cluster_id] = RollingStats(window_size=self.config.z_window_hours)
        return self.cluster_stats[cluster_id]

    def _compute_z_score(
        self,
        value: float,
        mean: float,
        std: float,
        min_std: float,
    ) -> float:
        """Compute Z-score with minimum std protection."""
        if std < min_std:
            return 0.0  # Not enough variance to detect anomaly
        return (value - mean) / std

    def update(self, cluster_id: int, timestamp: datetime, count: int, engagement: int) -> None:
        """
        Update statistics for a cluster without checking for anomalies.

        Use this for warm-up periods or when you just want to update baselines.

        Args:
            cluster_id: Cluster identifier
            timestamp: Observation timestamp
            count: Tweet count in this period
            engagement: Total engagement (RT + likes)
        """
        stats = self._get_or_create_stats(cluster_id)
        stats.add(count, engagement, timestamp)

    def detect(
        self,
        cluster_id: int,
        timestamp: datetime,
        count: int,
        engagement: int,
    ) -> AnomalyEvent | None:
        """
        Detect anomalies and update statistics.

        Returns AnomalyEvent if EITHER Z-score OR Kleinberg triggers.

        Args:
            cluster_id: Cluster identifier
            timestamp: Observation timestamp
            count: Tweet count in this period
            engagement: Total engagement (RT + likes)

        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        if not self.config.enabled:
            return None

        # Skip if below minimum activity thresholds
        if count < self.config.min_count or engagement < self.config.min_engagement:
            # Still update stats for baseline
            self.update(cluster_id, timestamp, count, engagement)
            return None

        stats = self._get_or_create_stats(cluster_id)

        # Check if we have enough observations
        if stats.n_observations < self.config.min_observations:
            stats.add(count, engagement, timestamp)
            return None

        # Compute Z-scores
        count_mean, count_std = stats.get_count_stats()
        engagement_mean, engagement_std = stats.get_engagement_stats()

        z_count = self._compute_z_score(count, count_mean, count_std, self.config.min_std_count)
        z_engagement = self._compute_z_score(
            engagement, engagement_mean, engagement_std, self.config.min_std_engagement
        )

        # Composite Z-score
        z_composite = self.config.count_weight * z_count + self.config.engagement_weight * z_engagement

        # Z-score anomaly check
        z_anomaly = z_composite > self.config.z_threshold

        # Kleinberg burst detection
        kleinberg_state = 0
        if self.config.kleinberg_enabled:
            # Use count mean as base rate for Kleinberg
            base_rate = max(1.0, count_mean)
            kleinberg_state = self.kleinberg.detect(cluster_id, count, base_rate)

        kleinberg_anomaly = kleinberg_state >= self.config.kleinberg_trigger_state

        # Update statistics AFTER computing anomaly (use past data only)
        stats.add(count, engagement, timestamp)

        # Determine trigger type
        if z_anomaly and kleinberg_anomaly:
            trigger: Literal["zscore", "kleinberg", "both"] = "both"
            self.both_triggers += 1
        elif z_anomaly:
            trigger = "zscore"
            self.zscore_triggers += 1
        elif kleinberg_anomaly:
            trigger = "kleinberg"
            self.kleinberg_triggers += 1
        else:
            return None  # No anomaly

        self.total_detections += 1

        return AnomalyEvent(
            cluster_id=cluster_id,
            timestamp=timestamp,
            count=count,
            engagement=engagement,
            z_score=z_composite,
            z_score_count=z_count,
            z_score_engagement=z_engagement,
            kleinberg_state=kleinberg_state,
            trigger=trigger,
        )

    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            "total_detections": self.total_detections,
            "zscore_triggers": self.zscore_triggers,
            "kleinberg_triggers": self.kleinberg_triggers,
            "both_triggers": self.both_triggers,
            "clusters_tracked": len(self.cluster_stats),
            "config": {
                "z_threshold": self.config.z_threshold,
                "kleinberg_enabled": self.config.kleinberg_enabled,
                "kleinberg_trigger_state": self.config.kleinberg_trigger_state,
            },
        }

    def get_cluster_baseline(self, cluster_id: int) -> dict:
        """Get baseline statistics for a specific cluster."""
        if cluster_id not in self.cluster_stats:
            return {"error": "Cluster not tracked"}

        stats = self.cluster_stats[cluster_id]
        count_mean, count_std = stats.get_count_stats()
        eng_mean, eng_std = stats.get_engagement_stats()

        return {
            "cluster_id": cluster_id,
            "n_observations": stats.n_observations,
            "count_mean": count_mean,
            "count_std": count_std,
            "engagement_mean": eng_mean,
            "engagement_std": eng_std,
            "kleinberg_state": self.kleinberg.get_state(cluster_id),
        }
