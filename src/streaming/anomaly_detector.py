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


# ==============================================================================
# GLOBAL STATISTICS TRACKER (for Bayesian Cold-Start)
# ==============================================================================


@dataclass
class GlobalStats:
    """
    Global statistics across ALL clusters for Bayesian prior estimation.

    This enables "borrowing strength" from the global distribution when
    individual clusters have insufficient observations.

    Uses Welford's online algorithm for numerically stable variance computation.
    """

    # Running statistics (Welford's algorithm)
    count_n: int = 0
    count_mean: float = 0.0
    count_M2: float = 0.0  # Sum of squared deviations

    engagement_n: int = 0
    engagement_mean: float = 0.0
    engagement_M2: float = 0.0

    # Clipped statistics (for robust estimation)
    # We track both raw and clipped to handle outliers
    count_clipped_n: int = 0
    count_clipped_mean: float = 0.0
    count_clipped_M2: float = 0.0

    def update(self, count: int, engagement: int, clip_percentile: float = 0.99) -> None:
        """
        Update global statistics with a new observation.

        Uses Welford's online algorithm for numerically stable updates.

        Args:
            count: Tweet count for this cluster-window
            engagement: Total engagement for this cluster-window
            clip_percentile: Clip extreme values (for robust stats)
        """
        # Update count stats (Welford's algorithm)
        self.count_n += 1
        delta = count - self.count_mean
        self.count_mean += delta / self.count_n
        delta2 = count - self.count_mean
        self.count_M2 += delta * delta2

        # Update engagement stats
        self.engagement_n += 1
        delta = engagement - self.engagement_mean
        self.engagement_mean += delta / self.engagement_n
        delta2 = engagement - self.engagement_mean
        self.engagement_M2 += delta * delta2

    @property
    def count_variance(self) -> float:
        """Sample variance of count observations."""
        if self.count_n < 2:
            return 0.0
        return self.count_M2 / (self.count_n - 1)

    @property
    def count_std(self) -> float:
        """Sample standard deviation of count observations."""
        return math.sqrt(max(0, self.count_variance))

    @property
    def engagement_variance(self) -> float:
        """Sample variance of engagement observations."""
        if self.engagement_n < 2:
            return 0.0
        return self.engagement_M2 / (self.engagement_n - 1)

    @property
    def engagement_std(self) -> float:
        """Sample standard deviation of engagement observations."""
        return math.sqrt(max(0, self.engagement_variance))


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

    # Evaluation mode: compute scores for ALL clusters regardless of min_observations
    # When True: z-scores are computed for all clusters (with relaxed thresholds for sparse)
    #            but anomaly triggers still require min_observations
    # When False: original behavior - skip score computation for sparse clusters entirely
    # Recommended: True for training classifiers and full evaluation
    compute_scores_for_all: bool = True

    # === BAYESIAN COLD-START DETECTOR (new) ===
    # Addresses the critical issue: 96.9% of clusters have <24 observations
    # Uses global statistics as empirical Bayes prior for sparse clusters
    bayesian_cold_start_enabled: bool = True  # Enable Bayesian detector
    bayesian_prior_strength: int = 24  # Effective prior sample size (higher = more shrinkage)
    bayesian_min_global_obs: int = 100  # Wait for global stats before enabling
    bayesian_min_activity_count: int = 5  # Lower threshold for cold-start (more aggressive)
    bayesian_min_activity_engagement: int = 20

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


# ==============================================================================
# BAYESIAN COLD-START DETECTOR (addresses 96.9% sparse cluster problem)
# ==============================================================================


class BayesianColdStartDetector:
    """
    Bayesian Cold-Start Anomaly Detector using hierarchical priors.

    PROBLEM: 96.9% of clusters have <24 observations, making classical
    Z-score detection impossible (can't compute meaningful statistics).

    SOLUTION: Use global statistics as an empirical Bayes prior.
    - New clusters: use global mean/std
    - As observations accumulate: posterior converges to cluster-specific stats
    - Weighted combination: weight = n / (prior_strength + n)

    This is essentially empirical Bayes with a conjugate normal-normal model.

    Example:
        detector = BayesianColdStartDetector(prior_strength=24)
        # For a brand-new cluster with 3 observations:
        # posterior_mean = (24 * global_mean + 3 * cluster_mean) / 27
        # Mostly global prior, but cluster data has some influence

    Reference: Casella, G. (1985). "An Introduction to Empirical Bayes Data Analysis"
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        prior_strength: int = 24,
        min_global_observations: int = 100,
        count_weight: float = 0.4,
        engagement_weight: float = 0.6,
        min_activity_count: int = 5,
        min_activity_engagement: int = 20,
    ):
        """
        Initialize the Bayesian cold-start detector.

        Args:
            z_threshold: Z-score threshold for anomaly detection
            prior_strength: Effective sample size of the global prior (higher = more shrinkage)
            min_global_observations: Minimum global obs before enabling detection
            count_weight: Weight for count Z-score in composite
            engagement_weight: Weight for engagement Z-score in composite
            min_activity_count: Minimum count to consider (filters noise)
            min_activity_engagement: Minimum engagement to consider
        """
        self.z_threshold = z_threshold
        self.prior_strength = prior_strength
        self.min_global_observations = min_global_observations
        self.count_weight = count_weight
        self.engagement_weight = engagement_weight
        self.min_activity_count = min_activity_count
        self.min_activity_engagement = min_activity_engagement

        # Global statistics (the "prior")
        self.global_stats = GlobalStats()

        # Per-cluster statistics (the "likelihood")
        self.cluster_stats: dict[int, RollingStats] = {}

        # Detection statistics
        self.total_detections = 0
        self.cold_start_detections = 0  # Detections on clusters with <24 obs
        self.warm_detections = 0  # Detections on clusters with ≥24 obs

        # Last computed scores for evaluation
        self._last_scores: dict[int, dict] = {}

    def _get_or_create_stats(self, cluster_id: int) -> RollingStats:
        """Get or create rolling statistics for a cluster."""
        if cluster_id not in self.cluster_stats:
            self.cluster_stats[cluster_id] = RollingStats(window_size=168)
        return self.cluster_stats[cluster_id]

    def _compute_posterior_stats(
        self, cluster_stats: RollingStats, metric: str = "count"
    ) -> tuple[float, float]:
        """
        Compute posterior mean and std using Bayesian shrinkage.

        For a Normal-Normal conjugate model:
        - posterior_mean = (prior_strength * prior_mean + n * sample_mean) / (prior_strength + n)
        - posterior_variance = prior_variance / (prior_strength + n)  # simplified

        Args:
            cluster_stats: Rolling statistics for this cluster
            metric: "count" or "engagement"

        Returns:
            (posterior_mean, posterior_std)
        """
        n = cluster_stats.n_observations

        if metric == "count":
            global_mean = self.global_stats.count_mean
            global_std = self.global_stats.count_std
            cluster_mean, cluster_std = cluster_stats.get_count_stats()
        else:
            global_mean = self.global_stats.engagement_mean
            global_std = self.global_stats.engagement_std
            cluster_mean, cluster_std = cluster_stats.get_engagement_stats()

        # Edge case: no global data yet
        if self.global_stats.count_n < self.min_global_observations:
            # Fall back to cluster-only (or zeros if no data)
            return cluster_mean, max(cluster_std, 1.0)

        # Bayesian shrinkage weight
        # weight = 0 → use only global prior
        # weight = 1 → use only cluster data
        weight = n / (self.prior_strength + n)

        # Posterior mean (weighted combination)
        posterior_mean = (1 - weight) * global_mean + weight * cluster_mean

        # Posterior std (conservative: use larger of shrunk and global)
        # This prevents false positives from underestimated variance
        if n < 3:
            posterior_std = global_std
        else:
            # Shrink cluster std toward global std
            shrunk_std = (1 - weight) * global_std + weight * cluster_std
            posterior_std = max(shrunk_std, global_std * 0.5)  # Floor at 50% of global

        return posterior_mean, max(posterior_std, 1.0)  # Minimum std of 1.0

    def detect(
        self,
        cluster_id: int,
        timestamp: datetime,
        count: int,
        engagement: int,
    ) -> AnomalyEvent | None:
        """
        Detect anomalies using Bayesian cold-start approach.

        Unlike classical Z-score, this CAN detect anomalies for clusters
        with very few observations by leveraging the global prior.

        Args:
            cluster_id: Cluster identifier
            timestamp: Observation timestamp
            count: Tweet count in this period
            engagement: Total engagement (RT + likes)

        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        # Update global statistics FIRST (before cluster update)
        self.global_stats.update(count, engagement)

        # Get or create cluster stats
        stats = self._get_or_create_stats(cluster_id)
        n_obs = stats.n_observations
        is_cold_start = n_obs < 24

        # Skip if below minimum activity (reduce noise)
        if count < self.min_activity_count or engagement < self.min_activity_engagement:
            stats.add(count, engagement, timestamp)
            return None

        # Skip if global stats insufficient
        if self.global_stats.count_n < self.min_global_observations:
            stats.add(count, engagement, timestamp)
            return None

        # Compute Bayesian posterior statistics
        count_mean, count_std = self._compute_posterior_stats(stats, "count")
        eng_mean, eng_std = self._compute_posterior_stats(stats, "engagement")

        # Compute Z-scores against posterior
        z_count = (count - count_mean) / count_std if count_std > 0 else 0.0
        z_engagement = (engagement - eng_mean) / eng_std if eng_std > 0 else 0.0

        # Composite Z-score
        z_composite = self.count_weight * z_count + self.engagement_weight * z_engagement

        # Store scores for evaluation
        self._last_scores[cluster_id] = {
            "z_score": z_composite,
            "z_score_count": z_count,
            "z_score_engagement": z_engagement,
            "n_observations": n_obs,
            "is_cold_start": is_cold_start,
            "posterior_count_mean": count_mean,
            "posterior_count_std": count_std,
            "shrinkage_weight": n_obs / (self.prior_strength + n_obs),
        }

        # Update cluster stats AFTER scoring
        stats.add(count, engagement, timestamp)

        # Check anomaly threshold
        if z_composite > self.z_threshold:
            self.total_detections += 1
            if is_cold_start:
                self.cold_start_detections += 1
            else:
                self.warm_detections += 1

            return AnomalyEvent(
                cluster_id=cluster_id,
                timestamp=timestamp,
                count=count,
                engagement=engagement,
                z_score=z_composite,
                z_score_count=z_count,
                z_score_engagement=z_engagement,
                kleinberg_state=0,  # Not used in Bayesian detector
                trigger="bayesian_cold_start" if is_cold_start else "bayesian_warm",
            )

        return None

    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            "total_detections": self.total_detections,
            "cold_start_detections": self.cold_start_detections,
            "warm_detections": self.warm_detections,
            "cold_start_ratio": (
                self.cold_start_detections / self.total_detections
                if self.total_detections > 0
                else 0.0
            ),
            "clusters_tracked": len(self.cluster_stats),
            "global_observations": self.global_stats.count_n,
            "global_count_mean": self.global_stats.count_mean,
            "global_count_std": self.global_stats.count_std,
            "global_engagement_mean": self.global_stats.engagement_mean,
            "global_engagement_std": self.global_stats.engagement_std,
        }

    def get_cluster_stats(self, cluster_id: int) -> dict:
        """Get last computed scores for a cluster."""
        return self._last_scores.get(cluster_id, {})


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining Z-score, Kleinberg burst detection,
    and Bayesian cold-start detection.

    Detection strategy:
    - Sparse clusters (< min_observations): Use Bayesian cold-start detector
    - Mature clusters (>= min_observations): Use Z-score + Kleinberg ensemble

    Flags an anomaly if ANY detector triggers, providing robustness
    against different types of anomalous patterns AND sparse data.
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

        # Bayesian cold-start detector (addresses 96.9% sparse cluster problem)
        self.bayesian: BayesianColdStartDetector | None = None
        if self.config.bayesian_cold_start_enabled:
            self.bayesian = BayesianColdStartDetector(
                z_threshold=self.config.z_threshold,
                prior_strength=self.config.bayesian_prior_strength,
                min_global_observations=self.config.bayesian_min_global_obs,
                count_weight=self.config.count_weight,
                engagement_weight=self.config.engagement_weight,
                min_activity_count=self.config.bayesian_min_activity_count,
                min_activity_engagement=self.config.bayesian_min_activity_engagement,
            )

        # Statistics
        self.total_detections = 0
        self.zscore_triggers = 0
        self.kleinberg_triggers = 0
        self.both_triggers = 0
        self.bayesian_cold_triggers = 0
        self.bayesian_warm_triggers = 0

        # Store last computed scores for ALL clusters (enables full recall evaluation)
        self._last_scores: dict[int, dict] = {}

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

        Behavior depends on config.compute_scores_for_all:
        - True (default): Z-scores computed for ALL clusters (relaxed thresholds for sparse)
          Enables full precision/recall evaluation and classifier training.
        - False: Original behavior - skip score computation for sparse clusters entirely.

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

        stats = self._get_or_create_stats(cluster_id)
        n_obs = stats.n_observations
        is_sparse = n_obs < self.config.min_observations

        # === BAYESIAN COLD-START PATH (for sparse clusters) ===
        # If Bayesian detector is enabled and cluster is sparse, use it INSTEAD of classical methods
        # This addresses the critical issue: 96.9% of clusters have <24 observations
        if self.bayesian is not None and is_sparse:
            event = self.bayesian.detect(cluster_id, timestamp, count, engagement)

            # Store Bayesian scores for evaluation
            bayesian_scores = self.bayesian.get_cluster_stats(cluster_id)
            self._last_scores[cluster_id] = {
                **bayesian_scores,
                "detector": "bayesian",
            }

            # Also update classical stats (so they're ready when cluster matures)
            stats.add(count, engagement, timestamp)

            if event:
                if event.trigger == "bayesian_cold_start":
                    self.bayesian_cold_triggers += 1
                else:
                    self.bayesian_warm_triggers += 1
                self.total_detections += 1

            return event

        # === ORIGINAL BEHAVIOR (compute_scores_for_all=False) ===
        if not self.config.compute_scores_for_all:
            # Skip if below minimum activity thresholds
            if count < self.config.min_count or engagement < self.config.min_engagement:
                self.update(cluster_id, timestamp, count, engagement)
                return None

            # Skip if not enough observations
            if is_sparse:
                stats.add(count, engagement, timestamp)
                return None

            # Compute Z-scores only for non-sparse clusters
            count_mean, count_std = stats.get_count_stats()
            engagement_mean, engagement_std = stats.get_engagement_stats()

            z_count = self._compute_z_score(count, count_mean, count_std, self.config.min_std_count)
            z_engagement = self._compute_z_score(
                engagement, engagement_mean, engagement_std, self.config.min_std_engagement
            )
            z_composite = self.config.count_weight * z_count + self.config.engagement_weight * z_engagement

            kleinberg_state = 0
            if self.config.kleinberg_enabled:
                base_rate = max(1.0, count_mean)
                kleinberg_state = self.kleinberg.detect(cluster_id, count, base_rate)

            self._last_scores[cluster_id] = {
                "z_score": z_composite,
                "z_score_count": z_count,
                "z_score_engagement": z_engagement,
                "kleinberg_state": kleinberg_state,
            }

            stats.add(count, engagement, timestamp)

        # === NEW BEHAVIOR (compute_scores_for_all=True) ===
        else:
            # ALWAYS compute Z-scores (even for sparse clusters)
            count_mean, count_std = stats.get_count_stats()
            engagement_mean, engagement_std = stats.get_engagement_stats()

            # For sparse clusters, use relaxed std thresholds
            if is_sparse:
                effective_min_std_count = 0.1  # Very permissive
                effective_min_std_engagement = 0.1
            else:
                effective_min_std_count = self.config.min_std_count
                effective_min_std_engagement = self.config.min_std_engagement

            z_count = self._compute_z_score(count, count_mean, count_std, effective_min_std_count)
            z_engagement = self._compute_z_score(
                engagement, engagement_mean, engagement_std, effective_min_std_engagement
            )
            z_composite = self.config.count_weight * z_count + self.config.engagement_weight * z_engagement

            # ALWAYS compute Kleinberg state
            kleinberg_state = 0
            if self.config.kleinberg_enabled:
                base_rate = max(1.0, count_mean) if count_mean > 0 else 1.0
                kleinberg_state = self.kleinberg.detect(cluster_id, count, base_rate)

            # ALWAYS store scores (enables full evaluation)
            self._last_scores[cluster_id] = {
                "z_score": z_composite,
                "z_score_count": z_count,
                "z_score_engagement": z_engagement,
                "kleinberg_state": kleinberg_state,
                "n_observations": n_obs,
                "is_sparse": is_sparse,
            }

            stats.add(count, engagement, timestamp)

            # Skip TRIGGERING if below thresholds (but scores are still stored)
            if count < self.config.min_count or engagement < self.config.min_engagement:
                return None
            if is_sparse:
                return None

        # === CHECK ANOMALY CONDITIONS (both modes) ===
        z_anomaly = z_composite > self.config.z_threshold
        kleinberg_anomaly = kleinberg_state >= self.config.kleinberg_trigger_state

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
        stats = {
            "total_detections": self.total_detections,
            "zscore_triggers": self.zscore_triggers,
            "kleinberg_triggers": self.kleinberg_triggers,
            "both_triggers": self.both_triggers,
            "bayesian_cold_triggers": self.bayesian_cold_triggers,
            "bayesian_warm_triggers": self.bayesian_warm_triggers,
            "clusters_tracked": len(self.cluster_stats),
            "config": {
                "z_threshold": self.config.z_threshold,
                "kleinberg_enabled": self.config.kleinberg_enabled,
                "kleinberg_trigger_state": self.config.kleinberg_trigger_state,
                "bayesian_enabled": self.config.bayesian_cold_start_enabled,
            },
        }

        # Add Bayesian detector stats if enabled
        if self.bayesian is not None:
            stats["bayesian_detector"] = self.bayesian.get_stats()

        return stats

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

    def get_cluster_stats(self, cluster_id: int) -> dict:
        """
        Get last computed z-scores for a cluster.

        Returns scores from the most recent detect() call for this cluster.
        Enables tracking non-anomalous clusters for recall evaluation.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Dict with z_score, z_score_count, z_score_engagement, kleinberg_state
            or empty dict if cluster hasn't been through detect() yet
        """
        return self._last_scores.get(cluster_id, {})
