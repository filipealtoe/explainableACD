#!/usr/bin/env python3
"""Temperature Sanity Check for Checkworthiness Pipeline.

=============================================================================
PURPOSE: VALIDATE DEFAULT ASSUMPTION (Not "optimize" temperature)
=============================================================================

This is NOT a full optimization study. It's a sanity check to confirm that
our default choice of T=0.0 is reasonable across all models.

## Default Assumption

For classification tasks with structured JSON output, T=0.0 (deterministic)
is the standard choice because:
1. Reproducibility - same input produces same output
2. Stability - no random variation in predictions
3. Convention - widely accepted for classification tasks

## What This Study Checks

Are there any models where T=0.0 causes CATASTROPHIC failure compared to T=0.7?
- "Catastrophic" = >20% absolute difference in F2 score
- If no catastrophic failures, we stick with T=0.0 (convention)
- If a model fails badly at T=0.0, we investigate further

## Why We're NOT Doing a Full Study

Power analysis shows:
- N=20 samples can only detect effects >= 28% (e.g., 70% -> 98%)
- Detecting 10% differences would require N=292 samples (~$500+)
- If the effect is <10%, it's probably not worth optimizing

=============================================================================
PRE-REGISTRATION (AHIRT Framework)
=============================================================================

## Hypothesis (Falsifiable)

H0 (Default): T=0.0 works well for all models (no catastrophic failures)

Falsification criteria:
- If ANY model shows >20% F2 drop at T=0.0 compared to T=0.7, investigate

Note: We're NOT testing "which is better" - we're testing "is T=0.0 safe?"

## Assumptions

1. CT24 dev set is representative enough for a sanity check
2. 20 samples can detect catastrophic failures (>20% effect)
3. If T=0.0 vs T=0.7 shows no big difference, T=0.3 won't either

## Primary Metric

F2 Score (weights recall 2x) because False Negatives > False Positives

## Success Criteria

- If |F2(T=0.0) - F2(T=0.7)| < 20% for all models: ✓ Use T=0.0
- If any model shows >20% difference: Investigate that model further

## Known Limitations

- Cannot detect small effects (<20%)
- This is a sanity check, not publication-grade rigor
- Assumes extreme (T=0.7) would reveal any issues with T=0.3 too

=============================================================================
EXPERIMENT DESIGN
=============================================================================

- 20 samples (10 Yes + 10 No, stratified)
- 5 models (GPT-4.1, GPT-4.1-mini, DeepSeek Chat, DeepSeek Reasoner, Grok 4.1)
- 2 temperatures (0.0 vs 0.7 - extremes only)
- 1 run per config (no variance estimation - this is a sanity check)
- Zero-shot prompts

Total API calls: 20 x 5 x 2 x 3 = 600 calls
Estimated cost: ~$10-15

Key insight: We're validating our assumption, not optimizing.
"""

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from openai import OpenAI

from checkworthiness.config import ModelConfig, ModelProvider
from checkworthiness.metrics import calculate_all_metrics
from checkworthiness.prompting_baseline import PromptingBaseline
from checkworthiness.statistical_tests import (
    bootstrap_ci,
    bootstrap_f2_ci,
    cohens_h,
    holm_bonferroni_correction,
    interpret_effect_size,
    mcnemar_test,
    summarize_power_for_experiment,
    wilson_score_ci,
)

# =============================================================================
# Reasoning Alignment Analysis
# =============================================================================


def compute_reasoning_alignment(
    internal_reasoning: str,
    stated_reasoning: str,
    client: OpenAI | None = None,
    model: str = "text-embedding-3-large",
) -> float | None:
    """Compute cosine similarity between internal and stated reasoning.

    Uses OpenAI's text-embedding-3-large (SOTA quality) to embed both texts
    and compute cosine similarity.

    Args:
        internal_reasoning: The internal chain-of-thought from reasoning models
        stated_reasoning: The reasoning included in the final JSON output
        client: OpenAI client instance (creates one if not provided)
        model: Embedding model to use (default: text-embedding-3-large)

    Returns:
        Cosine similarity score between 0 and 1 (higher = more aligned)
        Returns None if either input is empty/whitespace-only (not computable)

    Cost: ~$0.0001 per embedding pair (negligible for small experiments)
    """
    # Check for empty or whitespace-only inputs - alignment not computable
    # Return None (not 0.0) so these are excluded from aggregates
    internal_clean = internal_reasoning.strip() if internal_reasoning else ""
    stated_clean = stated_reasoning.strip() if stated_reasoning else ""

    if not internal_clean or not stated_clean:
        return None  # Not computable - exclude from aggregates

    if client is None:
        client = OpenAI()

    # Get embeddings for both texts
    response = client.embeddings.create(
        model=model,
        input=[internal_clean, stated_clean],
    )

    emb_internal = response.data[0].embedding
    emb_stated = response.data[1].embedding

    # Cosine similarity
    dot_product = sum(a * b for a, b in zip(emb_internal, emb_stated))
    norm_internal = sum(a * a for a in emb_internal) ** 0.5
    norm_stated = sum(b * b for b in emb_stated) ** 0.5

    if norm_internal == 0 or norm_stated == 0:
        return None  # Degenerate case - exclude from aggregates

    return dot_product / (norm_internal * norm_stated)


def compute_alignment_for_predictions(
    predictions: list,  # list[PredictionResult]
    verbose: bool = True,
) -> list:  # list[PredictionResult]
    """Compute reasoning alignment scores for all predictions with separate reasoning.

    Only computes alignment for predictions where has_separate_reasoning=True
    (currently only DeepSeek Reasoner).

    Args:
        predictions: List of PredictionResult objects
        verbose: Print progress messages

    Returns:
        Updated list of PredictionResult objects with alignment scores populated
    """
    # Filter to predictions with separate reasoning
    with_reasoning = [p for p in predictions if p.has_separate_reasoning and p.internal_reasoning]

    if not with_reasoning:
        if verbose:
            print("No predictions with separate reasoning found. Skipping alignment computation.")
        return predictions

    if verbose:
        print(f"\nComputing alignment scores for {len(with_reasoning)} predictions...")

    # Initialize OpenAI client once for efficiency
    client = OpenAI()

    for i, pred in enumerate(with_reasoning):
        # Combine stated reasoning from all modules
        stated_reasoning = "\n\n".join([
            f"[Checkability] {pred.checkability_reasoning}",
            f"[Verifiability] {pred.verifiability_reasoning}",
            f"[Harm] {pred.harm_reasoning}",
        ])

        try:
            alignment = compute_reasoning_alignment(
                internal_reasoning=pred.internal_reasoning,
                stated_reasoning=stated_reasoning,
                client=client,
            )
            pred.reasoning_alignment_score = alignment

            if verbose and (i + 1) % 5 == 0:
                print(f"  Computed alignment for {i + 1}/{len(with_reasoning)} predictions")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to compute alignment for {pred.sample_id}: {e}")
            pred.reasoning_alignment_score = None

    if verbose:
        # Summary stats
        scores = [p.reasoning_alignment_score for p in with_reasoning if p.reasoning_alignment_score is not None]
        if scores:
            print(f"  Alignment scores: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
            print(f"  Range: [{min(scores):.3f}, {max(scores):.3f}]")

    return predictions


def compute_reasoning_calibration_metrics(
    predictions: list,  # list[PredictionResult]
    verbose: bool = True,
) -> dict:
    """Compute calibration metrics for reasoning models.

    For models with separate reasoning (DeepSeek Reasoner), computes:
    - ECE (Expected Calibration Error) from final answer confidence
    - Brier score from final answer confidence
    - Token statistics for reasoning vs final answer

    NOTE: We do NOT compare "internal reasoning calibration" vs "final answer calibration"
    because there's no meaningful way to extract a scalar confidence from the internal
    reasoning chain. The internal reasoning is a sequence of tokens that inform the final
    answer, but doesn't produce a separate probability estimate. Computing ECE on both
    using the same p.confidence value would be scientifically meaningless (always 0 difference).

    Future work could explore:
    - Perplexity-based uncertainty from reasoning tokens
    - Semantic consistency between reasoning and conclusion
    - Token-level entropy analysis

    Args:
        predictions: List of PredictionResult objects
        verbose: Print progress messages

    Returns:
        Dictionary with calibration metrics:
        {
            "has_reasoning_models": bool,
            "n_with_reasoning": int,
            "final_ece": float | None,
            "final_brier": float | None,
            "avg_reasoning_tokens": float | None,
            "avg_final_tokens": float | None,
        }
    """
    # Filter to predictions with separate reasoning
    with_reasoning = [
        p for p in predictions
        if p.has_separate_reasoning and p.internal_reasoning
    ]

    result = {
        "has_reasoning_models": len(with_reasoning) > 0,
        "n_with_reasoning": len(with_reasoning),
        "final_ece": None,
        "final_brier": None,
        "avg_reasoning_tokens": None,
        "avg_final_tokens": None,
    }

    if not with_reasoning:
        if verbose:
            print("No predictions with separate reasoning found. Skipping calibration metrics.")
        return result

    if verbose:
        print(f"\nComputing calibration metrics for {len(with_reasoning)} predictions with separate reasoning...")

    # Compute average token counts
    reasoning_tokens = [p.reasoning_tokens for p in with_reasoning]
    final_tokens = [p.final_answer_tokens for p in with_reasoning]
    result["avg_reasoning_tokens"] = np.mean(reasoning_tokens) if reasoning_tokens else 0
    result["avg_final_tokens"] = np.mean(final_tokens) if final_tokens else 0

    # For ECE/Brier, we need:
    # 1. The model's confidence (from final answer logprobs)
    # 2. The ground truth label

    # Extract confidences and ground truths
    final_confidences = []
    ground_truths = []  # 1 for Yes, 0 for No

    for p in with_reasoning:
        # Ground truth
        gt = 1 if p.ground_truth == "Yes" else 0
        ground_truths.append(gt)

        # Final answer confidence (computed from logprobs in p_true)
        final_confidences.append(p.confidence / 100.0)  # Convert 0-100 to 0-1

    # Convert to numpy arrays
    final_conf = np.array(final_confidences)
    gt = np.array(ground_truths)

    # Compute ECE (Expected Calibration Error)
    def compute_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int | None = None) -> float:
        """Compute Expected Calibration Error with adaptive binning.

        For small N, uses sqrt(N) bins clamped to [3, 10] range.
        This avoids unreliable estimates from sparsely populated bins.

        Uses np.digitize for consistent edge handling (includes both 0.0 and 1.0).
        """
        if len(confidences) == 0:
            return 0.0

        # Adaptive binning for small samples
        if n_bins is None:
            n_bins = max(3, min(10, int(np.sqrt(len(confidences)))))

        # Use np.digitize for consistent edge handling
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries[1:-1])  # Use internal boundaries

        ece = 0.0
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() == 0:
                continue
            bin_conf = confidences[mask].mean()
            bin_acc = labels[mask].mean()
            ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
        return ece

    # Compute Brier score
    def compute_brier(confidences: np.ndarray, labels: np.ndarray) -> float:
        """Compute Brier score (mean squared error of probabilistic predictions)."""
        return np.mean((confidences - labels) ** 2)

    result["final_ece"] = compute_ece(final_conf, gt)
    result["final_brier"] = compute_brier(final_conf, gt)

    if verbose:
        print(f"  Final answer ECE: {result['final_ece']:.4f}")
        print(f"  Final answer Brier: {result['final_brier']:.4f}")
        print(f"  Avg reasoning tokens: {result['avg_reasoning_tokens']:.1f}")
        print(f"  Avg final tokens: {result['avg_final_tokens']:.1f}")

    return result


def get_reasoning_analysis_summary(predictions: list, verbose: bool = True) -> dict:
    """Get comprehensive reasoning analysis summary.

    Combines alignment scores and calibration metrics into a single summary.

    Args:
        predictions: List of PredictionResult objects
        verbose: Print progress messages

    Returns:
        Dictionary with full reasoning analysis:
        {
            "alignment": {...},
            "calibration": {...},
            "reasoning_models": [...],
            "n_total_predictions": int,
        }
    """
    # Get predictions with separate reasoning
    with_reasoning = [p for p in predictions if p.has_separate_reasoning and p.internal_reasoning]

    # Get alignment scores
    alignment_scores = [p.reasoning_alignment_score for p in with_reasoning if p.reasoning_alignment_score is not None]

    alignment_summary = {
        "n_with_alignment": len(alignment_scores),
        "mean": np.mean(alignment_scores) if alignment_scores else None,
        "std": np.std(alignment_scores) if alignment_scores else None,
        "min": min(alignment_scores) if alignment_scores else None,
        "max": max(alignment_scores) if alignment_scores else None,
        "median": np.median(alignment_scores) if alignment_scores else None,
    }

    # Get calibration metrics
    calibration_summary = compute_reasoning_calibration_metrics(predictions, verbose=False)

    # Get reasoning models
    reasoning_models = list(set(p.model for p in with_reasoning))

    return {
        "alignment": alignment_summary,
        "calibration": calibration_summary,
        "reasoning_models": reasoning_models,
        "n_total_predictions": len(predictions),
        "n_with_reasoning": len(with_reasoning),
    }


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
N_SAMPLES = 20  # 10 Yes + 10 No (stratified) - sanity check size
TEMPERATURES = [0.0, 0.7]  # Extremes only - validates our T=0.0 assumption
N_RUNS_PER_CONFIG = 1  # Single run (this is a sanity check, not optimization)

# CATASTROPHIC_THRESHOLD: If |F2(T=0.0) - F2(T=0.7)| > this, investigate further
CATASTROPHIC_THRESHOLD = 0.20  # 20% absolute difference

# MIN_VALID_SAMPLE_RATIO: If valid samples < this * N_SAMPLES, results are unreliable
# Protects against rate limit failures silently producing misleading metrics
MIN_VALID_SAMPLE_RATIO = 0.80  # Need at least 80% of samples for valid results

# =============================================================================
# Logprob Drift Probe
# =============================================================================
# Detects if model changed during experiment by comparing logprobs before/after.
# S ≈ 0 means stable, S large means potential drift.
# p-value tells us if the difference is real or just random noise.
#
# Probe prompt should trigger reasoning to detect drift in reasoning models.
# Simple prompts like "x" won't engage the reasoning pathway in models like
# DeepSeek Reasoner, missing potential drift in the reasoning module.
# =============================================================================
LOGPROB_PROBE_PROMPT = "If A>B and B>C, is A>C? Answer Yes or No."  # Triggers reasoning
LOGPROB_PROBE_SAMPLES = 10  # Samples before + after (20 total per model)
LOGPROB_PERMUTATIONS = 200  # Shuffles for p-value calculation
LOGPROB_ALPHA = 0.05  # p < this = statistically significant
LOGPROB_DRIFT_MIN_S = 0.5  # S > this = meaningful effect size

# Dataset path
CT24_DEV_PATH = project_root / "data" / "raw" / "CT24_checkworthy_english" / "CT24_checkworthy_english_dev.tsv"

# Zero-shot V2 prompts (ternary output: true/false/uncertain)
PROMPTS_PATH = project_root / "prompts" / "checkworthiness_prompts_zeroshot_v2.yaml"

# Model configurations
# =============================================================================
# IJCAI Paper Model Selection Rationale:
# -----------------------------------------------------------------------------
# 1. gpt-4.1 vs gpt-4.1-mini  → Size comparison (same architecture, different scale)
# 2. deepseek-chat vs deepseek-reasoner  → Reasoning comparison (V3 vs R1)
# 3. grok-4.1-fast-reasoning  → Third provider for robustness
#
# All models verified to support logprobs (see verify_logprobs.py)
# =============================================================================
MODEL_CONFIGS = {
    # OpenAI - Size comparison
    "gpt-4.1": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4.1",
        api_key_env="OPENAI_API_KEY",
        max_tokens=512,
        logprobs=True,
        top_logprobs=5,
    ),
    "gpt-4.1-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
        max_tokens=512,
        logprobs=True,
        top_logprobs=5,
    ),
    # DeepSeek - Reasoning comparison (open-weight models)
    "deepseek-chat": ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-chat",  # DeepSeek V3 (non-reasoning)
        api_key_env="DEEPSEEK_API_KEY",
        api_base="https://api.deepseek.com/beta",  # Beta API for logprobs
        max_tokens=512,
        logprobs=True,
        top_logprobs=5,
    ),
    "deepseek-reasoner": ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-reasoner",  # DeepSeek R1 (reasoning model)
        api_key_env="DEEPSEEK_API_KEY",
        api_base="https://api.deepseek.com/beta",  # Beta API for logprobs
        max_tokens=512,  # For final answer (reasoning tokens are additional)
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=True,  # Reasoning model - no assistant prefill
    ),
    # xAI - Third provider for robustness
    "grok-4.1-fast-reasoning": ModelConfig(
        provider=ModelProvider.XAI,
        model_name="grok-4-1-fast-reasoning",  # Grok 4.1 reasoning variant
        api_key_env="XAI_API_KEY",
        api_base="https://api.x.ai/v1",
        max_tokens=512,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=True,  # Reasoning model - no assistant prefill
    ),
}


# =============================================================================
# Logprob Drift Probe Functions
# =============================================================================


@dataclass
class DriftProbeResult:
    """Result of logprob drift probe for one model."""

    model: str
    pre_samples: int
    post_samples: int
    statistic_S: float | None  # None if skipped
    p_value: float | None  # None if skipped or test degenerate
    drift_detected: bool | None  # None if test not applicable (degenerate)
    skipped: bool
    skip_reason: str | None
    pre_timestamp: str
    post_timestamp: str
    # Raw data for debugging
    vocabulary_size: int = 0
    pre_mean_logprobs: dict | None = None
    post_mean_logprobs: dict | None = None


def collect_logprob_probe_samples(
    model_config: ModelConfig,
    n_samples: int = LOGPROB_PROBE_SAMPLES,
) -> list[dict[str, float]]:
    """Collect first-token logprob samples for drift detection.

    Sends a minimal prompt and collects the logprob distribution over tokens.

    Args:
        model_config: Model configuration
        n_samples: Number of samples to collect

    Returns:
        List of dicts, each mapping token -> logprob for that sample.
        Empty list if logprobs not supported.
    """
    from openai import OpenAI

    api_key = model_config.get_api_key()
    if not api_key:
        return []

    client = OpenAI(api_key=api_key, base_url=model_config.api_base)

    samples = []
    for _ in range(n_samples):
        try:
            response = client.chat.completions.create(
                model=model_config.model_name,
                messages=[{"role": "user", "content": LOGPROB_PROBE_PROMPT}],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=model_config.top_logprobs or 5,
            )

            choice = response.choices[0]
            if choice.logprobs and choice.logprobs.content:
                first_token_info = choice.logprobs.content[0]
                # Build dict of token -> logprob from top_logprobs
                token_logprobs = {}
                if first_token_info.top_logprobs:
                    for lp in first_token_info.top_logprobs:
                        token_logprobs[lp.token] = lp.logprob
                # Also include the actual token if not in top_logprobs
                if first_token_info.token not in token_logprobs:
                    token_logprobs[first_token_info.token] = first_token_info.logprob
                samples.append(token_logprobs)

        except Exception as e:
            # Log but continue - some samples may fail
            print(f"    Probe sample failed: {e}")
            continue

    return samples


def compute_drift_statistic(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
) -> tuple[float, set[str]]:
    """Compute drift statistic S between two sets of logprob samples.

    S = mean absolute difference of per-token mean logprobs.

    Args:
        samples_a: First set of samples (e.g., pre-experiment)
        samples_b: Second set of samples (e.g., post-experiment)

    Returns:
        Tuple of (S statistic, vocabulary set)
    """
    if not samples_a or not samples_b:
        return 0.0, set()

    # Build vocabulary = union of all tokens seen
    vocabulary = set()
    for sample in samples_a + samples_b:
        vocabulary.update(sample.keys())

    if not vocabulary:
        return 0.0, set()

    # For each set, compute mean logprob per token
    # If token missing from a sample, impute with fixed floor value
    # Using fixed floor (-100.0) ensures consistent penalty regardless of sample confidence
    # This avoids confounding "drift" with "change in confidence distribution"
    IMPUTATION_FLOOR = -100.0  # Fixed penalty for missing tokens

    def compute_means(samples: list[dict[str, float]]) -> dict[str, float]:
        means = {}
        for token in vocabulary:
            token_values = []
            for sample in samples:
                if token in sample:
                    token_values.append(sample[token])
                else:
                    # Fixed floor imputation - consistent across all samples
                    token_values.append(IMPUTATION_FLOOR)
            means[token] = np.mean(token_values)
        return means

    means_a = compute_means(samples_a)
    means_b = compute_means(samples_b)

    # S = mean |mean_a - mean_b| across all tokens
    diffs = [abs(means_a[t] - means_b[t]) for t in vocabulary]
    S = np.mean(diffs)

    return S, vocabulary


def permutation_test(
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    observed_S: float,
    n_permutations: int = LOGPROB_PERMUTATIONS,
    random_state: int | None = None,
) -> float | None:
    """Compute p-value via permutation test.

    Answers: "Could this S happen by random chance?"

    Args:
        samples_a: First set of samples
        samples_b: Second set of samples
        observed_S: The observed drift statistic
        n_permutations: Number of random shuffles
        random_state: Random seed for reproducibility (uses local RNG)

    Returns:
        p-value (fraction of permuted S >= observed S)
        Returns None if test is degenerate (zero variance - all permutations identical)
    """
    if not samples_a or not samples_b:
        return None  # Cannot compute - insufficient data

    # Use local RNG for reproducibility independent of global state
    rng = random.Random(random_state)

    pooled = samples_a + samples_b
    n_a = len(samples_a)
    count_ge = 0
    perm_S_values = []

    for _ in range(n_permutations):
        # Random shuffle and split using local RNG
        shuffled = rng.sample(pooled, len(pooled))
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]

        perm_S, _ = compute_drift_statistic(perm_a, perm_b)
        perm_S_values.append(perm_S)
        if perm_S >= observed_S:
            count_ge += 1

    # Detect degenerate case: zero variance in permutation distribution
    # This happens when all samples are identical (e.g., deterministic model)
    # In this case, the test cannot distinguish drift from no-drift
    if len(set(round(s, 10) for s in perm_S_values)) <= 1:
        return None  # Test not applicable - degenerate distribution

    return count_ge / n_permutations


def run_drift_probe(
    model_key: str,
    model_config: ModelConfig,
    phase: str,
    verbose: bool = True,
) -> tuple[list[dict[str, float]], str]:
    """Run drift probe for a model (either 'pre' or 'post' phase).

    Args:
        model_key: Model identifier
        model_config: Model configuration
        phase: 'pre' or 'post'
        verbose: Print progress

    Returns:
        Tuple of (samples list, timestamp string)
    """
    timestamp = datetime.now().isoformat()

    if verbose:
        print(f"  [{phase.upper()}] Collecting {LOGPROB_PROBE_SAMPLES} probe samples...", end=" ", flush=True)

    samples = collect_logprob_probe_samples(model_config, LOGPROB_PROBE_SAMPLES)

    if verbose:
        if samples:
            print(f"got {len(samples)} samples")
        else:
            print("skipped (no logprobs)")

    return samples, timestamp


def analyze_drift_probe(
    model_key: str,
    pre_samples: list[dict[str, float]],
    post_samples: list[dict[str, float]],
    pre_timestamp: str,
    post_timestamp: str,
    verbose: bool = True,
) -> DriftProbeResult:
    """Analyze pre/post probe samples to detect drift.

    Args:
        model_key: Model identifier
        pre_samples: Samples from before experiment
        post_samples: Samples from after experiment
        pre_timestamp: When pre-samples were collected
        post_timestamp: When post-samples were collected
        verbose: Print results

    Returns:
        DriftProbeResult with statistics and detection result
    """
    # Check if we have enough samples
    if not pre_samples or not post_samples:
        skip_reason = "logprobs not supported" if not pre_samples else "post-samples missing"
        return DriftProbeResult(
            model=model_key,
            pre_samples=len(pre_samples),
            post_samples=len(post_samples),
            statistic_S=None,
            p_value=None,
            drift_detected=False,
            skipped=True,
            skip_reason=skip_reason,
            pre_timestamp=pre_timestamp,
            post_timestamp=post_timestamp,
        )

    # Compute drift statistic
    S, vocabulary = compute_drift_statistic(pre_samples, post_samples)

    # Compute p-value (may return None if test is degenerate)
    # Use RANDOM_SEED for reproducibility independent of earlier random calls
    p_value = permutation_test(pre_samples, post_samples, S, random_state=RANDOM_SEED)

    # Detect drift: significant AND meaningful effect
    # If p_value is None (degenerate test), we cannot detect drift
    if p_value is None:
        drift_detected = None  # Test not applicable
    else:
        drift_detected = (p_value < LOGPROB_ALPHA) and (S > LOGPROB_DRIFT_MIN_S)

    if verbose:
        if drift_detected is None:
            status = "⚠️  TEST N/A (degenerate)"
            print(f"  [RESULT] S={S:.4f}, p=N/A → {status}")
        elif drift_detected:
            status = "⚠️  DRIFT DETECTED"
            print(f"  [RESULT] S={S:.4f}, p={p_value:.3f} → {status}")
        else:
            status = "✓ Stable"
            print(f"  [RESULT] S={S:.4f}, p={p_value:.3f} → {status}")

    return DriftProbeResult(
        model=model_key,
        pre_samples=len(pre_samples),
        post_samples=len(post_samples),
        statistic_S=S,
        p_value=p_value,
        drift_detected=drift_detected,
        skipped=False,
        skip_reason=None,
        pre_timestamp=pre_timestamp,
        post_timestamp=post_timestamp,
        vocabulary_size=len(vocabulary),
    )


# =============================================================================
# Data Loading
# =============================================================================


def load_ct24_samples(n_samples: int = 20, seed: int = RANDOM_SEED) -> list[dict]:
    """Load stratified samples from CT24 dev set.

    Args:
        n_samples: Total samples (half Yes, half No)
        seed: Random seed for reproducibility

    Returns:
        List of dicts with 'id', 'text', 'label'
    """
    df = pl.read_csv(CT24_DEV_PATH, separator="\t")

    yes_rows = df.filter(pl.col("class_label") == "Yes").to_dicts()
    no_rows = df.filter(pl.col("class_label") == "No").to_dicts()

    random.seed(seed)
    n_per_class = n_samples // 2

    yes_selected = random.sample(yes_rows, min(n_per_class, len(yes_rows)))
    no_selected = random.sample(no_rows, min(n_per_class, len(no_rows)))

    samples = []
    for row in yes_selected + no_selected:
        samples.append({
            "id": str(row["Sentence_id"]),
            "text": row["Text"],
            "label": row["class_label"],
        })

    random.shuffle(samples)
    print(f"Loaded {len(samples)} samples: {len(yes_selected)} Yes, {len(no_selected)} No")
    return samples


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are available."""
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "XAI_API_KEY": bool(os.environ.get("XAI_API_KEY")),
    }


# =============================================================================
# Experiment Runner
# =============================================================================


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    sample_id: str
    text: str  # Full claim text (not truncated)
    ground_truth: str  # "Yes" or "No"
    model: str
    temperature: float
    run_id: int  # Run number for variance estimation (1, 2, 3, ...)
    confidence: float  # average p_true across modules (0-100)
    prediction: bool  # True if checkworthy (confidence > 50)

    # Per-module confidences
    checkability_confidence: float
    verifiability_confidence: float
    harm_confidence: float

    # Per-module reasoning (full text)
    checkability_reasoning: str
    verifiability_reasoning: str
    harm_reasoning: str

    # Harm sub-scores
    harm_social_fragmentation: float
    harm_spurs_action: float
    harm_believability: float
    harm_exploitativeness: float

    # Metadata
    tokens_used: int
    latency_ms: float
    error: str | None = None

    # NEW: Internal reasoning analysis (DeepSeek Reasoner only)
    internal_reasoning: str | None = None  # reasoning_content from API
    has_separate_reasoning: bool = False  # True only for reasoning models with separate CoT

    # NEW: Token-level logprobs (stored as JSON strings for parquet compatibility)
    final_answer_logprobs: str | None = None  # JSON list of {token, logprob}
    internal_reasoning_logprobs: str | None = None  # JSON list (DeepSeek Reasoner only)

    # NEW: Token counts by type
    reasoning_tokens: int = 0  # Tokens in reasoning_content
    final_answer_tokens: int = 0  # Tokens in content

    # NEW: Alignment metrics (computed via embeddings)
    reasoning_alignment_score: float | None = None  # Cosine similarity internal vs stated


@dataclass
class ExperimentResults:
    """Results from temperature experiment."""

    predictions: list[PredictionResult] = field(default_factory=list)
    drift_probes: list[DriftProbeResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    intended_models: list[str] = field(default_factory=list)  # Models we attempted to run

    def add(self, result: PredictionResult):
        self.predictions.append(result)

    def add_drift_probe(self, probe: DriftProbeResult):
        self.drift_probes.append(probe)

    def to_dataframe(self, truncate_text: bool = False) -> pl.DataFrame:
        """Convert predictions to DataFrame.

        Args:
            truncate_text: If True, truncate text/reasoning for display. If False, keep full text.

        Returns:
            DataFrame with all prediction data including:
            - Basic prediction info (sample_id, text, ground_truth, model, temperature, etc.)
            - Per-module confidences and reasoning
            - Harm sub-scores
            - NEW: Internal reasoning (for reasoning models like DeepSeek Reasoner)
            - NEW: Token-level logprobs (JSON strings for parquet compatibility)
            - NEW: Alignment metrics
        """
        records = []
        for p in self.predictions:
            text = p.text[:60] + "..." if truncate_text and len(p.text) > 60 else p.text
            records.append({
                "sample_id": p.sample_id,
                "text": text,
                "ground_truth": p.ground_truth,
                "model": p.model,
                "temperature": p.temperature,
                "run_id": p.run_id,
                "confidence": p.confidence,
                "prediction": "Yes" if p.prediction else "No",
                "correct": (p.ground_truth == "Yes") == p.prediction,
                # Per-module confidences
                "checkability_confidence": p.checkability_confidence,
                "verifiability_confidence": p.verifiability_confidence,
                "harm_confidence": p.harm_confidence,
                # Per-module reasoning (stated reasoning from JSON output)
                "checkability_reasoning": p.checkability_reasoning,
                "verifiability_reasoning": p.verifiability_reasoning,
                "harm_reasoning": p.harm_reasoning,
                # Harm sub-scores
                "harm_social_fragmentation": p.harm_social_fragmentation,
                "harm_spurs_action": p.harm_spurs_action,
                "harm_believability": p.harm_believability,
                "harm_exploitativeness": p.harm_exploitativeness,
                # Metadata
                "tokens": p.tokens_used,
                "latency_ms": p.latency_ms,
                "error": p.error,
                # NEW: Internal reasoning (DeepSeek Reasoner only)
                "internal_reasoning": p.internal_reasoning,
                "has_separate_reasoning": p.has_separate_reasoning,
                # NEW: Token-level logprobs (JSON strings)
                "final_answer_logprobs": p.final_answer_logprobs,
                "internal_reasoning_logprobs": p.internal_reasoning_logprobs,
                # NEW: Token counts
                "reasoning_tokens": p.reasoning_tokens,
                "final_answer_tokens": p.final_answer_tokens,
                # NEW: Alignment metrics
                "reasoning_alignment_score": p.reasoning_alignment_score,
            })

        # Use explicit schema to handle None/str mixed types
        schema = {
            "sample_id": pl.Utf8,
            "text": pl.Utf8,
            "ground_truth": pl.Utf8,
            "model": pl.Utf8,
            "temperature": pl.Float64,
            "run_id": pl.Int64,
            "confidence": pl.Float64,
            "prediction": pl.Utf8,
            "correct": pl.Boolean,
            "checkability_confidence": pl.Float64,
            "verifiability_confidence": pl.Float64,
            "harm_confidence": pl.Float64,
            "checkability_reasoning": pl.Utf8,
            "verifiability_reasoning": pl.Utf8,
            "harm_reasoning": pl.Utf8,
            "harm_social_fragmentation": pl.Float64,
            "harm_spurs_action": pl.Float64,
            "harm_believability": pl.Float64,
            "harm_exploitativeness": pl.Float64,
            "tokens": pl.Int64,
            "latency_ms": pl.Float64,
            "error": pl.Utf8,  # Explicit string type handles None → null correctly
            # NEW: Internal reasoning columns
            "internal_reasoning": pl.Utf8,
            "has_separate_reasoning": pl.Boolean,
            # NEW: Logprobs columns (JSON strings)
            "final_answer_logprobs": pl.Utf8,
            "internal_reasoning_logprobs": pl.Utf8,
            # NEW: Token counts
            "reasoning_tokens": pl.Int64,
            "final_answer_tokens": pl.Int64,
            # NEW: Alignment metrics
            "reasoning_alignment_score": pl.Float64,
        }
        return pl.DataFrame(records, schema=schema)

    def get_metrics_by_config(self) -> pl.DataFrame:
        """Calculate metrics for each model x temperature configuration.

        For configs with multiple runs, computes:
        - Mean metrics across runs
        - Standard deviation across runs
        - 95% Wilson score CIs for accuracy/recall/precision
        """
        df = self.to_dataframe()
        if df.is_empty():
            return pl.DataFrame()

        df_ok = df.filter(pl.col("error").is_null())
        if df_ok.is_empty():
            return pl.DataFrame()

        results = []
        models = df_ok["model"].unique().to_list()
        temps = sorted(df_ok["temperature"].unique().to_list())

        for model in models:
            for temp in temps:
                subset = df_ok.filter((pl.col("model") == model) & (pl.col("temperature") == temp))

                if subset.is_empty():
                    continue

                # Get unique runs
                runs = subset["run_id"].unique().to_list()
                n_runs = len(runs)

                # Compute per-run metrics for variance estimation
                run_metrics = []
                run_sample_counts = []  # Track actual samples per run
                for run_id in runs:
                    run_subset = subset.filter(pl.col("run_id") == run_id)
                    run_sample_counts.append(run_subset.height)
                    preds = run_subset.select(["ground_truth", "prediction", "confidence"]).to_dicts()
                    metrics = calculate_all_metrics(preds, confidence_is_correct=False)
                    run_metrics.append(metrics)

                # Aggregate metrics across runs
                # Use pooled predictions for main metrics
                all_preds = subset.select(["ground_truth", "prediction", "confidence"]).to_dicts()
                pooled_metrics = calculate_all_metrics(all_preds, confidence_is_correct=False)

                # Calculate variance across runs (if multiple runs)
                if n_runs > 1:
                    f2_values = [m["f2_score"] for m in run_metrics]
                    acc_values = [m["accuracy"] for m in run_metrics]
                    recall_values = [m["recall"] for m in run_metrics]

                    f2_std = float(np.std(f2_values, ddof=1))
                    acc_std = float(np.std(acc_values, ddof=1))
                    recall_std = float(np.std(recall_values, ddof=1))
                else:
                    f2_std = acc_std = recall_std = 0.0

                # Wilson score CI for proportions (more accurate for small samples)
                # Use actual per-run counts instead of integer division
                n_samples_per_run_min = min(run_sample_counts) if run_sample_counts else 0
                n_samples_per_run_mean = sum(run_sample_counts) / len(run_sample_counts) if run_sample_counts else 0
                tp = pooled_metrics["confusion_matrix"]["tp"]
                fn = pooled_metrics["confusion_matrix"]["fn"]
                fp = pooled_metrics["confusion_matrix"]["fp"]
                tn = pooled_metrics["confusion_matrix"]["tn"]

                # CI for accuracy
                acc_ci = wilson_score_ci(tp + tn, tp + tn + fp + fn)
                # CI for recall (among positives)
                recall_ci = wilson_score_ci(tp, tp + fn) if (tp + fn) > 0 else wilson_score_ci(0, 1)
                # CI for precision (among predicted positives)
                prec_ci = wilson_score_ci(tp, tp + fp) if (tp + fp) > 0 else wilson_score_ci(0, 1)

                # Bootstrap CI for F2 score (compound metric needs bootstrap, not Wilson)
                f2_ci = bootstrap_f2_ci(
                    all_preds,
                    n_bootstrap=5000,  # Fewer than default 10k for speed, still accurate
                    ci_level=0.95,
                    random_state=RANDOM_SEED,
                )

                avg_tokens = float(subset["tokens"].mean()) if subset["tokens"].mean() is not None else 0.0
                avg_latency = (
                    float(subset["latency_ms"].mean()) if subset["latency_ms"].mean() is not None else 0.0
                )

                results.append({
                    "model": model,
                    "temperature": temp,
                    "n_runs": n_runs,
                    "n_samples_total": subset.height,
                    "n_samples_per_run_min": n_samples_per_run_min,
                    "n_samples_per_run_mean": n_samples_per_run_mean,
                    # Main metrics
                    "accuracy": pooled_metrics["accuracy"],
                    "recall": pooled_metrics["recall"],
                    "precision": pooled_metrics["precision"],
                    "f1_score": pooled_metrics["f1_score"],
                    "f2_score": pooled_metrics["f2_score"],
                    # Variance across runs (if any)
                    "f2_std": f2_std,
                    "accuracy_std": acc_std,
                    "recall_std": recall_std,
                    # 95% Wilson CIs
                    "accuracy_ci_lower": acc_ci.ci_lower,
                    "accuracy_ci_upper": acc_ci.ci_upper,
                    "recall_ci_lower": recall_ci.ci_lower,
                    "recall_ci_upper": recall_ci.ci_upper,
                    "precision_ci_lower": prec_ci.ci_lower,
                    "precision_ci_upper": prec_ci.ci_upper,
                    # F2 bootstrap CI
                    "f2_ci_lower": f2_ci.ci_lower,
                    "f2_ci_upper": f2_ci.ci_upper,
                    # Calibration
                    "ece": pooled_metrics["ece"],
                    "brier_score": pooled_metrics["brier_score"],
                    # Confusion matrix
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "tn": tn,
                    # Cost
                    "avg_tokens": avg_tokens,
                    "avg_latency_ms": avg_latency,
                })

        return pl.DataFrame(results)


def run_single_prediction(
    baseline: PromptingBaseline,
    sample: dict,
    model_name: str,
    temperature: float,
    run_id: int = 1,
) -> PredictionResult:
    """Run full checkworthiness pipeline for a single sample.

    For reasoning models (e.g., DeepSeek Reasoner), this also captures:
    - Internal reasoning (reasoning_content from API)
    - Token-level logprobs for both final answer and internal reasoning
    - Token counts for reasoning vs final answer
    """
    try:
        start = time.time()
        # New signature returns: (result, usage, all_logprobs, combined_reasoning, combined_reasoning_logprobs)
        result, usage, all_logprobs, internal_reasoning, internal_reasoning_lp = baseline(sample["text"])
        latency = (time.time() - start) * 1000

        # Extract all confidences
        check_conf = float(result.checkability.confidence)
        verif_conf = float(result.verifiability.confidence)
        harm_conf = float(result.harm_potential.confidence)
        avg_conf = float(result.average_confidence)

        # Extract harm sub-scores
        harm_sub = result.harm_potential.sub_scores

        # Serialize logprobs to JSON strings for parquet storage
        # Combine all module logprobs into single list for final answer
        final_logprobs_list: list[dict] = []
        for module_name, module_logprobs in all_logprobs.items():
            if module_logprobs:
                final_logprobs_list.extend(module_logprobs)

        final_answer_logprobs_json = json.dumps(final_logprobs_list) if final_logprobs_list else None
        internal_reasoning_logprobs_json = json.dumps(internal_reasoning_lp) if internal_reasoning_lp else None

        # Calculate token counts with fallback to usage-based counts
        # Prefer logprobs length (exact) but fall back to API usage data if logprobs missing
        final_answer_tokens = len(final_logprobs_list) if final_logprobs_list else usage.completion_tokens
        if internal_reasoning_lp:
            reasoning_tokens = len(internal_reasoning_lp)
        elif usage.reasoning_tokens > 0:
            # Fallback to API-reported reasoning tokens
            reasoning_tokens = usage.reasoning_tokens
        else:
            reasoning_tokens = 0

        # Determine if model has separate reasoning (DeepSeek Reasoner)
        has_separate_reasoning = internal_reasoning is not None and len(internal_reasoning) > 0

        return PredictionResult(
            sample_id=sample["id"],
            text=sample["text"],  # Full text
            ground_truth=sample["label"],
            model=model_name,
            temperature=temperature,
            run_id=run_id,
            confidence=avg_conf,
            prediction=result.prediction == "Yes",
            # Per-module confidences
            checkability_confidence=check_conf,
            verifiability_confidence=verif_conf,
            harm_confidence=harm_conf,
            # Per-module reasoning (stated reasoning from JSON output)
            checkability_reasoning=result.checkability.reasoning or "",
            verifiability_reasoning=result.verifiability.reasoning or "",
            harm_reasoning=result.harm_potential.reasoning or "",
            # Harm sub-scores
            harm_social_fragmentation=float(harm_sub.social_fragmentation),
            harm_spurs_action=float(harm_sub.spurs_action),
            harm_believability=float(harm_sub.believability),
            harm_exploitativeness=float(harm_sub.exploitativeness),
            # Metadata
            tokens_used=usage.total_tokens,
            latency_ms=latency,
            # NEW: Internal reasoning (DeepSeek Reasoner only)
            internal_reasoning=internal_reasoning,
            has_separate_reasoning=has_separate_reasoning,
            # NEW: Token-level logprobs (serialized as JSON)
            final_answer_logprobs=final_answer_logprobs_json,
            internal_reasoning_logprobs=internal_reasoning_logprobs_json,
            # NEW: Token counts
            reasoning_tokens=reasoning_tokens,
            final_answer_tokens=final_answer_tokens,
            # Alignment score computed later via compute_reasoning_alignment()
            reasoning_alignment_score=None,
        )
    except Exception as e:
        return PredictionResult(
            sample_id=sample["id"],
            text=sample["text"],
            ground_truth=sample["label"],
            model=model_name,
            temperature=temperature,
            run_id=run_id,
            confidence=0.0,
            prediction=False,
            checkability_confidence=0.0,
            verifiability_confidence=0.0,
            harm_confidence=0.0,
            checkability_reasoning="",
            verifiability_reasoning="",
            harm_reasoning="",
            harm_social_fragmentation=0.0,
            harm_spurs_action=0.0,
            harm_believability=0.0,
            harm_exploitativeness=0.0,
            tokens_used=0,
            latency_ms=0.0,
            error=str(e)[:200],  # Extended error message
            # Default values for new fields on error
            internal_reasoning=None,
            has_separate_reasoning=False,
            final_answer_logprobs=None,
            internal_reasoning_logprobs=None,
            reasoning_tokens=0,
            final_answer_tokens=0,
            reasoning_alignment_score=None,
        )


def run_experiment(
    models: list[str] | None = None,
    temperatures: list[float] | None = None,
    samples: list[dict] | None = None,
    verbose: bool = True,
) -> ExperimentResults:
    """Run the temperature optimization experiment."""
    # Ensure reproducibility - set seed at experiment start
    # (Also set in load_ct24_samples, but this guards against direct sample injection)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if models is None:
        models = list(MODEL_CONFIGS.keys())
    if temperatures is None:
        temperatures = TEMPERATURES
    if samples is None:
        samples = load_ct24_samples(N_SAMPLES)

    api_keys = check_api_keys()
    if verbose:
        print("\n" + "=" * 60)
        print("API KEY STATUS")
        print("=" * 60)
        for key, available in api_keys.items():
            status = "OK" if available else "MISSING"
            print(f"  {status} {key}")

    available_models = []
    for model_name in models:
        config = MODEL_CONFIGS[model_name]
        if api_keys.get(config.api_key_env, False):
            available_models.append(model_name)
        elif verbose:
            print(f"\nWARNING: Skipping {model_name}: {config.api_key_env} not set")

    if not available_models:
        print("\nERROR: No models available. Please set API keys.")
        return ExperimentResults(intended_models=[])

    # Calculate total API calls (T=0 is deterministic, others get multiple runs)
    calls_per_sample = 3  # checkability, verifiability, harm
    n_configs = 0
    for temp in temperatures:
        n_runs = 1 if temp == 0.0 else N_RUNS_PER_CONFIG
        n_configs += n_runs
    total_calls = len(samples) * len(available_models) * n_configs * calls_per_sample

    if verbose:
        print("\n" + "=" * 60)
        print("TEMPERATURE OPTIMIZATION EXPERIMENT")
        print("=" * 60)
        print(f"  Models: {available_models}")
        print(f"  Temperatures: {temperatures}")
        print(f"  Runs per stochastic config: {N_RUNS_PER_CONFIG}")
        print(f"  Samples: {len(samples)} (stratified)")
        print(f"  Calls per sample: {calls_per_sample} (checkability, verifiability, harm)")
        print(f"  Total API calls: {total_calls}")

        # Power analysis
        n_comparisons = len(available_models) * (len(temperatures) * (len(temperatures) - 1)) // 2
        power_info = summarize_power_for_experiment(
            n_samples=len(samples),
            n_comparisons=max(1, n_comparisons),
            baseline_f2=0.7,
        )
        print(f"\n  POWER ANALYSIS:")
        print(f"    {power_info['interpretation']}")
        print(f"    Sample size needed for 10% effect: {power_info['sample_size_for_10pct_effect']}")
        print("=" * 60)

    results = ExperimentResults(intended_models=available_models.copy())

    # Store pre-probe samples for each model (collected before predictions)
    pre_probe_data: dict[str, tuple[list[dict[str, float]], str]] = {}

    for model_name in available_models:
        config = MODEL_CONFIGS[model_name]

        # =====================================================================
        # PRE-EXPERIMENT DRIFT PROBE
        # =====================================================================
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"DRIFT PROBE: {model_name}")
            print("=" * 60)

        pre_samples, pre_timestamp = run_drift_probe(model_name, config, "pre", verbose)
        pre_probe_data[model_name] = (pre_samples, pre_timestamp)

        # =====================================================================
        # MAIN EXPERIMENT PREDICTIONS
        # =====================================================================
        for temp in temperatures:
            # Temperature 0.0 is deterministic, only 1 run needed
            n_runs = 1 if temp == 0.0 else N_RUNS_PER_CONFIG

            for run_id in range(1, n_runs + 1):
                if verbose:
                    print(f"\n{'-' * 40}")
                    print(f"Model: {model_name} | Temperature: {temp} | Run: {run_id}/{n_runs}")
                    print("-" * 40)

                baseline = PromptingBaseline(
                    config,
                    threshold=50.0,
                    temperature=temp,
                    prompts_path=str(PROMPTS_PATH),
                )

                for i, sample in enumerate(samples):
                    if verbose:
                        print(f"  [{i + 1}/{len(samples)}] {sample['text'][:40]}... ({sample['label']})")

                    result = run_single_prediction(baseline, sample, model_name, temp, run_id)
                    results.add(result)

                    # Print error immediately so user knows which model failed
                    if result.error:
                        print(f"    ⚠️  ERROR: {result.error[:100]}")

                    time.sleep(0.5)

        # =====================================================================
        # POST-EXPERIMENT DRIFT PROBE
        # =====================================================================
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"DRIFT PROBE (POST): {model_name}")
            print("=" * 60)

        post_samples, post_timestamp = run_drift_probe(model_name, config, "post", verbose)

        # Analyze drift between pre and post
        pre_samples_stored, pre_timestamp_stored = pre_probe_data[model_name]
        probe_result = analyze_drift_probe(
            model_name,
            pre_samples_stored,
            post_samples,
            pre_timestamp_stored,
            post_timestamp,
            verbose,
        )
        results.add_drift_probe(probe_result)

        # Warn loudly if drift detected or test was inconclusive
        if probe_result.drift_detected is True:
            print(f"\n⚠️  WARNING: Model drift detected for {model_name}!")
            print(f"    S={probe_result.statistic_S:.4f}, p={probe_result.p_value:.4f}")
            print("    Results for this model may be unreliable.\n")
        elif probe_result.drift_detected is None and not probe_result.skipped:
            print(f"\n⚠️  NOTE: Drift test inconclusive for {model_name} (degenerate distribution)")
            print(f"    S={probe_result.statistic_S:.4f}, p=N/A")
            print("    All logprob samples were identical - cannot assess drift.\n")

    results.end_time = datetime.now()
    return results


# =============================================================================
# Visualization
# =============================================================================


def _write_dashboard(figures: list[tuple[str, go.Figure]], output_path: Path, title: str) -> None:
    html_parts = []
    include_js = True

    for section_title, fig in figures:
        fig.update_layout(title=section_title)
        section_html = pio.to_html(fig, include_plotlyjs=include_js, full_html=False)
        include_js = False
        html_parts.append(f"<section><h2>{section_title}</h2>{section_html}</section>")

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\">
    <title>{title}</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; color: #111; }}
      h1 {{ margin-bottom: 8px; }}
      h2 {{ margin-top: 32px; }}
      section {{ background: #fff; padding: 16px; border: 1px solid #ddd; border-radius: 8px; }}
    </style>
  </head>
  <body>
    <h1>{title}</h1>
    {''.join(html_parts)}
  </body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def create_visualizations(results: ExperimentResults, output_dir: Path):
    """Generate analysis visualizations (Plotly HTML dashboard)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = results.get_metrics_by_config()
    if metrics_df.is_empty():
        print("No metrics to visualize")
        return

    models = metrics_df["model"].unique().to_list()
    temps = sorted(metrics_df["temperature"].unique().to_list())

    # 1. F2 Score by Model and Temperature
    fig_f2 = go.Figure()
    for model in models:
        model_data = metrics_df.filter(pl.col("model") == model).sort("temperature")
        fig_f2.add_trace(
            go.Scatter(
                x=model_data["temperature"].to_list(),
                y=model_data["f2_score"].to_list(),
                mode="lines+markers",
                name=model,
            )
        )
    fig_f2.update_layout(
        xaxis_title="Temperature",
        yaxis_title="F2 Score",
        yaxis=dict(range=[0, 1]),
    )

    # 2. Recall vs Precision Trade-off
    fig_rp = make_subplots(
        rows=1,
        cols=len(temps),
        shared_yaxes=True,
        subplot_titles=[f"T={temp}" for temp in temps],
    )
    for col_index, temp in enumerate(temps, start=1):
        temp_data = metrics_df.filter(pl.col("temperature") == temp)
        for row in temp_data.to_dicts():
            fig_rp.add_trace(
                go.Scatter(
                    x=[row["precision"]],
                    y=[row["recall"]],
                    mode="markers+text",
                    text=[row["model"]],
                    textposition="top center",
                    marker=dict(size=12),
                    name=row["model"],
                    showlegend=(col_index == 1),
                ),
                row=1,
                col=col_index,
            )
        fig_rp.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=0.8,
            y1=0.8,
            line=dict(color="red", dash="dash"),
            row=1,
            col=col_index,
        )

    fig_rp.update_xaxes(title_text="Precision", range=[0, 1])
    fig_rp.update_yaxes(title_text="Recall", range=[0, 1], row=1, col=1)

    # 3. Calibration Metrics (ECE and Brier)
    fig_cal = make_subplots(rows=1, cols=2, subplot_titles=["ECE", "Brier Score"])
    for model in models:
        model_data = metrics_df.filter(pl.col("model") == model).sort("temperature")
        fig_cal.add_trace(
            go.Scatter(
                x=model_data["temperature"].to_list(),
                y=model_data["ece"].to_list(),
                mode="lines+markers",
                name=model,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig_cal.add_trace(
            go.Scatter(
                x=model_data["temperature"].to_list(),
                y=model_data["brier_score"].to_list(),
                mode="lines+markers",
                name=model,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig_cal.update_xaxes(title_text="Temperature", row=1, col=1)
    fig_cal.update_xaxes(title_text="Temperature", row=1, col=2)
    fig_cal.update_yaxes(title_text="ECE (lower is better)", row=1, col=1)
    fig_cal.update_yaxes(title_text="Brier (lower is better)", row=1, col=2)

    # 4. Confusion Matrix Heatmaps
    fig_cm = make_subplots(
        rows=len(models),
        cols=len(temps),
        subplot_titles=[f"{model} T={temp}" for model in models for temp in temps],
    )
    for i, model in enumerate(models, start=1):
        for j, temp in enumerate(temps, start=1):
            row_data = metrics_df.filter((pl.col("model") == model) & (pl.col("temperature") == temp))
            if row_data.is_empty():
                continue
            row = row_data.to_dicts()[0]
            z = [[row["tp"], row["fn"]], [row["fp"], row["tn"]]]
            fig_cm.add_trace(
                go.Heatmap(
                    z=z,
                    x=["Yes", "No"],
                    y=["Yes", "No"],
                    colorscale="Blues",
                    showscale=False,
                    text=z,
                    texttemplate="%{text}",
                ),
                row=i,
                col=j,
            )
            if i == len(models):
                fig_cm.update_xaxes(title_text="Predicted", row=i, col=j)
            if j == 1:
                fig_cm.update_yaxes(title_text="Actual", row=i, col=j)

    # 5. F2 Heatmap
    f2_rows = []
    for model in models:
        row = []
        for temp in temps:
            cell = metrics_df.filter((pl.col("model") == model) & (pl.col("temperature") == temp))
            if cell.is_empty():
                row.append(None)
            else:
                row.append(float(cell["f2_score"][0]))
        f2_rows.append(row)

    fig_f2_heat = go.Figure(
        data=[
            go.Heatmap(
                z=f2_rows,
                x=temps,
                y=models,
                colorscale="RdYlGn",
                zmin=0,
                zmax=1,
                text=f2_rows,
                texttemplate="%{text:.3f}",
            )
        ]
    )
    fig_f2_heat.update_xaxes(title_text="Temperature")
    fig_f2_heat.update_yaxes(title_text="Model")

    # 6. Results table
    table_df = metrics_df.sort(["model", "temperature"]).to_dicts()
    headers = ["Model", "Temp", "Acc", "Recall", "Prec", "F2", "ECE", "Brier", "FN"]
    cells = [
        [row["model"] for row in table_df],
        [f"{row['temperature']:.1f}" for row in table_df],
        [f"{row['accuracy']:.2%}" for row in table_df],
        [f"{row['recall']:.2%}" for row in table_df],
        [f"{row['precision']:.2%}" for row in table_df],
        [f"{row['f2_score']:.3f}" for row in table_df],
        [f"{row['ece']:.3f}" for row in table_df],
        [f"{row['brier_score']:.3f}" for row in table_df],
        [str(int(row["fn"])) for row in table_df],
    ]
    fig_table = go.Figure(data=[go.Table(header=dict(values=headers), cells=dict(values=cells))])

    figures = [
        ("F2 Score by Temperature", fig_f2),
        ("Recall vs Precision", fig_rp),
        ("Calibration Metrics", fig_cal),
        ("Confusion Matrices", fig_cm),
        ("F2 Score Heatmap", fig_f2_heat),
        ("Results Table", fig_table),
    ]

    dashboard_path = output_dir / "temperature_dashboard.html"
    _write_dashboard(figures, dashboard_path, title="Temperature Study Dashboard")

    print(f"Dashboard saved to {dashboard_path}")


def create_reasoning_visualizations(results: ExperimentResults, output_dir: Path):
    """Generate reasoning analysis visualizations for models with separate internal reasoning.

    Creates visualizations for:
    1. Alignment score distribution (histogram)
    2. Reasoning length vs confidence
    3. Model comparison (reasoning vs non-reasoning)

    Only generates if there are predictions with separate reasoning (e.g., DeepSeek Reasoner).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get predictions with separate reasoning
    with_reasoning = [
        p for p in results.predictions
        if p.has_separate_reasoning and p.internal_reasoning
    ]

    if not with_reasoning:
        print("No predictions with separate reasoning. Skipping reasoning visualizations.")
        return

    print(f"\nGenerating reasoning analysis visualizations for {len(with_reasoning)} predictions...")

    figures = []

    # 1. Alignment Score Distribution (Histogram)
    alignment_scores = [
        p.reasoning_alignment_score
        for p in with_reasoning
        if p.reasoning_alignment_score is not None
    ]

    if alignment_scores:
        fig_alignment = go.Figure()
        fig_alignment.add_trace(
            go.Histogram(
                x=alignment_scores,
                nbinsx=20,
                name="Alignment Scores",
                marker_color="rgb(55, 83, 109)",
            )
        )
        fig_alignment.add_vline(
            x=np.mean(alignment_scores),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {np.mean(alignment_scores):.3f}",
            annotation_position="top right",
        )
        fig_alignment.update_layout(
            xaxis_title="Cosine Similarity (Internal vs Stated Reasoning)",
            yaxis_title="Count",
            showlegend=False,
        )
        figures.append(("Reasoning Alignment Distribution", fig_alignment))

    # 2. Reasoning Tokens vs Confidence
    reasoning_tokens = [p.reasoning_tokens for p in with_reasoning]
    confidences = [p.confidence for p in with_reasoning]
    correct = [
        (p.ground_truth == "Yes") == p.prediction
        for p in with_reasoning
    ]

    if reasoning_tokens and confidences:
        fig_tokens = go.Figure()
        fig_tokens.add_trace(
            go.Scatter(
                x=reasoning_tokens,
                y=confidences,
                mode="markers",
                marker=dict(
                    size=10,
                    color=[1 if c else 0 for c in correct],
                    colorscale=[[0, "red"], [1, "green"]],
                    showscale=False,
                ),
                text=[f"{'Correct' if c else 'Wrong'}" for c in correct],
                name="Predictions",
            )
        )
        fig_tokens.update_layout(
            xaxis_title="Reasoning Tokens",
            yaxis_title="Confidence (%)",
            yaxis=dict(range=[0, 100]),
        )
        figures.append(("Reasoning Length vs Confidence", fig_tokens))

    # 3. Model Comparison: Reasoning vs Non-Reasoning Models
    df = results.to_dataframe()
    if not df.is_empty():
        # Group by model and compute average metrics
        model_summary = (
            df.filter(pl.col("error").is_null())
            .group_by("model")
            .agg([
                pl.col("confidence").mean().alias("avg_confidence"),
                pl.col("correct").mean().alias("accuracy"),
                pl.col("has_separate_reasoning").first().alias("is_reasoning_model"),
                pl.col("reasoning_tokens").mean().alias("avg_reasoning_tokens"),
            ])
            .sort("model")
        )

        if model_summary.height > 1:
            fig_comparison = go.Figure()

            models = model_summary["model"].to_list()
            accuracies = model_summary["accuracy"].to_list()
            is_reasoning = model_summary["is_reasoning_model"].to_list()

            colors = ["rgb(26, 118, 255)" if r else "rgb(55, 83, 109)" for r in is_reasoning]
            patterns = ["/" if r else "" for r in is_reasoning]

            fig_comparison.add_trace(
                go.Bar(
                    x=models,
                    y=accuracies,
                    marker_color=colors,
                    text=[f"{acc:.1%}" for acc in accuracies],
                    textposition="outside",
                    name="Accuracy",
                )
            )
            fig_comparison.update_layout(
                xaxis_title="Model",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1]),
                showlegend=False,
                annotations=[
                    dict(
                        x=0.5,
                        y=1.05,
                        xref="paper",
                        yref="paper",
                        text="Blue = Reasoning Model | Gray = Standard Model",
                        showarrow=False,
                    )
                ],
            )
            figures.append(("Model Accuracy Comparison", fig_comparison))

    # 4. Calibration by Model Type
    with_reasoning_data = [
        {"confidence": p.confidence / 100, "correct": (p.ground_truth == "Yes") == p.prediction}
        for p in results.predictions
        if p.has_separate_reasoning
    ]
    without_reasoning_data = [
        {"confidence": p.confidence / 100, "correct": (p.ground_truth == "Yes") == p.prediction}
        for p in results.predictions
        if not p.has_separate_reasoning and p.error is None
    ]

    if with_reasoning_data and without_reasoning_data:
        fig_calibration = make_subplots(rows=1, cols=2, subplot_titles=["Reasoning Models", "Standard Models"])

        # Reasoning models
        fig_calibration.add_trace(
            go.Scatter(
                x=[d["confidence"] for d in with_reasoning_data],
                y=[1 if d["correct"] else 0 for d in with_reasoning_data],
                mode="markers",
                marker=dict(size=8, opacity=0.6),
                name="Reasoning Models",
            ),
            row=1,
            col=1,
        )

        # Standard models
        fig_calibration.add_trace(
            go.Scatter(
                x=[d["confidence"] for d in without_reasoning_data],
                y=[1 if d["correct"] else 0 for d in without_reasoning_data],
                mode="markers",
                marker=dict(size=8, opacity=0.6),
                name="Standard Models",
            ),
            row=1,
            col=2,
        )

        # Add diagonal line for perfect calibration
        for col in [1, 2]:
            fig_calibration.add_shape(
                type="line",
                x0=0, x1=1, y0=0, y1=1,
                line=dict(color="gray", dash="dash"),
                row=1, col=col,
            )

        fig_calibration.update_xaxes(title_text="Confidence", range=[0, 1])
        fig_calibration.update_yaxes(title_text="Correct (0/1)")
        figures.append(("Calibration by Model Type", fig_calibration))

    # Write dashboard
    if figures:
        dashboard_path = output_dir / "reasoning_analysis_dashboard.html"
        _write_dashboard(figures, dashboard_path, title="Reasoning Analysis Dashboard")
        print(f"Reasoning analysis dashboard saved to {dashboard_path}")
    else:
        print("No figures generated for reasoning analysis.")


# =============================================================================
# Reporting
# =============================================================================


def print_summary(results: ExperimentResults):
    """Print a formatted summary for the temperature sanity check."""
    metrics_df = results.get_metrics_by_config()

    print("\n" + "=" * 80)
    print("TEMPERATURE SANITY CHECK RESULTS")
    print("=" * 80)
    print("\nQuestion: Is T=0.0 safe to use, or does any model fail catastrophically?")
    print(f"Catastrophic threshold: >{CATASTROPHIC_THRESHOLD:.0%} absolute F2 difference")

    df = results.to_dataframe()
    total = df.height
    successful = df.filter(pl.col("error").is_null()).height if not df.is_empty() else 0
    failed = df.filter(pl.col("error").is_not_null()).height if not df.is_empty() else 0

    print(f"\nTotal predictions: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Show error details by model if any failed
    if failed > 0:
        print("\n⚠️  ERRORS BY MODEL:")
        error_df = df.filter(pl.col("error").is_not_null())
        error_summary = (
            error_df.group_by("model")
            .agg([
                pl.len().alias("error_count"),
                pl.col("error").first().alias("sample_error"),
            ])
            .sort("model")
        )
        for row in error_summary.iter_rows(named=True):
            print(f"  {row['model']}: {row['error_count']} errors")
            print(f"    Example: {row['sample_error'][:80]}...")

    # === CHECK FOR MODELS WITH 100% FAILURE RATE ===
    # These models disappear from metrics_df since they have no successful predictions
    if results.intended_models:
        successful_df = df.filter(pl.col("error").is_null()) if not df.is_empty() else df
        if not successful_df.is_empty():
            models_with_predictions = set(successful_df["model"].unique().to_list())
        else:
            models_with_predictions = set()

        failed_models = []
        for intended_model in results.intended_models:
            if intended_model not in models_with_predictions:
                # Model was intended but has 0 successful predictions
                model_df = df.filter(pl.col("model") == intended_model) if not df.is_empty() else df
                total_attempts = model_df.height if not model_df.is_empty() else 0
                failed_models.append((intended_model, total_attempts))

        if failed_models:
            print("\n🚫 MODELS WITH 100% FAILURE RATE:")
            for model_name, n_attempts in failed_models:
                print(f"  {model_name}: {n_attempts} attempts, 0 successful predictions")
                print(f"    ⚠️  This model is EXCLUDED from all statistical comparisons!")

    if metrics_df.is_empty():
        print("\n❌ No successful predictions to analyze. Check API keys.")
        return

    # === MAIN COMPARISON TABLE WITH STATISTICS ===
    print("\n" + "-" * 90)
    print("STATISTICAL COMPARISON (T=0.0 vs T=0.7)")
    print("-" * 90)
    print(f"{'Model':<15} {'F2(T=0)':>8} {'F2(T=0.7)':>10} {'Δ':>7} {'Cohen h':>8} {'p-value':>8} {'Status':>12}")
    print("-" * 90)

    models = metrics_df["model"].unique().to_list()
    catastrophic_failures = []
    missing_data_models = []  # Models with incomplete temperature coverage
    insufficient_samples_models = []  # Models with too few valid samples (rate limit failures)
    min_required_samples = int(MIN_VALID_SAMPLE_RATIO * N_SAMPLES)

    for model in sorted(models):
        model_data = metrics_df.filter(pl.col("model") == model)

        # Get F2 for each temperature
        t0_data = model_data.filter(pl.col("temperature") == 0.0)
        t7_data = model_data.filter(pl.col("temperature") == 0.7)

        f2_t0 = t0_data["f2_score"][0] if not t0_data.is_empty() else None
        f2_t7 = t7_data["f2_score"][0] if not t7_data.is_empty() else None

        # Check sample counts - detect rate limit failures
        n_t0 = t0_data["n_samples_total"][0] if not t0_data.is_empty() and "n_samples_total" in t0_data.columns else 0
        n_t7 = t7_data["n_samples_total"][0] if not t7_data.is_empty() and "n_samples_total" in t7_data.columns else 0
        has_sufficient_samples = n_t0 >= min_required_samples and n_t7 >= min_required_samples

        if f2_t0 is not None and f2_t7 is not None:
            delta = f2_t7 - f2_t0
            abs_delta = abs(delta)

            # Calculate Cohen's h (effect size for proportions)
            h = cohens_h(f2_t7, f2_t0)

            # Run McNemar's test
            # CRITICAL: Join on sample_id to ensure perfect alignment
            # Without this, different samples could succeed at different temperatures
            model_df = df.filter((pl.col("model") == model) & (pl.col("error").is_null()))
            preds_t0 = model_df.filter((pl.col("temperature") == 0.0) & (pl.col("run_id") == 1)).select(
                pl.col("sample_id"),
                pl.col("prediction").alias("pred_t0"),
                pl.col("ground_truth"),
            )
            preds_t7 = model_df.filter((pl.col("temperature") == 0.7) & (pl.col("run_id") == 1)).select(
                pl.col("sample_id"),
                pl.col("prediction").alias("pred_t7"),
            )

            # Inner join keeps only samples that succeeded at BOTH temperatures
            paired = preds_t0.join(preds_t7, on="sample_id", how="inner").sort("sample_id")

            p_value = None
            if not paired.is_empty():
                try:
                    pred_t0_list = [p == "Yes" for p in paired["pred_t0"].to_list()]
                    pred_t7_list = [p == "Yes" for p in paired["pred_t7"].to_list()]
                    truth_list = [g == "Yes" for g in paired["ground_truth"].to_list()]
                    mcnemar_result = mcnemar_test(pred_t0_list, pred_t7_list, truth_list)
                    p_value = mcnemar_result.p_value
                except Exception:
                    p_value = None

            # Determine status - check sample count first
            if not has_sufficient_samples:
                status = "⚠️  LOW N"
                insufficient_samples_models.append(f"{model} (T=0:{n_t0}, T=0.7:{n_t7})")
            elif abs_delta > CATASTROPHIC_THRESHOLD:
                status = "⚠️  INVESTIGATE"
                catastrophic_failures.append(model)
            else:
                status = "✓ OK"

            p_str = f"{p_value:.4f}" if p_value is not None else "N/A"
            print(f"{model:<15} {f2_t0:>8.3f} {f2_t7:>10.3f} {delta:>+7.3f} {h:>+8.3f} {p_str:>8} {status:>12}")
        else:
            missing_data_models.append(model)
            print(f"{model:<15} {'N/A':>8} {'N/A':>10} {'N/A':>7} {'N/A':>8} {'N/A':>8} {'MISSING':>12}")

    print("-" * 90)
    print("\nEffect size interpretation (Cohen's h): |h|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")
    print(f"Power note: With N={N_SAMPLES}, only effects |h|>0.6 (~28% difference) detectable at 80% power")

    # === CONCLUSION ===
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if missing_data_models:
        print("\n⚠️  INCOMPLETE DATA - CANNOT DRAW CONCLUSIONS")
        print(f"  Models with missing temperature data: {', '.join(missing_data_models)}")
        print("  → Recommendation: Check API keys and re-run experiment")
    elif insufficient_samples_models:
        print("\n⚠️  INSUFFICIENT SAMPLES - RESULTS UNRELIABLE")
        print(f"  Required: {min_required_samples}/{N_SAMPLES} samples ({MIN_VALID_SAMPLE_RATIO:.0%})")
        print(f"  Models with low sample counts:")
        for model_info in insufficient_samples_models:
            print(f"    - {model_info}")
        print("  → Recommendation: Check rate limits, add retry logic, and re-run")
    elif catastrophic_failures:
        print("\n⚠️  SANITY CHECK FAILED - INVESTIGATE FURTHER")
        print(f"  Models with >20% F2 difference: {', '.join(catastrophic_failures)}")
        print("  → Recommendation: Run larger study on flagged models")
    else:
        print("\n✓ SANITY CHECK PASSED")
        print("  No model shows catastrophic failure at T=0.0")
        print("  → Recommendation: Use T=0.0 (deterministic) for all models")

    # === DETAILED METRICS (for reference) ===
    print("\n" + "-" * 95)
    print("DETAILED METRICS (All Configurations)")
    print("-" * 95)
    print(
        f"{'Model':<15} {'Temp':>5} {'F2':>6} {'Acc':>6} {'Recall':>7} {'Prec':>6} "
        f"{'FN':>4} {'ECE':>6} {'Latency':>10}"
    )
    print("-" * 95)

    for row in metrics_df.sort(["model", "temperature"]).to_dicts():
        latency_str = f"{row['avg_latency_ms']:.0f}ms" if row.get('avg_latency_ms') else "N/A"
        print(
            f"{row['model']:<15} {row['temperature']:>5.1f} "
            f"{row['f2_score']:>6.3f} {row['accuracy']:>6.1%} "
            f"{row['recall']:>7.1%} {row['precision']:>6.1%} "
            f"{int(row['fn']):>4} {row['ece']:>6.3f} {latency_str:>10}"
        )

    print("-" * 95)


def save_results(results: ExperimentResults, output_dir: Path):
    """Save results to files (both parquet and CSV for all data).

    Output files:
    - predictions_{timestamp}.parquet: Full data with complete text/reasoning
    - predictions_{timestamp}.csv: Full data with truncated text (all columns)
    - predictions_compact_{timestamp}.csv: Compact version without long text fields
    - metrics_{timestamp}.parquet: Metrics by model/temperature
    - metrics_{timestamp}.csv: Same as parquet in CSV format
    - reasoning_details_{timestamp}.parquet: Reasoning-specific data (if available)
    - reasoning_details_{timestamp}.csv: Same as parquet in CSV format
    - summary_{timestamp}.json: Experiment summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==========================================================================
    # 1. Predictions - Full data (Parquet)
    # ==========================================================================
    df_full = results.to_dataframe(truncate_text=False)
    df_full.write_parquet(output_dir / f"predictions_{timestamp}.parquet")

    # ==========================================================================
    # 2. Predictions - Full data with truncated text (CSV)
    # ==========================================================================
    df_display = results.to_dataframe(truncate_text=True)
    df_display.write_csv(output_dir / f"predictions_{timestamp}.csv")

    # ==========================================================================
    # 3. Predictions - Compact version without long fields (CSV)
    # ==========================================================================
    # Drop columns that are too long for quick inspection
    compact_exclude = [
        "text", "internal_reasoning", "checkability_reasoning",
        "verifiability_reasoning", "harm_reasoning", "final_answer_logprobs",
        "internal_reasoning_logprobs", "error"
    ]
    compact_cols = [c for c in df_display.columns if c not in compact_exclude]
    df_display.select(compact_cols).write_csv(output_dir / f"predictions_compact_{timestamp}.csv")

    # ==========================================================================
    # 4. Metrics by config (both formats)
    # ==========================================================================
    metrics_df = results.get_metrics_by_config()
    if not metrics_df.is_empty():
        metrics_df.write_csv(output_dir / f"metrics_{timestamp}.csv")
        metrics_df.write_parquet(output_dir / f"metrics_{timestamp}.parquet")

    # ==========================================================================
    # 5. Reasoning details - for predictions with internal reasoning
    # ==========================================================================
    # Filter to predictions with separate reasoning (DeepSeek Reasoner)
    reasoning_preds = [p for p in results.predictions if p.has_separate_reasoning]
    if reasoning_preds:
        reasoning_data = []
        for p in reasoning_preds:
            reasoning_data.append({
                "sample_id": p.sample_id,
                "model": p.model,
                "temperature": p.temperature,
                "prediction": p.prediction,
                "confidence": p.confidence,
                "ground_truth": p.ground_truth,
                "correct": p.prediction == (p.ground_truth == "Yes"),  # Both are now bool
                "internal_reasoning": p.internal_reasoning[:500] + "..." if p.internal_reasoning and len(p.internal_reasoning) > 500 else p.internal_reasoning,
                "internal_reasoning_full_length": len(p.internal_reasoning) if p.internal_reasoning else 0,
                "reasoning_tokens": p.reasoning_tokens,
                "final_answer_tokens": p.final_answer_tokens,
                "reasoning_alignment_score": p.reasoning_alignment_score,
                "checkability_confidence": p.checkability_confidence,
                "verifiability_confidence": p.verifiability_confidence,
                "harm_confidence": p.harm_confidence,
            })

        reasoning_df = pl.DataFrame(reasoning_data)
        reasoning_df.write_csv(output_dir / f"reasoning_details_{timestamp}.csv")
        reasoning_df.write_parquet(output_dir / f"reasoning_details_{timestamp}.parquet")

        # Also save full reasoning text separately (parquet only - too large for CSV)
        full_reasoning_data = [
            {
                "sample_id": p.sample_id,
                "model": p.model,
                "temperature": p.temperature,
                "internal_reasoning": p.internal_reasoning,
                "checkability_reasoning": p.checkability_reasoning,
                "verifiability_reasoning": p.verifiability_reasoning,
                "harm_reasoning": p.harm_reasoning,
            }
            for p in reasoning_preds
        ]
        pl.DataFrame(full_reasoning_data).write_parquet(
            output_dir / f"reasoning_full_text_{timestamp}.parquet"
        )

    # ==========================================================================
    # 6. Per-model reasoning statistics
    # ==========================================================================
    if reasoning_preds:
        model_reasoning_stats = []
        for model in set(p.model for p in reasoning_preds):
            model_preds = [p for p in reasoning_preds if p.model == model]
            alignment_scores = [p.reasoning_alignment_score for p in model_preds if p.reasoning_alignment_score is not None]
            model_reasoning_stats.append({
                "model": model,
                "n_predictions": len(model_preds),
                "n_with_alignment": len(alignment_scores),
                "alignment_mean": np.mean(alignment_scores) if alignment_scores else None,
                "alignment_std": np.std(alignment_scores) if alignment_scores else None,
                "alignment_min": min(alignment_scores) if alignment_scores else None,
                "alignment_max": max(alignment_scores) if alignment_scores else None,
                "avg_reasoning_tokens": np.mean([p.reasoning_tokens for p in model_preds]),
                "avg_final_tokens": np.mean([p.final_answer_tokens for p in model_preds]),
            })

        stats_df = pl.DataFrame(model_reasoning_stats)
        stats_df.write_csv(output_dir / f"reasoning_stats_{timestamp}.csv")
        stats_df.write_parquet(output_dir / f"reasoning_stats_{timestamp}.parquet")

    # ==========================================================================
    # 7. Drift probe results
    # ==========================================================================
    drift_probes_saved = False
    if results.drift_probes:
        probe_data = []
        for probe in results.drift_probes:
            probe_data.append({
                "model": probe.model,
                "pre_samples": probe.pre_samples,
                "post_samples": probe.post_samples,
                "statistic_S": probe.statistic_S,
                "p_value": probe.p_value,
                "drift_detected": probe.drift_detected,
                "skipped": probe.skipped,
                "skip_reason": probe.skip_reason,
                "vocabulary_size": probe.vocabulary_size,
                "pre_timestamp": probe.pre_timestamp,
                "post_timestamp": probe.post_timestamp,
            })

        probe_df = pl.DataFrame(probe_data)
        probe_df.write_csv(output_dir / f"drift_probes_{timestamp}.csv")
        probe_df.write_parquet(output_dir / f"drift_probes_{timestamp}.parquet")
        drift_probes_saved = True

    # ==========================================================================
    # 8. Build comprehensive summary
    # ==========================================================================
    summary = {
        "timestamp": timestamp,
        "experiment": {
            "name": "temperature_optimization",
            "version": "1.0",
            "prompt_type": "zero-shot",
            "prompts_path": str(PROMPTS_PATH),
            "dataset": str(CT24_DEV_PATH),
            "random_seed": RANDOM_SEED,
        },
        "parameters": {
            "temperatures": TEMPERATURES,
            "n_samples": N_SAMPLES,
            "threshold": 50.0,
        },
        "results": {
            "n_predictions": df_full.height,
            "n_successful": df_full.filter(pl.col("error").is_null()).height if not df_full.is_empty() else 0,
            "n_failed": df_full.filter(pl.col("error").is_not_null()).height if not df_full.is_empty() else 0,
            "models_tested": metrics_df["model"].unique().to_list() if not metrics_df.is_empty() else [],
            "duration_seconds": (results.end_time - results.start_time).total_seconds() if results.end_time else None,
        },
        "files": {
            "predictions_parquet": f"predictions_{timestamp}.parquet",
            "predictions_csv": f"predictions_{timestamp}.csv",
            "predictions_compact_csv": f"predictions_compact_{timestamp}.csv",
            "metrics_csv": f"metrics_{timestamp}.csv",
            "metrics_parquet": f"metrics_{timestamp}.parquet",
            "reasoning_details_csv": f"reasoning_details_{timestamp}.csv" if reasoning_preds else None,
            "reasoning_details_parquet": f"reasoning_details_{timestamp}.parquet" if reasoning_preds else None,
            "reasoning_full_text_parquet": f"reasoning_full_text_{timestamp}.parquet" if reasoning_preds else None,
            "reasoning_stats_csv": f"reasoning_stats_{timestamp}.csv" if reasoning_preds else None,
            "reasoning_stats_parquet": f"reasoning_stats_{timestamp}.parquet" if reasoning_preds else None,
            "drift_probes_csv": f"drift_probes_{timestamp}.csv" if drift_probes_saved else None,
            "drift_probes_parquet": f"drift_probes_{timestamp}.parquet" if drift_probes_saved else None,
            "dashboard": "temperature_dashboard.html",
        },
    }

    # Add drift probe summary to results
    if results.drift_probes:
        drift_summary = []
        for probe in results.drift_probes:
            drift_summary.append({
                "model": probe.model,
                "statistic_S": probe.statistic_S,
                "p_value": probe.p_value,
                "drift_detected": probe.drift_detected,
                "skipped": probe.skipped,
            })
        summary["drift_probes"] = drift_summary

    if not metrics_df.is_empty():
        best = metrics_df.sort("f2_score", descending=True).to_dicts()[0]
        summary["best_config"] = {
            "model": best["model"],
            "temperature": float(best["temperature"]),
            "f2_score": float(best["f2_score"]),
            "recall": float(best["recall"]),
            "precision": float(best["precision"]),
            "accuracy": float(best["accuracy"]),
            "ece": float(best["ece"]),
            "brier_score": float(best["brier_score"]),
            "false_negatives": int(best["fn"]),
            "false_positives": int(best["fp"]),
        }

        summary["metrics_by_config"] = metrics_df.to_dicts()

    with open(output_dir / f"summary_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_dir}")

    return output_dir, timestamp


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the temperature sanity check."""
    print("\n" + "=" * 60)
    print("TEMPERATURE SANITY CHECK")
    print("Checkworthiness Pipeline - Zero-Shot")
    print("=" * 60)
    print("\nObjective: Validate that T=0.0 is safe for all models")
    print("Question: Does any model catastrophically fail at T=0.0?")
    print(f"Threshold: >{CATASTROPHIC_THRESHOLD:.0%} F2 difference = investigate further")
    print("\nNote: This is NOT an optimization study.")
    print("      With N=20, we can only detect effects >28%.")

    samples = load_ct24_samples(N_SAMPLES)

    results = run_experiment(samples=samples, verbose=True)

    if results.predictions:
        # Compute reasoning alignment for models with separate internal reasoning
        print("\n" + "=" * 60)
        print("REASONING ANALYSIS")
        print("=" * 60)
        compute_alignment_for_predictions(results.predictions, verbose=True)

        # Get comprehensive reasoning analysis summary
        reasoning_summary = get_reasoning_analysis_summary(results.predictions, verbose=True)

        # Print reasoning analysis summary
        if reasoning_summary["n_with_reasoning"] > 0:
            print(f"\nReasoning Models Found: {reasoning_summary['reasoning_models']}")
            print(f"Predictions with Reasoning: {reasoning_summary['n_with_reasoning']}")
            if reasoning_summary["alignment"]["mean"] is not None:
                print(f"Alignment Score: {reasoning_summary['alignment']['mean']:.3f} ± {reasoning_summary['alignment']['std']:.3f}")
        else:
            print("No reasoning models with separate internal reasoning detected.")

        print_summary(results)

        output_dir = project_root / "experiments" / "results" / "temperature_sanity_check"
        output_dir, _ = save_results(results, output_dir)

        # Create standard visualizations
        create_visualizations(results, output_dir)

        # Create reasoning analysis visualizations (if applicable)
        create_reasoning_visualizations(results, output_dir)

        # Save reasoning analysis summary (JSON, CSV, and Parquet)
        if reasoning_summary["n_with_reasoning"] > 0:
            # JSON - nested structure
            reasoning_json_path = output_dir / "reasoning_analysis.json"
            with open(reasoning_json_path, "w") as f:
                json.dump(reasoning_summary, f, indent=2, default=str)

            # Flatten for CSV/Parquet - one row summary
            flat_summary = {
                "n_total_predictions": reasoning_summary["n_total_predictions"],
                "n_with_reasoning": reasoning_summary["n_with_reasoning"],
                "reasoning_models": ",".join(reasoning_summary["reasoning_models"]),
                # Alignment metrics
                "alignment_n": reasoning_summary["alignment"]["n_with_alignment"],
                "alignment_mean": reasoning_summary["alignment"]["mean"],
                "alignment_std": reasoning_summary["alignment"]["std"],
                "alignment_min": reasoning_summary["alignment"]["min"],
                "alignment_max": reasoning_summary["alignment"]["max"],
                "alignment_median": reasoning_summary["alignment"]["median"],
            }

            # Add calibration metrics if available
            if reasoning_summary.get("calibration"):
                cal = reasoning_summary["calibration"]
                # Only final answer calibration is meaningful (see compute_reasoning_calibration_metrics docstring)
                if cal.get("final_ece") is not None:
                    flat_summary["final_ece"] = cal["final_ece"]
                if cal.get("final_brier") is not None:
                    flat_summary["final_brier"] = cal["final_brier"]
                if cal.get("avg_reasoning_tokens") is not None:
                    flat_summary["avg_reasoning_tokens"] = cal["avg_reasoning_tokens"]
                if cal.get("avg_final_tokens") is not None:
                    flat_summary["avg_final_tokens"] = cal["avg_final_tokens"]

            summary_df = pl.DataFrame([flat_summary])
            summary_df.write_csv(output_dir / "reasoning_analysis_summary.csv")
            summary_df.write_parquet(output_dir / "reasoning_analysis_summary.parquet")

            print(f"Reasoning analysis saved to {output_dir}:")

        print("\n" + "=" * 60)
        print("SANITY CHECK COMPLETE")
        print("=" * 60)
        print(f"\nResults directory: {output_dir}")
        print("\nOutput files:")
        print("  Predictions:")
        print(f"    - predictions_*.parquet (full data)")
        print(f"    - predictions_*.csv (full data, truncated text)")
        print(f"    - predictions_compact_*.csv (compact, no long fields)")
        print("  Metrics:")
        print(f"    - metrics_*.csv / metrics_*.parquet")
        print("  Dashboards:")
        print(f"    - temperature_dashboard.html")
        if reasoning_summary["n_with_reasoning"] > 0:
            print("  Reasoning Analysis:")
            print(f"    - reasoning_details_*.csv / .parquet (per-prediction)")
            print(f"    - reasoning_stats_*.csv / .parquet (per-model)")
            print(f"    - reasoning_full_text_*.parquet (full internal reasoning)")
            print(f"    - reasoning_analysis_summary.csv / .parquet / .json")
            print(f"    - reasoning_analysis_dashboard.html")
        print("  Drift Probes:")
        print(f"    - drift_probes_*.csv / .parquet (model stability check)")

        # Print drift probe summary
        if results.drift_probes:
            print("\n  DRIFT PROBE SUMMARY:")
            any_drift = False
            any_inconclusive = False
            for probe in results.drift_probes:
                if probe.skipped:
                    status = "⏭️  Skipped"
                elif probe.drift_detected is True:
                    status = "⚠️  DRIFT"
                    any_drift = True
                elif probe.drift_detected is None:
                    status = "⚠️  N/A (degenerate)"
                    any_inconclusive = True
                else:
                    status = "✓ Stable"
                s_str = f"S={probe.statistic_S:.4f}" if probe.statistic_S is not None else "S=N/A"
                p_str = f"p={probe.p_value:.3f}" if probe.p_value is not None else "p=N/A"
                print(f"    {probe.model}: {status} ({s_str}, {p_str})")
            if any_drift:
                print("    → Some models showed drift - results may be unreliable ⚠️")
            elif any_inconclusive:
                print("    → Some tests inconclusive (degenerate distribution) ⚠️")
            else:
                print("    → All models stable during experiment ✓")
    else:
        print("\nERROR: No predictions completed. Check API keys and try again.")


if __name__ == "__main__":
    main()
