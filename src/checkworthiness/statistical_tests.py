"""Statistical tests for rigorous experiment evaluation.

Implements:
- McNemar's test for paired binary classifiers
- Bootstrap confidence intervals
- Effect size measures (Cohen's h, Cohen's d)
- Multiple comparison corrections (Holm-Bonferroni)
- Power analysis

Following research discipline principles:
- No single-run reporting
- Confidence intervals > point estimates
- Effect sizes alongside p-values
- Multiple comparison corrections
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats


@dataclass
class StatisticalResult:
    """Result from a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    ci_lower: float
    ci_upper: float
    significant: bool  # After correction if applicable
    interpretation: str


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""

    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_bootstrap: int = 10000

    @property
    def margin_of_error(self) -> float:
        return (self.ci_upper - self.ci_lower) / 2


@dataclass
class ComparisonResult:
    """Result from comparing two configurations."""

    config_a: str
    config_b: str
    metric: str
    value_a: float
    value_b: float
    difference: float  # B - A
    ci_lower: float
    ci_upper: float
    p_value: float
    p_value_corrected: float
    effect_size: float
    significant: bool
    interpretation: str


@dataclass
class MultipleComparisonResult:
    """Results from multiple pairwise comparisons with correction."""

    method: str  # "holm-bonferroni", "bonferroni", "fdr-bh"
    n_comparisons: int
    alpha_original: float
    comparisons: list[ComparisonResult] = field(default_factory=list)

    @property
    def significant_count(self) -> int:
        return sum(1 for c in self.comparisons if c.significant)


# =============================================================================
# McNemar's Test (Paired Binary Classifiers)
# =============================================================================


def mcnemar_test(
    predictions_a: list[bool],
    predictions_b: list[bool],
    ground_truth: list[bool],
    correction: bool = True,
) -> StatisticalResult:
    """McNemar's test for comparing paired binary classifiers.

    This is the correct test for comparing two models on the SAME test set.
    It only considers cases where the two models disagree.

    Args:
        predictions_a: Predictions from model A (True = positive class)
        predictions_b: Predictions from model B (True = positive class)
        ground_truth: True labels
        correction: Apply continuity correction (Edwards' correction)

    Returns:
        StatisticalResult with chi-square statistic and p-value
    """
    n = len(predictions_a)
    if n != len(predictions_b) or n != len(ground_truth):
        raise ValueError("All lists must have same length")

    # Build contingency table based on correctness
    correct_a = [p == g for p, g in zip(predictions_a, ground_truth)]
    correct_b = [p == g for p, g in zip(predictions_b, ground_truth)]

    # Count disagreements
    # b01: A wrong, B correct
    # b10: A correct, B wrong
    b01 = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)
    b10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)

    if b01 + b10 == 0:
        # No disagreements - models are equivalent on this dataset
        return StatisticalResult(
            test_name="McNemar's test",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_size_name="Odds Ratio",
            ci_lower=0.0,
            ci_upper=0.0,
            significant=False,
            interpretation="Models agree on all samples - no difference detectable",
        )

    # McNemar's chi-square statistic
    if correction:
        # Edwards' continuity correction
        statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    else:
        statistic = (b01 - b10) ** 2 / (b01 + b10)

    # p-value from chi-square distribution with 1 df
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    # Effect size: odds ratio
    if b10 == 0:
        odds_ratio = float("inf") if b01 > 0 else 1.0
    else:
        odds_ratio = b01 / b10

    # Interpretation
    if p_value < 0.05:
        if b01 > b10:
            interpretation = f"Model B significantly better (B correct {b01}x when A wrong, A correct {b10}x when B wrong)"
        else:
            interpretation = f"Model A significantly better (A correct {b10}x when B wrong, B correct {b01}x when A wrong)"
    else:
        interpretation = f"No significant difference (disagreements: A→B: {b01}, B→A: {b10})"

    return StatisticalResult(
        test_name="McNemar's test (Edwards corrected)" if correction else "McNemar's test",
        statistic=statistic,
        p_value=p_value,
        effect_size=odds_ratio,
        effect_size_name="Odds Ratio",
        ci_lower=0.0,  # Would need exact CI calculation
        ci_upper=0.0,
        significant=p_value < 0.05,
        interpretation=interpretation,
    )


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================


def bootstrap_ci(
    data: list[float] | np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int | None = None,
) -> BootstrapCI:
    """Calculate bootstrap confidence interval for any statistic.

    Uses BCa (bias-corrected and accelerated) bootstrap for better coverage.

    Args:
        data: Sample data
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default: 0.95)
        random_state: Random seed for reproducibility

    Returns:
        BootstrapCI with estimate and bounds
    """
    rng = np.random.RandomState(random_state)
    data = np.asarray(data)
    n = len(data)

    # Original estimate
    estimate = float(statistic_fn(data))

    # Bootstrap resamples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = data[rng.randint(0, n, size=n)]
        bootstrap_stats.append(statistic_fn(resample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method (simple but effective for most cases)
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


def bootstrap_proportion_ci(
    successes: int,
    total: int,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int | None = None,
) -> BootstrapCI:
    """Bootstrap CI for a proportion (e.g., accuracy, recall).

    Better than normal approximation for small samples or extreme proportions.

    Args:
        successes: Number of successes
        total: Total number of trials
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        random_state: Random seed

    Returns:
        BootstrapCI with proportion estimate and bounds
    """
    rng = np.random.RandomState(random_state)

    # Original estimate
    p = successes / total if total > 0 else 0.0

    # Bootstrap: resample from Bernoulli(p)
    bootstrap_props = []
    for _ in range(n_bootstrap):
        resample_successes = rng.binomial(total, p)
        bootstrap_props.append(resample_successes / total)

    bootstrap_props = np.array(bootstrap_props)

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_props, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_props, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        estimate=p,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


def bootstrap_f2_ci(
    predictions: list[dict],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int | None = None,
    prediction_key: str = "prediction",
    ground_truth_key: str = "ground_truth",
    positive_label: str = "Yes",
) -> BootstrapCI:
    """Bootstrap CI for F2 score (compound metric requiring special handling).

    F2 = (5 × P × R) / (4P + R) where P=precision, R=recall.
    Unlike simple proportions, F2 depends on the joint distribution of TP/FP/FN,
    so we use bootstrap resampling over individual predictions.

    Args:
        predictions: List of dicts with prediction and ground_truth keys
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default: 0.95)
        random_state: Random seed for reproducibility
        prediction_key: Key for prediction in each dict
        ground_truth_key: Key for ground truth in each dict
        positive_label: Value representing positive class

    Returns:
        BootstrapCI with F2 estimate and bounds
    """
    rng = np.random.RandomState(random_state)
    n = len(predictions)

    if n == 0:
        return BootstrapCI(estimate=0.0, ci_lower=0.0, ci_upper=0.0, ci_level=ci_level, n_bootstrap=n_bootstrap)

    def compute_f2(preds: list[dict]) -> float:
        """Compute F2 score from prediction list."""
        tp = fn = fp = 0
        for p in preds:
            is_positive = p[ground_truth_key] == positive_label
            pred_positive = p[prediction_key]
            if isinstance(pred_positive, str):
                pred_positive = pred_positive == positive_label

            if is_positive and pred_positive:
                tp += 1
            elif is_positive and not pred_positive:
                fn += 1
            elif not is_positive and pred_positive:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (4 * precision + recall) == 0:
            return 0.0
        return (5 * precision * recall) / (4 * precision + recall)

    # Original estimate
    estimate = compute_f2(predictions)

    # Bootstrap resamples
    bootstrap_f2s = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        resample = [predictions[i] for i in indices]
        bootstrap_f2s.append(compute_f2(resample))

    bootstrap_f2s = np.array(bootstrap_f2s)

    # Percentile CI
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_f2s, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_f2s, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


def wilson_score_ci(
    successes: int,
    total: int,
    ci_level: float = 0.95,
) -> BootstrapCI:
    """Wilson score interval for proportions.

    Preferred over normal approximation for small samples.
    Handles edge cases (p=0 or p=1) correctly.

    Args:
        successes: Number of successes
        total: Total trials
        ci_level: Confidence level

    Returns:
        BootstrapCI-like object (not actually bootstrapped)
    """
    if total == 0:
        return BootstrapCI(estimate=0.0, ci_lower=0.0, ci_upper=0.0, ci_level=ci_level, n_bootstrap=0)

    p = successes / total
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return BootstrapCI(
        estimate=p,
        ci_lower=max(0.0, center - margin),
        ci_upper=min(1.0, center + margin),
        ci_level=ci_level,
        n_bootstrap=0,  # Not bootstrap, but same structure
    )


# =============================================================================
# Effect Size Measures
# =============================================================================


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for comparing two proportions.

    Effect size interpretation:
    - |h| < 0.2: negligible
    - 0.2 ≤ |h| < 0.5: small
    - 0.5 ≤ |h| < 0.8: medium
    - |h| ≥ 0.8: large

    Args:
        p1: First proportion (0-1)
        p2: Second proportion (0-1)

    Returns:
        Cohen's h (can be negative)
    """

    def arcsin_transform(p: float) -> float:
        # Handle edge cases
        p = max(0.0, min(1.0, p))
        return 2 * np.arcsin(np.sqrt(p))

    return arcsin_transform(p1) - arcsin_transform(p2)


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Cohen's d for comparing two continuous distributions.

    Effect size interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large

    Uses pooled standard deviation.

    Args:
        group1: First group's values
        group2: Second group's values

    Returns:
        Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def interpret_effect_size(effect_size: float, metric: str = "cohens_d") -> str:
    """Interpret effect size magnitude.

    Args:
        effect_size: The effect size value
        metric: "cohens_d", "cohens_h", or "r"

    Returns:
        Interpretation string
    """
    abs_es = abs(effect_size)

    if metric in ("cohens_d", "cohens_h"):
        if abs_es < 0.2:
            return "negligible"
        elif abs_es < 0.5:
            return "small"
        elif abs_es < 0.8:
            return "medium"
        else:
            return "large"
    elif metric == "r":
        if abs_es < 0.1:
            return "negligible"
        elif abs_es < 0.3:
            return "small"
        elif abs_es < 0.5:
            return "medium"
        else:
            return "large"
    else:
        return "unknown"


# =============================================================================
# Multiple Comparison Corrections
# =============================================================================


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Holm-Bonferroni step-down procedure for multiple comparisons.

    Less conservative than Bonferroni while still controlling family-wise error rate (FWER).

    Args:
        p_values: List of p-values from multiple tests
        alpha: Desired family-wise error rate

    Returns:
        List of (corrected_alpha, is_significant) tuples in original order
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value, keeping track of original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [None] * n
    rejected_so_far = True

    for rank, (original_idx, pval) in enumerate(indexed):
        # Holm's adjusted alpha for this rank
        adjusted_alpha = alpha / (n - rank)

        # Once we fail to reject, all subsequent (larger p-values) are also not rejected
        if rejected_so_far and pval <= adjusted_alpha:
            results[original_idx] = (adjusted_alpha, True)
        else:
            rejected_so_far = False
            results[original_idx] = (adjusted_alpha, False)

    return results


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Simple Bonferroni correction (more conservative than Holm).

    Args:
        p_values: List of p-values
        alpha: Desired FWER

    Returns:
        List of (corrected_alpha, is_significant) tuples
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    return [(corrected_alpha, p <= corrected_alpha) for p in p_values]


def fdr_bh_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Benjamini-Hochberg False Discovery Rate correction.

    Less conservative than FWER methods. Controls expected proportion of false positives
    among rejected hypotheses.

    Args:
        p_values: List of p-values
        alpha: Desired FDR

    Returns:
        List of (adjusted_p_value, is_significant) tuples in original order
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [None] * n

    # Find the largest k where p(k) <= k/n * alpha
    max_significant_rank = 0
    for rank, (original_idx, pval) in enumerate(indexed, start=1):
        threshold = (rank / n) * alpha
        if pval <= threshold:
            max_significant_rank = rank

    # All tests with rank <= max_significant_rank are significant
    for rank, (original_idx, pval) in enumerate(indexed, start=1):
        threshold = (rank / n) * alpha
        results[original_idx] = (threshold, rank <= max_significant_rank)

    return results


# =============================================================================
# Power Analysis
# =============================================================================


def power_analysis_proportions(
    p1: float,
    p2: float,
    n: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> float:
    """Calculate statistical power for comparing two proportions.

    Answers: "With n samples, what's the probability of detecting a difference
    from p1 to p2 if it truly exists?"

    Args:
        p1: Proportion under null hypothesis (baseline)
        p2: Proportion under alternative (expected effect)
        n: Sample size per group
        alpha: Significance level
        alternative: Type of alternative hypothesis

    Returns:
        Power (0-1)
    """
    # Effect size
    h = abs(cohens_h(p1, p2))

    # Standard error under pooled proportion
    p_pooled = (p1 + p2) / 2
    se = np.sqrt(2 * p_pooled * (1 - p_pooled) / n)

    if se == 0:
        return 1.0 if p1 != p2 else 0.0

    # z-value for the effect
    z_effect = h / se

    # Critical value
    if alternative == "two-sided":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        power = 1 - stats.norm.cdf(z_crit - z_effect) + stats.norm.cdf(-z_crit - z_effect)
    elif alternative == "larger":
        z_crit = stats.norm.ppf(1 - alpha)
        power = 1 - stats.norm.cdf(z_crit - z_effect)
    else:  # smaller
        z_crit = stats.norm.ppf(1 - alpha)
        power = stats.norm.cdf(-z_crit - z_effect)

    return float(np.clip(power, 0, 1))


def sample_size_for_power(
    p1: float,
    p2: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Calculate required sample size to achieve desired power.

    Args:
        p1: Baseline proportion
        p2: Expected proportion under alternative
        power: Desired power (default 0.8)
        alpha: Significance level

    Returns:
        Required sample size per group
    """
    h = abs(cohens_h(p1, p2))

    if h == 0:
        return float("inf")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    # Sample size formula for proportions
    n = 2 * ((z_alpha + z_power) / h) ** 2

    return int(np.ceil(n))


def minimum_detectable_effect(
    n: int,
    baseline_p: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Calculate minimum detectable effect size (as proportion difference).

    Answers: "With n samples at 80% power, what's the smallest effect I can reliably detect?"

    Args:
        n: Sample size per group
        baseline_p: Baseline proportion
        alpha: Significance level
        power: Desired power

    Returns:
        Minimum detectable proportion (absolute change from baseline)
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    # Solve for h
    h = (z_alpha + z_power) / np.sqrt(n / 2)

    # Convert h back to proportion difference (approximate)
    # This is a numerical search since h -> p is not analytic
    for delta in np.linspace(0, 1 - baseline_p, 1000):
        p2 = baseline_p + delta
        if abs(cohens_h(baseline_p, p2)) >= h:
            return delta

    return 1 - baseline_p


# =============================================================================
# Convenience Functions for Experiment Analysis
# =============================================================================


def compare_configurations(
    predictions_a: list[dict],
    predictions_b: list[dict],
    config_a_name: str,
    config_b_name: str,
    metric: str = "f2_score",
    n_bootstrap: int = 10000,
    random_state: int | None = 42,
) -> ComparisonResult:
    """Compare two model configurations on the same test set.

    Args:
        predictions_a: Predictions from config A
        predictions_b: Predictions from config B
        config_a_name: Name of config A
        config_b_name: Name of config B
        metric: Metric to compare ("f2_score", "accuracy", "recall", etc.)
        n_bootstrap: Bootstrap samples for CI
        random_state: Random seed

    Returns:
        ComparisonResult with statistical analysis
    """
    from .metrics import confusion_matrix_from_predictions

    # Get metric values
    cm_a = confusion_matrix_from_predictions(predictions_a)
    cm_b = confusion_matrix_from_predictions(predictions_b)

    metric_map = {
        "accuracy": "accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1_score": "f1_score",
        "f2_score": "f2_score",
    }

    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric}")

    value_a = getattr(cm_a, metric_map[metric])
    value_b = getattr(cm_b, metric_map[metric])
    difference = value_b - value_a

    # McNemar's test for binary predictions
    preds_a = [p["prediction"] == "Yes" for p in predictions_a]
    preds_b = [p["prediction"] == "Yes" for p in predictions_b]
    truth = [p["ground_truth"] == "Yes" for p in predictions_a]

    mcnemar = mcnemar_test(preds_a, preds_b, truth)

    # Effect size (Cohen's h for proportions)
    effect_size = cohens_h(value_b, value_a)

    # Bootstrap CI for the difference
    # Create paired differences
    rng = np.random.RandomState(random_state)
    n = len(predictions_a)
    correct_a = [p["prediction"] == p["ground_truth"] for p in predictions_a]
    correct_b = [p["prediction"] == p["ground_truth"] for p in predictions_b]

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        acc_a = np.mean([correct_a[i] for i in indices])
        acc_b = np.mean([correct_b[i] for i in indices])
        bootstrap_diffs.append(acc_b - acc_a)

    ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5))

    # Interpretation
    effect_interp = interpret_effect_size(effect_size, "cohens_h")
    if mcnemar.significant:
        if difference > 0:
            interpretation = f"{config_b_name} significantly better ({effect_interp} effect, h={effect_size:.3f})"
        else:
            interpretation = f"{config_a_name} significantly better ({effect_interp} effect, h={effect_size:.3f})"
    else:
        interpretation = f"No significant difference ({effect_interp} effect, h={effect_size:.3f})"

    return ComparisonResult(
        config_a=config_a_name,
        config_b=config_b_name,
        metric=metric,
        value_a=value_a,
        value_b=value_b,
        difference=difference,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=mcnemar.p_value,
        p_value_corrected=mcnemar.p_value,  # Will be updated by multiple comparison
        effect_size=effect_size,
        significant=mcnemar.significant,
        interpretation=interpretation,
    )


def run_pairwise_comparisons(
    all_predictions: dict[str, list[dict]],
    metric: str = "f2_score",
    correction_method: str = "holm-bonferroni",
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """Run all pairwise comparisons between configurations with multiple testing correction.

    Args:
        all_predictions: Dict mapping config name to predictions
        metric: Metric to compare
        correction_method: "holm-bonferroni", "bonferroni", or "fdr-bh"
        alpha: Significance level

    Returns:
        MultipleComparisonResult with all comparisons
    """
    config_names = list(all_predictions.keys())
    comparisons = []

    # Run all pairwise comparisons
    for i, config_a in enumerate(config_names):
        for config_b in config_names[i + 1 :]:
            result = compare_configurations(
                all_predictions[config_a],
                all_predictions[config_b],
                config_a,
                config_b,
                metric=metric,
            )
            comparisons.append(result)

    if not comparisons:
        return MultipleComparisonResult(
            method=correction_method,
            n_comparisons=0,
            alpha_original=alpha,
            comparisons=[],
        )

    # Apply multiple comparison correction
    p_values = [c.p_value for c in comparisons]

    if correction_method == "holm-bonferroni":
        corrections = holm_bonferroni_correction(p_values, alpha)
    elif correction_method == "bonferroni":
        corrections = bonferroni_correction(p_values, alpha)
    elif correction_method == "fdr-bh":
        corrections = fdr_bh_correction(p_values, alpha)
    else:
        raise ValueError(f"Unknown correction method: {correction_method}")

    # Update comparisons with corrected significance
    for comp, (threshold, significant) in zip(comparisons, corrections):
        comp.p_value_corrected = threshold
        comp.significant = significant

    return MultipleComparisonResult(
        method=correction_method,
        n_comparisons=len(comparisons),
        alpha_original=alpha,
        comparisons=comparisons,
    )


def summarize_power_for_experiment(
    n_samples: int,
    n_comparisons: int,
    baseline_f2: float = 0.7,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict:
    """Summarize statistical power for an experiment design.

    Args:
        n_samples: Samples per configuration
        n_comparisons: Number of pairwise comparisons
        baseline_f2: Expected baseline F2 score
        alpha: Significance level
        power: Desired power

    Returns:
        Dict with power analysis results
    """
    # Adjust alpha for multiple comparisons (Bonferroni)
    alpha_corrected = alpha / n_comparisons

    # Minimum detectable effect with corrected alpha
    mde = minimum_detectable_effect(
        n=n_samples,
        baseline_p=baseline_f2,
        alpha=alpha_corrected,
        power=power,
    )

    # Sample size needed to detect various effect sizes
    small_effect_n = sample_size_for_power(baseline_f2, baseline_f2 + 0.1, power, alpha_corrected)
    medium_effect_n = sample_size_for_power(baseline_f2, baseline_f2 + 0.2, power, alpha_corrected)

    return {
        "n_samples": n_samples,
        "n_comparisons": n_comparisons,
        "alpha_original": alpha,
        "alpha_corrected": alpha_corrected,
        "desired_power": power,
        "baseline_metric": baseline_f2,
        "minimum_detectable_effect": mde,
        "interpretation": f"With {n_samples} samples and {n_comparisons} comparisons, "
        f"you can detect effects of {mde:.1%} or larger at {power:.0%} power",
        "sample_size_for_10pct_effect": small_effect_n,
        "sample_size_for_20pct_effect": medium_effect_n,
    }
