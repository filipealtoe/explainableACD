"""Evaluation metrics for checkworthiness classification.

Implements calibration metrics (ECE, Brier Score) and classification metrics
with emphasis on recall (F2 score) since False Negatives are worse than False Positives.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification."""

    tp: int  # True Positives (correctly predicted checkworthy)
    fn: int  # False Negatives (missed checkworthy claims) - CRITICAL
    fp: int  # False Positives (incorrectly flagged as checkworthy)
    tn: int  # True Negatives (correctly identified as not checkworthy)

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def accuracy(self) -> float:
        """(TP + TN) / Total"""
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total

    @property
    def recall(self) -> float:
        """TP / (TP + FN) - Critical metric, minimize FN."""
        if (self.tp + self.fn) == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def precision(self) -> float:
        """TP / (TP + FP)"""
        if (self.tp + self.fp) == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        if (self.precision + self.recall) == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def f2_score(self) -> float:
        """F2 score - weights recall 2x more than precision.

        Formula: (5 × P × R) / (4P + R)
        This is the key metric for checkworthiness where FN > FP in cost.
        """
        p, r = self.precision, self.recall
        if (4 * p + r) == 0:
            return 0.0
        return (5 * p * r) / (4 * p + r)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fn": self.fn,
            "fp": self.fp,
            "tn": self.tn,
            "accuracy": self.accuracy,
            "recall": self.recall,
            "precision": self.precision,
            "f1_score": self.f1_score,
            "f2_score": self.f2_score,
        }

    def __str__(self) -> str:
        return (
            f"Confusion Matrix:\n"
            f"                    Predicted\n"
            f"                 Yes    No\n"
            f"Actual  Yes       {self.tp:2d}     {self.fn:2d}   ← {self.fn} FN (worse!)\n"
            f"        No        {self.fp:2d}     {self.tn:2d}\n"
            f"\n"
            f"Accuracy:  {self.accuracy:.1%}\n"
            f"Recall:    {self.recall:.1%}  (minimize FN)\n"
            f"Precision: {self.precision:.1%}\n"
            f"F1 Score:  {self.f1_score:.3f}\n"
            f"F2 Score:  {self.f2_score:.3f}  (key metric)"
        )


def confusion_matrix_from_predictions(
    predictions: list[dict],
    ground_truth_key: str = "ground_truth",
    prediction_key: str = "prediction",
    positive_label: str = "Yes",
) -> ConfusionMatrix:
    """Build confusion matrix from prediction results.

    Args:
        predictions: List of dicts with ground truth and prediction
        ground_truth_key: Key for ground truth label in each dict
        prediction_key: Key for prediction (bool or "Yes"/"No")
        positive_label: Value that represents "checkworthy"

    Returns:
        ConfusionMatrix with counts
    """
    tp = fn = fp = tn = 0

    for pred in predictions:
        is_positive = pred[ground_truth_key] == positive_label
        predicted_positive = pred[prediction_key]

        # Handle both bool and string predictions
        if isinstance(predicted_positive, str):
            predicted_positive = predicted_positive == positive_label

        if is_positive and predicted_positive:
            tp += 1
        elif is_positive and not predicted_positive:
            fn += 1
        elif not is_positive and predicted_positive:
            fp += 1
        else:
            tn += 1

    return ConfusionMatrix(tp=tp, fn=fn, fp=fp, tn=tn)


def expected_calibration_error(
    confidences: list[float],
    outcomes: list[bool],
    n_bins: int | None = None,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    ECE measures how well the model's confidence matches its accuracy.
    A well-calibrated model should be correct 80% of the time when it says 80% confident.

    Formula: ECE = sum(|bin_acc - bin_conf| * bin_size / total) for each bin

    Note on small samples: With small N (e.g., N=20), using 10 bins results in
    ~2 samples per bin on average, making ECE unreliable. This function uses
    adaptive binning: sqrt(N) bins, clamped to [3, 10] range.

    For N=20: sqrt(20) ≈ 4.5 → 5 bins (4 samples per bin on average)
    For N=100: sqrt(100) = 10 → 10 bins

    Args:
        confidences: Model confidence scores (0-100 scale)
        outcomes: Ground truth (True if prediction was correct)
        n_bins: Number of bins for calibration. If None, uses adaptive binning.

    Returns:
        ECE score (0-1 scale, lower is better)
    """
    if len(confidences) == 0:
        return 0.0

    # Adaptive binning for small samples
    # Rule of thumb: sqrt(N) bins, clamped to reasonable range
    if n_bins is None:
        n_bins = max(3, min(10, int(np.sqrt(len(confidences)))))

    # Normalize confidences to 0-1 scale
    confidences_norm = np.array(confidences) / 100.0
    outcomes_arr = np.array(outcomes, dtype=float)

    # Create bins using np.digitize for consistent edge handling
    # This properly includes both 0.0 and 1.0 in their respective bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    # np.digitize returns bin index (1-indexed); clip to handle edge case at 1.0
    bin_indices = np.digitize(confidences_norm, bin_boundaries[1:-1])  # Use internal boundaries

    ece = 0.0
    for bin_idx in range(n_bins):
        in_bin = bin_indices == bin_idx
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(outcomes_arr[in_bin])
            bin_conf = np.mean(confidences_norm[in_bin])
            ece += np.abs(bin_acc - bin_conf) * (bin_size / len(confidences))

    return float(ece)


def brier_score(confidences: list[float], outcomes: list[bool]) -> float:
    """Calculate Brier Score.

    Brier Score is a proper scoring rule that measures the accuracy of probabilistic predictions.
    Formula: mean((confidence - outcome)²)

    For binary classification:
    - outcome = 1 if positive (or correct, depending on how you supply outcomes)
    - confidence is the model's probability for that outcome

    Args:
        confidences: Model confidence scores (0-100 scale)
        outcomes: Ground truth (True if prediction was correct)

    Returns:
        Brier score (0-1 scale, lower is better)
    """
    if len(confidences) == 0:
        return 0.0

    # Normalize confidences to 0-1 scale
    confidences_norm = np.array(confidences) / 100.0
    outcomes_arr = np.array(outcomes, dtype=float)

    return float(np.mean((confidences_norm - outcomes_arr) ** 2))


def calculate_calibration_metrics(
    predictions: list[dict],
    confidence_key: str = "confidence",
    ground_truth_key: str = "ground_truth",
    prediction_key: str = "prediction",
    positive_label: str = "Yes",
    confidence_is_correct: bool = False,
) -> dict:
    """Calculate calibration metrics from predictions.

    Assumes confidence represents P(positive_label) by default.
    Set confidence_is_correct=True if confidence represents P(correct).

    Args:
        predictions: List of prediction dicts
        confidence_key: Key for confidence score
        ground_truth_key: Key for ground truth label
        prediction_key: Key for prediction
        positive_label: Value representing positive class
        confidence_is_correct: True if confidence is P(correct), else P(positive)

    Returns:
        Dict with ECE and Brier score
    """
    confidences = []
    outcomes = []

    for pred in predictions:
        conf = pred[confidence_key]
        is_positive = pred[ground_truth_key] == positive_label
        predicted_positive = pred[prediction_key]

        if isinstance(predicted_positive, str):
            predicted_positive = predicted_positive == positive_label

        if confidence_is_correct:
            outcome = is_positive == predicted_positive
        else:
            outcome = is_positive

        outcomes.append(outcome)
        confidences.append(conf)

    return {
        "ece": expected_calibration_error(confidences, outcomes),
        "brier_score": brier_score(confidences, outcomes),
    }


def calculate_all_metrics(
    predictions: list[dict],
    confidence_key: str = "confidence",
    ground_truth_key: str = "ground_truth",
    prediction_key: str = "prediction",
    positive_label: str = "Yes",
    confidence_is_correct: bool = False,
) -> dict:
    """Calculate all metrics: confusion matrix + calibration.

    Args:
        predictions: List of prediction dicts

    Returns:
        Dict with all metrics
    """
    cm = confusion_matrix_from_predictions(
        predictions,
        ground_truth_key=ground_truth_key,
        prediction_key=prediction_key,
        positive_label=positive_label,
    )

    calibration = calculate_calibration_metrics(
        predictions,
        confidence_key=confidence_key,
        ground_truth_key=ground_truth_key,
        prediction_key=prediction_key,
        positive_label=positive_label,
        confidence_is_correct=confidence_is_correct,
    )

    return {
        "confusion_matrix": cm.to_dict(),
        "accuracy": cm.accuracy,
        "recall": cm.recall,
        "precision": cm.precision,
        "f1_score": cm.f1_score,
        "f2_score": cm.f2_score,
        "ece": calibration["ece"],
        "brier_score": calibration["brier_score"],
    }
