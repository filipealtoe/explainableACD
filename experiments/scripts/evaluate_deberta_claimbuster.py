#!/usr/bin/env python3
"""
Evaluate fine-tuned DeBERTa on ClaimBuster dataset (cross-dataset generalization test).

This tests whether the model trained on CT24 generalizes to ClaimBuster's
presidential debate data.

ClaimBuster labels:
  - Verdict = 1  -> CFS (Check-worthy Factual Statement) -> "Yes"
  - Verdict = 0  -> UFS (Unimportant Factual Statement)  -> "No"
  - Verdict = -1 -> NFS (Non-Factual Statement)          -> "No"

Usage:
    python experiments/scripts/evaluate_deberta_claimbuster.py
    python experiments/scripts/evaluate_deberta_claimbuster.py --threshold 0.3
    python experiments/scripts/evaluate_deberta_claimbuster.py --auto-threshold
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

# Paths
MODEL_PATH = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "claim_buster"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "claimbuster_eval"


def find_optimal_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, dict]:
    """
    Find the optimal classification threshold.

    Args:
        y_true: Ground truth binary labels
        probs: Predicted probabilities for positive class
        metric: Optimization target ("f1", "f2", "youden", "precision_at_recall")
        thresholds: Custom thresholds to try (default: 0.05 to 0.95 in 0.01 steps)

    Returns:
        (best_threshold, best_score, all_results)
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    results = {}

    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)

        # Skip if all predictions are same class
        if len(np.unique(y_pred)) == 1:
            continue

        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # F2 score (weights recall higher than precision)
        beta = 2
        f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0

        # Youden's J statistic (sensitivity + specificity - 1)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = recall + specificity - 1

        results[thresh] = {
            "f1": f1,
            "f2": f2,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "youden": youden,
            "accuracy": accuracy_score(y_true, y_pred),
        }

    # Find best threshold for chosen metric
    best_thresh = max(results.keys(), key=lambda t: results[t][metric])
    best_score = results[best_thresh][metric]

    return best_thresh, best_score, results


def load_claimbuster(file_path: Path) -> tuple[list[str], np.ndarray, pl.DataFrame]:
    """Load ClaimBuster CSV and convert to binary labels."""
    df = pl.read_csv(file_path)

    # Convert 3-class to binary: CFS (1) -> Yes, UFS (0) and NFS (-1) -> No
    texts = df["Text"].to_list()
    verdicts = df["Verdict"].to_list()
    y_true = np.array([1 if v == 1 else 0 for v in verdicts])

    return texts, y_true, df


def evaluate_dataset(
    model,
    tokenizer,
    device: torch.device,
    texts: list[str],
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return predictions and probabilities."""
    probs = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Inference"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            probs.append(prob)

    probs = np.array(probs)
    y_pred = (probs >= threshold).astype(int)

    return y_pred, probs


def print_results(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    threshold: float,
):
    """Print comprehensive evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Basic stats
    n_total = len(y_true)
    n_positive = sum(y_true)
    print(f"\nDataset: {n_total} samples, {n_positive} positive ({100*n_positive/n_total:.1f}%)")
    print(f"Threshold: {threshold}")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TP={tp:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TN={tn:4d}")

    # Metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nMetrics:")
    print(f"  F1 (positive class):  {f1:.4f}")
    print(f"  Precision:            {precision:.4f}")
    print(f"  Recall:               {recall:.4f}")
    print(f"  Accuracy:             {accuracy:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No (NFS/UFS)", "Yes (CFS)"]))

    # Probability distribution
    print(f"Probability Distribution:")
    print(f"  Mean prob (true=0): {probs[y_true == 0].mean():.4f}")
    print(f"  Mean prob (true=1): {probs[y_true == 1].mean():.4f}")
    print(f"  Std prob (true=0):  {probs[y_true == 0].std():.4f}")
    print(f"  Std prob (true=1):  {probs[y_true == 1].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeBERTa on ClaimBuster")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--auto-threshold", action="store_true", help="Automatically find optimal threshold")
    parser.add_argument("--optimize-metric", type=str, default="f1", choices=["f1", "f2", "youden"],
                        help="Metric to optimize when using --auto-threshold")
    parser.add_argument("--save", action="store_true", help="Save predictions to CSV")
    args = parser.parse_args()

    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")

    # Load model
    print("\nLoading model...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate on both datasets
    for dataset_name in ["groundtruth", "crowdsourced"]:
        file_path = DATA_DIR / f"{dataset_name}.csv"

        if not file_path.exists():
            print(f"\nSkipping {dataset_name}: file not found at {file_path}")
            continue

        print(f"\n{'#'*60}")
        print(f"# Processing: {dataset_name}.csv")
        print(f"{'#'*60}")

        # Load data
        texts, y_true, df = load_claimbuster(file_path)

        # Run inference (get probabilities first, threshold later)
        _, probs = evaluate_dataset(model, tokenizer, device, texts, y_true, threshold=0.5)

        # Determine threshold
        if args.auto_threshold:
            best_thresh, best_score, all_results = find_optimal_threshold(
                y_true, probs, metric=args.optimize_metric
            )
            threshold = best_thresh

            print(f"\n>>> Auto-threshold optimization ({args.optimize_metric}):")
            print(f"    Best threshold: {best_thresh:.2f} -> {args.optimize_metric.upper()} = {best_score:.4f}")

            # Show top 5 thresholds
            sorted_results = sorted(all_results.items(), key=lambda x: x[1][args.optimize_metric], reverse=True)[:5]
            print(f"\n    Top 5 thresholds by {args.optimize_metric}:")
            for t, metrics in sorted_results:
                print(f"      {t:.2f}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
        else:
            threshold = args.threshold

        # Apply threshold
        y_pred = (probs >= threshold).astype(int)

        # Print results
        print_results(f"ClaimBuster ({dataset_name})", y_true, y_pred, probs, threshold)

        # Also show results at default 0.5 for comparison if auto-threshold was used
        if args.auto_threshold and threshold != 0.5:
            y_pred_default = (probs >= 0.5).astype(int)
            print(f"\n--- Comparison: Results @ default threshold 0.50 ---")
            f1_default = f1_score(y_true, y_pred_default)
            f1_optimal = f1_score(y_true, y_pred)
            print(f"  F1 @ 0.50: {f1_default:.4f}")
            print(f"  F1 @ {threshold:.2f}: {f1_optimal:.4f} ({'+' if f1_optimal > f1_default else ''}{f1_optimal - f1_default:.4f})")

        # Analyze by original verdict class
        verdicts = df["Verdict"].to_list()
        print(f"\nBreakdown by Original ClaimBuster Class:")
        for verdict_val, verdict_name in [(-1, "NFS"), (0, "UFS"), (1, "CFS")]:
            mask = np.array([v == verdict_val for v in verdicts])
            if mask.sum() > 0:
                pred_positive_rate = probs[mask].mean()
                n_predicted_yes = (y_pred[mask] == 1).sum()
                print(f"  {verdict_name} (n={mask.sum():4d}): {n_predicted_yes:4d} predicted Yes ({100*n_predicted_yes/mask.sum():.1f}%), mean prob={pred_positive_rate:.3f}")

        # Save predictions if requested
        if args.save:
            output_df = df.with_columns([
                pl.Series("prob", probs),
                pl.Series("pred_binary", y_pred),
                pl.Series("pred_label", ["Yes" if p == 1 else "No" for p in y_pred]),
                pl.Series("true_binary", y_true),
                pl.lit(threshold).alias("threshold_used"),
            ])
            suffix = f"_thresh{threshold:.2f}" if args.auto_threshold else ""
            output_path = OUTPUT_DIR / f"{dataset_name}_predictions{suffix}.csv"
            output_df.write_csv(output_path)
            print(f"\nSaved predictions to: {output_path}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("  SUMMARY: Cross-Dataset Generalization")
    print(f"{'='*60}")
    print(f"Model trained on: CT24 (CheckThat! 2024)")
    print(f"Evaluated on:     ClaimBuster (US Presidential Debates 1960-2016)")
    print(f"\nExpected behavior:")
    print(f"  - CFS claims should have HIGH probability (ideally >0.5)")
    print(f"  - UFS claims should have LOW probability (these are claims, but not checkworthy)")
    print(f"  - NFS claims should have LOW probability (not even claims)")
    print(f"\nIf F1 drops significantly vs CT24 dev-test, the model learned")
    print(f"CT24-specific patterns rather than generalizable checkworthiness.")


if __name__ == "__main__":
    main()
