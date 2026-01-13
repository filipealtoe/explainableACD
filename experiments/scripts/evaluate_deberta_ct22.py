#!/usr/bin/env python3
"""
Evaluate fine-tuned DeBERTa on CheckThat! 2022 Task 1A test set.

CT22 is COVID-19 Twitter data — tests topic/domain generalization.
Model trained on CT24 (mixed Twitter + debates) evaluated on CT22 (COVID tweets).

Key differences from CT23/CT24:
- Text column is 'tweet_text' (not 'Text')
- Labels are numeric (0/1) not strings (Yes/No)

Usage:
    python experiments/scripts/evaluate_deberta_ct22.py
    python experiments/scripts/evaluate_deberta_ct22.py --auto-threshold
    python experiments/scripts/evaluate_deberta_ct22.py --auto-threshold --save
    python experiments/scripts/evaluate_deberta_ct22.py --split dev_test --auto-threshold
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

# Paths
MODEL_PATH = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "check_that_22"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "ct22_eval"

# Available splits
SPLITS = {
    "train": "CT22_english_1A_checkworthy_train.tsv",
    "dev": "CT22_english_1A_checkworthy_dev.tsv",
    "dev_test": "CT22_english_1A_checkworthy_dev_test.tsv",
    "test_gold": "CT22_english_1A_checkworthy_test_gold.tsv",
}


def find_optimal_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, dict]:
    """Find the optimal classification threshold."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    results = {}

    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)

        if len(np.unique(y_pred)) == 1:
            continue

        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        beta = 2
        f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0

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

    best_thresh = max(results.keys(), key=lambda t: results[t][metric])
    best_score = results[best_thresh][metric]

    return best_thresh, best_score, results


def load_ct22(file_path: Path) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Load CT22 TSV with pandas (handles malformed rows)."""
    df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip', engine='python')

    texts = df["tweet_text"].tolist()
    # Labels are numeric in CT22
    y_true = df["class_label"].values.astype(int)

    return texts, y_true, df


def run_inference(
    model,
    tokenizer,
    device: torch.device,
    texts: list[str],
    max_length: int = 128,
) -> np.ndarray:
    """Run inference and return probabilities."""
    probs = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Inference"):
            # Handle potential NaN or non-string texts
            if pd.isna(text):
                text = ""
            text = str(text)

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            probs.append(prob)

    return np.array(probs)


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

    n_total = len(y_true)
    n_positive = sum(y_true)
    print(f"\nDataset: {n_total} samples, {n_positive} positive ({100*n_positive/n_total:.1f}%)")
    print(f"Threshold: {threshold:.2f}")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TP={tp:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TN={tn:4d}")

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nMetrics:")
    print(f"  F1 (positive class):  {f1:.4f}")
    print(f"  Precision:            {precision:.4f}")
    print(f"  Recall:               {recall:.4f}")
    print(f"  Accuracy:             {accuracy:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No (0)", "Yes (1)"]))

    print(f"Probability Distribution:")
    print(f"  Mean prob (true=0): {probs[y_true == 0].mean():.4f} (std={probs[y_true == 0].std():.4f})")
    print(f"  Mean prob (true=1): {probs[y_true == 1].mean():.4f} (std={probs[y_true == 1].std():.4f})")

    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeBERTa on CheckThat! 2022")
    parser.add_argument("--split", type=str, default="test_gold", choices=list(SPLITS.keys()),
                        help="Which split to evaluate (default: test_gold)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--auto-threshold", action="store_true", help="Automatically find optimal threshold")
    parser.add_argument("--optimize-metric", type=str, default="f1", choices=["f1", "f2", "youden"],
                        help="Metric to optimize when using --auto-threshold")
    parser.add_argument("--save", action="store_true", help="Save predictions to CSV")
    parser.add_argument("--all-splits", action="store_true", help="Evaluate on all splits with labels")
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

    # Determine which splits to evaluate
    if args.all_splits:
        splits_to_eval = ["train", "dev", "dev_test", "test_gold"]
    else:
        splits_to_eval = [args.split]

    all_results = {}

    for split_name in splits_to_eval:
        file_path = DATA_DIR / SPLITS[split_name]

        if not file_path.exists():
            print(f"\nSkipping {split_name}: file not found at {file_path}")
            continue

        print(f"\n{'#'*60}")
        print(f"# Evaluating: {split_name}")
        print(f"# File: {SPLITS[split_name]}")
        print(f"# Domain: COVID-19 Tweets")
        print(f"{'#'*60}")

        # Load data
        texts, y_true, df = load_ct22(file_path)
        print(f"Loaded {len(texts)} samples")

        # Run inference
        probs = run_inference(model, tokenizer, device, texts)

        # Determine threshold
        if args.auto_threshold:
            best_thresh, best_score, thresh_results = find_optimal_threshold(
                y_true, probs, metric=args.optimize_metric
            )
            threshold = best_thresh

            print(f"\n>>> Auto-threshold optimization ({args.optimize_metric}):")
            print(f"    Best threshold: {best_thresh:.2f} -> {args.optimize_metric.upper()} = {best_score:.4f}")

            sorted_results = sorted(thresh_results.items(), key=lambda x: x[1][args.optimize_metric], reverse=True)[:5]
            print(f"\n    Top 5 thresholds by {args.optimize_metric}:")
            for t, metrics in sorted_results:
                print(f"      {t:.2f}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
        else:
            threshold = args.threshold

        # Apply threshold
        y_pred = (probs >= threshold).astype(int)

        # Print results
        metrics = print_results(f"CT22 {split_name} (COVID-19 Tweets)", y_true, y_pred, probs, threshold)
        all_results[split_name] = metrics

        # Show comparison with default threshold
        if args.auto_threshold and abs(threshold - 0.5) > 0.01:
            y_pred_default = (probs >= 0.5).astype(int)
            f1_default = f1_score(y_true, y_pred_default)
            f1_optimal = f1_score(y_true, y_pred)
            print(f"\n--- Comparison: Results @ default threshold 0.50 ---")
            print(f"  F1 @ 0.50: {f1_default:.4f}")
            print(f"  F1 @ {threshold:.2f}: {f1_optimal:.4f} ({'+' if f1_optimal > f1_default else ''}{f1_optimal - f1_default:.4f})")

        # Save predictions
        if args.save:
            df_out = df.copy()
            df_out["prob"] = probs
            df_out["pred_binary"] = y_pred
            df_out["pred_label"] = ["Yes" if p == 1 else "No" for p in y_pred]
            df_out["true_binary"] = y_true
            df_out["threshold_used"] = threshold

            suffix = f"_thresh{threshold:.2f}" if args.auto_threshold else ""
            output_path = OUTPUT_DIR / f"{split_name}_predictions{suffix}.csv"
            df_out.to_csv(output_path, index=False)
            print(f"\nSaved predictions to: {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY: CheckThat! 2022 Evaluation")
    print(f"{'='*60}")
    print(f"\nModel trained on: CT24 (CheckThat! 2024 - Mixed Twitter + Debates)")
    print(f"Evaluated on:     CT22 Task 1A (COVID-19 Tweets)")

    if len(all_results) > 1:
        print(f"\nResults by split:")
        print(f"  {'Split':<12} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")
        for split_name, metrics in all_results.items():
            print(f"  {split_name:<12} {metrics['f1']:>8.4f} {metrics['precision']:>10.4f} {metrics['recall']:>8.4f} {metrics['accuracy']:>10.4f}")

    print(f"\nContext:")
    print(f"  - CT22 = COVID-19 tweets (Twitter domain, pandemic topic)")
    print(f"  - CT24 = Mixed sources (Twitter + debates, various topics)")
    print(f"  - This tests COVID-topic and temporal generalization (2022 data)")


if __name__ == "__main__":
    main()
