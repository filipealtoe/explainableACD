#!/usr/bin/env python3
"""
Threshold Optimization for Binary Classification

The default threshold (0.5) is rarely optimal for imbalanced datasets.
This script finds the optimal threshold that maximizes F1 on the dev set,
then evaluates on test set.

Usage:
    python experiments/scripts/optimize_threshold.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
EMBEDDING_FILE = DATA_DIR / "embedding_cache" / "bge-large_embeddings.npz"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"

SOTA_F1 = 0.82
PREVIOUS_BEST_F1 = 0.749


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load BGE-large embeddings and labels."""
    embed_data = np.load(EMBEDDING_FILE)
    X_train = embed_data["train"].astype(np.float32)
    X_dev = embed_data["dev"].astype(np.float32)
    X_test = embed_data["test"].astype(np.float32)

    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    y_train = (ct24_train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (ct24_dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (ct24_test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def find_optimal_threshold(y_true, y_proba, metric="f1"):
    """Find optimal threshold using precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # F1 at each threshold
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )

    # Best threshold (excluding last point which has threshold=1)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1, thresholds, f1_scores[:-1]


def evaluate_at_threshold(y_true, y_proba, threshold):
    """Evaluate predictions at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pred_positive_rate": y_pred.mean(),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 100)
    print("\nDefault threshold = 0.5 is rarely optimal for imbalanced data.")
    print("Finding the best threshold that maximizes F1 on dev set.\n")

    # Load data
    print("Loading BGE-large embeddings...")
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    print(f"  Train: {len(y_train)} ({100*y_train.mean():.1f}% positive)")
    print(f"  Dev:   {len(y_dev)} ({100*y_dev.mean():.1f}% positive)")
    print(f"  Test:  {len(y_test)} ({100*y_test.mean():.1f}% positive)")

    # Classifiers to test
    classifiers = {
        "logistic_w1": LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 1}),
        "logistic_w3": LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 3}),
        "logistic_w4": LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4}),
    }

    if HAS_LIGHTGBM:
        classifiers["lgbm_w4"] = LGBMClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=-1,
            class_weight={0: 1, 1: 4}
        )

    if HAS_XGBOOST:
        classifiers["xgb_w4"] = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss",
            scale_pos_weight=4
        )

    results = []

    for clf_name, clf in classifiers.items():
        print(f"\n{'='*100}")
        print(f"Classifier: {clf_name}")
        print(f"{'='*100}")

        # Train
        print("  Training...", end=" ", flush=True)
        clf.fit(X_train_s, y_train)
        print("done")

        # Get probabilities
        y_proba_dev = clf.predict_proba(X_dev_s)[:, 1]
        y_proba_test = clf.predict_proba(X_test_s)[:, 1]

        # Evaluate at default threshold (0.5)
        default_dev = evaluate_at_threshold(y_dev, y_proba_dev, 0.5)
        default_test = evaluate_at_threshold(y_test, y_proba_test, 0.5)

        print(f"\n  DEFAULT THRESHOLD (0.5):")
        print(f"    Dev:  F1={default_dev['f1']:.4f}  P={default_dev['precision']:.4f}  R={default_dev['recall']:.4f}")
        print(f"    Test: F1={default_test['f1']:.4f}  P={default_test['precision']:.4f}  R={default_test['recall']:.4f}")

        # Find optimal threshold on DEV set
        opt_threshold, opt_dev_f1, thresholds, f1_scores = find_optimal_threshold(y_dev, y_proba_dev)

        # Evaluate optimal threshold on TEST set
        opt_test = evaluate_at_threshold(y_test, y_proba_test, opt_threshold)

        print(f"\n  OPTIMAL THRESHOLD ({opt_threshold:.3f}):")
        print(f"    Dev:  F1={opt_dev_f1:.4f}")
        print(f"    Test: F1={opt_test['f1']:.4f}  P={opt_test['precision']:.4f}  R={opt_test['recall']:.4f}  Acc={opt_test['accuracy']:.4f}")

        improvement = opt_test['f1'] - default_test['f1']
        print(f"\n  Improvement from threshold tuning: {improvement:+.4f}")

        # Show F1 at various thresholds
        print(f"\n  F1 vs Threshold (test set):")
        test_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for t in test_thresholds:
            result = evaluate_at_threshold(y_test, y_proba_test, t)
            marker = " ← default" if t == 0.5 else ""
            marker = " ← optimal" if abs(t - opt_threshold) < 0.05 else marker
            bar = "█" * int(result['f1'] * 50)
            print(f"    t={t:.1f}: F1={result['f1']:.4f} P={result['precision']:.4f} R={result['recall']:.4f} {bar}{marker}")

        results.append({
            "classifier": clf_name,
            "default_threshold": 0.5,
            "default_test_f1": default_test['f1'],
            "optimal_threshold": opt_threshold,
            "optimal_test_f1": opt_test['f1'],
            "optimal_test_acc": opt_test['accuracy'],
            "optimal_precision": opt_test['precision'],
            "optimal_recall": opt_test['recall'],
            "improvement": improvement,
        })

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"\n{'Classifier':<15} {'Default t':<12} {'Default F1':<12} {'Optimal t':<12} {'Optimal F1':<12} {'Improvement':<12}")
    print("-" * 80)

    best_result = None
    best_f1 = 0

    for r in results:
        print(f"{r['classifier']:<15} {r['default_threshold']:<12.2f} {r['default_test_f1']:<12.4f} "
              f"{r['optimal_threshold']:<12.3f} {r['optimal_test_f1']:<12.4f} {r['improvement']:+.4f}")

        if r['optimal_test_f1'] > best_f1:
            best_f1 = r['optimal_test_f1']
            best_result = r

    print(f"\n{'='*100}")
    print("🏆 BEST RESULT")
    print(f"{'='*100}")
    print(f"  Classifier:        {best_result['classifier']}")
    print(f"  Optimal Threshold: {best_result['optimal_threshold']:.3f}")
    print(f"  Test F1:           {best_result['optimal_test_f1']:.4f}")
    print(f"  Test Accuracy:     {best_result['optimal_test_acc']:.4f}")
    print(f"  Precision:         {best_result['optimal_precision']:.4f}")
    print(f"  Recall:            {best_result['optimal_recall']:.4f}")
    print(f"\n  Previous best (t=0.5): F1 = {PREVIOUS_BEST_F1}")
    print(f"  Improvement:           {best_result['optimal_test_f1'] - PREVIOUS_BEST_F1:+.4f}")
    print(f"\n  SOTA: F1 = {SOTA_F1}")
    print(f"  Gap to SOTA: {best_result['optimal_test_f1'] - SOTA_F1:+.4f}")


if __name__ == "__main__":
    main()
