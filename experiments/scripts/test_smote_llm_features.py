#!/usr/bin/env python3
"""
Test SMOTE with Best LLM Features

SMOTE works better in low dimensions (6 features) than high dimensions (1024 embeddings).
Tests SMOTE + various classifiers on scores + entropy features.

Usage:
    python experiments/scripts/test_smote_llm_features.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("WARNING: imblearn not installed. Run: pip install imbalanced-learn")

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
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"

# Best LLM features: scores + entropy
BEST_FEATURES = [
    "check_score", "verif_score", "harm_score",
    "check_entropy", "verif_entropy", "harm_entropy",
]

SOTA_F1 = 0.82
PREVIOUS_BEST_F1 = 0.708  # LLM features without SMOTE


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load LLM features and labels."""
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    # Join on sentence_id
    llm_train = llm_train.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_test = llm_test.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    train = llm_train.join(ct24_train.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")
    test = llm_test.join(ct24_test.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")

    X_train = train.select(BEST_FEATURES).to_numpy().astype(np.float32)
    X_test = test.select(BEST_FEATURES).to_numpy().astype(np.float32)

    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, X_test, y_train, y_test


# =============================================================================
# Main
# =============================================================================

def main():
    if not HAS_IMBLEARN:
        print("ERROR: imbalanced-learn not installed. Run: pip install imbalanced-learn")
        return

    print("=" * 100)
    print("SMOTE TEST with LLM Features (6 dimensions)")
    print("=" * 100)
    print(f"\nFeatures: {BEST_FEATURES}")
    print(f"Previous best (no SMOTE): F1 = {PREVIOUS_BEST_F1}")

    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test = load_data()

    print(f"  Train: {len(y_train)} samples ({100*y_train.mean():.1f}% positive)")
    print(f"  Test:  {len(y_test)} samples ({100*y_test.mean():.1f}% positive)")

    # Define resampling strategies
    resamplers = {
        "None (baseline)": None,
        "SMOTE": SMOTE(random_state=42),
        "SMOTE (k=3)": SMOTE(k_neighbors=3, random_state=42),
        "SMOTE (k=7)": SMOTE(k_neighbors=7, random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "SMOTETomek": SMOTETomek(random_state=42),
    }

    # Define classifiers (with class_weight for comparison)
    classifiers = {
        "logistic": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "logistic_w4": LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4}),
    }

    if HAS_LIGHTGBM:
        classifiers["lgbm"] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        classifiers["lgbm_w4"] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, verbose=-1, class_weight={0: 1, 1: 4}
        )

    if HAS_XGBOOST:
        classifiers["xgb"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42, eval_metric="logloss"
        )

    results = []

    # Test each combination
    for resampler_name, resampler in resamplers.items():
        print(f"\n{'='*100}")
        print(f"Resampling: {resampler_name}")
        print(f"{'='*100}")

        # Apply resampling
        if resampler is not None:
            X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
            print(f"  After resampling: {len(y_train_res)} samples ({100*y_train_res.mean():.1f}% positive)")
        else:
            X_train_res, y_train_res = X_train, y_train

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_res)
        X_test_s = scaler.transform(X_test)

        for clf_name, clf in classifiers.items():
            # Skip weighted classifiers with SMOTE (redundant)
            if resampler is not None and "_w4" in clf_name:
                continue

            clf_copy = clf.__class__(**clf.get_params())
            clf_copy.fit(X_train_s, y_train_res)

            y_pred = clf_copy.predict(X_test_s)

            test_f1 = f1_score(y_test, y_pred)
            test_acc = accuracy_score(y_test, y_pred)
            test_p = precision_score(y_test, y_pred, zero_division=0)
            test_r = recall_score(y_test, y_pred, zero_division=0)

            results.append({
                "resampler": resampler_name,
                "classifier": clf_name,
                "test_f1": test_f1,
                "test_acc": test_acc,
                "precision": test_p,
                "recall": test_r,
            })

            delta = test_f1 - PREVIOUS_BEST_F1
            marker = "🔥" if test_f1 > PREVIOUS_BEST_F1 else ""
            print(f"  {clf_name:<15} F1={test_f1:.4f} ({delta:+.4f}) Acc={test_acc:.4f} P={test_p:.4f} R={test_r:.4f} {marker}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY - All Results")
    print("=" * 100)

    # Sort by F1
    results.sort(key=lambda x: x["test_f1"], reverse=True)

    print(f"\n{'Rank':<5} {'Resampler':<20} {'Classifier':<15} {'Test F1':<10} {'Acc':<10} {'P':<8} {'R':<8}")
    print("-" * 85)

    for rank, r in enumerate(results[:15], 1):
        marker = "🔥" if r["test_f1"] > PREVIOUS_BEST_F1 else ""
        print(f"{rank:<5} {r['resampler']:<20} {r['classifier']:<15} {r['test_f1']:<10.4f} "
              f"{r['test_acc']:<10.4f} {r['precision']:<8.4f} {r['recall']:<8.4f} {marker}")

    # Best result
    best = results[0]
    print(f"\n{'='*100}")
    print("🏆 BEST RESULT")
    print(f"{'='*100}")
    print(f"  Resampler:   {best['resampler']}")
    print(f"  Classifier:  {best['classifier']}")
    print(f"  Test F1:     {best['test_f1']:.4f}")
    print(f"  Test Acc:    {best['test_acc']:.4f}")
    print(f"  Precision:   {best['precision']:.4f}")
    print(f"  Recall:      {best['recall']:.4f}")
    print(f"\n  Previous best: F1 = {PREVIOUS_BEST_F1}")
    print(f"  Improvement:   {best['test_f1'] - PREVIOUS_BEST_F1:+.4f}")
    print(f"\n  SOTA: F1 = {SOTA_F1}")
    print(f"  Gap to SOTA: {best['test_f1'] - SOTA_F1:+.4f}")

    # Analysis
    print("\n" + "=" * 100)
    print("ANALYSIS: SMOTE vs Class Weights")
    print("=" * 100)

    # Compare SMOTE alone vs class weights alone
    smote_results = [r for r in results if "SMOTE" in r["resampler"] and "_w" not in r["classifier"]]
    weight_results = [r for r in results if r["resampler"] == "None (baseline)" and "_w4" in r["classifier"]]

    if smote_results:
        best_smote = max(smote_results, key=lambda x: x["test_f1"])
        print(f"\n  Best SMOTE:        F1 = {best_smote['test_f1']:.4f} ({best_smote['resampler']} + {best_smote['classifier']})")

    if weight_results:
        best_weight = max(weight_results, key=lambda x: x["test_f1"])
        print(f"  Best Class Weight: F1 = {best_weight['test_f1']:.4f} ({best_weight['classifier']})")

    if smote_results and weight_results:
        if best_smote["test_f1"] > best_weight["test_f1"]:
            print(f"\n  → SMOTE wins by {best_smote['test_f1'] - best_weight['test_f1']:+.4f}")
        else:
            print(f"\n  → Class weights win by {best_weight['test_f1'] - best_smote['test_f1']:+.4f}")


if __name__ == "__main__":
    main()
