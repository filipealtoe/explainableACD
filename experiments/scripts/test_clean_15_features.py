#!/usr/bin/env python3
"""
Test Clean 15 LLM Features

Only raw features from LLM:
- 3 self-reported confidences (check_score, verif_score, harm_score)
- 9 probabilities from logprobs (p_true, p_false, p_uncertain × 3 modules)
- 3 entropy values (check_entropy, verif_entropy, harm_entropy)

No derived features, no redundancy.

Usage:
    python experiments/scripts/test_clean_15_features.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

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

SOTA_F1 = 0.82
SOTA_ACC = 0.905

# =============================================================================
# The Clean 15 Features
# =============================================================================

CLEAN_15_FEATURES = [
    # Self-reported confidences (3)
    "check_score", "verif_score", "harm_score",

    # Probabilities from logprobs (9)
    "check_p_true", "check_p_false", "check_p_uncertain",
    "verif_p_true", "verif_p_false", "verif_p_uncertain",
    "harm_p_true", "harm_p_false", "harm_p_uncertain",

    # Entropy (3)
    "check_entropy", "verif_entropy", "harm_entropy",
]


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load data and extract clean 15 features."""
    # LLM features
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    # Extract features
    X_train = llm_train.select(CLEAN_15_FEATURES).to_numpy().astype(np.float32)
    X_dev = llm_dev.select(CLEAN_15_FEATURES).to_numpy().astype(np.float32)
    X_test = llm_test.select(CLEAN_15_FEATURES).to_numpy().astype(np.float32)

    # Labels
    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    y_train = (ct24_train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (ct24_dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (ct24_test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, X_dev, X_test, y_train, y_dev, y_test


# =============================================================================
# Classifiers
# =============================================================================

def get_classifiers():
    """Get all classifiers to test."""
    classifiers = {
        "logistic": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42,
            n_jobs=-1, class_weight="balanced"
        ),
    }

    if HAS_LIGHTGBM:
        classifiers["lightgbm"] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
        )
        classifiers["lgbm_deep"] = LGBMClassifier(
            n_estimators=400, max_depth=12, learning_rate=0.03,
            random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
        )

    if HAS_CATBOOST:
        classifiers["catboost"] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            random_state=42, auto_class_weights="Balanced", verbose=False
        )
        classifiers["cb_deep"] = CatBoostClassifier(
            iterations=400, depth=8, learning_rate=0.03,
            random_state=42, auto_class_weights="Balanced", verbose=False
        )

    if HAS_XGBOOST:
        classifiers["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss", scale_pos_weight=3.0
        )

    return classifiers


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("CLEAN 15 LLM FEATURES TEST")
    print("=" * 90)
    print("\nFeatures used (15 total):")
    print("  • Self-reported confidences: check_score, verif_score, harm_score")
    print("  • Logprob probabilities: p_true, p_false, p_uncertain × 3 modules")
    print("  • Entropy: check_entropy, verif_entropy, harm_entropy")
    print(f"\nSOTA targets: F1={SOTA_F1}, Acc={SOTA_ACC}")

    # Load data
    print("\n" + "-" * 90)
    print("Loading data...")
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Dev:   {X_dev.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Positive rate: {100*y_train.mean():.1f}%")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    # Get classifiers
    classifiers = get_classifiers()
    print(f"\nClassifiers: {list(classifiers.keys())}")

    # Results
    results = []

    print("\n" + "-" * 90)
    print("Training classifiers...")

    for clf_name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)

        y_pred_train = clf.predict(X_train_s)
        y_pred_dev = clf.predict(X_dev_s)
        y_pred_test = clf.predict(X_test_s)

        train_f1 = f1_score(y_train, y_pred_train)
        dev_f1 = f1_score(y_dev, y_pred_dev)
        test_f1 = f1_score(y_test, y_pred_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_p = precision_score(y_test, y_pred_test, zero_division=0)
        test_r = recall_score(y_test, y_pred_test, zero_division=0)

        results.append({
            "classifier": clf_name,
            "train_f1": train_f1,
            "dev_f1": dev_f1,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "test_p": test_p,
            "test_r": test_r,
            "gap": dev_f1 - test_f1,
        })

        print(f"  {clf_name}: Test F1={test_f1:.3f}, Acc={test_acc:.3f}")

    # Sort by test F1
    results.sort(key=lambda x: x["test_f1"], reverse=True)

    # Display results
    print("\n" + "=" * 90)
    print("RESULTS (sorted by Test F1)")
    print("=" * 90)

    print(f"\n{'Classifier':<12} {'Train F1':<10} {'Dev F1':<10} {'Test F1':<10} {'Test Acc':<10} {'P':<8} {'R':<8} {'Gap':<8}")
    print("-" * 90)

    for r in results:
        print(f"{r['classifier']:<12} {r['train_f1']:<10.3f} {r['dev_f1']:<10.3f} "
              f"{r['test_f1']:<10.3f} {r['test_acc']:<10.3f} {r['test_p']:<8.3f} {r['test_r']:<8.3f} {r['gap']:+.3f}")

    # Best result
    best = results[0]
    print(f"\n" + "=" * 90)
    print(f"🏆 BEST: {best['classifier']}")
    print("=" * 90)
    print(f"  Test F1:  {best['test_f1']:.4f}  (SOTA: {SOTA_F1}, Δ = {best['test_f1']-SOTA_F1:+.4f})")
    print(f"  Test Acc: {best['test_acc']:.4f}  (SOTA: {SOTA_ACC}, Δ = {best['test_acc']-SOTA_ACC:+.4f})")
    print(f"  Test P:   {best['test_p']:.4f}")
    print(f"  Test R:   {best['test_r']:.4f}")
    print(f"  Gap:      {best['gap']:+.4f}")

    # Confusion matrix for best
    best_clf = classifiers[best["classifier"]]
    y_pred_best = best_clf.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred_best)

    print(f"\n  Confusion Matrix:")
    print(f"                Pred No  Pred Yes")
    print(f"    True No       {cm[0,0]:5}     {cm[0,1]:5}")
    print(f"    True Yes      {cm[1,0]:5}     {cm[1,1]:5}")

    # Feature importance for best model
    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\n  Feature Importances ({best['classifier']}):")
        for i in indices:
            bar = "█" * int(importances[i] * 40 / importances[indices[0]])
            print(f"    {CLEAN_15_FEATURES[i]:<25} {importances[i]:.4f} {bar}")

    elif hasattr(best_clf, "coef_"):
        coefs = np.abs(best_clf.coef_[0])
        indices = np.argsort(coefs)[::-1]

        print(f"\n  Feature Coefficients ({best['classifier']}):")
        for i in indices:
            bar = "█" * int(coefs[i] * 40 / coefs[indices[0]])
            print(f"    {CLEAN_15_FEATURES[i]:<25} {coefs[i]:.4f} {bar}")

    # Comparison note
    print("\n" + "=" * 90)
    print("COMPARISON")
    print("=" * 90)
    print(f"\n  Clean 15 features:      Test F1 = {best['test_f1']:.3f}")
    print(f"  top_importance_15:      Test F1 = 0.711 (from earlier grid search)")
    print(f"  all_79 features:        Test F1 = 0.703 (from earlier grid search)")

    diff = best["test_f1"] - 0.711
    if diff > 0:
        print(f"\n  ✅ Clean 15 BEATS top_importance_15 by {diff:+.3f}!")
    elif diff > -0.02:
        print(f"\n  ≈ Clean 15 within 0.02 of top_importance_15 ({diff:+.3f})")
    else:
        print(f"\n  ❌ top_importance_15 still better by {-diff:.3f}")


if __name__ == "__main__":
    main()
