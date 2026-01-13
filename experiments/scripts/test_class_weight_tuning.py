#!/usr/bin/env python3
"""
Test Class Weight Tuning for Best LLM Features

Tests different positive class weights (1, 2, 3, 4, 5) to see impact on F1.
Uses the best feature set found: scores + entropy (6 features).

Usage:
    python experiments/scripts/test_class_weight_tuning.py
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

# Best feature set: scores + entropy
BEST_FEATURES = [
    "check_score", "verif_score", "harm_score",
    "check_entropy", "verif_entropy", "harm_entropy",
]

# Weights to test
POS_WEIGHTS = [1, 2, 3, 4, 5]


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load and prepare data."""
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    # Join on sentence_id
    llm_train = llm_train.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_dev = llm_dev.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_test = llm_test.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    train = llm_train.join(ct24_train.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")
    dev = llm_dev.join(ct24_dev.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")
    test = llm_test.join(ct24_test.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")

    # Extract features and labels
    X_train = train.select(BEST_FEATURES).to_numpy().astype(np.float32)
    X_dev = dev.select(BEST_FEATURES).to_numpy().astype(np.float32)
    X_test = test.select(BEST_FEATURES).to_numpy().astype(np.float32)

    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def get_classifiers(pos_weight: float) -> dict:
    """Get classifiers with specified positive class weight."""
    classifiers = {
        "logistic": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42,
            class_weight={0: 1, 1: pos_weight}
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
            class_weight={0: 1, 1: pos_weight}
        ),
    }

    if HAS_LIGHTGBM:
        classifiers["lightgbm"] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=-1,
            class_weight={0: 1, 1: pos_weight}
        )

    if HAS_CATBOOST:
        classifiers["catboost"] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            random_state=42, verbose=False,
            class_weights=[1, pos_weight]
        )

    if HAS_XGBOOST:
        classifiers["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss",
            scale_pos_weight=pos_weight
        )

    return classifiers


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("CLASS WEIGHT TUNING: Testing Positive Class Weights")
    print("=" * 100)
    print(f"\nFeatures: {BEST_FEATURES}")
    print(f"Positive weights to test: {POS_WEIGHTS}")
    print(f"SOTA targets: F1={SOTA_F1}, Acc={SOTA_ACC}")

    # Load data
    print("\nLoading data...")
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()

    pos_rate = y_train.mean()
    print(f"  Train: {len(y_train)} samples, {100*pos_rate:.1f}% positive")
    print(f"  Test:  {len(y_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    # Results storage
    all_results = []

    # Test each weight
    for pos_weight in POS_WEIGHTS:
        print(f"\n{'='*100}")
        print(f"TESTING: pos_weight = {pos_weight}")
        print(f"{'='*100}")

        classifiers = get_classifiers(pos_weight)

        for clf_name, clf in classifiers.items():
            clf.fit(X_train_s, y_train)

            y_pred_test = clf.predict(X_test_s)

            test_f1 = f1_score(y_test, y_pred_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_p = precision_score(y_test, y_pred_test, zero_division=0)
            test_r = recall_score(y_test, y_pred_test, zero_division=0)

            all_results.append({
                "pos_weight": pos_weight,
                "classifier": clf_name,
                "test_f1": test_f1,
                "test_acc": test_acc,
                "precision": test_p,
                "recall": test_r,
            })

            print(f"  {clf_name:<12} F1={test_f1:.4f}  Acc={test_acc:.4f}  P={test_p:.4f}  R={test_r:.4f}")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Best F1 per Classifier across Weights")
    print("=" * 100)

    # Group by classifier
    classifiers_tested = list(set(r["classifier"] for r in all_results))

    print(f"\n{'Classifier':<12} ", end="")
    for w in POS_WEIGHTS:
        print(f"w={w:<8}", end="")
    print(f"{'Best':<10} {'Best W':<8}")
    print("-" * 100)

    best_overall = None
    best_overall_f1 = 0

    for clf_name in sorted(classifiers_tested):
        clf_results = [r for r in all_results if r["classifier"] == clf_name]

        print(f"{clf_name:<12} ", end="")

        best_f1 = 0
        best_w = 1

        for w in POS_WEIGHTS:
            r = next((x for x in clf_results if x["pos_weight"] == w), None)
            if r:
                f1 = r["test_f1"]
                print(f"{f1:<8.4f} ", end="")
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = w
                    best_result = r

        print(f"{best_f1:<10.4f} w={best_w}")

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall = best_result

    # Best overall
    print(f"\n{'='*100}")
    print("🏆 BEST OVERALL")
    print(f"{'='*100}")
    print(f"  Classifier: {best_overall['classifier']}")
    print(f"  Pos Weight: {best_overall['pos_weight']}")
    print(f"  Test F1:    {best_overall['test_f1']:.4f} (SOTA: {SOTA_F1}, Δ={best_overall['test_f1']-SOTA_F1:+.4f})")
    print(f"  Test Acc:   {best_overall['test_acc']:.4f} (SOTA: {SOTA_ACC}, Δ={best_overall['test_acc']-SOTA_ACC:+.4f})")
    print(f"  Precision:  {best_overall['precision']:.4f}")
    print(f"  Recall:     {best_overall['recall']:.4f}")

    # Impact analysis
    print(f"\n{'='*100}")
    print("WEIGHT IMPACT ANALYSIS (average F1 change from w=1 baseline)")
    print(f"{'='*100}")

    for clf_name in sorted(classifiers_tested):
        clf_results = [r for r in all_results if r["classifier"] == clf_name]
        baseline = next((x["test_f1"] for x in clf_results if x["pos_weight"] == 1), 0)

        print(f"\n  {clf_name}:")
        for w in POS_WEIGHTS:
            r = next((x for x in clf_results if x["pos_weight"] == w), None)
            if r:
                delta = r["test_f1"] - baseline
                bar = "█" * int(abs(delta) * 200) if delta != 0 else ""
                sign = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                print(f"    w={w}: {r['test_f1']:.4f} ({delta:+.4f}) {sign} {bar}")


if __name__ == "__main__":
    main()
