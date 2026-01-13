#!/usr/bin/env python3
"""
Test Class Weight Tuning with BGE-Large Embeddings

Tests different positive class weights (1-5) with BGE-large embeddings (1024 dim).
Previous best: bge-large + logistic = F1 0.749

Usage:
    python experiments/scripts/test_bge_class_weight.py
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
EMBEDDING_FILE = DATA_DIR / "embedding_cache" / "bge-large_embeddings.npz"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"

SOTA_F1 = 0.82
SOTA_ACC = 0.905
PREVIOUS_BEST_F1 = 0.749  # bge-large + logistic (no weight tuning)

# Weights to test
POS_WEIGHTS = [1, 2, 3, 4, 5]


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load BGE-large embeddings and labels."""
    # Load embeddings
    embed_data = np.load(EMBEDDING_FILE)
    X_train = embed_data["train"].astype(np.float32)
    X_dev = embed_data["dev"].astype(np.float32)
    X_test = embed_data["test"].astype(np.float32)

    # Load labels
    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    y_train = (ct24_train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (ct24_dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (ct24_test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def get_classifiers(pos_weight: float) -> dict:
    """Get classifiers with specified positive class weight."""
    classifiers = {
        "logistic": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42,
            class_weight={0: 1, 1: pos_weight}
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
            class_weight={0: 1, 1: pos_weight}
        ),
    }

    if HAS_LIGHTGBM:
        classifiers["lightgbm"] = LGBMClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=-1,
            class_weight={0: 1, 1: pos_weight}
        )

    if HAS_CATBOOST:
        classifiers["catboost"] = CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.05,
            random_state=42, verbose=False,
            class_weights=[1, pos_weight]
        )

    if HAS_XGBOOST:
        classifiers["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss",
            scale_pos_weight=pos_weight
        )

    return classifiers


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("CLASS WEIGHT TUNING: BGE-Large Embeddings (1024 dim)")
    print("=" * 100)
    print(f"\nPrevious best (no weight tuning): F1 = {PREVIOUS_BEST_F1}")
    print(f"Positive weights to test: {POS_WEIGHTS}")
    print(f"SOTA targets: F1={SOTA_F1}, Acc={SOTA_ACC}")

    # Load data
    print("\nLoading data...")
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()

    pos_rate = y_train.mean()
    print(f"  Embeddings: {X_train.shape[1]} dimensions")
    print(f"  Train: {len(y_train)} samples, {100*pos_rate:.1f}% positive")
    print(f"  Dev:   {len(y_dev)} samples")
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
            print(f"  Training {clf_name}...", end=" ", flush=True)
            clf.fit(X_train_s, y_train)

            y_pred_dev = clf.predict(X_dev_s)
            y_pred_test = clf.predict(X_test_s)

            dev_f1 = f1_score(y_dev, y_pred_dev)
            test_f1 = f1_score(y_test, y_pred_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_p = precision_score(y_test, y_pred_test, zero_division=0)
            test_r = recall_score(y_test, y_pred_test, zero_division=0)

            all_results.append({
                "pos_weight": pos_weight,
                "classifier": clf_name,
                "dev_f1": dev_f1,
                "test_f1": test_f1,
                "test_acc": test_acc,
                "precision": test_p,
                "recall": test_r,
            })

            delta = test_f1 - PREVIOUS_BEST_F1
            marker = "🔥" if test_f1 > PREVIOUS_BEST_F1 else ""
            print(f"F1={test_f1:.4f} (Δ={delta:+.4f}) Acc={test_acc:.4f} P={test_p:.4f} R={test_r:.4f} {marker}")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Test F1 per Classifier across Weights")
    print("=" * 100)

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
        best_result = None

        for w in POS_WEIGHTS:
            r = next((x for x in clf_results if x["pos_weight"] == w), None)
            if r:
                f1 = r["test_f1"]
                marker = "*" if f1 > PREVIOUS_BEST_F1 else " "
                print(f"{f1:<7.4f}{marker} ", end="")
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = w
                    best_result = r

        beat_prev = "🔥" if best_f1 > PREVIOUS_BEST_F1 else ""
        print(f"{best_f1:<10.4f} w={best_w} {beat_prev}")

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall = best_result

    # Best overall
    print(f"\n{'='*100}")
    print("🏆 BEST OVERALL")
    print(f"{'='*100}")
    print(f"  Classifier: {best_overall['classifier']}")
    print(f"  Pos Weight: {best_overall['pos_weight']}")
    print(f"  Dev F1:     {best_overall['dev_f1']:.4f}")
    print(f"  Test F1:    {best_overall['test_f1']:.4f}")
    print(f"  Test Acc:   {best_overall['test_acc']:.4f}")
    print(f"  Precision:  {best_overall['precision']:.4f}")
    print(f"  Recall:     {best_overall['recall']:.4f}")

    print(f"\n  Previous best (logistic, no weighting): F1 = {PREVIOUS_BEST_F1}")
    print(f"  Improvement: {best_overall['test_f1'] - PREVIOUS_BEST_F1:+.4f}")
    print(f"\n  SOTA: F1 = {SOTA_F1}")
    print(f"  Gap to SOTA: {best_overall['test_f1'] - SOTA_F1:+.4f}")

    # Weight impact analysis
    print(f"\n{'='*100}")
    print("WEIGHT IMPACT ANALYSIS")
    print(f"{'='*100}")

    for clf_name in sorted(classifiers_tested):
        clf_results = [r for r in all_results if r["classifier"] == clf_name]
        baseline = next((x["test_f1"] for x in clf_results if x["pos_weight"] == 1), 0)

        print(f"\n  {clf_name}:")
        for w in POS_WEIGHTS:
            r = next((x for x in clf_results if x["pos_weight"] == w), None)
            if r:
                delta = r["test_f1"] - baseline
                bar = "█" * int(abs(delta) * 100) if delta != 0 else ""
                sign = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                beat = "🔥" if r["test_f1"] > PREVIOUS_BEST_F1 else ""
                print(f"    w={w}: F1={r['test_f1']:.4f} ({delta:+.4f}) {sign} {bar} {beat}")


if __name__ == "__main__":
    main()
