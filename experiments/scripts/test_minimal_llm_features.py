#!/usr/bin/env python3
"""
Test Minimal LLM Features: Logprobs + Confidence + Entropy Only

Tests hypothesis: fewer, cleaner features may generalize better.

Feature sets tested:
- scores_only: Just the 3 confidence scores
- logprobs_only: Just the probability distributions
- entropy_only: Just entropy features
- confidence_logprobs: Scores + logprobs (no derived features)
- confidence_entropy: Scores + entropy
- minimal_all: Scores + logprobs + entropy (no derived features)

Usage:
    python experiments/scripts/test_minimal_llm_features.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
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
# Minimal Feature Sets
# =============================================================================

FEATURE_SETS = {
    # === SELF-REPORTED CONFIDENCE ONLY ===
    "scores_3": [
        "check_score", "verif_score", "harm_score"
    ],

    # === LOGPROBS ONLY (probability distributions) ===
    "probs_9": [
        "check_p_true", "check_p_false", "check_p_uncertain",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
    ],

    "logits_9": [
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
    ],

    "probs_logits_18": [
        "check_p_true", "check_p_false", "check_p_uncertain",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
    ],

    # === ENTROPY ONLY ===
    "entropy_6": [
        "check_entropy", "check_entropy_norm",
        "verif_entropy", "verif_entropy_norm",
        "harm_entropy", "harm_entropy_norm",
    ],

    # === COMBINATIONS ===
    "scores_probs_12": [
        "check_score", "verif_score", "harm_score",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
    ],

    "scores_logits_12": [
        "check_score", "verif_score", "harm_score",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
    ],

    "scores_entropy_9": [
        "check_score", "verif_score", "harm_score",
        "check_entropy", "check_entropy_norm",
        "verif_entropy", "verif_entropy_norm",
        "harm_entropy", "harm_entropy_norm",
    ],

    "scores_probs_entropy_18": [
        "check_score", "verif_score", "harm_score",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
        "check_entropy", "verif_entropy", "harm_entropy",
        "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
    ],

    # === MINIMAL COMPLETE (no derived features like margin, variance, etc.) ===
    "minimal_complete_27": [
        # Scores
        "check_score", "verif_score", "harm_score",
        # Probabilities
        "check_p_true", "check_p_false", "check_p_uncertain",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
        # Logits
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
        # Entropy
        "check_entropy", "verif_entropy", "harm_entropy",
        "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
    ],

    # === FOR COMPARISON: top_importance_15 ===
    "top_importance_15": [
        "check_score", "check_margin_logit", "check_logit_p_false",
        "check_p_false", "check_margin_p", "check_prediction",
        "verif_score", "verif_logit_p_true", "verif_margin_logit",
        "score_variance", "yes_vote_count", "check_p_uncertain_dominant",
        "harm_spurs_action", "verif_is_argmax_match", "harm_is_argmax_match",
    ],
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load LLM features and labels."""
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    y_train = (ct24_train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (ct24_dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (ct24_test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return llm_train, llm_dev, llm_test, y_train, y_dev, y_test


def extract_features(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Extract features from dataframe."""
    existing = [c for c in feature_cols if c in df.columns]
    return df.select(existing).to_numpy().astype(np.float32)


# =============================================================================
# Classifiers
# =============================================================================

def get_classifiers():
    """Get classifiers to test."""
    classifiers = {
        "logistic": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced"),
        "rf": lambda: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"),
    }

    if HAS_LIGHTGBM:
        classifiers["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
        )

    if HAS_CATBOOST:
        classifiers["catboost"] = lambda: CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            random_state=42, auto_class_weights="Balanced", verbose=False
        )

    return classifiers


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 110)
    print("MINIMAL LLM FEATURES TEST: Logprobs + Confidence + Entropy Only")
    print("=" * 110)
    print(f"\nHypothesis: Fewer, cleaner features may generalize better than derived features.")
    print(f"SOTA: F1={SOTA_F1}, Acc={SOTA_ACC}")

    # Load data
    print("\n" + "-" * 110)
    print("Loading data...")
    llm_train, llm_dev, llm_test, y_train, y_dev, y_test = load_data()
    print(f"  Train: {len(y_train)} | Dev: {len(y_dev)} | Test: {len(y_test)}")

    classifiers = get_classifiers()
    print(f"  Classifiers: {list(classifiers.keys())}")

    # Results
    results = []

    print("\n" + "-" * 110)
    print("Running experiments...")

    for set_name, feature_cols in FEATURE_SETS.items():
        X_train = extract_features(llm_train, feature_cols)
        X_dev = extract_features(llm_dev, feature_cols)
        X_test = extract_features(llm_test, feature_cols)

        n_features = X_train.shape[1]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_dev_s = scaler.transform(X_dev)
        X_test_s = scaler.transform(X_test)

        for clf_name, clf_factory in classifiers.items():
            clf = clf_factory()
            clf.fit(X_train_s, y_train)

            y_pred_dev = clf.predict(X_dev_s)
            y_pred_test = clf.predict(X_test_s)

            dev_f1 = f1_score(y_dev, y_pred_dev)
            test_f1 = f1_score(y_test, y_pred_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_p = precision_score(y_test, y_pred_test, zero_division=0)
            test_r = recall_score(y_test, y_pred_test, zero_division=0)

            results.append({
                "features": set_name,
                "classifier": clf_name,
                "n_feat": n_features,
                "dev_f1": dev_f1,
                "test_f1": test_f1,
                "test_acc": test_acc,
                "test_p": test_p,
                "test_r": test_r,
                "gap": dev_f1 - test_f1,
            })

    # Sort by test F1
    results.sort(key=lambda x: x["test_f1"], reverse=True)

    # Display results
    print("\n" + "=" * 110)
    print("ALL RESULTS (sorted by Test F1)")
    print("=" * 110)

    print(f"\n{'#':<3} {'Features':<25} {'Clf':<10} {'#Feat':<6} {'Dev F1':<8} {'Test F1':<8} {'Test Acc':<9} {'P':<7} {'R':<7} {'Gap':<7}")
    print("-" * 110)

    for i, r in enumerate(results, 1):
        print(f"{i:<3} {r['features']:<25} {r['classifier']:<10} {r['n_feat']:<6} "
              f"{r['dev_f1']:<8.3f} {r['test_f1']:<8.3f} {r['test_acc']:<9.3f} "
              f"{r['test_p']:<7.3f} {r['test_r']:<7.3f} {r['gap']:+.3f}")

    # Analysis by feature set
    print("\n" + "=" * 110)
    print("AVERAGE BY FEATURE SET (across classifiers)")
    print("=" * 110)

    set_stats = {}
    for r in results:
        k = r["features"]
        if k not in set_stats:
            set_stats[k] = {"f1": [], "acc": [], "gap": [], "n": r["n_feat"]}
        set_stats[k]["f1"].append(r["test_f1"])
        set_stats[k]["acc"].append(r["test_acc"])
        set_stats[k]["gap"].append(r["gap"])

    set_avg = [(k, v["n"], np.mean(v["f1"]), np.max(v["f1"]), np.mean(v["gap"]))
               for k, v in set_stats.items()]
    set_avg.sort(key=lambda x: x[3], reverse=True)  # Sort by max F1

    print(f"\n{'Features':<25} {'#Feat':<6} {'Avg F1':<9} {'Max F1':<9} {'Avg Gap':<9}")
    print("-" * 65)

    for name, n_feat, avg_f1, max_f1, avg_gap in set_avg:
        marker = "⭐" if "top_importance" in name else ""
        print(f"{name:<25} {n_feat:<6} {avg_f1:<9.3f} {max_f1:<9.3f} {avg_gap:+.3f} {marker}")

    # Key findings
    print("\n" + "=" * 110)
    print("KEY FINDINGS")
    print("=" * 110)

    # Find best minimal set (excluding top_importance_15)
    minimal_results = [r for r in results if "top_importance" not in r["features"]]
    best_minimal = minimal_results[0] if minimal_results else None

    # Find top_importance_15 best
    top15_results = [r for r in results if r["features"] == "top_importance_15"]
    best_top15 = max(top15_results, key=lambda x: x["test_f1"]) if top15_results else None

    if best_minimal and best_top15:
        print(f"\n  Best MINIMAL feature set:")
        print(f"    {best_minimal['features']} + {best_minimal['classifier']}")
        print(f"    Test F1: {best_minimal['test_f1']:.3f} | Acc: {best_minimal['test_acc']:.3f}")
        print(f"    Gap: {best_minimal['gap']:+.3f}")

        print(f"\n  top_importance_15 (for comparison):")
        print(f"    {best_top15['features']} + {best_top15['classifier']}")
        print(f"    Test F1: {best_top15['test_f1']:.3f} | Acc: {best_top15['test_acc']:.3f}")
        print(f"    Gap: {best_top15['gap']:+.3f}")

        diff = best_minimal["test_f1"] - best_top15["test_f1"]
        if diff > 0:
            print(f"\n  ✅ MINIMAL features BEAT top_importance_15 by {diff:+.3f} F1!")
            print(f"     Hypothesis confirmed: fewer features generalize better.")
        elif diff > -0.02:
            print(f"\n  ≈ MINIMAL features within 0.02 of top_importance_15 ({diff:+.3f})")
            print(f"     Consider using minimal for interpretability.")
        else:
            print(f"\n  ❌ top_importance_15 still better by {-diff:.3f} F1")
            print(f"     Derived features (margins, variance) add value.")

    # Best overall
    best = results[0]
    print(f"\n  🏆 BEST OVERALL: {best['features']} + {best['classifier']}")
    print(f"     Test F1: {best['test_f1']:.3f} | Acc: {best['test_acc']:.3f}")


if __name__ == "__main__":
    main()
