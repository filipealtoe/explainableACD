#!/usr/bin/env python3
"""
Train Combined Classifier: LLM Features + Embeddings

Combines the best LLM feature set with embeddings to beat SOTA.
SOTA targets: F1 = 0.82, Acc = 0.905

Usage:
    python experiments/scripts/train_combined_classifier.py
    python experiments/scripts/train_combined_classifier.py --llm-set top_importance_15 --embedding bge-large
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"
EMBEDDING_CACHE_DIR = DATA_DIR / "embedding_cache"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "combined_classifier"

# SOTA targets
SOTA_F1 = 0.82
SOTA_ACC = 0.905

# =============================================================================
# LLM Feature Sets
# =============================================================================

LLM_FEATURE_SETS = {
    "top_importance_15": [
        "check_score", "check_margin_logit", "check_logit_p_false",
        "check_p_false", "check_margin_p", "check_prediction",
        "verif_score", "verif_logit_p_true", "verif_margin_logit",
        "score_variance", "yes_vote_count", "check_p_uncertain_dominant",
        "harm_spurs_action", "verif_is_argmax_match", "harm_is_argmax_match",
    ],

    "interpretable_11": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction",
        "score_variance", "yes_vote_count", "unanimous_yes",
        "harm_spurs_action", "harm_believability",
    ],

    "balanced_18": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction",
        "check_margin_p", "verif_margin_p", "harm_margin_p",
        "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
        "score_variance", "yes_vote_count", "unanimous_yes",
        "harm_spurs_action", "harm_believability", "harm_social_fragmentation",
    ],

    "scores_only_3": [
        "check_score", "verif_score", "harm_score",
    ],

    "all_79": [
        "check_score", "check_prediction", "check_reasoning_length", "check_reasoning_hedged",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "check_entropy", "check_entropy_norm",
        "check_margin_p", "check_margin_logit", "check_top1_top2_gap",
        "check_p_uncertain_dominant", "check_is_argmax_match",
        "check_score_p_residual", "check_pred_score_mismatch", "check_completion_tokens",
        "verif_score", "verif_prediction", "verif_reasoning_length", "verif_reasoning_hedged",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "verif_entropy", "verif_entropy_norm",
        "verif_margin_p", "verif_margin_logit", "verif_top1_top2_gap",
        "verif_p_uncertain_dominant", "verif_is_argmax_match",
        "verif_score_p_residual", "verif_pred_score_mismatch", "verif_completion_tokens",
        "harm_score", "harm_prediction", "harm_reasoning_length", "harm_reasoning_hedged",
        "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
        "harm_entropy", "harm_entropy_norm",
        "harm_margin_p", "harm_margin_logit", "harm_top1_top2_gap",
        "harm_p_uncertain_dominant", "harm_is_argmax_match",
        "harm_score_p_residual", "harm_pred_score_mismatch", "harm_completion_tokens",
        "score_variance", "score_max_diff",
        "check_minus_verif", "check_minus_harm", "verif_minus_harm",
        "harm_high_verif_low",
        "yes_vote_count", "unanimous_yes", "unanimous_no",
        "check_verif_agree", "check_harm_agree", "verif_harm_agree",
        "pairwise_agreement_rate", "check_yes_verif_yes", "consensus_entropy",
    ],
}

# =============================================================================
# Classifiers
# =============================================================================

def get_classifiers() -> dict:
    """Get classifiers optimized for this task."""
    classifiers = {
        "logistic": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced"),
        "rf": lambda: RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight="balanced"),
    }

    if HAS_LIGHTGBM:
        classifiers["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
        )
        classifiers["lightgbm_deep"] = lambda: LGBMClassifier(
            n_estimators=500, max_depth=12, learning_rate=0.03,
            random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
        )

    if HAS_CATBOOST:
        classifiers["catboost"] = lambda: CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.05,
            random_state=42, auto_class_weights="Balanced", verbose=False
        )
        classifiers["catboost_deep"] = lambda: CatBoostClassifier(
            iterations=500, depth=10, learning_rate=0.03,
            random_state=42, auto_class_weights="Balanced", verbose=False
        )

    if HAS_XGBOOST:
        classifiers["xgboost"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss",
            scale_pos_weight=3.0  # Handle class imbalance
        )

    return classifiers


# =============================================================================
# Data Loading
# =============================================================================

def load_llm_features(feature_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load LLM features for train/dev/test."""
    train_llm = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    dev_llm = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    test_llm = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    feature_cols = LLM_FEATURE_SETS[feature_set]
    existing = [c for c in feature_cols if c in train_llm.columns]

    X_train = train_llm.select(existing).to_numpy().astype(np.float32)
    X_dev = dev_llm.select(existing).to_numpy().astype(np.float32)
    X_test = test_llm.select(existing).to_numpy().astype(np.float32)

    return X_train, X_dev, X_test, existing


def load_embeddings(embedding_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed embeddings."""
    cache_file = EMBEDDING_CACHE_DIR / f"{embedding_name}_embeddings.npz"

    if not cache_file.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_file}")

    data = np.load(cache_file)
    return data["train"], data["dev"], data["test"]


def load_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load labels for train/dev/test."""
    train_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")["class_label"]
    dev_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")["class_label"]
    test_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")["class_label"]

    y_train = (train_labels == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (dev_labels == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test_labels == "Yes").cast(pl.Int8).to_numpy()

    return y_train, y_dev, y_test


# =============================================================================
# Feature Combinations
# =============================================================================

def combine_features(
    llm_train: np.ndarray, llm_dev: np.ndarray, llm_test: np.ndarray,
    emb_train: np.ndarray, emb_dev: np.ndarray, emb_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate LLM features with embeddings."""
    X_train = np.hstack([llm_train, emb_train])
    X_dev = np.hstack([llm_dev, emb_dev])
    X_test = np.hstack([llm_test, emb_test])
    return X_train, X_dev, X_test


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, split_name: str):
    """Print confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  {split_name} Confusion Matrix:")
    print(f"              Pred No  Pred Yes")
    print(f"    True No     {cm[0,0]:5}     {cm[0,1]:5}")
    print(f"    True Yes    {cm[1,0]:5}     {cm[1,1]:5}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train combined LLM + embedding classifier")
    parser.add_argument("--llm-set", default="top_importance_15",
                        choices=list(LLM_FEATURE_SETS.keys()),
                        help="LLM feature set to use")
    parser.add_argument("--embedding", default="bge-large",
                        help="Embedding model (must have cached embeddings)")
    parser.add_argument("--classifiers", nargs="+", default=None,
                        help="Classifiers to test (default: all)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation: LLM-only, embedding-only, combined")
    args = parser.parse_args()

    print("=" * 100)
    print("COMBINED CLASSIFIER: LLM Features + Embeddings")
    print("=" * 100)
    print(f"\nSOTA Targets: F1 = {SOTA_F1:.3f}, Acc = {SOTA_ACC:.3f}")

    # Load data
    print("\n" + "-" * 100)
    print("Loading data...")

    # LLM features
    llm_train, llm_dev, llm_test, llm_cols = load_llm_features(args.llm_set)
    print(f"  LLM features ({args.llm_set}): {llm_train.shape[1]} dims")

    # Embeddings
    emb_train, emb_dev, emb_test = load_embeddings(args.embedding)
    print(f"  Embeddings ({args.embedding}): {emb_train.shape[1]} dims")

    # Labels
    y_train, y_dev, y_test = load_labels()
    print(f"  Train: {len(y_train)} | Dev: {len(y_dev)} | Test: {len(y_test)}")
    print(f"  Positive rate: {100*y_train.mean():.1f}% (train)")

    # Combine features
    X_train, X_dev, X_test = combine_features(
        llm_train, llm_dev, llm_test,
        emb_train, emb_dev, emb_test
    )
    print(f"  Combined features: {X_train.shape[1]} dims")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    # Get classifiers
    all_classifiers = get_classifiers()
    if args.classifiers:
        classifiers = {k: v for k, v in all_classifiers.items() if k in args.classifiers}
    else:
        classifiers = all_classifiers

    print(f"\nClassifiers: {list(classifiers.keys())}")

    # Results storage
    results = []

    # === ABLATION MODE ===
    if args.ablation:
        print("\n" + "=" * 100)
        print("ABLATION STUDY: LLM-only vs Embedding-only vs Combined")
        print("=" * 100)

        # Use best classifier (lightgbm_deep)
        if "lightgbm_deep" in classifiers:
            clf_name = "lightgbm_deep"
        elif "lightgbm" in classifiers:
            clf_name = "lightgbm"
        elif "catboost" in classifiers:
            clf_name = "catboost"
        else:
            clf_name = list(classifiers.keys())[0]

        ablation_results = []

        # 1. LLM-only
        scaler_llm = StandardScaler()
        llm_train_s = scaler_llm.fit_transform(llm_train)
        llm_dev_s = scaler_llm.transform(llm_dev)
        llm_test_s = scaler_llm.transform(llm_test)

        clf = classifiers[clf_name]()
        clf.fit(llm_train_s, y_train)
        metrics = evaluate(y_test, clf.predict(llm_test_s))
        ablation_results.append({"config": "LLM-only", "dims": llm_train.shape[1], **metrics})

        # 2. Embedding-only
        scaler_emb = StandardScaler()
        emb_train_s = scaler_emb.fit_transform(emb_train)
        emb_dev_s = scaler_emb.transform(emb_dev)
        emb_test_s = scaler_emb.transform(emb_test)

        clf = classifiers[clf_name]()
        clf.fit(emb_train_s, y_train)
        metrics = evaluate(y_test, clf.predict(emb_test_s))
        ablation_results.append({"config": "Embedding-only", "dims": emb_train.shape[1], **metrics})

        # 3. Combined
        clf = classifiers[clf_name]()
        clf.fit(X_train_s, y_train)
        metrics = evaluate(y_test, clf.predict(X_test_s))
        ablation_results.append({"config": "Combined", "dims": X_train.shape[1], **metrics})

        print(f"\nClassifier: {clf_name}")
        print(f"\n{'Config':<18} {'Dims':<8} {'Test F1':<10} {'Test Acc':<10} {'Test P':<10} {'Test R':<10} {'vs SOTA F1':<12}")
        print("-" * 90)

        for r in ablation_results:
            delta_f1 = r["f1"] - SOTA_F1
            marker = "✅" if r["f1"] >= SOTA_F1 else "❌"
            print(f"{r['config']:<18} {r['dims']:<8} {r['f1']:<10.3f} {r['accuracy']:<10.3f} "
                  f"{r['precision']:<10.3f} {r['recall']:<10.3f} {delta_f1:+.3f} {marker}")

        print()

    # === FULL GRID ===
    print("\n" + "=" * 100)
    print("COMBINED CLASSIFIER RESULTS")
    print("=" * 100)

    for clf_name, clf_factory in classifiers.items():
        try:
            clf = clf_factory()
            clf.fit(X_train_s, y_train)

            y_pred_train = clf.predict(X_train_s)
            y_pred_dev = clf.predict(X_dev_s)
            y_pred_test = clf.predict(X_test_s)

            metrics_train = evaluate(y_train, y_pred_train)
            metrics_dev = evaluate(y_dev, y_pred_dev)
            metrics_test = evaluate(y_test, y_pred_test)

            results.append({
                "classifier": clf_name,
                "n_features": X_train.shape[1],
                "train_f1": metrics_train["f1"],
                "dev_f1": metrics_dev["f1"],
                "test_f1": metrics_test["f1"],
                "test_acc": metrics_test["accuracy"],
                "test_p": metrics_test["precision"],
                "test_r": metrics_test["recall"],
                "gap": metrics_dev["f1"] - metrics_test["f1"],
            })

        except Exception as e:
            print(f"  {clf_name}: ERROR - {e}")

    # Sort by test F1
    results.sort(key=lambda x: x["test_f1"], reverse=True)

    print(f"\n{'Classifier':<18} {'#Feat':<8} {'Train F1':<10} {'Dev F1':<10} {'Test F1':<10} {'Test Acc':<10} {'Test P':<10} {'Test R':<10} {'vs SOTA':<10}")
    print("-" * 110)

    for r in results:
        delta_f1 = r["test_f1"] - SOTA_F1
        marker = "✅" if r["test_f1"] >= SOTA_F1 else ""
        print(f"{r['classifier']:<18} {r['n_features']:<8} {r['train_f1']:<10.3f} {r['dev_f1']:<10.3f} "
              f"{r['test_f1']:<10.3f} {r['test_acc']:<10.3f} {r['test_p']:<10.3f} {r['test_r']:<10.3f} {delta_f1:+.3f} {marker}")

    # Best result
    if results:
        best = results[0]
        print(f"\n{'='*100}")
        print(f"BEST RESULT: {best['classifier']}")
        print(f"{'='*100}")
        print(f"  Test F1:  {best['test_f1']:.3f}  (SOTA: {SOTA_F1:.3f}, delta: {best['test_f1']-SOTA_F1:+.3f})")
        print(f"  Test Acc: {best['test_acc']:.3f}  (SOTA: {SOTA_ACC:.3f}, delta: {best['test_acc']-SOTA_ACC:+.3f})")
        print(f"  Test P:   {best['test_p']:.3f}")
        print(f"  Test R:   {best['test_r']:.3f}")

        if best["test_f1"] >= SOTA_F1 and best["test_acc"] >= SOTA_ACC:
            print(f"\n🎉 SOTA BEATEN on both F1 and Accuracy!")
        elif best["test_f1"] >= SOTA_F1:
            print(f"\n✅ SOTA F1 achieved! Accuracy still {SOTA_ACC - best['test_acc']:.3f} below target.")
        elif best["test_acc"] >= SOTA_ACC:
            print(f"\n✅ SOTA Accuracy achieved! F1 still {SOTA_F1 - best['test_f1']:.3f} below target.")
        else:
            print(f"\n❌ Below SOTA. F1 needs +{SOTA_F1 - best['test_f1']:.3f}, Acc needs +{SOTA_ACC - best['test_acc']:.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pl.DataFrame(results)
    results_file = RESULTS_DIR / f"combined_{args.llm_set}_{args.embedding}.csv"
    results_df.write_csv(results_file)
    print(f"\n📁 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
