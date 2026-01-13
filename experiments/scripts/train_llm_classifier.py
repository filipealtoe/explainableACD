#!/usr/bin/env python3
"""
Train Classifiers on LLM Features for Checkworthiness Detection

Tests various classifiers on the 79 LLM-extracted features.

Usage:
    python experiments/scripts/train_llm_classifier.py
    python experiments/scripts/train_llm_classifier.py --include-flags
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
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
RESULTS_DIR = Path(__file__).parent.parent / "results" / "llm_classifier"

# Metadata columns (exclude from features)
METADATA_COLS = ["sentence_id"]

# Flag columns (optional, can be included or excluded)
FLAG_COLS = [
    "check_parse_issue", "check_pred_derived",
    "verif_parse_issue", "verif_pred_derived",
    "harm_parse_issue", "harm_pred_derived", "harm_subscore_missing",
    "row_has_parse_issues", "row_has_uncertain_pred"
]

# Core LLM features (79 columns)
CORE_LLM_FEATURES = [
    # Checkability module (20 features)
    "check_score", "check_prediction", "check_reasoning_length", "check_reasoning_hedged",
    "check_p_true", "check_p_false", "check_p_uncertain",
    "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
    "check_entropy", "check_entropy_norm",
    "check_margin_p", "check_margin_logit", "check_top1_top2_gap",
    "check_p_uncertain_dominant", "check_is_argmax_match",
    "check_score_p_residual", "check_pred_score_mismatch", "check_completion_tokens",

    # Verifiability module (20 features)
    "verif_score", "verif_prediction", "verif_reasoning_length", "verif_reasoning_hedged",
    "verif_p_true", "verif_p_false", "verif_p_uncertain",
    "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
    "verif_entropy", "verif_entropy_norm",
    "verif_margin_p", "verif_margin_logit", "verif_top1_top2_gap",
    "verif_p_uncertain_dominant", "verif_is_argmax_match",
    "verif_score_p_residual", "verif_pred_score_mismatch", "verif_completion_tokens",

    # Harm module (24 features)
    "harm_score", "harm_prediction", "harm_reasoning_length", "harm_reasoning_hedged",
    "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
    "harm_p_true", "harm_p_false", "harm_p_uncertain",
    "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
    "harm_entropy", "harm_entropy_norm",
    "harm_margin_p", "harm_margin_logit", "harm_top1_top2_gap",
    "harm_p_uncertain_dominant", "harm_is_argmax_match",
    "harm_score_p_residual", "harm_pred_score_mismatch", "harm_completion_tokens",

    # Cross-module features (15 features)
    "score_variance", "score_max_diff",
    "check_minus_verif", "check_minus_harm", "verif_minus_harm",
    "harm_high_verif_low",
    "yes_vote_count", "unanimous_yes", "unanimous_no",
    "check_verif_agree", "check_harm_agree", "verif_harm_agree",
    "pairwise_agreement_rate", "check_yes_verif_yes", "consensus_entropy",
]


# =============================================================================
# Data Loading
# =============================================================================

def load_data(include_flags: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str]]:
    """Load train/dev/test LLM features and labels."""

    # Load LLM features
    train_llm = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    dev_llm = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    test_llm = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    # Load labels from original features
    train_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet").select(["Sentence_id", "class_label"])
    dev_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet").select(["Sentence_id", "class_label"])
    test_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet").select(["Sentence_id", "class_label"])

    # Join labels (handle case sensitivity)
    train_llm = train_llm.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    dev_llm = dev_llm.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    test_llm = test_llm.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    train = train_llm.join(train_labels, on="Sentence_id", how="left")
    dev = dev_llm.join(dev_labels, on="Sentence_id", how="left")
    test = test_llm.join(test_labels, on="Sentence_id", how="left")

    # Determine feature columns
    feature_cols = CORE_LLM_FEATURES.copy()
    if include_flags:
        feature_cols.extend(FLAG_COLS)

    # Filter to columns that exist
    feature_cols = [c for c in feature_cols if c in train.columns]

    return train, dev, test, feature_cols


def prepare_data(df: pl.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels."""
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = (df["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    return X, y


# =============================================================================
# Classifiers
# =============================================================================

def get_classifier(name: str):
    """Get classifier by name."""
    if name == "logistic":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced")
    elif name == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
    elif name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="logloss")
    elif name == "catboost":
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not installed")
        return CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, auto_class_weights="Balanced", verbose=False)
    elif name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        return LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {name}")


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


def get_feature_importances(clf, feature_names: list[str], top_n: int = 20) -> list[tuple[str, float]]:
    """Get top feature importances."""
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return []

    indices = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], importances[i]) for i in indices]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train classifiers on LLM features")
    parser.add_argument("--classifiers", nargs="+",
                        default=["logistic", "rf", "xgboost", "catboost", "lightgbm"],
                        choices=["logistic", "rf", "xgboost", "catboost", "lightgbm"],
                        help="Classifiers to test")
    parser.add_argument("--include-flags", action="store_true",
                        help="Include parse/uncertainty flag features")
    parser.add_argument("--show-importances", action="store_true",
                        help="Show feature importances for each classifier")
    args = parser.parse_args()

    # Check classifier availability
    if "xgboost" in args.classifiers and not HAS_XGBOOST:
        args.classifiers.remove("xgboost")
        print("Warning: XGBoost not available")
    if "catboost" in args.classifiers and not HAS_CATBOOST:
        args.classifiers.remove("catboost")
        print("Warning: CatBoost not available")
    if "lightgbm" in args.classifiers and not HAS_LIGHTGBM:
        args.classifiers.remove("lightgbm")
        print("Warning: LightGBM not available")

    print("=" * 70)
    print("LLM FEATURES CLASSIFIER TRAINING")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_df, dev_df, test_df, feature_cols = load_data(include_flags=args.include_flags)

    X_train, y_train = prepare_data(train_df, feature_cols)
    X_dev, y_dev = prepare_data(dev_df, feature_cols)
    X_test, y_test = prepare_data(test_df, feature_cols)

    print(f"  Train: {len(y_train)} samples, {len(feature_cols)} features")
    print(f"  Dev:   {len(y_dev)} samples")
    print(f"  Test:  {len(y_test)} samples")
    print(f"  Positive rate: {100*y_train.mean():.1f}% (train)")
    print(f"  Include flags: {args.include_flags}")

    # Results storage
    results = []

    # Train each classifier
    for clf_name in args.classifiers:
        print(f"\n{'='*70}")
        print(f"Classifier: {clf_name.upper()}")
        print(f"{'='*70}")

        try:
            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_dev_s = scaler.transform(X_dev)
            X_test_s = scaler.transform(X_test)

            # Train
            clf = get_classifier(clf_name)
            clf.fit(X_train_s, y_train)

            # Predict
            y_pred_train = clf.predict(X_train_s)
            y_pred_dev = clf.predict(X_dev_s)
            y_pred_test = clf.predict(X_test_s)

            # Evaluate
            metrics_train = evaluate(y_train, y_pred_train)
            metrics_dev = evaluate(y_dev, y_pred_dev)
            metrics_test = evaluate(y_test, y_pred_test)

            print(f"\n  Results:")
            print(f"  {'Split':<6} | {'Acc':<8} {'P':<8} {'R':<8} {'F1':<8}")
            print(f"  {'-'*42}")
            print(f"  {'Train':<6} | {metrics_train['accuracy']:<8.3f} {metrics_train['precision']:<8.3f} {metrics_train['recall']:<8.3f} {metrics_train['f1']:<8.3f}")
            print(f"  {'Dev':<6} | {metrics_dev['accuracy']:<8.3f} {metrics_dev['precision']:<8.3f} {metrics_dev['recall']:<8.3f} {metrics_dev['f1']:<8.3f}")
            print(f"  {'Test':<6} | {metrics_test['accuracy']:<8.3f} {metrics_test['precision']:<8.3f} {metrics_test['recall']:<8.3f} {metrics_test['f1']:<8.3f}")

            print_confusion_matrix(y_test, y_pred_test, "Test")

            # Feature importances
            if args.show_importances:
                importances = get_feature_importances(clf, feature_cols, top_n=15)
                if importances:
                    print(f"\n  Top 15 Feature Importances:")
                    max_imp = importances[0][1]
                    for name, imp in importances:
                        bar = "█" * int(imp * 40 / max_imp) if max_imp > 0 else ""
                        print(f"    {name:35} {imp:.4f} {bar}")

            results.append({
                "classifier": clf_name,
                "n_features": len(feature_cols),
                "include_flags": args.include_flags,
                **{f"train_{k}": v for k, v in metrics_train.items()},
                **{f"dev_{k}": v for k, v in metrics_dev.items()},
                **{f"test_{k}": v for k, v in metrics_test.items()},
            })

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY (sorted by Test F1)")
    print("=" * 100)

    results.sort(key=lambda x: x["test_f1"], reverse=True)

    print(f"\n{'Classifier':<12} {'#Feat':<6} | {'Train F1':<9} {'Dev F1':<9} | {'Test F1':<9} {'Test Acc':<9} {'Test P':<9} {'Test R':<9}")
    print("-" * 90)

    for r in results:
        print(f"{r['classifier']:<12} {r['n_features']:<6} | "
              f"{r['train_f1']:<9.3f} {r['dev_f1']:<9.3f} | "
              f"{r['test_f1']:<9.3f} {r['test_accuracy']:<9.3f} {r['test_precision']:<9.3f} {r['test_recall']:<9.3f}")

    # Best result
    if results:
        best = results[0]
        print(f"\n✅ Best: {best['classifier']} → Test F1: {best['test_f1']:.3f} | Test Acc: {best['test_accuracy']:.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pl.DataFrame(results)
    results_file = RESULTS_DIR / "llm_classifier_results.csv"
    results_df.write_csv(results_file)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
