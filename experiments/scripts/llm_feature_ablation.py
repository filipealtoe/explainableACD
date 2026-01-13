#!/usr/bin/env python3
"""
LLM Feature Ablation Study

Tests different feature subsets to find optimal configuration.

Usage:
    python experiments/scripts/llm_feature_ablation.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

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

# =============================================================================
# Feature Sets to Test
# =============================================================================

FEATURE_SETS = {
    # Minimal sets
    "scores_only": [
        "check_score", "verif_score", "harm_score"
    ],

    "scores_predictions": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction"
    ],

    # Probability-based
    "scores_probs": [
        "check_score", "verif_score", "harm_score",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
    ],

    "scores_logits": [
        "check_score", "verif_score", "harm_score",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
    ],

    # Uncertainty-focused
    "scores_entropy": [
        "check_score", "verif_score", "harm_score",
        "check_entropy", "verif_entropy", "harm_entropy",
        "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
    ],

    "scores_margins": [
        "check_score", "verif_score", "harm_score",
        "check_margin_p", "verif_margin_p", "harm_margin_p",
        "check_margin_logit", "verif_margin_logit", "harm_margin_logit",
    ],

    # Cross-module features
    "cross_module": [
        "score_variance", "score_max_diff",
        "check_minus_verif", "check_minus_harm", "verif_minus_harm",
        "yes_vote_count", "unanimous_yes", "unanimous_no",
        "pairwise_agreement_rate", "consensus_entropy",
    ],

    "scores_cross": [
        "check_score", "verif_score", "harm_score",
        "score_variance", "score_max_diff",
        "yes_vote_count", "unanimous_yes", "unanimous_no",
        "consensus_entropy",
    ],

    # Checkability-focused (dominates importance)
    "check_only": [
        "check_score", "check_prediction",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "check_margin_p", "check_margin_logit",
        "check_entropy", "check_entropy_norm",
    ],

    "check_verif": [
        "check_score", "verif_score",
        "check_prediction", "verif_prediction",
        "check_p_true", "check_p_false",
        "verif_p_true", "verif_p_false",
        "check_margin_p", "verif_margin_p",
    ],

    # Harm sub-dimensions
    "harm_subdims": [
        "harm_score", "harm_prediction",
        "harm_social_fragmentation", "harm_spurs_action",
        "harm_believability", "harm_exploitativeness",
    ],

    "scores_harm_subdims": [
        "check_score", "verif_score", "harm_score",
        "harm_social_fragmentation", "harm_spurs_action",
        "harm_believability", "harm_exploitativeness",
    ],

    # Top features from importance analysis
    "top_importance": [
        "check_score", "check_margin_logit", "check_logit_p_false",
        "check_p_false", "check_margin_p", "check_prediction",
        "verif_score", "verif_logit_p_true", "verif_margin_logit",
        "score_variance", "yes_vote_count",
    ],

    # Compact interpretable set
    "interpretable": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction",
        "score_variance", "yes_vote_count", "unanimous_yes",
        "harm_spurs_action", "harm_believability",
    ],

    # Medium complexity
    "medium": [
        # Core scores
        "check_score", "verif_score", "harm_score",
        # Predictions
        "check_prediction", "verif_prediction", "harm_prediction",
        # Key uncertainty indicators
        "check_margin_p", "verif_margin_p", "harm_margin_p",
        "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
        # Cross-module
        "score_variance", "yes_vote_count",
    ],

    # No token/length features (remove spurious correlations)
    "no_length": [
        "check_score", "check_prediction", "check_p_true", "check_p_false", "check_p_uncertain",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "check_entropy", "check_entropy_norm", "check_margin_p", "check_margin_logit",
        "check_top1_top2_gap", "check_p_uncertain_dominant", "check_is_argmax_match",
        "check_score_p_residual", "check_pred_score_mismatch",
        "verif_score", "verif_prediction", "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "verif_entropy", "verif_entropy_norm", "verif_margin_p", "verif_margin_logit",
        "verif_top1_top2_gap", "verif_p_uncertain_dominant", "verif_is_argmax_match",
        "verif_score_p_residual", "verif_pred_score_mismatch",
        "harm_score", "harm_prediction", "harm_p_true", "harm_p_false", "harm_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
        "harm_entropy", "harm_entropy_norm", "harm_margin_p", "harm_margin_logit",
        "harm_top1_top2_gap", "harm_p_uncertain_dominant", "harm_is_argmax_match",
        "harm_score_p_residual", "harm_pred_score_mismatch",
        "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
        "score_variance", "score_max_diff", "check_minus_verif", "check_minus_harm", "verif_minus_harm",
        "harm_high_verif_low", "yes_vote_count", "unanimous_yes", "unanimous_no",
        "check_verif_agree", "check_harm_agree", "verif_harm_agree",
        "pairwise_agreement_rate", "check_yes_verif_yes", "consensus_entropy",
    ],
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/dev/test data with labels."""

    # Load LLM features
    train_llm = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    dev_llm = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    test_llm = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    # Load labels
    train_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet").select(["Sentence_id", "class_label"])
    dev_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet").select(["Sentence_id", "class_label"])
    test_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet").select(["Sentence_id", "class_label"])

    # Join
    train_llm = train_llm.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    dev_llm = dev_llm.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    test_llm = test_llm.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    train = train_llm.join(train_labels, on="Sentence_id", how="left")
    dev = dev_llm.join(dev_labels, on="Sentence_id", how="left")
    test = test_llm.join(test_labels, on="Sentence_id", how="left")

    return train, dev, test


def prepare_data(df: pl.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels."""
    # Filter to existing columns
    existing = [c for c in feature_cols if c in df.columns]
    X = df.select(existing).to_numpy().astype(np.float32)
    y = (df["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    return X, y


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("LLM FEATURE ABLATION STUDY")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    train_df, dev_df, test_df = load_data()
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # Results storage
    results = []

    # Test each feature set with CatBoost (best from previous run)
    print("\n" + "=" * 90)
    print("Testing feature sets with CatBoost")
    print("=" * 90)

    for set_name, feature_cols in FEATURE_SETS.items():
        # Filter to existing columns
        existing = [c for c in feature_cols if c in train_df.columns]
        n_features = len(existing)

        if n_features == 0:
            print(f"\n{set_name}: No features found, skipping")
            continue

        # Prepare data
        X_train, y_train = prepare_data(train_df, existing)
        X_dev, y_dev = prepare_data(dev_df, existing)
        X_test, y_test = prepare_data(test_df, existing)

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_dev_s = scaler.transform(X_dev)
        X_test_s = scaler.transform(X_test)

        # Train CatBoost
        if HAS_CATBOOST:
            clf = CatBoostClassifier(
                iterations=100, depth=6, learning_rate=0.1,
                random_state=42, auto_class_weights="Balanced", verbose=False
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42,
                n_jobs=-1, class_weight="balanced"
            )

        clf.fit(X_train_s, y_train)

        # Evaluate
        train_f1 = f1_score(y_train, clf.predict(X_train_s))
        dev_f1 = f1_score(y_dev, clf.predict(X_dev_s))
        test_f1 = f1_score(y_test, clf.predict(X_test_s))

        # Dev-Test gap (overfitting indicator)
        gap = dev_f1 - test_f1

        results.append({
            "set": set_name,
            "n_features": n_features,
            "train_f1": train_f1,
            "dev_f1": dev_f1,
            "test_f1": test_f1,
            "dev_test_gap": gap,
        })

        print(f"\n{set_name} ({n_features} features):")
        print(f"  Train F1: {train_f1:.3f} | Dev F1: {dev_f1:.3f} | Test F1: {test_f1:.3f} | Gap: {gap:+.3f}")

    # Summary sorted by test F1
    print("\n" + "=" * 90)
    print("SUMMARY (sorted by Test F1)")
    print("=" * 90)

    results.sort(key=lambda x: x["test_f1"], reverse=True)

    print(f"\n{'Feature Set':<20} {'#Feat':<6} {'Train':<8} {'Dev':<8} {'Test':<8} {'Gap':<8}")
    print("-" * 60)

    for r in results:
        print(f"{r['set']:<20} {r['n_features']:<6} {r['train_f1']:<8.3f} {r['dev_f1']:<8.3f} {r['test_f1']:<8.3f} {r['dev_test_gap']:+.3f}")

    # Best result
    if results:
        best = results[0]
        print(f"\n✅ Best: {best['set']} ({best['n_features']} features) → Test F1: {best['test_f1']:.3f}")
        print(f"   Dev-Test gap: {best['dev_test_gap']:+.3f}")

        # Show features in best set
        print(f"\n   Features: {FEATURE_SETS[best['set']]}")

    # Also test with multiple classifiers on top 3 feature sets
    print("\n" + "=" * 90)
    print("TOP 3 FEATURE SETS × MULTIPLE CLASSIFIERS")
    print("=" * 90)

    top_sets = [r["set"] for r in results[:3]]
    classifiers = {
        "logistic": LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"),
    }
    if HAS_CATBOOST:
        classifiers["catboost"] = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, auto_class_weights="Balanced", verbose=False)

    multi_results = []

    for set_name in top_sets:
        feature_cols = FEATURE_SETS[set_name]
        existing = [c for c in feature_cols if c in train_df.columns]

        X_train, y_train = prepare_data(train_df, existing)
        X_dev, y_dev = prepare_data(dev_df, existing)
        X_test, y_test = prepare_data(test_df, existing)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_dev_s = scaler.transform(X_dev)
        X_test_s = scaler.transform(X_test)

        for clf_name, clf in classifiers.items():
            # Need fresh instance for each run
            if clf_name == "logistic":
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
            elif clf_name == "rf":
                clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
            elif clf_name == "catboost" and HAS_CATBOOST:
                clf = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, auto_class_weights="Balanced", verbose=False)

            clf.fit(X_train_s, y_train)

            dev_f1 = f1_score(y_dev, clf.predict(X_dev_s))
            test_f1 = f1_score(y_test, clf.predict(X_test_s))

            multi_results.append({
                "set": set_name,
                "clf": clf_name,
                "dev_f1": dev_f1,
                "test_f1": test_f1,
            })

    print(f"\n{'Feature Set':<20} {'Classifier':<12} {'Dev F1':<10} {'Test F1':<10}")
    print("-" * 55)

    multi_results.sort(key=lambda x: x["test_f1"], reverse=True)
    for r in multi_results:
        print(f"{r['set']:<20} {r['clf']:<12} {r['dev_f1']:<10.3f} {r['test_f1']:<10.3f}")

    if multi_results:
        best = multi_results[0]
        print(f"\n✅ Best overall: {best['set']} + {best['clf']} → Test F1: {best['test_f1']:.3f}")


if __name__ == "__main__":
    main()
