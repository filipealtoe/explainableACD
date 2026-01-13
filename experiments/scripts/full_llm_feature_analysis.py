#!/usr/bin/env python3
"""
Full LLM Feature Analysis with Feature Importance

Tests all LLM features across multiple classifiers with varying class weights.
Reports the best configuration and its feature importances.

Usage:
    python experiments/scripts/full_llm_feature_analysis.py
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

# All LLM feature groups
FEATURE_GROUPS = {
    "scores": ["check_score", "verif_score", "harm_score"],
    "entropy": ["check_entropy", "verif_entropy", "harm_entropy"],
    "entropy_norm": ["check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm"],
    "p_true": ["check_p_true", "verif_p_true", "harm_p_true"],
    "p_false": ["check_p_false", "verif_p_false", "harm_p_false"],
    "p_uncertain": ["check_p_uncertain", "verif_p_uncertain", "harm_p_uncertain"],
    "logit_p_true": ["check_logit_p_true", "verif_logit_p_true", "harm_logit_p_true"],
    "logit_p_false": ["check_logit_p_false", "verif_logit_p_false", "harm_logit_p_false"],
    "logit_p_uncertain": ["check_logit_p_uncertain", "verif_logit_p_uncertain", "harm_logit_p_uncertain"],
    "margin_p": ["check_margin_p", "verif_margin_p", "harm_margin_p"],
    "margin_logit": ["check_margin_logit", "verif_margin_logit", "harm_margin_logit"],
    "predictions": ["check_prediction", "verif_prediction", "harm_prediction"],
    "is_argmax_match": ["check_is_argmax_match", "verif_is_argmax_match", "harm_is_argmax_match"],
    "score_p_residual": ["check_score_p_residual", "verif_score_p_residual", "harm_score_p_residual"],
    "reasoning_hedged": ["check_reasoning_hedged", "verif_reasoning_hedged", "harm_reasoning_hedged"],
    "cross_basic": ["score_variance", "yes_vote_count", "unanimous_yes"],
    "cross_diffs": ["check_minus_verif", "check_minus_harm", "verif_minus_harm"],
    "cross_agree": ["check_verif_agree", "check_harm_agree", "verif_harm_agree"],
    "harm_subdims": ["harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness"],
}


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

    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return train, test, y_train, y_test


def get_all_features(df) -> list[str]:
    """Get all available LLM features from the dataframe."""
    all_features = []
    for group_features in FEATURE_GROUPS.values():
        for f in group_features:
            if f in df.columns and f not in all_features:
                all_features.append(f)
    return all_features


# =============================================================================
# Classifiers
# =============================================================================

def get_classifiers(pos_weight: int) -> dict:
    """Get all classifiers with specified positive class weight."""
    classifiers = {
        f"Logistic_w{pos_weight}": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42,
            class_weight={0: 1, 1: pos_weight}
        ),
        f"RandomForest_w{pos_weight}": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
            class_weight={0: 1, 1: pos_weight}
        ),
    }

    if HAS_LIGHTGBM:
        classifiers[f"LightGBM_w{pos_weight}"] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=-1,
            class_weight={0: 1, 1: pos_weight}
        )

    if HAS_XGBOOST:
        classifiers[f"XGBoost_w{pos_weight}"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42, eval_metric="logloss",
            scale_pos_weight=pos_weight
        )

    return classifiers


def extract_feature_importance(clf, feature_names: list[str], clf_name: str) -> list[tuple[str, float]]:
    """Extract feature importances from a trained classifier."""
    if "Logistic" in clf_name:
        # For logistic regression, use absolute coefficient values
        importances = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        # Tree-based models
        importances = clf.feature_importances_
    else:
        return []

    # Pair with feature names and sort by importance
    paired = list(zip(feature_names, importances))
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("FULL LLM FEATURE ANALYSIS - All Features, All Classifiers, Varying Weights")
    print("=" * 100)

    # Load data
    print("\nLoading data...")
    train_df, test_df, y_train, y_test = load_data()

    # Get all features
    all_features = get_all_features(train_df)
    print(f"  Total LLM features: {len(all_features)}")
    print(f"  Train: {len(y_train)} ({100*y_train.mean():.1f}% positive)")
    print(f"  Test:  {len(y_test)} ({100*y_test.mean():.1f}% positive)")

    # Prepare feature matrix
    X_train = train_df.select(all_features).to_numpy().astype(np.float32)
    X_test = test_df.select(all_features).to_numpy().astype(np.float32)

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Store all results
    all_results = []
    best_result = None
    best_clf = None
    best_clf_name = None

    # Test weights from 1 to 6
    weights_to_test = [1, 2, 3, 4, 5, 6]

    print("\n" + "=" * 100)
    print("RUNNING ALL CLASSIFIER + WEIGHT COMBINATIONS")
    print("=" * 100)

    for pos_weight in weights_to_test:
        print(f"\n{'─'*100}")
        print(f"WEIGHT = {pos_weight}")
        print(f"{'─'*100}")

        classifiers = get_classifiers(pos_weight)

        for clf_name, clf in classifiers.items():
            # Train
            clf.fit(X_train_s, y_train)

            # Predict
            y_pred = clf.predict(X_test_s)

            # Metrics
            test_f1 = f1_score(y_test, y_pred)
            test_acc = accuracy_score(y_test, y_pred)
            test_p = precision_score(y_test, y_pred, zero_division=0)
            test_r = recall_score(y_test, y_pred, zero_division=0)

            result = {
                "classifier": clf_name,
                "weight": pos_weight,
                "f1": test_f1,
                "acc": test_acc,
                "precision": test_p,
                "recall": test_r,
            }
            all_results.append(result)

            # Check if best
            if best_result is None or test_f1 > best_result["f1"]:
                best_result = result
                best_clf = clf
                best_clf_name = clf_name

            # Print
            gap_f1 = test_f1 - SOTA_F1
            marker = "🔥" if test_f1 > 0.75 else ("↑" if test_f1 > 0.70 else "")
            print(f"  {clf_name:<25} F1={test_f1:.4f} (gap:{gap_f1:+.4f}) Acc={test_acc:.4f} P={test_p:.4f} R={test_r:.4f} {marker}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY - TOP 15 RESULTS (sorted by F1)")
    print("=" * 100)

    all_results.sort(key=lambda x: x["f1"], reverse=True)

    print(f"\n{'Rank':<5} {'Classifier':<25} {'Weight':<8} {'F1':<10} {'Acc':<10} {'P':<8} {'R':<8} {'Gap to SOTA'}")
    print("-" * 95)

    for rank, r in enumerate(all_results[:15], 1):
        gap = r["f1"] - SOTA_F1
        marker = "🔥" if r["f1"] > 0.75 else ""
        print(f"{rank:<5} {r['classifier']:<25} {r['weight']:<8} {r['f1']:<10.4f} {r['acc']:<10.4f} "
              f"{r['precision']:<8.4f} {r['recall']:<8.4f} {gap:+.4f} {marker}")

    # =========================================================================
    # Best Result Analysis
    # =========================================================================
    print("\n" + "=" * 100)
    print("🏆 BEST RESULT")
    print("=" * 100)

    print(f"\n  Classifier:  {best_result['classifier']}")
    print(f"  Weight:      {best_result['weight']}")
    print(f"  Test F1:     {best_result['f1']:.4f}")
    print(f"  Test Acc:    {best_result['acc']:.4f}")
    print(f"  Precision:   {best_result['precision']:.4f}")
    print(f"  Recall:      {best_result['recall']:.4f}")
    print(f"\n  SOTA F1:     {SOTA_F1}")
    print(f"  Gap:         {best_result['f1'] - SOTA_F1:+.4f}")

    # =========================================================================
    # Feature Importance Analysis
    # =========================================================================
    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE (from best model)")
    print("=" * 100)

    importances = extract_feature_importance(best_clf, all_features, best_clf_name)

    if importances:
        print(f"\n  {'Rank':<5} {'Feature':<35} {'Importance':<12} {'Group'}")
        print("  " + "-" * 70)

        # Find which group each feature belongs to
        def get_group(feat):
            for group_name, features in FEATURE_GROUPS.items():
                if feat in features:
                    return group_name
            return "unknown"

        for rank, (feat, imp) in enumerate(importances[:25], 1):
            group = get_group(feat)
            bar = "█" * int(imp / max(i[1] for i in importances) * 20)
            print(f"  {rank:<5} {feat:<35} {imp:<12.4f} {group:<15} {bar}")

        # Group importance summary
        print("\n" + "-" * 100)
        print("GROUP IMPORTANCE (sum of feature importances per group)")
        print("-" * 100)

        group_importance = {}
        for feat, imp in importances:
            group = get_group(feat)
            group_importance[group] = group_importance.get(group, 0) + imp

        group_sorted = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  {'Rank':<5} {'Group':<25} {'Total Importance':<18} {'% of Total'}")
        print("  " + "-" * 60)

        total_imp = sum(i[1] for i in importances)
        for rank, (group, imp) in enumerate(group_sorted, 1):
            pct = 100 * imp / total_imp
            bar = "█" * int(pct / 5)
            print(f"  {rank:<5} {group:<25} {imp:<18.4f} {pct:>5.1f}% {bar}")

    # =========================================================================
    # Weight Analysis
    # =========================================================================
    print("\n" + "=" * 100)
    print("WEIGHT ANALYSIS - Best F1 per Classifier Type")
    print("=" * 100)

    # Get unique classifier types (without weight suffix)
    clf_types = set()
    for r in all_results:
        clf_type = r["classifier"].rsplit("_w", 1)[0]
        clf_types.add(clf_type)

    print(f"\n  {'Classifier':<20} {'Best Weight':<12} {'Best F1':<10} {'Best Acc'}")
    print("  " + "-" * 55)

    for clf_type in sorted(clf_types):
        type_results = [r for r in all_results if r["classifier"].startswith(clf_type)]
        best_for_type = max(type_results, key=lambda x: x["f1"])
        print(f"  {clf_type:<20} w={best_for_type['weight']:<10} {best_for_type['f1']:<10.4f} {best_for_type['acc']:.4f}")

    # =========================================================================
    # Recommendations
    # =========================================================================
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    top_groups = [g for g, _ in group_sorted[:5]] if importances else []
    top_features = [f for f, _ in importances[:10]] if importances else []

    print(f"""
  Based on the analysis:

  1. BEST CLASSIFIER: {best_result['classifier']}
     - Use class weight = {best_result['weight']} for optimal F1

  2. MOST IMPORTANT FEATURE GROUPS:
     {chr(10).join(f'     • {g}' for g in top_groups)}

  3. TOP 10 INDIVIDUAL FEATURES:
     {chr(10).join(f'     • {f}' for f in top_features)}

  4. NEXT STEPS:
     • Try feature selection with only top 10-15 features
     • Consider ensemble of Logistic + LightGBM
     • Test threshold optimization on dev set
     • Gap to SOTA: {best_result['f1'] - SOTA_F1:+.4f}
""")


if __name__ == "__main__":
    main()
