#!/usr/bin/env python3
"""
LLM Feature Grid Search

Tests all combinations of feature sets × classifiers.

Usage:
    python experiments/scripts/llm_feature_grid_search.py
    python experiments/scripts/llm_feature_grid_search.py --top 10
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
RESULTS_DIR = Path(__file__).parent.parent / "results" / "llm_grid_search"

# =============================================================================
# Feature Sets
# =============================================================================

FEATURE_SETS = {
    # === MINIMAL (1-6 features) ===
    "scores_3": [
        "check_score", "verif_score", "harm_score"
    ],

    "scores_preds_6": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction"
    ],

    # === SMALL (7-12 features) ===
    "scores_entropy_9": [
        "check_score", "verif_score", "harm_score",
        "check_entropy", "verif_entropy", "harm_entropy",
        "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
    ],

    "scores_margins_9": [
        "check_score", "verif_score", "harm_score",
        "check_margin_p", "verif_margin_p", "harm_margin_p",
        "check_margin_logit", "verif_margin_logit", "harm_margin_logit",
    ],

    "scores_probs_12": [
        "check_score", "verif_score", "harm_score",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
    ],

    "scores_cross_10": [
        "check_score", "verif_score", "harm_score",
        "score_variance", "score_max_diff",
        "yes_vote_count", "unanimous_yes", "unanimous_no",
        "pairwise_agreement_rate", "consensus_entropy",
    ],

    "interpretable_11": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction",
        "score_variance", "yes_vote_count", "unanimous_yes",
        "harm_spurs_action", "harm_believability",
    ],

    # === MEDIUM (13-25 features) ===
    "check_focused_17": [
        "check_score", "check_prediction",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "check_entropy", "check_entropy_norm",
        "check_margin_p", "check_margin_logit", "check_top1_top2_gap",
        "check_p_uncertain_dominant", "check_is_argmax_match",
        "check_score_p_residual", "check_pred_score_mismatch",
    ],

    "top_importance_15": [
        "check_score", "check_margin_logit", "check_logit_p_false",
        "check_p_false", "check_margin_p", "check_prediction",
        "verif_score", "verif_logit_p_true", "verif_margin_logit",
        "score_variance", "yes_vote_count", "check_p_uncertain_dominant",
        "harm_spurs_action", "verif_is_argmax_match", "harm_is_argmax_match",
    ],

    "balanced_18": [
        # Core scores + predictions
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction",
        # Best uncertainty indicators
        "check_margin_p", "verif_margin_p", "harm_margin_p",
        "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
        # Cross-module
        "score_variance", "yes_vote_count", "unanimous_yes",
        # Harm specifics
        "harm_spurs_action", "harm_believability", "harm_social_fragmentation",
    ],

    "scores_all_probs_21": [
        "check_score", "verif_score", "harm_score",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
    ],

    # === LARGE (26-50 features) ===
    "no_tokens_51": [
        # Checkability (17)
        "check_score", "check_prediction",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "check_entropy", "check_entropy_norm",
        "check_margin_p", "check_margin_logit", "check_top1_top2_gap",
        "check_p_uncertain_dominant", "check_is_argmax_match",
        "check_score_p_residual", "check_pred_score_mismatch",
        # Verifiability (17)
        "verif_score", "verif_prediction",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "verif_entropy", "verif_entropy_norm",
        "verif_margin_p", "verif_margin_logit", "verif_top1_top2_gap",
        "verif_p_uncertain_dominant", "verif_is_argmax_match",
        "verif_score_p_residual", "verif_pred_score_mismatch",
        # Harm (17 - no tokens/length)
        "harm_score", "harm_prediction",
        "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
        "harm_entropy", "harm_entropy_norm",
        "harm_margin_p", "harm_margin_logit", "harm_top1_top2_gap",
        "harm_p_uncertain_dominant", "harm_is_argmax_match",
        "harm_score_p_residual", "harm_pred_score_mismatch",
    ],

    "no_tokens_cross_66": [
        # Same as no_tokens_51 + cross-module
        "check_score", "check_prediction",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "check_entropy", "check_entropy_norm",
        "check_margin_p", "check_margin_logit", "check_top1_top2_gap",
        "check_p_uncertain_dominant", "check_is_argmax_match",
        "check_score_p_residual", "check_pred_score_mismatch",
        "verif_score", "verif_prediction",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "verif_entropy", "verif_entropy_norm",
        "verif_margin_p", "verif_margin_logit", "verif_top1_top2_gap",
        "verif_p_uncertain_dominant", "verif_is_argmax_match",
        "verif_score_p_residual", "verif_pred_score_mismatch",
        "harm_score", "harm_prediction",
        "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
        "harm_entropy", "harm_entropy_norm",
        "harm_margin_p", "harm_margin_logit", "harm_top1_top2_gap",
        "harm_p_uncertain_dominant", "harm_is_argmax_match",
        "harm_score_p_residual", "harm_pred_score_mismatch",
        # Cross-module (15)
        "score_variance", "score_max_diff",
        "check_minus_verif", "check_minus_harm", "verif_minus_harm",
        "harm_high_verif_low",
        "yes_vote_count", "unanimous_yes", "unanimous_no",
        "check_verif_agree", "check_harm_agree", "verif_harm_agree",
        "pairwise_agreement_rate", "check_yes_verif_yes", "consensus_entropy",
    ],

    # === FULL (all 79) ===
    "all_79": [
        # Checkability (20)
        "check_score", "check_prediction", "check_reasoning_length", "check_reasoning_hedged",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "check_entropy", "check_entropy_norm",
        "check_margin_p", "check_margin_logit", "check_top1_top2_gap",
        "check_p_uncertain_dominant", "check_is_argmax_match",
        "check_score_p_residual", "check_pred_score_mismatch", "check_completion_tokens",
        # Verifiability (20)
        "verif_score", "verif_prediction", "verif_reasoning_length", "verif_reasoning_hedged",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "verif_entropy", "verif_entropy_norm",
        "verif_margin_p", "verif_margin_logit", "verif_top1_top2_gap",
        "verif_p_uncertain_dominant", "verif_is_argmax_match",
        "verif_score_p_residual", "verif_pred_score_mismatch", "verif_completion_tokens",
        # Harm (24)
        "harm_score", "harm_prediction", "harm_reasoning_length", "harm_reasoning_hedged",
        "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
        "harm_entropy", "harm_entropy_norm",
        "harm_margin_p", "harm_margin_logit", "harm_top1_top2_gap",
        "harm_p_uncertain_dominant", "harm_is_argmax_match",
        "harm_score_p_residual", "harm_pred_score_mismatch", "harm_completion_tokens",
        # Cross-module (15)
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
    """Get all available classifiers."""
    classifiers = {
        "logistic": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced"),
        "logistic_l1": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42, penalty="l1", solver="saga", class_weight="balanced"),
        "rf": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"),
        "rf_deep": lambda: RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1, class_weight="balanced"),
        "gb": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        "svm_rbf": lambda: SVC(kernel="rbf", C=1.0, random_state=42, class_weight="balanced"),
    }

    if HAS_XGBOOST:
        classifiers["xgboost"] = lambda: XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="logloss")
        classifiers["xgboost_deep"] = lambda: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric="logloss")

    if HAS_CATBOOST:
        classifiers["catboost"] = lambda: CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, auto_class_weights="Balanced", verbose=False)
        classifiers["catboost_deep"] = lambda: CatBoostClassifier(iterations=200, depth=8, learning_rate=0.05, random_state=42, auto_class_weights="Balanced", verbose=False)

    if HAS_LIGHTGBM:
        classifiers["lightgbm"] = lambda: LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1)
        classifiers["lightgbm_deep"] = lambda: LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1)

    return classifiers


# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/dev/test data with labels."""
    train_llm = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    dev_llm = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    test_llm = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    train_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet").select(["Sentence_id", "class_label"])
    dev_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet").select(["Sentence_id", "class_label"])
    test_labels = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet").select(["Sentence_id", "class_label"])

    train_llm = train_llm.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    dev_llm = dev_llm.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    test_llm = test_llm.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    train = train_llm.join(train_labels, on="Sentence_id", how="left")
    dev = dev_llm.join(dev_labels, on="Sentence_id", how="left")
    test = test_llm.join(test_labels, on="Sentence_id", how="left")

    return train, dev, test


def prepare_data(df: pl.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels."""
    existing = [c for c in feature_cols if c in df.columns]
    X = df.select(existing).to_numpy().astype(np.float32)
    y = (df["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    return X, y


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Feature Grid Search")
    parser.add_argument("--top", type=int, default=20, help="Show top N results")
    parser.add_argument("--save", action="store_true", help="Save results to CSV")
    args = parser.parse_args()

    print("=" * 100)
    print("LLM FEATURE GRID SEARCH: Feature Sets × Classifiers")
    print("=" * 100)

    # Load data
    print("\nLoading data...")
    train_df, dev_df, test_df = load_data()
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # Get classifiers
    classifiers = get_classifiers()
    print(f"\nClassifiers: {list(classifiers.keys())}")
    print(f"Feature sets: {list(FEATURE_SETS.keys())}")

    total_combinations = len(FEATURE_SETS) * len(classifiers)
    print(f"\nTotal combinations: {total_combinations}")

    # Results storage
    results = []
    count = 0

    # Grid search
    print("\n" + "-" * 100)

    for set_name, feature_cols in FEATURE_SETS.items():
        # Filter to existing columns
        existing = [c for c in feature_cols if c in train_df.columns]
        n_features = len(existing)

        if n_features == 0:
            continue

        # Prepare data once per feature set
        X_train, y_train = prepare_data(train_df, existing)
        X_dev, y_dev = prepare_data(dev_df, existing)
        X_test, y_test = prepare_data(test_df, existing)

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_dev_s = scaler.transform(X_dev)
        X_test_s = scaler.transform(X_test)

        for clf_name, clf_factory in classifiers.items():
            count += 1
            try:
                clf = clf_factory()
                clf.fit(X_train_s, y_train)

                y_pred_dev = clf.predict(X_dev_s)
                y_pred_test = clf.predict(X_test_s)

                dev_f1 = f1_score(y_dev, y_pred_dev)
                test_f1 = f1_score(y_test, y_pred_test)
                test_acc = accuracy_score(y_test, y_pred_test)
                test_p = precision_score(y_test, y_pred_test, zero_division=0)
                test_r = recall_score(y_test, y_pred_test, zero_division=0)

                gap = dev_f1 - test_f1

                results.append({
                    "features": set_name,
                    "classifier": clf_name,
                    "n_feat": n_features,
                    "dev_f1": dev_f1,
                    "test_f1": test_f1,
                    "test_acc": test_acc,
                    "test_p": test_p,
                    "test_r": test_r,
                    "gap": gap,
                })

                # Progress indicator
                if count % 10 == 0:
                    print(f"  [{count}/{total_combinations}] {set_name} + {clf_name}: Test F1={test_f1:.3f}")

            except Exception as e:
                print(f"  [{count}/{total_combinations}] {set_name} + {clf_name}: ERROR - {e}")

    # Sort by test F1
    results.sort(key=lambda x: x["test_f1"], reverse=True)

    # Summary
    print("\n" + "=" * 100)
    print(f"TOP {args.top} RESULTS (sorted by Test F1)")
    print("=" * 100)

    print(f"\n{'Rank':<5} {'Features':<22} {'Classifier':<15} {'#Feat':<6} {'Dev F1':<8} {'Test F1':<8} {'Test P':<8} {'Test R':<8} {'Gap':<8}")
    print("-" * 95)

    for i, r in enumerate(results[:args.top], 1):
        print(f"{i:<5} {r['features']:<22} {r['classifier']:<15} {r['n_feat']:<6} "
              f"{r['dev_f1']:<8.3f} {r['test_f1']:<8.3f} {r['test_p']:<8.3f} {r['test_r']:<8.3f} {r['gap']:+.3f}")

    # Best result
    if results:
        best = results[0]
        print(f"\n✅ BEST: {best['features']} + {best['classifier']}")
        print(f"   Test F1: {best['test_f1']:.3f} | Test Acc: {best['test_acc']:.3f} | P: {best['test_p']:.3f} | R: {best['test_r']:.3f}")
        print(f"   Dev-Test gap: {best['gap']:+.3f}")

    # Analysis by feature set (average across classifiers)
    print("\n" + "=" * 100)
    print("AVERAGE BY FEATURE SET")
    print("=" * 100)

    feature_stats = {}
    for r in results:
        fs = r["features"]
        if fs not in feature_stats:
            feature_stats[fs] = {"test_f1": [], "gap": [], "n_feat": r["n_feat"]}
        feature_stats[fs]["test_f1"].append(r["test_f1"])
        feature_stats[fs]["gap"].append(r["gap"])

    feature_avg = []
    for fs, stats in feature_stats.items():
        feature_avg.append({
            "features": fs,
            "n_feat": stats["n_feat"],
            "avg_f1": np.mean(stats["test_f1"]),
            "max_f1": np.max(stats["test_f1"]),
            "avg_gap": np.mean(stats["gap"]),
        })

    feature_avg.sort(key=lambda x: x["avg_f1"], reverse=True)

    print(f"\n{'Features':<22} {'#Feat':<6} {'Avg F1':<10} {'Max F1':<10} {'Avg Gap':<10}")
    print("-" * 60)

    for r in feature_avg:
        print(f"{r['features']:<22} {r['n_feat']:<6} {r['avg_f1']:<10.3f} {r['max_f1']:<10.3f} {r['avg_gap']:+.3f}")

    # Analysis by classifier (average across feature sets)
    print("\n" + "=" * 100)
    print("AVERAGE BY CLASSIFIER")
    print("=" * 100)

    clf_stats = {}
    for r in results:
        clf = r["classifier"]
        if clf not in clf_stats:
            clf_stats[clf] = {"test_f1": [], "gap": []}
        clf_stats[clf]["test_f1"].append(r["test_f1"])
        clf_stats[clf]["gap"].append(r["gap"])

    clf_avg = []
    for clf, stats in clf_stats.items():
        clf_avg.append({
            "classifier": clf,
            "avg_f1": np.mean(stats["test_f1"]),
            "max_f1": np.max(stats["test_f1"]),
            "avg_gap": np.mean(stats["gap"]),
        })

    clf_avg.sort(key=lambda x: x["avg_f1"], reverse=True)

    print(f"\n{'Classifier':<18} {'Avg F1':<10} {'Max F1':<10} {'Avg Gap':<10}")
    print("-" * 50)

    for r in clf_avg:
        print(f"{r['classifier']:<18} {r['avg_f1']:<10.3f} {r['max_f1']:<10.3f} {r['avg_gap']:+.3f}")

    # Save results
    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df = pl.DataFrame(results)
        results_file = RESULTS_DIR / "grid_search_results.csv"
        results_df.write_csv(results_file)
        print(f"\n📁 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
