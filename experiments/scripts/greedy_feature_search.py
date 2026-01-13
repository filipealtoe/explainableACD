#!/usr/bin/env python3
"""
Greedy Forward Feature Selection for LLM Features

Systematically builds the best feature set by:
1. Finding the best single feature
2. Adding features one-by-one that most improve F1
3. Stopping when no improvement

Also tests feature GROUPS to find good combinations quickly.

Usage:
    python experiments/scripts/greedy_feature_search.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

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

# Define feature groups for interpretability
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

    # Join
    llm_train = llm_train.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_test = llm_test.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    train = llm_train.join(ct24_train.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")
    test = llm_test.join(ct24_test.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")

    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return train, test, y_train, y_test


def evaluate_features(train_df, test_df, y_train, y_test, features: list[str]) -> tuple[float, float]:
    """Evaluate a feature set and return (test_f1, test_acc)."""
    # Filter to existing features
    existing = [f for f in features if f in train_df.columns]
    if not existing:
        return 0.0, 0.0

    X_train = train_df.select(existing).to_numpy().astype(np.float32)
    X_test = test_df.select(existing).to_numpy().astype(np.float32)

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4})
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred)


# =============================================================================
# Search Algorithms
# =============================================================================

def greedy_forward_selection(train_df, test_df, y_train, y_test, candidate_features: list[str],
                             max_features: int = 20, verbose: bool = True):
    """Greedy forward selection: add best feature one at a time."""
    selected = []
    remaining = candidate_features.copy()
    history = []

    if verbose:
        print("\n  GREEDY FORWARD SELECTION")
        print("  " + "-" * 70)

    best_f1 = 0.0

    while remaining and len(selected) < max_features:
        best_candidate = None
        best_candidate_f1 = best_f1

        # Try adding each remaining feature
        for feat in remaining:
            test_features = selected + [feat]
            f1, acc = evaluate_features(train_df, test_df, y_train, y_test, test_features)

            if f1 > best_candidate_f1:
                best_candidate_f1 = f1
                best_candidate = feat
                best_acc = acc

        # If no improvement, stop
        if best_candidate is None:
            if verbose:
                print(f"\n  → No improvement found. Stopping.")
            break

        # Add best candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        improvement = best_candidate_f1 - best_f1
        best_f1 = best_candidate_f1

        history.append({
            "step": len(selected),
            "added": best_candidate,
            "f1": best_f1,
            "acc": best_acc,
            "improvement": improvement,
        })

        if verbose:
            marker = "🔥" if improvement > 0.01 else ("↑" if improvement > 0 else "→")
            print(f"  {len(selected):2d}. +{best_candidate:<30} F1={best_f1:.4f} ({improvement:+.4f}) {marker}")

    return selected, history


def group_based_search(train_df, test_df, y_train, y_test, verbose: bool = True):
    """Search by adding feature groups instead of individual features."""
    if verbose:
        print("\n  GROUP-BASED SEARCH")
        print("  " + "-" * 70)

    # First, evaluate each group individually
    group_scores = []

    if verbose:
        print("\n  Individual group performance:")
        print(f"  {'Group':<25} {'F1':<10} {'Acc':<10} {'#Features'}")
        print("  " + "-" * 55)

    for group_name, features in FEATURE_GROUPS.items():
        existing = [f for f in features if f in train_df.columns]
        if existing:
            f1, acc = evaluate_features(train_df, test_df, y_train, y_test, existing)
            group_scores.append({
                "group": group_name,
                "features": existing,
                "f1": f1,
                "acc": acc,
            })
            if verbose:
                print(f"  {group_name:<25} {f1:<10.4f} {acc:<10.4f} {len(existing)}")

    # Sort by F1
    group_scores.sort(key=lambda x: x["f1"], reverse=True)

    # Greedy forward selection on groups
    if verbose:
        print("\n  Greedy group combination:")
        print("  " + "-" * 70)

    selected_groups = []
    selected_features = []
    history = []
    best_f1 = 0.0

    for gs in group_scores:
        test_features = selected_features + gs["features"]
        f1, acc = evaluate_features(train_df, test_df, y_train, y_test, test_features)

        if f1 > best_f1:
            selected_groups.append(gs["group"])
            selected_features.extend(gs["features"])
            improvement = f1 - best_f1
            best_f1 = f1

            history.append({
                "group": gs["group"],
                "f1": f1,
                "acc": acc,
                "total_features": len(selected_features),
            })

            if verbose:
                marker = "🔥" if improvement > 0.01 else "↑"
                print(f"  +{gs['group']:<25} F1={f1:.4f} ({improvement:+.4f}) [{len(selected_features)} features] {marker}")

    return selected_groups, selected_features, history


def correlation_analysis(train_df, features: list[str], threshold: float = 0.9):
    """Find highly correlated feature pairs."""
    existing = [f for f in features if f in train_df.columns]
    X = train_df.select(existing).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    corr_matrix = np.corrcoef(X.T)

    highly_correlated = []
    for i in range(len(existing)):
        for j in range(i + 1, len(existing)):
            if abs(corr_matrix[i, j]) > threshold:
                highly_correlated.append((existing[i], existing[j], corr_matrix[i, j]))

    return highly_correlated


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("GREEDY FEATURE SEARCH for LLM Features")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    train_df, test_df, y_train, y_test = load_data()
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # Get all available features
    all_features = []
    for group_features in FEATURE_GROUPS.values():
        for f in group_features:
            if f in train_df.columns and f not in all_features:
                all_features.append(f)

    print(f"  Available features: {len(all_features)}")

    # =========================================================================
    # Part 1: Correlation analysis (find redundant features)
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. CORRELATION ANALYSIS (removing redundancy)")
    print("=" * 80)

    highly_corr = correlation_analysis(train_df, all_features, threshold=0.95)
    print(f"\n  Highly correlated pairs (r > 0.95): {len(highly_corr)}")
    for f1, f2, r in highly_corr[:10]:
        print(f"    {f1} ↔ {f2}: r={r:.3f}")

    if len(highly_corr) > 10:
        print(f"    ... and {len(highly_corr) - 10} more")

    # =========================================================================
    # Part 2: Group-based search
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. GROUP-BASED SEARCH")
    print("=" * 80)

    best_groups, best_group_features, group_history = group_based_search(
        train_df, test_df, y_train, y_test, verbose=True
    )

    # =========================================================================
    # Part 3: Greedy forward selection on individual features
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. GREEDY FORWARD SELECTION (individual features)")
    print("=" * 80)

    best_features, feature_history = greedy_forward_selection(
        train_df, test_df, y_train, y_test, all_features, max_features=15, verbose=True
    )

    # =========================================================================
    # Part 4: Start from best group, then add individual features
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. HYBRID: Best groups + individual refinement")
    print("=" * 80)

    # Start with top 2 groups
    top_groups = group_history[:2] if len(group_history) >= 2 else group_history
    starting_features = []
    for g in top_groups:
        for f in FEATURE_GROUPS.get(g["group"], []):
            if f in train_df.columns and f not in starting_features:
                starting_features.append(f)

    print(f"\n  Starting with groups: {[g['group'] for g in top_groups]}")
    print(f"  Starting features: {len(starting_features)}")

    remaining = [f for f in all_features if f not in starting_features]
    hybrid_features, hybrid_history = greedy_forward_selection(
        train_df, test_df, y_train, y_test, remaining, max_features=10, verbose=True
    )

    final_hybrid = starting_features + hybrid_features
    final_f1, final_acc = evaluate_features(train_df, test_df, y_train, y_test, final_hybrid)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Best from each approach
    best_group_f1 = group_history[-1]["f1"] if group_history else 0
    best_greedy_f1 = feature_history[-1]["f1"] if feature_history else 0

    print(f"\n  {'Approach':<40} {'Test F1':<10} {'#Features'}")
    print("  " + "-" * 60)
    print(f"  {'Group-based search':<40} {best_group_f1:<10.4f} {len(best_group_features)}")
    print(f"  {'Greedy forward (individual)':<40} {best_greedy_f1:<10.4f} {len(best_features)}")
    print(f"  {'Hybrid (groups + refinement)':<40} {final_f1:<10.4f} {len(final_hybrid)}")

    # Best overall
    results = [
        ("Group-based", best_group_f1, best_group_features),
        ("Greedy individual", best_greedy_f1, best_features),
        ("Hybrid", final_f1, final_hybrid),
    ]
    best_approach, best_f1, best_feature_set = max(results, key=lambda x: x[1])

    print(f"\n  🏆 BEST: {best_approach} with F1 = {best_f1:.4f}")
    print(f"\n  Best feature set ({len(best_feature_set)} features):")
    for f in best_feature_set:
        print(f"    • {f}")

    # SOTA comparison
    SOTA_F1 = 0.82
    print(f"\n  SOTA: F1 = {SOTA_F1}")
    print(f"  Gap: {best_f1 - SOTA_F1:+.4f}")


if __name__ == "__main__":
    main()
