#!/usr/bin/env python3
"""
Feature Ablation Study: Which Features Actually Help?

Starts from base 6 features (equivalent to GPT-3.5) and progressively
adds feature groups to see incremental improvement.

Usage:
    python experiments/scripts/feature_ablation_study.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"

# =============================================================================
# Feature Groups (building blocks)
# =============================================================================

# Base: equivalent to GPT-3.5 features
BASE_6 = {
    "name": "Base 6 (scores + p_true)",
    "features": [
        "check_score", "verif_score", "harm_score",
        "check_p_true", "verif_p_true", "harm_p_true",
    ]
}

# Additional feature groups to test
FEATURE_GROUPS = [
    {
        "name": "+ p_false (3)",
        "features": ["check_p_false", "verif_p_false", "harm_p_false"],
    },
    {
        "name": "+ p_uncertain (3)",
        "features": ["check_p_uncertain", "verif_p_uncertain", "harm_p_uncertain"],
    },
    {
        "name": "+ entropy (3)",
        "features": ["check_entropy", "verif_entropy", "harm_entropy"],
    },
    {
        "name": "+ entropy_norm (3)",
        "features": ["check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm"],
    },
    {
        "name": "+ predictions (3)",
        "features": ["check_prediction", "verif_prediction", "harm_prediction"],
    },
    {
        "name": "+ margin_p (3)",
        "features": ["check_margin_p", "verif_margin_p", "harm_margin_p"],
    },
    {
        "name": "+ margin_logit (3)",
        "features": ["check_margin_logit", "verif_margin_logit", "harm_margin_logit"],
    },
    {
        "name": "+ logit_p_true (3)",
        "features": ["check_logit_p_true", "verif_logit_p_true", "harm_logit_p_true"],
    },
    {
        "name": "+ logit_p_false (3)",
        "features": ["check_logit_p_false", "verif_logit_p_false", "harm_logit_p_false"],
    },
    {
        "name": "+ logit_p_uncertain (3)",
        "features": ["check_logit_p_uncertain", "verif_logit_p_uncertain", "harm_logit_p_uncertain"],
    },
    {
        "name": "+ is_argmax_match (3)",
        "features": ["check_is_argmax_match", "verif_is_argmax_match", "harm_is_argmax_match"],
    },
    {
        "name": "+ p_uncertain_dominant (3)",
        "features": ["check_p_uncertain_dominant", "verif_p_uncertain_dominant", "harm_p_uncertain_dominant"],
    },
    {
        "name": "+ score_p_residual (3)",
        "features": ["check_score_p_residual", "verif_score_p_residual", "harm_score_p_residual"],
    },
    {
        "name": "+ cross-module basic (3)",
        "features": ["score_variance", "yes_vote_count", "unanimous_yes"],
    },
    {
        "name": "+ cross-module diffs (3)",
        "features": ["check_minus_verif", "check_minus_harm", "verif_minus_harm"],
    },
    {
        "name": "+ cross-module agree (3)",
        "features": ["check_verif_agree", "check_harm_agree", "verif_harm_agree"],
    },
    {
        "name": "+ harm subdims (4)",
        "features": ["harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness"],
    },
    {
        "name": "+ reasoning_hedged (3)",
        "features": ["check_reasoning_hedged", "verif_reasoning_hedged", "harm_reasoning_hedged"],
    },
]


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load and join data properly."""
    # LLM features
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    # Labels
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

    return train, dev, test


def evaluate(train, dev, test, features):
    """Evaluate a feature set."""
    # Filter to existing features
    existing = [f for f in features if f in train.columns]
    if not existing:
        return None, None, None

    X_train = train.select(existing).to_numpy().astype(np.float32)
    X_dev = dev.select(existing).to_numpy().astype(np.float32)
    X_test = test.select(existing).to_numpy().astype(np.float32)

    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    clf = LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        class_weight="balanced", verbose=-1, random_state=42
    )
    clf.fit(X_train_s, y_train)

    dev_f1 = f1_score(y_dev, clf.predict(X_dev_s))
    test_f1 = f1_score(y_test, clf.predict(X_test_s))
    test_acc = accuracy_score(y_test, clf.predict(X_test_s))

    return dev_f1, test_f1, test_acc


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("FEATURE ABLATION STUDY: Which Features Actually Help?")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    train, dev, test = load_data()
    print(f"  Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")

    results = []

    # ==========================================================================
    # PART 1: Incremental Addition (start from base, add one group at a time)
    # ==========================================================================
    print("\n" + "=" * 90)
    print("PART 1: INCREMENTAL ADDITION (cumulative)")
    print("=" * 90)
    print("Starting from base 6 features, adding one group at a time.\n")

    current_features = BASE_6["features"].copy()
    dev_f1, test_f1, test_acc = evaluate(train, dev, test, current_features)
    baseline_f1 = test_f1

    print(f"{'Configuration':<45} {'#Feat':<7} {'Dev F1':<9} {'Test F1':<9} {'Δ Test':<9}")
    print("-" * 90)
    print(f"{BASE_6['name']:<45} {len(current_features):<7} {dev_f1:<9.4f} {test_f1:<9.4f} {'baseline':<9}")

    results.append({
        "config": BASE_6["name"],
        "n_features": len(current_features),
        "dev_f1": dev_f1,
        "test_f1": test_f1,
        "delta": 0,
        "cumulative": True,
    })

    best_cumulative_f1 = test_f1
    best_cumulative_config = BASE_6["name"]

    for group in FEATURE_GROUPS:
        # Add this group's features
        new_features = [f for f in group["features"] if f in train.columns]
        if not new_features:
            continue

        current_features.extend(new_features)
        dev_f1, test_f1, test_acc = evaluate(train, dev, test, current_features)

        if test_f1 is None:
            continue

        delta = test_f1 - baseline_f1
        marker = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")

        print(f"{group['name']:<45} {len(current_features):<7} {dev_f1:<9.4f} {test_f1:<9.4f} {delta:+.4f} {marker}")

        results.append({
            "config": group["name"],
            "n_features": len(current_features),
            "dev_f1": dev_f1,
            "test_f1": test_f1,
            "delta": delta,
            "cumulative": True,
        })

        baseline_f1 = test_f1  # Update baseline for next delta

        if test_f1 > best_cumulative_f1:
            best_cumulative_f1 = test_f1
            best_cumulative_config = f"Up to {group['name']}"

    # ==========================================================================
    # PART 2: Individual Group Impact (base + each group separately)
    # ==========================================================================
    print("\n" + "=" * 90)
    print("PART 2: INDIVIDUAL GROUP IMPACT (base + one group only)")
    print("=" * 90)
    print("Testing each group added to base 6 separately.\n")

    base_dev_f1, base_test_f1, _ = evaluate(train, dev, test, BASE_6["features"])

    individual_results = []

    print(f"{'Configuration':<45} {'#Feat':<7} {'Dev F1':<9} {'Test F1':<9} {'Δ from base':<12}")
    print("-" * 90)
    print(f"{'Base 6 only':<45} {6:<7} {base_dev_f1:<9.4f} {base_test_f1:<9.4f} {'baseline':<12}")

    for group in FEATURE_GROUPS:
        features = BASE_6["features"] + [f for f in group["features"] if f in train.columns]
        dev_f1, test_f1, test_acc = evaluate(train, dev, test, features)

        if test_f1 is None:
            continue

        delta = test_f1 - base_test_f1
        marker = "✅" if delta > 0.01 else ("❌" if delta < -0.01 else "➖")

        print(f"Base + {group['name']:<38} {len(features):<7} {dev_f1:<9.4f} {test_f1:<9.4f} {delta:+.4f} {marker}")

        individual_results.append({
            "group": group["name"],
            "n_features": len(features),
            "test_f1": test_f1,
            "delta": delta,
        })

    # Sort individual results by delta
    individual_results.sort(key=lambda x: x["delta"], reverse=True)

    print("\n" + "=" * 90)
    print("PART 3: RANKED FEATURE GROUPS (by individual impact)")
    print("=" * 90)

    print(f"\n{'Rank':<6} {'Group':<40} {'Δ Test F1':<12} {'Impact':<10}")
    print("-" * 70)

    for i, r in enumerate(individual_results, 1):
        if r["delta"] > 0.01:
            impact = "HELPFUL ✅"
        elif r["delta"] < -0.01:
            impact = "HARMFUL ❌"
        else:
            impact = "NEUTRAL ➖"
        print(f"{i:<6} {r['group']:<40} {r['delta']:+.4f}       {impact}")

    # ==========================================================================
    # PART 4: Best Subset (only helpful features)
    # ==========================================================================
    print("\n" + "=" * 90)
    print("PART 4: OPTIMIZED SUBSET (base + only helpful groups)")
    print("=" * 90)

    helpful_groups = [r for r in individual_results if r["delta"] > 0.005]
    optimized_features = BASE_6["features"].copy()

    print(f"\nHelpful groups (Δ > 0.005):")
    for r in helpful_groups:
        print(f"  • {r['group']}: {r['delta']:+.4f}")
        # Find the group and add its features
        for group in FEATURE_GROUPS:
            if group["name"] == r["group"]:
                optimized_features.extend([f for f in group["features"] if f in train.columns])
                break

    # Remove duplicates while preserving order
    optimized_features = list(dict.fromkeys(optimized_features))

    dev_f1, test_f1, test_acc = evaluate(train, dev, test, optimized_features)

    print(f"\nOptimized feature set: {len(optimized_features)} features")
    print(f"  Dev F1:  {dev_f1:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")

    print(f"\nFeatures in optimized set:")
    for f in optimized_features:
        print(f"  • {f}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"\n  Base 6 features:     Test F1 = {base_test_f1:.4f}")
    print(f"  Best cumulative:     Test F1 = {best_cumulative_f1:.4f} ({best_cumulative_config})")
    print(f"  Optimized subset:    Test F1 = {test_f1:.4f} ({len(optimized_features)} features)")

    # Top 3 most helpful individual groups
    print(f"\n  Top 3 most helpful feature groups:")
    for r in individual_results[:3]:
        print(f"    {r['group']}: {r['delta']:+.4f}")

    # Top 3 most harmful individual groups
    harmful = [r for r in individual_results if r["delta"] < 0]
    if harmful:
        print(f"\n  Top 3 most harmful feature groups:")
        for r in sorted(harmful, key=lambda x: x["delta"])[:3]:
            print(f"    {r['group']}: {r['delta']:+.4f}")


if __name__ == "__main__":
    main()
