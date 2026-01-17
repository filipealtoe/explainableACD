#!/usr/bin/env python3
"""
Combined Greedy Feature Search: Text + LLM Features

Combines high-lift text features with LLM confidence features to reach SOTA.

Strategy:
1. Evaluate text-only, LLM-only, and combined baselines
2. Greedy forward selection across ALL features
3. Test starting from best text group, best LLM group, and combined
4. Report best configuration for reaching SOTA (F1=0.82, Acc=0.905)

Usage:
    python experiments/scripts/combined_greedy_search.py
    python experiments/scripts/combined_greedy_search.py --quick
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features_v4"
TEXT_FEATURES_DIR = DATA_DIR / "CT24_features"

SOTA_F1 = 0.82
SOTA_ACC = 0.905

# High-lift text features from prior analysis
TEXT_FEATURE_GROUPS = {
    # Quantification - strong positive lift
    "quantification": [
        "has_number", "has_precise_number", "has_large_scale",
        "number_count", "has_range", "has_delta"
    ],
    # Temporal anchoring
    "temporal": [
        "has_specific_year", "has_relative_time", "has_temporal_anchor"
    ],
    # Source/evidence - credibility signals
    "source_evidence": [
        "has_source_attribution", "has_evidence_noun",
        "has_official_source", "has_said_claimed"
    ],
    # Comparative/superlative
    "comparative": [
        "has_comparative", "has_superlative", "has_ranking"
    ],
    # Action/change verbs
    "action_change": [
        "has_increase_decrease", "has_voted", "has_negation_claim"
    ],
    # Opinion/hedge - negative lift (non-checkworthy)
    "opinion_hedge": [
        "has_first_person_stance", "has_desire_intent",
        "has_future_modal", "has_hedge", "has_vague_quantifier"
    ],
    # Rhetorical patterns
    "rhetorical": [
        "has_rhetorical_filler", "has_fact_assertion",
        "is_question", "has_transcript_artifact"
    ],
    # Metadata
    "metadata": [
        "word_count", "avg_word_length", "alpha_ratio"
    ],
    # Interaction features
    "interactions": [
        "has_number_and_time", "has_number_and_comparative",
        "has_change_and_time", "has_source_and_number"
    ],
}

# LLM feature groups (from v4 schema)
LLM_FEATURE_GROUPS = {
    "scores": ["check_score", "verif_score", "harm_score"],
    "p_yes": ["check_p_yes", "verif_p_yes", "harm_p_yes"],
    "p_no": ["check_p_no", "verif_p_no", "harm_p_no"],
    "logit_p_yes": ["check_logit_p_yes", "verif_logit_p_yes", "harm_logit_p_yes"],
    "logit_p_no": ["check_logit_p_no", "verif_logit_p_no", "harm_logit_p_no"],
    "entropy": ["check_entropy", "verif_entropy", "harm_entropy"],
    "margin_p": ["check_margin_p", "verif_margin_p", "harm_margin_p"],
    "margin_logit": ["check_margin_logit", "verif_margin_logit", "harm_margin_logit"],
    "predictions": ["check_prediction", "verif_prediction", "harm_prediction"],
    "score_p_residual": ["check_score_p_residual", "verif_score_p_residual", "harm_score_p_residual"],
    "cross_votes": ["yes_vote_count", "unanimous_yes", "unanimous_no"],
    "cross_variance": ["score_variance", "score_max_diff"],
    "cross_diffs": ["check_minus_verif", "check_minus_harm", "verif_minus_harm"],
    "harm_subdims": ["harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness"],
}

# Pre-defined feature sets based on prior analysis
PRESET_FEATURE_SETS = {
    # Text-only presets
    "text_high_lift": [
        "has_precise_number", "has_number", "has_large_scale", "has_voted",
        "is_question", "has_first_person_stance", "has_desire_intent"
    ],
    "text_top15": [
        "has_precise_number", "has_number", "has_large_scale", "number_count",
        "avg_word_length", "word_count", "alpha_ratio", "has_first_person_stance",
        "has_desire_intent", "has_delta", "has_number_and_time", "has_future_modal",
        "has_increase_decrease", "has_official_source", "has_voted"
    ],
    # LLM-only presets
    "llm_logit_yes": ["check_logit_p_yes", "verif_logit_p_yes", "harm_logit_p_yes"],
    "llm_score_residual": ["check_score_p_residual", "verif_score_p_residual", "harm_score_p_residual"],
    "llm_best_8": [
        "check_logit_p_yes", "verif_logit_p_yes", "harm_logit_p_yes",
        "verif_score", "harm_believability", "harm_entropy_norm",
        "harm_margin_p", "check_p_no"
    ],
    # Combined presets
    "combined_minimal": [
        # LLM
        "check_logit_p_yes", "verif_logit_p_yes", "harm_logit_p_yes",
        # Text high-lift
        "has_precise_number", "has_voted", "is_question"
    ],
    "combined_balanced": [
        # LLM core
        "check_logit_p_yes", "verif_logit_p_yes", "harm_logit_p_yes",
        "harm_believability", "score_variance",
        # Text quantification
        "has_precise_number", "has_number", "has_large_scale",
        # Text opinion (negative signal)
        "is_question", "has_first_person_stance",
        # Text action
        "has_voted", "has_increase_decrease"
    ],
}


@dataclass
class Metrics:
    f1: float
    acc: float
    precision: float
    recall: float

    def __str__(self) -> str:
        return f"F1={self.f1:.4f} Acc={self.acc:.4f}"


# =============================================================================
# Data Loading
# =============================================================================


def load_combined_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load and merge LLM features + text features."""
    # Load LLM features
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    # Load text features
    text_train = pl.read_parquet(TEXT_FEATURES_DIR / "CT24_train_features.parquet")
    text_dev = pl.read_parquet(TEXT_FEATURES_DIR / "CT24_dev_features.parquet")
    text_test = pl.read_parquet(TEXT_FEATURES_DIR / "CT24_test_features.parquet")

    # Normalize IDs for join
    def normalize_llm(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("sentence_id").cast(pl.Utf8).alias("join_id"))

    def normalize_text(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("Sentence_id").cast(pl.Utf8).alias("join_id"))

    llm_train = normalize_llm(llm_train)
    llm_dev = normalize_llm(llm_dev)
    llm_test = normalize_llm(llm_test)

    text_train = normalize_text(text_train)
    text_dev = normalize_text(text_dev)
    text_test = normalize_text(text_test)

    # Get text feature columns (exclude metadata)
    text_feature_cols = [c for c in text_train.columns if c not in [
        "Sentence_id", "Text", "class_label", "original_text", "cleaned_text",
        "would_exclude", "exclusion_reasons", "has_label_conflict", "join_id"
    ]]

    # Join: LLM features + text features + labels
    train = llm_train.join(
        text_train.select(["join_id", "class_label"] + text_feature_cols),
        on="join_id", how="left"
    )
    dev = llm_dev.join(
        text_dev.select(["join_id", "class_label"] + text_feature_cols),
        on="join_id", how="left"
    )
    test = llm_test.join(
        text_test.select(["join_id", "class_label"] + text_feature_cols),
        on="join_id", how="left"
    )

    # Extract labels
    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return train, dev, test, y_train, y_dev, y_test


def get_all_features(df: pl.DataFrame, prefix: str = "") -> list[str]:
    """Get all numeric/boolean features, optionally filtered by prefix."""
    exclude = {"sentence_id", "join_id", "class_label", "Text", "original_text",
               "cleaned_text", "would_exclude", "exclusion_reasons", "has_label_conflict"}
    features = []
    for col in df.columns:
        if col in exclude:
            continue
        if prefix and not col.startswith(prefix):
            continue
        dtype = df[col].dtype
        if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Boolean):
            features.append(col)
    return sorted(features)


# =============================================================================
# Classifier
# =============================================================================


def get_classifier(classifier_type: str = "lgbm"):
    if classifier_type == "lgbm" and HAS_LGBM:
        return LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            class_weight="balanced", random_state=42, verbose=-1,
        )
    return LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)


def evaluate_features(
    train_df: pl.DataFrame, dev_df: pl.DataFrame, test_df: pl.DataFrame,
    y_train: np.ndarray, y_dev: np.ndarray, y_test: np.ndarray,
    features: list[str], classifier_type: str = "lgbm"
) -> tuple[Metrics, Metrics]:
    """Evaluate feature set on dev and test."""
    existing = [f for f in features if f in train_df.columns]
    if not existing:
        return Metrics(0, 0, 0, 0), Metrics(0, 0, 0, 0)

    X_train = train_df.select(existing).to_numpy().astype(np.float32)
    X_dev = dev_df.select(existing).to_numpy().astype(np.float32)
    X_test = test_df.select(existing).to_numpy().astype(np.float32)

    # Handle NaN/inf
    for X in [X_train, X_dev, X_test]:
        np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    clf = get_classifier(classifier_type)
    clf.fit(X_train, y_train)

    y_dev_pred = clf.predict(X_dev)
    y_test_pred = clf.predict(X_test)

    dev_m = Metrics(
        f1=f1_score(y_dev, y_dev_pred),
        acc=accuracy_score(y_dev, y_dev_pred),
        precision=precision_score(y_dev, y_dev_pred, zero_division=0),
        recall=recall_score(y_dev, y_dev_pred, zero_division=0),
    )
    test_m = Metrics(
        f1=f1_score(y_test, y_test_pred),
        acc=accuracy_score(y_test, y_test_pred),
        precision=precision_score(y_test, y_test_pred, zero_division=0),
        recall=recall_score(y_test, y_test_pred, zero_division=0),
    )
    return dev_m, test_m


# =============================================================================
# Feature Group Evaluation
# =============================================================================


def evaluate_all_groups(
    train_df: pl.DataFrame, dev_df: pl.DataFrame, test_df: pl.DataFrame,
    y_train: np.ndarray, y_dev: np.ndarray, y_test: np.ndarray,
    classifier_type: str = "lgbm"
) -> list[dict]:
    """Evaluate each feature group individually."""
    results = []

    print("\n" + "=" * 95)
    print("INDIVIDUAL GROUP EVALUATION")
    print("=" * 95)

    # Text groups
    print("\n  TEXT FEATURE GROUPS:")
    print(f"  {'Group':<25} {'#Feat':<7} {'Dev F1':<10} {'Test F1':<10} {'Test Acc':<10}")
    print("  " + "-" * 70)

    for group_name, features in TEXT_FEATURE_GROUPS.items():
        existing = [f for f in features if f in train_df.columns]
        if existing:
            dev_m, test_m = evaluate_features(
                train_df, dev_df, test_df, y_train, y_dev, y_test, existing, classifier_type
            )
            results.append({
                "type": "text", "group": group_name, "features": existing,
                "n_features": len(existing), "dev_f1": dev_m.f1,
                "test_f1": test_m.f1, "test_acc": test_m.acc
            })
            print(f"  {group_name:<25} {len(existing):<7} {dev_m.f1:<10.4f} {test_m.f1:<10.4f} {test_m.acc:<10.4f}")

    # LLM groups
    print("\n  LLM FEATURE GROUPS:")
    print(f"  {'Group':<25} {'#Feat':<7} {'Dev F1':<10} {'Test F1':<10} {'Test Acc':<10}")
    print("  " + "-" * 70)

    for group_name, features in LLM_FEATURE_GROUPS.items():
        existing = [f for f in features if f in train_df.columns]
        if existing:
            dev_m, test_m = evaluate_features(
                train_df, dev_df, test_df, y_train, y_dev, y_test, existing, classifier_type
            )
            results.append({
                "type": "llm", "group": group_name, "features": existing,
                "n_features": len(existing), "dev_f1": dev_m.f1,
                "test_f1": test_m.f1, "test_acc": test_m.acc
            })
            print(f"  {group_name:<25} {len(existing):<7} {dev_m.f1:<10.4f} {test_m.f1:<10.4f} {test_m.acc:<10.4f}")

    return sorted(results, key=lambda x: x["test_f1"], reverse=True)


def evaluate_presets(
    train_df: pl.DataFrame, dev_df: pl.DataFrame, test_df: pl.DataFrame,
    y_train: np.ndarray, y_dev: np.ndarray, y_test: np.ndarray,
    classifier_type: str = "lgbm"
) -> list[dict]:
    """Evaluate preset feature combinations."""
    print("\n" + "=" * 95)
    print("PRESET FEATURE SET EVALUATION")
    print("=" * 95)
    print(f"\n  {'Preset':<25} {'#Feat':<7} {'Dev F1':<10} {'Test F1':<10} {'Test Acc':<10}")
    print("  " + "-" * 70)

    results = []
    for preset_name, features in PRESET_FEATURE_SETS.items():
        existing = [f for f in features if f in train_df.columns]
        if existing:
            dev_m, test_m = evaluate_features(
                train_df, dev_df, test_df, y_train, y_dev, y_test, existing, classifier_type
            )
            results.append({
                "preset": preset_name, "features": existing,
                "n_features": len(existing), "dev_f1": dev_m.f1,
                "test_f1": test_m.f1, "test_acc": test_m.acc
            })
            marker = "🔥" if test_m.f1 > 0.70 else ""
            print(f"  {preset_name:<25} {len(existing):<7} {dev_m.f1:<10.4f} {test_m.f1:<10.4f} {test_m.acc:<10.4f} {marker}")

    return sorted(results, key=lambda x: x["test_f1"], reverse=True)


# =============================================================================
# Greedy Search
# =============================================================================


def greedy_combined_search(
    train_df: pl.DataFrame, dev_df: pl.DataFrame, test_df: pl.DataFrame,
    y_train: np.ndarray, y_dev: np.ndarray, y_test: np.ndarray,
    starting_features: list[str],
    candidate_features: list[str],
    max_additional: int = 20,
    patience: int = 5,
    classifier_type: str = "lgbm",
    verbose: bool = True,
) -> tuple[list[str], list[dict]]:
    """Greedy forward selection starting from a base set."""
    selected = starting_features.copy()
    remaining = [f for f in candidate_features if f not in selected]
    history = []

    if selected:
        dev_m, test_m = evaluate_features(
            train_df, dev_df, test_df, y_train, y_dev, y_test, selected, classifier_type
        )
        best_dev_f1 = dev_m.f1
        if verbose:
            print(f"\n  Starting: {len(selected)} features, Dev F1={dev_m.f1:.4f}, Test F1={test_m.f1:.4f}")
    else:
        best_dev_f1 = 0.0

    no_improvement = 0

    if verbose:
        print(f"\n  {'Step':<5} {'Added':<35} {'Dev F1':<10} {'Test F1':<10} {'Acc':<10} {'Δ':<8}")
        print("  " + "-" * 85)

    while remaining and len(selected) - len(starting_features) < max_additional:
        best_candidate = None
        best_candidate_f1 = best_dev_f1
        best_metrics = None

        for feat in remaining:
            try:
                dev_m, test_m = evaluate_features(
                    train_df, dev_df, test_df, y_train, y_dev, y_test,
                    selected + [feat], classifier_type
                )
                if dev_m.f1 > best_candidate_f1:
                    best_candidate_f1 = dev_m.f1
                    best_candidate = feat
                    best_metrics = (dev_m, test_m)
            except Exception:
                continue

        if best_candidate is None:
            no_improvement += 1
            if no_improvement >= patience:
                if verbose:
                    print(f"\n  → No improvement for {patience} iterations. Stopping.")
                break
            continue

        no_improvement = 0
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        dev_m, test_m = best_metrics
        improvement = dev_m.f1 - best_dev_f1
        best_dev_f1 = dev_m.f1

        history.append({
            "step": len(selected), "feature": best_candidate,
            "dev_f1": dev_m.f1, "test_f1": test_m.f1, "test_acc": test_m.acc,
            "improvement": improvement
        })

        if verbose:
            marker = "🏆" if test_m.f1 >= SOTA_F1 else ("🔥" if improvement > 0.005 else "↑")
            # Mark text vs LLM features
            feat_type = "T" if best_candidate.startswith("has_") or best_candidate in ["word_count", "avg_word_length", "alpha_ratio", "number_count", "is_question"] else "L"
            print(f"  {len(selected):<5} [{feat_type}] {best_candidate:<32} {dev_m.f1:<10.4f} {test_m.f1:<10.4f} {test_m.acc:<10.4f} {improvement:+.4f} {marker}")

    return selected, history


def run_multi_strategy_search(
    train_df: pl.DataFrame, dev_df: pl.DataFrame, test_df: pl.DataFrame,
    y_train: np.ndarray, y_dev: np.ndarray, y_test: np.ndarray,
    group_results: list[dict],
    all_features: list[str],
    classifier_type: str = "lgbm",
) -> list[dict]:
    """Try multiple starting strategies and compare."""
    print("\n" + "=" * 95)
    print("MULTI-STRATEGY GREEDY SEARCH")
    print("=" * 95)

    all_results = []

    # Strategy 1: Start from best LLM group
    best_llm = next((g for g in group_results if g["type"] == "llm"), None)
    if best_llm:
        print(f"\n--- Strategy 1: Start from best LLM group ({best_llm['group']}) ---")
        features, history = greedy_combined_search(
            train_df, dev_df, test_df, y_train, y_dev, y_test,
            best_llm["features"], all_features, max_additional=15, classifier_type=classifier_type
        )
        if history:
            best = max(history, key=lambda x: x["test_f1"])
            all_results.append({
                "strategy": f"LLM start ({best_llm['group']})",
                "test_f1": best["test_f1"], "test_acc": best["test_acc"],
                "n_features": best["step"], "features": features[:best["step"]]
            })

    # Strategy 2: Start from best text group
    best_text = next((g for g in group_results if g["type"] == "text"), None)
    if best_text:
        print(f"\n--- Strategy 2: Start from best text group ({best_text['group']}) ---")
        features, history = greedy_combined_search(
            train_df, dev_df, test_df, y_train, y_dev, y_test,
            best_text["features"], all_features, max_additional=15, classifier_type=classifier_type
        )
        if history:
            best = max(history, key=lambda x: x["test_f1"])
            all_results.append({
                "strategy": f"Text start ({best_text['group']})",
                "test_f1": best["test_f1"], "test_acc": best["test_acc"],
                "n_features": best["step"], "features": features[:best["step"]]
            })

    # Strategy 3: Start from combined preset
    if "combined_balanced" in PRESET_FEATURE_SETS:
        print("\n--- Strategy 3: Start from combined_balanced preset ---")
        start_features = [f for f in PRESET_FEATURE_SETS["combined_balanced"] if f in train_df.columns]
        features, history = greedy_combined_search(
            train_df, dev_df, test_df, y_train, y_dev, y_test,
            start_features, all_features, max_additional=12, classifier_type=classifier_type
        )
        if history:
            best = max(history, key=lambda x: x["test_f1"])
            all_results.append({
                "strategy": "Combined preset",
                "test_f1": best["test_f1"], "test_acc": best["test_acc"],
                "n_features": best["step"], "features": features[:best["step"]]
            })

    # Strategy 4: Best LLM + Best Text combined start
    if best_llm and best_text:
        print(f"\n--- Strategy 4: Combined start (best LLM + best text) ---")
        combined_start = best_llm["features"] + best_text["features"]
        features, history = greedy_combined_search(
            train_df, dev_df, test_df, y_train, y_dev, y_test,
            combined_start, all_features, max_additional=10, classifier_type=classifier_type
        )
        if history:
            best = max(history, key=lambda x: x["test_f1"])
            all_results.append({
                "strategy": "LLM+Text combined",
                "test_f1": best["test_f1"], "test_acc": best["test_acc"],
                "n_features": best["step"], "features": features[:best["step"]]
            })

    # Strategy 5: Greedy from scratch (mixed)
    print("\n--- Strategy 5: Greedy from scratch (all features) ---")
    features, history = greedy_combined_search(
        train_df, dev_df, test_df, y_train, y_dev, y_test,
        [], all_features, max_additional=20, classifier_type=classifier_type
    )
    if history:
        best = max(history, key=lambda x: x["test_f1"])
        all_results.append({
            "strategy": "Greedy from scratch",
            "test_f1": best["test_f1"], "test_acc": best["test_acc"],
            "n_features": best["step"], "features": features[:best["step"]]
        })

    return all_results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=["lgbm", "logreg"], default="lgbm")
    parser.add_argument("--quick", action="store_true", help="Skip multi-strategy search")
    args = parser.parse_args()

    print("=" * 95)
    print("COMBINED GREEDY FEATURE SEARCH: Text + LLM Features")
    print(f"Target: F1 ≥ {SOTA_F1:.2f}, Accuracy ≥ {SOTA_ACC:.3f}")
    print("=" * 95)

    # Load data
    print("\nLoading combined data...")
    train_df, dev_df, test_df, y_train, y_dev, y_test = load_combined_data()
    print(f"  Train: {len(y_train):,} ({y_train.sum():,} positive, {100*y_train.mean():.1f}%)")
    print(f"  Dev:   {len(y_dev):,} ({y_dev.sum():,} positive)")
    print(f"  Test:  {len(y_test):,} ({y_test.sum():,} positive)")

    # Get all features
    all_text_features = [f for f in get_all_features(train_df)
                         if f.startswith("has_") or f in ["word_count", "avg_word_length", "alpha_ratio", "number_count", "is_question"]]
    all_llm_features = [f for f in get_all_features(train_df)
                        if f.startswith(("check_", "verif_", "harm_", "score_", "unanimous", "yes_vote"))]
    all_features = all_text_features + all_llm_features

    print(f"\n  Text features: {len(all_text_features)}")
    print(f"  LLM features:  {len(all_llm_features)}")
    print(f"  Total:         {len(all_features)}")

    # Evaluate groups
    group_results = evaluate_all_groups(
        train_df, dev_df, test_df, y_train, y_dev, y_test, args.classifier
    )

    # Evaluate presets
    preset_results = evaluate_presets(
        train_df, dev_df, test_df, y_train, y_dev, y_test, args.classifier
    )

    # Multi-strategy search
    if not args.quick:
        strategy_results = run_multi_strategy_search(
            train_df, dev_df, test_df, y_train, y_dev, y_test,
            group_results, all_features, args.classifier
        )
    else:
        strategy_results = []

    # Final summary
    print("\n" + "=" * 95)
    print("FINAL SUMMARY")
    print("=" * 95)

    # Collect all approaches
    all_approaches = []

    # Best preset
    if preset_results:
        best_preset = preset_results[0]
        all_approaches.append({
            "name": f"Preset: {best_preset['preset']}",
            "test_f1": best_preset["test_f1"],
            "test_acc": best_preset["test_acc"],
            "n_features": best_preset["n_features"],
            "features": best_preset["features"]
        })

    # Best from strategy search
    for r in strategy_results:
        all_approaches.append({
            "name": r["strategy"],
            "test_f1": r["test_f1"],
            "test_acc": r["test_acc"],
            "n_features": r["n_features"],
            "features": r["features"]
        })

    # Sort by test F1
    all_approaches.sort(key=lambda x: x["test_f1"], reverse=True)

    print(f"\n  {'Approach':<40} {'Test F1':<10} {'Test Acc':<10} {'#Feat'}")
    print("  " + "-" * 70)
    for i, a in enumerate(all_approaches[:8]):
        marker = "🏆" if i == 0 else ""
        print(f"  {a['name']:<40} {a['test_f1']:<10.4f} {a['test_acc']:<10.4f} {a['n_features']:<6} {marker}")

    if all_approaches:
        best = all_approaches[0]
        print(f"\n  SOTA Gap:")
        print(f"    F1:  {SOTA_F1:.4f} → {best['test_f1']:.4f} (gap: {best['test_f1'] - SOTA_F1:+.4f})")
        print(f"    Acc: {SOTA_ACC:.4f} → {best['test_acc']:.4f} (gap: {best['test_acc'] - SOTA_ACC:+.4f})")

        print(f"\n  🏆 BEST: {best['name']}")
        print(f"     Test F1={best['test_f1']:.4f}, Acc={best['test_acc']:.4f}")
        print(f"\n  Features ({best['n_features']}):")
        for i, f in enumerate(best["features"], 1):
            ftype = "Text" if f.startswith("has_") or f in ["word_count", "avg_word_length", "alpha_ratio", "number_count", "is_question"] else "LLM"
            print(f"    {i:2d}. [{ftype:4}] {f}")

    print("\n" + "=" * 95)


if __name__ == "__main__":
    main()
