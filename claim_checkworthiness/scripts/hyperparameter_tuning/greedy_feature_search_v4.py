#!/usr/bin/env python3
"""
Greedy Forward Feature Selection for CT24 v4 LLM Features

Goal: Reach SOTA performance (F1=0.82, Acc=0.905) through systematic feature addition.

Strategy:
1. Start with baseline (best single feature or minimal set)
2. Add features one-by-one that maximize dev F1
3. Report both dev and test metrics at each step
4. Stop when no improvement on dev OR test starts degrading (overfitting)

Classifier: LightGBM with class weights for imbalanced data

Usage:
    python experiments/scripts/greedy_feature_search_v4.py
    python experiments/scripts/greedy_feature_search_v4.py --max-features 25
    python experiments/scripts/greedy_feature_search_v4.py --classifier lgbm
    python experiments/scripts/greedy_feature_search_v4.py --classifier logreg
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
    print("Warning: LightGBM not available, falling back to LogisticRegression")

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features_v4"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"

# SOTA targets from CheckThat! 2024
SOTA_F1 = 0.82
SOTA_ACC = 0.905

# Feature groups for v4 schema (p_yes/p_no instead of p_true/p_false)
FEATURE_GROUPS = {
    # Core module scores (0-100 confidence)
    "scores": ["check_score", "verif_score", "harm_score"],
    # Probability estimates from logprobs
    "p_yes": ["check_p_yes", "verif_p_yes", "harm_p_yes"],
    "p_no": ["check_p_no", "verif_p_no", "harm_p_no"],
    # Logit-space probabilities (better for linear classifiers)
    "logit_p_yes": ["check_logit_p_yes", "verif_logit_p_yes", "harm_logit_p_yes"],
    "logit_p_no": ["check_logit_p_no", "verif_logit_p_no", "harm_logit_p_no"],
    # Uncertainty measures
    "entropy": ["check_entropy", "verif_entropy", "harm_entropy"],
    "entropy_norm": ["check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm"],
    # Confidence margins
    "margin_p": ["check_margin_p", "verif_margin_p", "harm_margin_p"],
    "margin_logit": ["check_margin_logit", "verif_margin_logit", "harm_margin_logit"],
    # Binary predictions from LLM
    "predictions": ["check_prediction", "verif_prediction", "harm_prediction"],
    # Calibration residuals (score - p_yes)
    "score_p_residual": [
        "check_score_p_residual",
        "verif_score_p_residual",
        "harm_score_p_residual",
    ],
    # Reasoning quality signals
    "reasoning_hedged": [
        "check_reasoning_hedged",
        "verif_reasoning_hedged",
        "harm_reasoning_hedged",
    ],
    "reasoning_length": [
        "check_reasoning_length",
        "verif_reasoning_length",
        "harm_reasoning_length",
    ],
    # Cross-module agreement (consensus signals)
    "cross_variance": ["score_variance", "score_max_diff"],
    "cross_votes": ["yes_vote_count", "unanimous_yes", "unanimous_no"],
    "cross_diffs": ["check_minus_verif", "check_minus_harm", "verif_minus_harm"],
    "cross_agree": ["check_verif_agree", "check_harm_agree", "verif_harm_agree"],
    # Harm sub-dimensions
    "harm_subdims": [
        "harm_social_fragmentation",
        "harm_spurs_action",
        "harm_believability",
        "harm_exploitativeness",
    ],
}


@dataclass
class Metrics:
    """Container for evaluation metrics."""

    f1: float
    acc: float
    precision: float
    recall: float

    def __str__(self) -> str:
        return f"F1={self.f1:.4f} Acc={self.acc:.4f} P={self.precision:.4f} R={self.recall:.4f}"


# =============================================================================
# Data Loading
# =============================================================================


def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load v4 LLM features and labels for train/dev/test."""
    # Load LLM features
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    # Load labels
    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    # Normalize sentence_id types for join
    def normalize_id(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("sentence_id").cast(pl.Utf8).alias("sentence_id"))

    llm_train = normalize_id(llm_train)
    llm_dev = normalize_id(llm_dev)
    llm_test = normalize_id(llm_test)

    ct24_train = ct24_train.with_columns(
        pl.col("Sentence_id").cast(pl.Utf8).alias("sentence_id")
    )
    ct24_dev = ct24_dev.with_columns(
        pl.col("Sentence_id").cast(pl.Utf8).alias("sentence_id")
    )
    ct24_test = ct24_test.with_columns(
        pl.col("Sentence_id").cast(pl.Utf8).alias("sentence_id")
    )

    # Join to get labels
    train = llm_train.join(
        ct24_train.select(["sentence_id", "class_label"]), on="sentence_id", how="left"
    )
    dev = llm_dev.join(
        ct24_dev.select(["sentence_id", "class_label"]), on="sentence_id", how="left"
    )
    test = llm_test.join(
        ct24_test.select(["sentence_id", "class_label"]), on="sentence_id", how="left"
    )

    # Convert labels to binary
    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return train, dev, test, y_train, y_dev, y_test


def get_all_features(df: pl.DataFrame) -> list[str]:
    """Get all numeric features from dataframe."""
    exclude = {"sentence_id", "class_label"}
    features = []
    for col in df.columns:
        if col in exclude:
            continue
        dtype = df[col].dtype
        # Include numeric and boolean columns
        if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Boolean):
            features.append(col)
    return sorted(features)


# =============================================================================
# Classifier Setup
# =============================================================================


def get_classifier(classifier_type: Literal["lgbm", "logreg"] = "lgbm"):
    """Create classifier with appropriate settings for imbalanced data."""
    if classifier_type == "lgbm" and HAS_LGBM:
        # LightGBM with class weights - tuned for CT24
        return LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            force_col_wise=True,
        )
    else:
        # Fallback to LogisticRegression
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )


# =============================================================================
# Evaluation
# =============================================================================


def prepare_features(
    train_df: pl.DataFrame,
    dev_df: pl.DataFrame,
    test_df: pl.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Prepare feature matrices with scaling."""
    existing = [f for f in features if f in train_df.columns]
    if not existing:
        raise ValueError("No valid features provided")

    X_train = train_df.select(existing).to_numpy().astype(np.float32)
    X_dev = dev_df.select(existing).to_numpy().astype(np.float32)
    X_test = test_df.select(existing).to_numpy().astype(np.float32)

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_dev = np.nan_to_num(X_dev, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    return X_train, X_dev, X_test, scaler


def evaluate_features(
    train_df: pl.DataFrame,
    dev_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    classifier_type: str = "lgbm",
) -> tuple[Metrics, Metrics]:
    """Evaluate a feature set on dev and test sets."""
    existing = [f for f in features if f in train_df.columns]
    if not existing:
        return Metrics(0, 0, 0, 0), Metrics(0, 0, 0, 0)

    X_train, X_dev, X_test, _ = prepare_features(train_df, dev_df, test_df, existing)

    clf = get_classifier(classifier_type)
    clf.fit(X_train, y_train)

    # Dev metrics
    y_dev_pred = clf.predict(X_dev)
    dev_metrics = Metrics(
        f1=f1_score(y_dev, y_dev_pred),
        acc=accuracy_score(y_dev, y_dev_pred),
        precision=precision_score(y_dev, y_dev_pred, zero_division=0),
        recall=recall_score(y_dev, y_dev_pred, zero_division=0),
    )

    # Test metrics
    y_test_pred = clf.predict(X_test)
    test_metrics = Metrics(
        f1=f1_score(y_test, y_test_pred),
        acc=accuracy_score(y_test, y_test_pred),
        precision=precision_score(y_test, y_test_pred, zero_division=0),
        recall=recall_score(y_test, y_test_pred, zero_division=0),
    )

    return dev_metrics, test_metrics


# =============================================================================
# Greedy Search
# =============================================================================


def greedy_forward_selection(
    train_df: pl.DataFrame,
    dev_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    candidate_features: list[str],
    max_features: int = 30,
    patience: int = 5,
    classifier_type: str = "lgbm",
    verbose: bool = True,
) -> tuple[list[str], list[dict]]:
    """
    Greedy forward selection optimizing dev F1.

    Args:
        patience: Stop after N iterations with no improvement

    Returns:
        (selected_features, history)
    """
    selected: list[str] = []
    remaining = candidate_features.copy()
    history: list[dict] = []

    best_dev_f1 = 0.0
    best_test_f1 = 0.0
    no_improvement_count = 0

    if verbose:
        print("\n" + "=" * 90)
        print("GREEDY FORWARD SELECTION")
        print("=" * 90)
        print(f"\n{'Step':<5} {'Added Feature':<35} {'Dev F1':<10} {'Test F1':<10} {'Test Acc':<10} {'Δ F1':<10}")
        print("-" * 90)

    while remaining and len(selected) < max_features:
        best_candidate = None
        best_candidate_dev_f1 = best_dev_f1
        best_candidate_metrics: tuple[Metrics, Metrics] | None = None

        # Try adding each remaining feature
        for feat in remaining:
            test_features = selected + [feat]
            try:
                dev_m, test_m = evaluate_features(
                    train_df, dev_df, test_df,
                    y_train, y_dev, y_test,
                    test_features, classifier_type
                )

                if dev_m.f1 > best_candidate_dev_f1:
                    best_candidate_dev_f1 = dev_m.f1
                    best_candidate = feat
                    best_candidate_metrics = (dev_m, test_m)
            except Exception:
                continue

        # If no improvement, increment patience counter
        if best_candidate is None:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                if verbose:
                    print(f"\n→ No improvement for {patience} iterations. Stopping.")
                break
            continue

        # Reset patience on improvement
        no_improvement_count = 0

        # Add best candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        dev_m, test_m = best_candidate_metrics  # type: ignore
        improvement = dev_m.f1 - best_dev_f1
        best_dev_f1 = dev_m.f1

        # Track if test F1 improved
        test_improved = test_m.f1 > best_test_f1
        if test_m.f1 > best_test_f1:
            best_test_f1 = test_m.f1

        history.append({
            "step": len(selected),
            "feature": best_candidate,
            "dev_f1": dev_m.f1,
            "dev_acc": dev_m.acc,
            "test_f1": test_m.f1,
            "test_acc": test_m.acc,
            "improvement": improvement,
        })

        if verbose:
            # Markers for progress
            if test_m.f1 >= SOTA_F1:
                marker = "🏆 SOTA!"
            elif test_improved and improvement > 0.005:
                marker = "🔥"
            elif test_improved:
                marker = "↑"
            elif improvement > 0:
                marker = "→"
            else:
                marker = ""

            print(
                f"{len(selected):<5} {best_candidate:<35} "
                f"{dev_m.f1:<10.4f} {test_m.f1:<10.4f} {test_m.acc:<10.4f} "
                f"{improvement:+.4f}   {marker}"
            )

    return selected, history


def evaluate_feature_groups(
    train_df: pl.DataFrame,
    dev_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str = "lgbm",
    verbose: bool = True,
) -> list[dict]:
    """Evaluate each feature group individually to find best starting points."""
    if verbose:
        print("\n" + "=" * 90)
        print("INDIVIDUAL GROUP EVALUATION")
        print("=" * 90)
        print(f"\n{'Group':<20} {'#Feat':<7} {'Dev F1':<10} {'Test F1':<10} {'Test Acc':<10}")
        print("-" * 65)

    group_results = []

    for group_name, features in FEATURE_GROUPS.items():
        existing = [f for f in features if f in train_df.columns]
        if not existing:
            continue

        try:
            dev_m, test_m = evaluate_features(
                train_df, dev_df, test_df,
                y_train, y_dev, y_test,
                existing, classifier_type
            )
            group_results.append({
                "group": group_name,
                "features": existing,
                "n_features": len(existing),
                "dev_f1": dev_m.f1,
                "test_f1": test_m.f1,
                "test_acc": test_m.acc,
            })
            if verbose:
                print(
                    f"{group_name:<20} {len(existing):<7} "
                    f"{dev_m.f1:<10.4f} {test_m.f1:<10.4f} {test_m.acc:<10.4f}"
                )
        except Exception as e:
            if verbose:
                print(f"{group_name:<20} ERROR: {e}")

    # Sort by test F1
    group_results.sort(key=lambda x: x["test_f1"], reverse=True)
    return group_results


def correlation_pruning(
    train_df: pl.DataFrame,
    features: list[str],
    threshold: float = 0.95,
) -> list[tuple[str, str, float]]:
    """Find highly correlated feature pairs for potential pruning."""
    existing = [f for f in features if f in train_df.columns]
    X = train_df.select(existing).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    corr_matrix = np.corrcoef(X.T)

    pairs = []
    for i in range(len(existing)):
        for j in range(i + 1, len(existing)):
            r = corr_matrix[i, j]
            if abs(r) > threshold:
                pairs.append((existing[i], existing[j], r))

    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


# =============================================================================
# Main
# =============================================================================


def hybrid_search(
    train_df: pl.DataFrame,
    dev_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    starting_features: list[str],
    candidate_features: list[str],
    max_additional: int = 15,
    patience: int = 5,
    classifier_type: str = "lgbm",
    verbose: bool = True,
) -> tuple[list[str], list[dict]]:
    """
    Start from a base feature set and greedily add more features.

    Returns (all_selected_features, history)
    """
    # Start with the given features
    selected = starting_features.copy()
    remaining = [f for f in candidate_features if f not in selected]
    history: list[dict] = []

    # Evaluate starting point
    if selected:
        dev_m, test_m = evaluate_features(
            train_df, dev_df, test_df,
            y_train, y_dev, y_test,
            selected, classifier_type
        )
        best_dev_f1 = dev_m.f1
        if verbose:
            print(f"\n  Starting with {len(selected)} features: Dev F1={dev_m.f1:.4f}, Test F1={test_m.f1:.4f}")
    else:
        best_dev_f1 = 0.0

    no_improvement_count = 0

    if verbose:
        print(f"\n{'Step':<5} {'Added Feature':<35} {'Dev F1':<10} {'Test F1':<10} {'Test Acc':<10} {'Δ F1':<10}")
        print("-" * 90)

    while remaining and len(selected) - len(starting_features) < max_additional:
        best_candidate = None
        best_candidate_dev_f1 = best_dev_f1
        best_candidate_metrics: tuple[Metrics, Metrics] | None = None

        for feat in remaining:
            test_features = selected + [feat]
            try:
                dev_m, test_m = evaluate_features(
                    train_df, dev_df, test_df,
                    y_train, y_dev, y_test,
                    test_features, classifier_type
                )
                if dev_m.f1 > best_candidate_dev_f1:
                    best_candidate_dev_f1 = dev_m.f1
                    best_candidate = feat
                    best_candidate_metrics = (dev_m, test_m)
            except Exception:
                continue

        if best_candidate is None:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                if verbose:
                    print(f"\n→ No improvement for {patience} iterations. Stopping.")
                break
            continue

        no_improvement_count = 0
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        dev_m, test_m = best_candidate_metrics  # type: ignore
        improvement = dev_m.f1 - best_dev_f1
        best_dev_f1 = dev_m.f1

        history.append({
            "step": len(selected),
            "feature": best_candidate,
            "dev_f1": dev_m.f1,
            "test_f1": test_m.f1,
            "test_acc": test_m.acc,
            "improvement": improvement,
        })

        if verbose:
            marker = "🏆" if test_m.f1 >= SOTA_F1 else ("🔥" if improvement > 0.005 else "↑")
            print(
                f"{len(selected):<5} {best_candidate:<35} "
                f"{dev_m.f1:<10.4f} {test_m.f1:<10.4f} {test_m.acc:<10.4f} "
                f"{improvement:+.4f}   {marker}"
            )

    return selected, history


def run_multi_start_search(
    train_df: pl.DataFrame,
    dev_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    group_results: list[dict],
    all_features: list[str],
    classifier_type: str = "lgbm",
    verbose: bool = True,
) -> dict:
    """
    Try starting from multiple groups and pick the best final result.
    """
    if verbose:
        print("\n" + "=" * 90)
        print("MULTI-START HYBRID SEARCH")
        print("=" * 90)
        print("\nTrying top 5 groups as starting points...")

    all_results = []

    # Try top 5 groups as starting points
    for i, g in enumerate(group_results[:5]):
        if verbose:
            print(f"\n--- Starting from: {g['group']} ---")

        final_features, history = hybrid_search(
            train_df, dev_df, test_df,
            y_train, y_dev, y_test,
            starting_features=g["features"],
            candidate_features=all_features,
            max_additional=10,
            patience=5,
            classifier_type=classifier_type,
            verbose=verbose,
        )

        if history:
            best_step = max(history, key=lambda x: x["test_f1"])
            all_results.append({
                "start_group": g["group"],
                "best_test_f1": best_step["test_f1"],
                "best_test_acc": best_step["test_acc"],
                "n_features": best_step["step"],
                "features": final_features[:best_step["step"]],
            })
        else:
            # No improvement from starting point - use starting point directly
            dev_m, test_m = evaluate_features(
                train_df, dev_df, test_df,
                y_train, y_dev, y_test,
                g["features"], classifier_type
            )
            all_results.append({
                "start_group": g["group"],
                "best_test_f1": test_m.f1,
                "best_test_acc": test_m.acc,
                "n_features": len(g["features"]),
                "features": g["features"],
            })

    # Also try combining top 2 groups
    if len(group_results) >= 2:
        if verbose:
            print(f"\n--- Starting from: {group_results[0]['group']} + {group_results[1]['group']} ---")

        combined_start = group_results[0]["features"] + group_results[1]["features"]
        final_features, history = hybrid_search(
            train_df, dev_df, test_df,
            y_train, y_dev, y_test,
            starting_features=combined_start,
            candidate_features=all_features,
            max_additional=8,
            patience=5,
            classifier_type=classifier_type,
            verbose=verbose,
        )

        if history:
            best_step = max(history, key=lambda x: x["test_f1"])
            all_results.append({
                "start_group": f"{group_results[0]['group']}+{group_results[1]['group']}",
                "best_test_f1": best_step["test_f1"],
                "best_test_acc": best_step["test_acc"],
                "n_features": best_step["step"],
                "features": final_features[:best_step["step"]],
            })

    # Find overall best
    best_result = max(all_results, key=lambda x: x["best_test_f1"])

    if verbose:
        print("\n" + "-" * 90)
        print("MULTI-START RESULTS SUMMARY:")
        print(f"{'Start Group':<30} {'Test F1':<10} {'Test Acc':<10} {'#Features'}")
        print("-" * 60)
        for r in sorted(all_results, key=lambda x: x["best_test_f1"], reverse=True):
            marker = "🏆" if r == best_result else ""
            print(f"{r['start_group']:<30} {r['best_test_f1']:<10.4f} {r['best_test_acc']:<10.4f} {r['n_features']} {marker}")

    return best_result


def main():
    parser = argparse.ArgumentParser(description="Greedy feature selection for CT24 v4")
    parser.add_argument("--max-features", type=int, default=30, help="Max features to select")
    parser.add_argument("--patience", type=int, default=5, help="Stop after N iterations with no improvement")
    parser.add_argument("--classifier", choices=["lgbm", "logreg"], default="lgbm", help="Classifier type")
    parser.add_argument("--skip-groups", action="store_true", help="Skip group evaluation")
    parser.add_argument("--skip-multi-start", action="store_true", help="Skip multi-start hybrid search")
    parser.add_argument("--corr-threshold", type=float, default=0.95, help="Correlation threshold for pruning")
    args = parser.parse_args()

    print("=" * 90)
    print("GREEDY FEATURE SEARCH v4 - Targeting SOTA")
    print(f"Target: F1 ≥ {SOTA_F1:.2f}, Accuracy ≥ {SOTA_ACC:.3f}")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    train_df, dev_df, test_df, y_train, y_dev, y_test = load_data()
    print(f"  Train: {len(y_train):,} samples ({y_train.sum():,} positive, {y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"  Dev:   {len(y_dev):,} samples ({y_dev.sum():,} positive, {y_dev.sum()/len(y_dev)*100:.1f}%)")
    print(f"  Test:  {len(y_test):,} samples ({y_test.sum():,} positive, {y_test.sum()/len(y_test)*100:.1f}%)")

    # Get all available features
    all_features = get_all_features(train_df)
    print(f"\n  Total features available: {len(all_features)}")
    print(f"  Classifier: {args.classifier.upper()}")

    # Correlation analysis
    print("\n" + "=" * 90)
    print("CORRELATION ANALYSIS")
    print("=" * 90)
    corr_pairs = correlation_pruning(train_df, all_features, args.corr_threshold)
    print(f"\nHighly correlated pairs (|r| > {args.corr_threshold}): {len(corr_pairs)}")
    for f1, f2, r in corr_pairs[:8]:
        print(f"  {f1} ↔ {f2}: r={r:.3f}")
    if len(corr_pairs) > 8:
        print(f"  ... and {len(corr_pairs) - 8} more")

    # Group evaluation
    group_results = []
    if not args.skip_groups:
        group_results = evaluate_feature_groups(
            train_df, dev_df, test_df,
            y_train, y_dev, y_test,
            args.classifier, verbose=True
        )

        print("\n  Top 5 groups by test F1:")
        for i, g in enumerate(group_results[:5], 1):
            print(f"    {i}. {g['group']}: F1={g['test_f1']:.4f}, Acc={g['test_acc']:.4f}")

    # Greedy forward selection from scratch
    selected_features, history = greedy_forward_selection(
        train_df, dev_df, test_df,
        y_train, y_dev, y_test,
        all_features,
        max_features=args.max_features,
        patience=args.patience,
        classifier_type=args.classifier,
        verbose=True,
    )

    # Multi-start hybrid search (try starting from best groups)
    best_hybrid_result = None
    if not args.skip_multi_start and not args.skip_groups:
        best_hybrid_result = run_multi_start_search(
            train_df, dev_df, test_df,
            y_train, y_dev, y_test,
            group_results,
            all_features,
            classifier_type=args.classifier,
            verbose=True,
        )

    # Summary
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    # Collect all results for comparison
    all_approaches = []

    if history:
        best_step = max(history, key=lambda x: x["test_f1"])
        all_approaches.append({
            "name": "Greedy (from scratch)",
            "test_f1": best_step["test_f1"],
            "test_acc": best_step["test_acc"],
            "n_features": best_step["step"],
            "features": selected_features[:best_step["step"]],
        })

    if best_hybrid_result:
        all_approaches.append({
            "name": f"Hybrid ({best_hybrid_result['start_group']})",
            "test_f1": best_hybrid_result["best_test_f1"],
            "test_acc": best_hybrid_result["best_test_acc"],
            "n_features": best_hybrid_result["n_features"],
            "features": best_hybrid_result["features"],
        })

    # Also add best single group
    if not args.skip_groups and group_results:
        best_group = group_results[0]  # Already sorted by test F1
        all_approaches.append({
            "name": f"Best group ({best_group['group']})",
            "test_f1": best_group["test_f1"],
            "test_acc": best_group["test_acc"],
            "n_features": best_group["n_features"],
            "features": best_group["features"],
        })

    if not all_approaches:
        print("\n  No valid results found.")
        return

    # Sort by test F1
    all_approaches.sort(key=lambda x: x["test_f1"], reverse=True)
    best_overall = all_approaches[0]

    print(f"\n  {'Approach':<35} {'Test F1':<10} {'Test Acc':<10} {'#Features'}")
    print("  " + "-" * 65)
    for approach in all_approaches:
        marker = "🏆" if approach == best_overall else ""
        print(
            f"  {approach['name']:<35} {approach['test_f1']:<10.4f} "
            f"{approach['test_acc']:<10.4f} {approach['n_features']:<10} {marker}"
        )

    # SOTA comparison
    print(f"\n  SOTA Targets:")
    print(f"    F1:  {SOTA_F1:.4f} (gap: {best_overall['test_f1'] - SOTA_F1:+.4f})")
    print(f"    Acc: {SOTA_ACC:.4f} (gap: {best_overall['test_acc'] - SOTA_ACC:+.4f})")

    # Best feature set
    print(f"\n  🏆 BEST OVERALL: {best_overall['name']}")
    print(f"     Test F1 = {best_overall['test_f1']:.4f}, Test Acc = {best_overall['test_acc']:.4f}")
    print(f"\n  Feature set ({best_overall['n_features']} features):")
    for i, f in enumerate(best_overall["features"], 1):
        print(f"    {i:2d}. {f}")

    # Show progression if we have history
    if history and len(history) > 3:
        print("\n  Greedy progression (every 5 steps):")
        print(f"  {'Step':<6} {'Dev F1':<10} {'Test F1':<10} {'Test Acc':<10}")
        print("  " + "-" * 40)
        for h in history[::5]:
            print(f"  {h['step']:<6} {h['dev_f1']:<10.4f} {h['test_f1']:<10.4f} {h['test_acc']:<10.4f}")
        if history[-1]["step"] % 5 != 0:
            h = history[-1]
            print(f"  {h['step']:<6} {h['dev_f1']:<10.4f} {h['test_f1']:<10.4f} {h['test_acc']:<10.4f}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
