#!/usr/bin/env python3
"""
Plot Learning Curves for XGBoost + SMOTE with LLM Features

Configuration:
- Model: XGBoost with scale_pos_weight=3
- Resampling: SMOTE with k_neighbors=5
- Features: All LLM features

Shows train vs test F1/accuracy as training size increases.
Helps diagnose overfitting (high variance) vs underfitting (high bias).

Usage:
    python experiments/scripts/plot_learning_curves.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("ERROR: xgboost not installed")

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("ERROR: imblearn not installed. Run: pip install imbalanced-learn")

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent / "visualizations"

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

SOTA_F1 = 0.82


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load all LLM features and labels."""
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    # Join on sentence_id
    llm_train = llm_train.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_test = llm_test.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    train = llm_train.join(ct24_train.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")
    test = llm_test.join(ct24_test.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")

    # Get all available features
    all_features = []
    for group_features in FEATURE_GROUPS.values():
        for f in group_features:
            if f in train.columns and f not in all_features:
                all_features.append(f)

    X_train = train.select(all_features).to_numpy().astype(np.float32)
    X_test = test.select(all_features).to_numpy().astype(np.float32)

    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    return X_train, X_test, y_train, y_test, all_features


def plot_train_vs_test_by_size(X_train, y_train, X_test, y_test, use_smote: bool, ax_pair):
    """Plot train and test scores as training size increases."""
    scaler = StandardScaler()
    smote = SMOTE(k_neighbors=5, random_state=42) if use_smote else None

    clf_name = "XGBoost (w=3) + SMOTE (k=5)" if use_smote else "XGBoost (w=3) No SMOTE"

    # Define training sizes to test
    train_sizes = [200, 500, 1000, 2000, 5000, 10000, 15000, len(X_train)]
    train_sizes = [s for s in train_sizes if s <= len(X_train)]

    train_f1s = []
    test_f1s = []
    train_accs = []
    test_accs = []
    actual_train_sizes = []

    for size in train_sizes:
        # Sample training data
        idx = np.random.RandomState(42).permutation(len(X_train))[:size]
        X_sub = X_train[idx]
        y_sub = y_train[idx]

        # Apply SMOTE if enabled
        if smote is not None:
            try:
                X_sub_res, y_sub_res = smote.fit_resample(X_sub, y_sub)
            except Exception:
                # If SMOTE fails (e.g., too few samples), skip
                continue
        else:
            X_sub_res, y_sub_res = X_sub, y_sub

        actual_train_sizes.append(size)

        # Scale
        X_sub_s = scaler.fit_transform(X_sub_res)
        X_test_s = scaler.transform(X_test)

        # Train XGBoost
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=3,
            use_label_encoder=False,
        )
        clf.fit(X_sub_s, y_sub_res)

        train_pred = clf.predict(X_sub_s)
        test_pred = clf.predict(X_test_s)

        train_f1s.append(f1_score(y_sub_res, train_pred))
        test_f1s.append(f1_score(y_test, test_pred))
        train_accs.append(accuracy_score(y_sub_res, train_pred))
        test_accs.append(accuracy_score(y_test, test_pred))

    # Plot F1
    ax_pair[0].plot(actual_train_sizes, train_f1s, 'o-', color='blue', label='Train F1', linewidth=2, markersize=8)
    ax_pair[0].plot(actual_train_sizes, test_f1s, 'o-', color='red', label='Test F1', linewidth=2, markersize=8)
    ax_pair[0].axhline(y=SOTA_F1, color='green', linestyle='--', alpha=0.7, label=f'SOTA F1={SOTA_F1}')
    ax_pair[0].set_xlabel('Training Size (before SMOTE)', fontsize=11)
    ax_pair[0].set_ylabel('F1 Score', fontsize=11)
    ax_pair[0].set_title(f'{clf_name} - F1 vs Training Size', fontsize=12, fontweight='bold')
    ax_pair[0].legend(loc='lower right')
    ax_pair[0].grid(True, alpha=0.3)
    ax_pair[0].set_xscale('log')

    # Add gap annotation
    gap_f1 = train_f1s[-1] - test_f1s[-1]
    ax_pair[0].annotate(f'Final Gap: {gap_f1:.3f}\nTest F1: {test_f1s[-1]:.3f}',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Plot Accuracy
    ax_pair[1].plot(actual_train_sizes, train_accs, 'o-', color='blue', label='Train Acc', linewidth=2, markersize=8)
    ax_pair[1].plot(actual_train_sizes, test_accs, 'o-', color='red', label='Test Acc', linewidth=2, markersize=8)
    ax_pair[1].set_xlabel('Training Size (before SMOTE)', fontsize=11)
    ax_pair[1].set_ylabel('Accuracy', fontsize=11)
    ax_pair[1].set_title(f'{clf_name} - Accuracy vs Training Size', fontsize=12, fontweight='bold')
    ax_pair[1].legend(loc='lower right')
    ax_pair[1].grid(True, alpha=0.3)
    ax_pair[1].set_xscale('log')

    gap_acc = train_accs[-1] - test_accs[-1]
    ax_pair[1].annotate(f'Final Gap: {gap_acc:.3f}\nTest Acc: {test_accs[-1]:.3f}',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    return actual_train_sizes, train_f1s, test_f1s, train_accs, test_accs


def plot_iterations_curve(X_train, y_train, X_test, y_test):
    """Plot performance vs number of XGBoost iterations (trees)."""
    scaler = StandardScaler()
    smote = SMOTE(k_neighbors=5, random_state=42)

    # Apply SMOTE
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    X_train_s = scaler.fit_transform(X_train_res)
    X_test_s = scaler.transform(X_test)

    iterations = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]

    train_f1s = []
    test_f1s = []

    for n_iter in iterations:
        clf = XGBClassifier(
            n_estimators=n_iter,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=3,
            use_label_encoder=False,
        )
        clf.fit(X_train_s, y_train_res)

        train_pred = clf.predict(X_train_s)
        test_pred = clf.predict(X_test_s)

        train_f1s.append(f1_score(y_train_res, train_pred))
        test_f1s.append(f1_score(y_test, test_pred))

    return iterations, train_f1s, test_f1s


def main():
    if not HAS_XGBOOST or not HAS_IMBLEARN:
        print("Missing required packages. Install xgboost and imbalanced-learn.")
        return

    print("=" * 80)
    print("LEARNING CURVES - XGBoost (w=3) + SMOTE (k=5) + All LLM Features")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading LLM features...")
    X_train, X_test, y_train, y_test, feature_names = load_data()

    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {len(y_train)} samples ({100*y_train.mean():.1f}% positive)")
    print(f"  Test:  {len(y_test)} samples ({100*y_test.mean():.1f}% positive)")

    # =========================================================================
    # Plot 1: Train vs Test by training size (with and without SMOTE)
    # =========================================================================
    print("\nGenerating learning curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # With SMOTE
    print("  Processing XGBoost + SMOTE...")
    sizes_smote, train_f1_smote, test_f1_smote, train_acc_smote, test_acc_smote = plot_train_vs_test_by_size(
        X_train, y_train, X_test, y_test, use_smote=True, ax_pair=axes[0]
    )

    # Without SMOTE (for comparison)
    print("  Processing XGBoost (no SMOTE)...")
    sizes_no_smote, train_f1_no_smote, test_f1_no_smote, train_acc_no_smote, test_acc_no_smote = plot_train_vs_test_by_size(
        X_train, y_train, X_test, y_test, use_smote=False, ax_pair=axes[1]
    )

    plt.suptitle('Learning Curves: XGBoost (w=3) with All LLM Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curves_xgb_smote_llm.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'learning_curves_xgb_smote_llm.png'}")
    plt.close()

    # =========================================================================
    # Plot 2: Iterations curve (number of trees)
    # =========================================================================
    print("\nGenerating iterations curve...")
    iters, train_f1s, test_f1s = plot_iterations_curve(X_train, y_train, X_test, y_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iters, train_f1s, 'o-', color='blue', label='Train F1', linewidth=2, markersize=8)
    ax.plot(iters, test_f1s, 'o-', color='red', label='Test F1', linewidth=2, markersize=8)
    ax.axhline(y=SOTA_F1, color='green', linestyle='--', alpha=0.7, label=f'SOTA F1={SOTA_F1}')

    # Find best test iteration
    best_iter_idx = np.argmax(test_f1s)
    best_iter = iters[best_iter_idx]
    best_test_f1 = test_f1s[best_iter_idx]

    ax.axvline(x=best_iter, color='purple', linestyle=':', alpha=0.7)
    ax.scatter([best_iter], [best_test_f1], color='purple', s=150, zorder=5, marker='*',
               label=f'Best Test @ {best_iter} trees')

    ax.set_xlabel('Number of Trees (n_estimators)', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('XGBoost (w=3) + SMOTE (k=5) - F1 vs Number of Trees', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    gap = train_f1s[-1] - test_f1s[-1]
    ax.annotate(f'Gap at {iters[-1]} trees: {gap:.3f}\nBest test F1: {best_test_f1:.3f} @ {best_iter} trees\nGap to SOTA: {best_test_f1 - SOTA_F1:+.3f}',
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iterations_curve_xgb_smote_llm.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'iterations_curve_xgb_smote_llm.png'}")
    plt.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    gap_smote = train_f1_smote[-1] - test_f1_smote[-1]
    gap_no_smote = train_f1_no_smote[-1] - test_f1_no_smote[-1]

    def diagnose(gap):
        if gap > 0.15:
            return "⚠️ HIGH VARIANCE (overfitting)"
        elif gap > 0.08:
            return "⚡ MODERATE overfitting"
        elif gap < 0.03:
            return "✓ Good generalization"
        else:
            return "~ Acceptable"

    print(f"\n  {'Configuration':<35} {'Train F1':<12} {'Test F1':<12} {'Gap':<10} {'Diagnosis'}")
    print(f"  {'-'*85}")
    print(f"  {'XGBoost (w=3) + SMOTE (k=5)':<35} {train_f1_smote[-1]:<12.4f} {test_f1_smote[-1]:<12.4f} {gap_smote:<10.4f} {diagnose(gap_smote)}")
    print(f"  {'XGBoost (w=3) No SMOTE':<35} {train_f1_no_smote[-1]:<12.4f} {test_f1_no_smote[-1]:<12.4f} {gap_no_smote:<10.4f} {diagnose(gap_no_smote)}")

    print(f"\n  Best iterations analysis:")
    print(f"    Best test F1: {best_test_f1:.4f} at {best_iter} trees")
    print(f"    Gap to SOTA: {best_test_f1 - SOTA_F1:+.4f}")

    print(f"\n  SMOTE Effect:")
    smote_delta = test_f1_smote[-1] - test_f1_no_smote[-1]
    print(f"    Test F1 change with SMOTE: {smote_delta:+.4f}")
    if smote_delta > 0:
        print(f"    → SMOTE HELPS")
    else:
        print(f"    → SMOTE HURTS (class weight alone may be sufficient)")

    print(f"\n  Plots saved to: {OUTPUT_DIR}/")
    print("""
  Interpretation Guide:

  HIGH GAP (Train >> Test):
    → Model is overfitting to training data
    → Solutions: More regularization, fewer features, early stopping

  BOTH LOW:
    → Model is underfitting (high bias)
    → Solutions: More features, less regularization, deeper model

  CURVES CONVERGING:
    → More training data would help
    → Solutions: Data augmentation, collect more data
    """)


if __name__ == "__main__":
    main()
