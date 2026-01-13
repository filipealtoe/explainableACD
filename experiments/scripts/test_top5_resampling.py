#!/usr/bin/env python3
"""
Test Top 5 LLM Features with Various Resampling Techniques

Features: check_score, check_p_false, check_margin_logit, yes_vote_count, verif_p_false
Base: XGBoost with scale_pos_weight=3

Tests: SMOTE, ADASYN, SMOTETomek, BorderlineSMOTE, SVMSMOTE, class weights

Usage:
    python experiments/scripts/test_top5_resampling.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
    from imblearn.combine import SMOTETomek, SMOTEENN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("WARNING: imblearn not installed. Run: pip install imbalanced-learn")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed")

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"

# Top 5 features from previous analysis
TOP5_FEATURES = [
    "check_score",
    "check_p_false",
    "check_margin_logit",
    "yes_vote_count",
    "verif_p_false",
]

SOTA_F1 = 0.82
SOTA_ACC = 0.905
BASELINE_F1 = 0.708  # Previous best with LLM features


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

    X_train = train.select(TOP5_FEATURES).to_numpy().astype(np.float32)
    X_test = test.select(TOP5_FEATURES).to_numpy().astype(np.float32)

    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, X_test, y_train, y_test


# =============================================================================
# Main
# =============================================================================

def main():
    if not HAS_XGBOOST:
        print("ERROR: xgboost not installed")
        return

    print("=" * 100)
    print("TOP 5 FEATURES + RESAMPLING TECHNIQUES")
    print("=" * 100)

    print(f"\nFeatures ({len(TOP5_FEATURES)}):")
    for f in TOP5_FEATURES:
        print(f"  • {f}")

    print(f"\nBaseline (XGBoost w=3, no resampling): F1 ≈ {BASELINE_F1}")
    print(f"Target SOTA: F1 = {SOTA_F1}")

    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test = load_data()

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    print(f"  Train: {len(y_train)} samples ({n_pos} pos, {n_neg} neg, ratio 1:{n_neg/n_pos:.1f})")
    print(f"  Test:  {len(y_test)} samples ({100*y_test.mean():.1f}% positive)")

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # Define resampling strategies
    resamplers = {
        "None (baseline w=3)": (None, 3),
        "None (w=1)": (None, 1),
        "None (w=4)": (None, 4),
        "None (w=5)": (None, 5),
    }

    if HAS_IMBLEARN:
        # Oversampling
        resamplers.update({
            "SMOTE (k=5)": (SMOTE(k_neighbors=5, random_state=42), 1),
            "SMOTE (k=3)": (SMOTE(k_neighbors=3, random_state=42), 1),
            "SMOTE (k=7)": (SMOTE(k_neighbors=7, random_state=42), 1),
            "BorderlineSMOTE": (BorderlineSMOTE(random_state=42), 1),
            "SVMSMOTE": (SVMSMOTE(random_state=42), 1),
            "ADASYN": (ADASYN(random_state=42), 1),
            # Combined (over + under)
            "SMOTETomek": (SMOTETomek(random_state=42), 1),
            "SMOTEENN": (SMOTEENN(random_state=42), 1),
            # Undersampling
            "RandomUnder (0.5)": (RandomUnderSampler(sampling_strategy=0.5, random_state=42), 1),
            "TomekLinks": (TomekLinks(), 3),
            "EditedNN": (EditedNearestNeighbours(), 3),
            # SMOTE + class weight combinations
            "SMOTE + w=2": (SMOTE(k_neighbors=5, random_state=42), 2),
            "SMOTE + w=3": (SMOTE(k_neighbors=5, random_state=42), 3),
        })

    results = []

    print("\n" + "=" * 100)
    print("RUNNING EXPERIMENTS")
    print("=" * 100)

    for name, (resampler, pos_weight) in resamplers.items():
        # Apply resampling
        if resampler is not None:
            try:
                X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
                n_pos_res = y_train_res.sum()
                n_neg_res = len(y_train_res) - n_pos_res
                resample_info = f"({n_pos_res} pos, {n_neg_res} neg)"
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                continue
        else:
            X_train_res, y_train_res = X_train, y_train
            resample_info = "(original)"

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_res)
        X_test_s = scaler.transform(X_test)

        # Train XGBoost
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            use_label_encoder=False,
        )
        clf.fit(X_train_s, y_train_res)

        # Predict
        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]

        # Metrics
        test_f1 = f1_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)
        test_p = precision_score(y_test, y_pred, zero_division=0)
        test_r = recall_score(y_test, y_pred, zero_division=0)

        results.append({
            "name": name,
            "pos_weight": pos_weight,
            "f1": test_f1,
            "acc": test_acc,
            "precision": test_p,
            "recall": test_r,
            "train_size": len(y_train_res),
        })

        delta = test_f1 - BASELINE_F1
        marker = "🔥" if test_f1 > BASELINE_F1 else ("↓" if test_f1 < BASELINE_F1 - 0.01 else "")
        print(f"  {name:<25} F1={test_f1:.4f} ({delta:+.4f}) Acc={test_acc:.4f} P={test_p:.4f} R={test_r:.4f} {resample_info} {marker}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY - All Results (sorted by F1)")
    print("=" * 100)

    results.sort(key=lambda x: x["f1"], reverse=True)

    print(f"\n{'Rank':<5} {'Method':<25} {'F1':<10} {'Acc':<10} {'P':<8} {'R':<8} {'Δ Baseline':<12} {'Δ SOTA'}")
    print("-" * 95)

    for rank, r in enumerate(results, 1):
        delta_base = r["f1"] - BASELINE_F1
        delta_sota = r["f1"] - SOTA_F1
        marker = "🔥" if r["f1"] > BASELINE_F1 else ""
        print(f"{rank:<5} {r['name']:<25} {r['f1']:<10.4f} {r['acc']:<10.4f} "
              f"{r['precision']:<8.4f} {r['recall']:<8.4f} {delta_base:+.4f}       {delta_sota:+.4f} {marker}")

    # Best result
    best = results[0]
    print(f"\n{'='*100}")
    print("🏆 BEST RESULT")
    print(f"{'='*100}")
    print(f"  Method:      {best['name']}")
    print(f"  Test F1:     {best['f1']:.4f}")
    print(f"  Test Acc:    {best['acc']:.4f}")
    print(f"  Precision:   {best['precision']:.4f}")
    print(f"  Recall:      {best['recall']:.4f}")
    print(f"\n  Baseline:    F1 = {BASELINE_F1}")
    print(f"  Improvement: {best['f1'] - BASELINE_F1:+.4f}")
    print(f"\n  SOTA:        F1 = {SOTA_F1}")
    print(f"  Gap:         {best['f1'] - SOTA_F1:+.4f}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    # Best oversampling vs best class weight
    oversample_results = [r for r in results if "SMOTE" in r["name"] or "ADASYN" in r["name"]]
    weight_results = [r for r in results if r["name"].startswith("None")]

    if oversample_results:
        best_oversample = max(oversample_results, key=lambda x: x["f1"])
        print(f"\n  Best Oversampling: {best_oversample['name']} → F1 = {best_oversample['f1']:.4f}")

    if weight_results:
        best_weight = max(weight_results, key=lambda x: x["f1"])
        print(f"  Best Class Weight: {best_weight['name']} → F1 = {best_weight['f1']:.4f}")

    # Precision-Recall tradeoff
    print("\n  Precision-Recall Analysis:")
    high_p = [r for r in results if r["precision"] > 0.5]
    high_r = [r for r in results if r["recall"] > 0.7]

    if high_p:
        best_p = max(high_p, key=lambda x: x["precision"])
        print(f"    Highest Precision: {best_p['name']} → P={best_p['precision']:.4f}, R={best_p['recall']:.4f}")

    if high_r:
        best_r = max(high_r, key=lambda x: x["recall"])
        print(f"    Highest Recall:    {best_r['name']} → P={best_r['precision']:.4f}, R={best_r['recall']:.4f}")

    # Recommendation
    print(f"""
  CONCLUSION:
  {'─'*60}
  With only 5 features ({len(TOP5_FEATURES)} dimensions), SMOTE should work well
  since it's designed for low-dimensional spaces.

  Key findings:
  • Best method: {best['name']}
  • Gap to SOTA remains: {best['f1'] - SOTA_F1:+.4f}

  Next steps if gap persists:
  • Add more features (entropy, harm dimensions)
  • Combine with BGE embeddings (late fusion)
  • Try ensemble methods
""")


if __name__ == "__main__":
    main()
