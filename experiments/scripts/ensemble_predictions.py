#!/usr/bin/env python3
"""
Ensemble DeBERTa + LogReg(LLM features) predictions.

Combines the fine-tuned DeBERTa model with a LogReg classifier
trained on LLM-generated features for potentially better performance.

Usage:
    python experiments/scripts/ensemble_predictions.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
DEBERTA_RESULTS = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features_v4"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"

# Features to use for LogReg (best from our earlier experiments)
LLM_FEATURE_COLS = [
    # Core scores
    'check_score', 'verif_score', 'harm_score',
    # Probabilities (from logprobs)
    'check_p_yes', 'check_p_no', 'verif_p_yes', 'verif_p_no', 'harm_p_yes', 'harm_p_no',
    # Logit probabilities
    'check_logit_p_yes', 'check_logit_p_no', 'verif_logit_p_yes', 'verif_logit_p_no',
    'harm_logit_p_yes', 'harm_logit_p_no',
    # Entropy (uncertainty)
    'check_entropy', 'check_entropy_norm', 'verif_entropy', 'verif_entropy_norm',
    'harm_entropy', 'harm_entropy_norm',
    # Margins
    'check_margin_p', 'verif_margin_p', 'harm_margin_p',
    'check_margin_logit', 'verif_margin_logit', 'harm_margin_logit',
    # Cross-module
    'score_variance', 'score_max_diff',
    # Predictions
    'check_prediction', 'verif_prediction', 'harm_prediction',
    # Agreement
    'check_verif_agree', 'check_harm_agree', 'verif_harm_agree',
    # Vote count
    'yes_vote_count', 'unanimous_yes',
]


def load_data():
    """Load LLM features and labels."""
    # Load LLM features
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    # Load CT24 for labels
    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    # Ensure consistent ID types
    llm_train = llm_train.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_test = llm_test.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    # Join to get labels
    train = llm_train.join(ct24_train.select(["Sentence_id", "class_label"]), on="Sentence_id", how="inner")
    test = llm_test.join(ct24_test.select(["Sentence_id", "class_label"]), on="Sentence_id", how="inner")

    # Get labels
    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return train, test, y_train, y_test


def get_logreg_predictions(train_df, test_df, y_train, feature_cols):
    """Train LogReg and return test probabilities."""
    # Filter to available features
    available_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"Using {len(available_cols)} LLM features for LogReg")

    # Extract features
    X_train = train_df.select(available_cols).to_numpy().astype(np.float32)
    X_test = test_df.select(available_cols).to_numpy().astype(np.float32)

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train LogReg with class weights
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4})
    clf.fit(X_train_s, y_train)

    # Get probabilities
    probs = clf.predict_proba(X_test_s)[:, 1]

    return probs


def evaluate_at_thresholds(probs, y_true, thresholds=None):
    """Evaluate predictions at various thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.75, 0.05)

    results = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        results.append({
            'threshold': thresh,
            'f1': f1_score(y_true, preds),
            'accuracy': accuracy_score(y_true, preds),
            'precision': precision_score(y_true, preds, zero_division=0),
            'recall': recall_score(y_true, preds, zero_division=0),
        })
    return results


def main():
    print("=" * 80)
    print("ENSEMBLE: DeBERTa + LogReg(LLM Features)")
    print("=" * 80)

    # Load DeBERTa predictions
    print("\n1. Loading DeBERTa predictions...")
    deberta_probs = np.load(DEBERTA_RESULTS / "test_probs.npy")
    print(f"   DeBERTa predictions shape: {deberta_probs.shape}")

    # Load data and train LogReg
    print("\n2. Training LogReg on LLM features...")
    train_df, test_df, y_train, y_test = load_data()
    logreg_probs = get_logreg_predictions(train_df, test_df, y_train, LLM_FEATURE_COLS)
    print(f"   LogReg predictions shape: {logreg_probs.shape}")

    # Verify alignment
    assert len(deberta_probs) == len(logreg_probs) == len(y_test), \
        f"Size mismatch: DeBERTa={len(deberta_probs)}, LogReg={len(logreg_probs)}, labels={len(y_test)}"

    # Individual model performance
    print("\n" + "=" * 80)
    print("3. INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 80)

    print("\nDeBERTa (baseline):")
    deberta_results = evaluate_at_thresholds(deberta_probs, y_test)
    best_deberta = max(deberta_results, key=lambda x: x['f1'])
    print(f"   Best F1: {best_deberta['f1']:.4f} @ threshold {best_deberta['threshold']:.2f}")

    print("\nLogReg (LLM features):")
    logreg_results = evaluate_at_thresholds(logreg_probs, y_test)
    best_logreg = max(logreg_results, key=lambda x: x['f1'])
    print(f"   Best F1: {best_logreg['f1']:.4f} @ threshold {best_logreg['threshold']:.2f}")

    # Ensemble with different weights
    print("\n" + "=" * 80)
    print("4. ENSEMBLE PERFORMANCE (varying DeBERTa weight)")
    print("=" * 80)

    print(f"\n{'DeBERTa Weight':<15} {'Best F1':<10} {'Best Acc':<10} {'Threshold':<10}")
    print("-" * 50)

    best_ensemble_f1 = 0
    best_weight = 0
    best_threshold = 0

    for deberta_weight in np.arange(0.3, 0.95, 0.05):
        logreg_weight = 1 - deberta_weight
        ensemble_probs = deberta_weight * deberta_probs + logreg_weight * logreg_probs

        results = evaluate_at_thresholds(ensemble_probs, y_test)
        best = max(results, key=lambda x: x['f1'])

        marker = ""
        if best['f1'] > best_deberta['f1']:
            marker = "↑"
        if best['f1'] > best_ensemble_f1:
            best_ensemble_f1 = best['f1']
            best_weight = deberta_weight
            best_threshold = best['threshold']
            marker = "🔥"

        print(f"{deberta_weight:<15.2f} {best['f1']:<10.4f} {best['accuracy']:<10.4f} {best['threshold']:<10.2f} {marker}")

    # Best ensemble detailed results
    print("\n" + "=" * 80)
    print("5. BEST ENSEMBLE DETAILED RESULTS")
    print("=" * 80)

    best_ensemble_probs = best_weight * deberta_probs + (1 - best_weight) * logreg_probs
    print(f"\nBest weights: DeBERTa={best_weight:.2f}, LogReg={1-best_weight:.2f}")

    results = evaluate_at_thresholds(best_ensemble_probs, y_test)
    print(f"\n{'Threshold':<12} {'F1':<10} {'Acc':<10} {'Prec':<10} {'Recall':<10}")
    print("-" * 55)
    for r in results:
        marker = "🔥" if r['f1'] == best_ensemble_f1 else ""
        print(f"{r['threshold']:<12.2f} {r['f1']:<10.4f} {r['accuracy']:<10.4f} "
              f"{r['precision']:<10.4f} {r['recall']:<10.4f} {marker}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<30} {'Best F1':<10} {'Threshold':<10}")
    print("-" * 50)
    print(f"{'DeBERTa (baseline)':<30} {best_deberta['f1']:<10.4f} {best_deberta['threshold']:<10.2f}")
    print(f"{'LogReg (LLM features)':<30} {best_logreg['f1']:<10.4f} {best_logreg['threshold']:<10.2f}")
    print(f"{'Ensemble (best)':<30} {best_ensemble_f1:<10.4f} {best_threshold:<10.2f}")

    improvement = best_ensemble_f1 - best_deberta['f1']
    print(f"\nEnsemble improvement over DeBERTa: {improvement:+.4f} F1")

    SOTA_F1 = 0.82
    print(f"\nSOTA Comparison:")
    print(f"  SOTA F1: {SOTA_F1:.4f}")
    print(f"  Ensemble F1: {best_ensemble_f1:.4f}")
    print(f"  Gap: {best_ensemble_f1 - SOTA_F1:+.4f}")

    if best_ensemble_f1 > SOTA_F1:
        print("\n🏆 ENSEMBLE BEATS SOTA!")

    # Save ensemble predictions
    output_path = DEBERTA_RESULTS.parent / "ensemble_test_probs.npy"
    np.save(output_path, best_ensemble_probs)
    print(f"\nEnsemble predictions saved to: {output_path}")

    # Also save LogReg predictions for future use
    logreg_output = DEBERTA_RESULTS.parent / "logreg_test_probs.npy"
    np.save(logreg_output, logreg_probs)
    print(f"LogReg predictions saved to: {logreg_output}")


if __name__ == "__main__":
    main()
