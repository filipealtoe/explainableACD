#!/usr/bin/env python3
"""
Diagnose Dev-Test Performance Gap

Investigates why dev F1 >> test F1. Possible causes:
1. Distribution shift between dev and test
2. Different class balance
3. Temporal/topical shift
4. Overfitting to dev set
5. Data leakage

Usage:
    python experiments/scripts/diagnose_dev_test_gap.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
EMBEDDING_FILE = DATA_DIR / "embedding_cache" / "bge-large_embeddings.npz"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"


# =============================================================================
# Diagnostic Functions
# =============================================================================

def load_all_data():
    """Load everything for analysis."""
    # Embeddings
    embed_data = np.load(EMBEDDING_FILE)
    X_train = embed_data["train"].astype(np.float32)
    X_dev = embed_data["dev"].astype(np.float32)
    X_test = embed_data["test"].astype(np.float32)

    # Full dataframes with metadata
    df_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    df_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    df_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    y_train = (df_train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (df_dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (df_test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, X_dev, X_test, y_train, y_dev, y_test, df_train, df_dev, df_test


def check_class_balance(y_train, y_dev, y_test):
    """Check if class distributions differ."""
    print("\n" + "=" * 80)
    print("1. CLASS BALANCE ANALYSIS")
    print("=" * 80)

    train_pos = y_train.mean()
    dev_pos = y_dev.mean()
    test_pos = y_test.mean()

    print(f"\n  {'Split':<10} {'Samples':<10} {'% Positive':<12} {'% Negative':<12}")
    print(f"  {'-'*45}")
    print(f"  {'Train':<10} {len(y_train):<10} {100*train_pos:<12.1f} {100*(1-train_pos):<12.1f}")
    print(f"  {'Dev':<10} {len(y_dev):<10} {100*dev_pos:<12.1f} {100*(1-dev_pos):<12.1f}")
    print(f"  {'Test':<10} {len(y_test):<10} {100*test_pos:<12.1f} {100*(1-test_pos):<12.1f}")

    # Statistical test
    _, p_train_dev = stats.chi2_contingency([
        [y_train.sum(), len(y_train) - y_train.sum()],
        [y_dev.sum(), len(y_dev) - y_dev.sum()]
    ])[:2]

    _, p_dev_test = stats.chi2_contingency([
        [y_dev.sum(), len(y_dev) - y_dev.sum()],
        [y_test.sum(), len(y_test) - y_test.sum()]
    ])[:2]

    print(f"\n  Chi-squared test for class balance:")
    print(f"    Train vs Dev: p={p_train_dev:.4f} {'⚠️ DIFFERENT' if p_train_dev < 0.05 else '✓ Similar'}")
    print(f"    Dev vs Test:  p={p_dev_test:.4f} {'⚠️ DIFFERENT' if p_dev_test < 0.05 else '✓ Similar'}")


def check_embedding_distribution(X_train, X_dev, X_test):
    """Check if embedding distributions differ."""
    print("\n" + "=" * 80)
    print("2. EMBEDDING DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Compare means
    train_mean = X_train.mean(axis=0)
    dev_mean = X_dev.mean(axis=0)
    test_mean = X_test.mean(axis=0)

    # L2 distance between mean embeddings
    dist_train_dev = np.linalg.norm(train_mean - dev_mean)
    dist_train_test = np.linalg.norm(train_mean - test_mean)
    dist_dev_test = np.linalg.norm(dev_mean - test_mean)

    print(f"\n  L2 distance between mean embeddings:")
    print(f"    Train ↔ Dev:  {dist_train_dev:.4f}")
    print(f"    Train ↔ Test: {dist_train_test:.4f}")
    print(f"    Dev ↔ Test:   {dist_dev_test:.4f}")

    if dist_dev_test > dist_train_dev * 1.5:
        print(f"\n  ⚠️ Dev and Test are more different than Train and Dev!")

    # Compare variances
    train_var = X_train.var(axis=0).mean()
    dev_var = X_dev.var(axis=0).mean()
    test_var = X_test.var(axis=0).mean()

    print(f"\n  Mean feature variance:")
    print(f"    Train: {train_var:.4f}")
    print(f"    Dev:   {dev_var:.4f}")
    print(f"    Test:  {test_var:.4f}")

    # KS test on first few principal components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_dev_pca = pca.transform(X_dev)
    X_test_pca = pca.transform(X_test)

    print(f"\n  KS test on top 3 PCA components:")
    for i in range(3):
        _, p_train_dev = stats.ks_2samp(X_train_pca[:, i], X_dev_pca[:, i])
        _, p_train_test = stats.ks_2samp(X_train_pca[:, i], X_test_pca[:, i])
        _, p_dev_test = stats.ks_2samp(X_dev_pca[:, i], X_test_pca[:, i])

        print(f"    PC{i+1}: Train-Dev p={p_train_dev:.3f}, Train-Test p={p_train_test:.3f}, Dev-Test p={p_dev_test:.3f}")


def check_cross_validation(X_train, y_train, X_dev, y_dev, X_test, y_test):
    """Compare CV score vs held-out performance."""
    print("\n" + "=" * 80)
    print("3. CROSS-VALIDATION vs HELD-OUT ANALYSIS")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4})

    # Cross-validation on train only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring='f1')

    # Fit on full train, evaluate on dev and test
    clf.fit(X_train_s, y_train)
    dev_pred = clf.predict(X_dev_s)
    test_pred = clf.predict(X_test_s)

    dev_f1 = f1_score(y_dev, dev_pred)
    test_f1 = f1_score(y_test, test_pred)

    print(f"\n  5-Fold CV on Train: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Dev F1:             {dev_f1:.4f}")
    print(f"  Test F1:            {test_f1:.4f}")

    print(f"\n  Gaps:")
    print(f"    CV → Dev:  {dev_f1 - cv_scores.mean():+.4f}")
    print(f"    CV → Test: {test_f1 - cv_scores.mean():+.4f}")
    print(f"    Dev → Test: {test_f1 - dev_f1:+.4f}")

    if dev_f1 - cv_scores.mean() > 0.1:
        print(f"\n  ⚠️ Dev is MUCH better than CV - possible data leakage or distribution match!")
    if test_f1 - dev_f1 < -0.1:
        print(f"\n  ⚠️ Test is MUCH worse than Dev - distribution shift!")


def check_train_dev_test_combined(X_train, X_dev, X_test, y_train, y_dev, y_test):
    """Train on train+dev, see if test improves."""
    print("\n" + "=" * 80)
    print("4. TRAIN+DEV COMBINED ANALYSIS")
    print("=" * 80)

    scaler = StandardScaler()

    # Original: Train on train only
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf1 = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4})
    clf1.fit(X_train_s, y_train)
    test_f1_train_only = f1_score(y_test, clf1.predict(X_test_s))

    # Combined: Train on train+dev
    X_combined = np.vstack([X_train, X_dev])
    y_combined = np.concatenate([y_train, y_dev])

    scaler2 = StandardScaler()
    X_combined_s = scaler2.fit_transform(X_combined)
    X_test_s2 = scaler2.transform(X_test)

    clf2 = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4})
    clf2.fit(X_combined_s, y_combined)
    test_f1_combined = f1_score(y_test, clf2.predict(X_test_s2))

    print(f"\n  Train on TRAIN only:    Test F1 = {test_f1_train_only:.4f}")
    print(f"  Train on TRAIN+DEV:     Test F1 = {test_f1_combined:.4f}")
    print(f"  Improvement:            {test_f1_combined - test_f1_train_only:+.4f}")

    if test_f1_combined > test_f1_train_only + 0.02:
        print(f"\n  ℹ️ Adding dev to training helps - dev data is useful, not leaked")
    else:
        print(f"\n  ℹ️ Adding dev doesn't help much - test is genuinely different")


def check_text_analysis(df_train, df_dev, df_test):
    """Analyze text characteristics."""
    print("\n" + "=" * 80)
    print("5. TEXT CHARACTERISTICS ANALYSIS")
    print("=" * 80)

    for name, df in [("Train", df_train), ("Dev", df_dev), ("Test", df_test)]:
        if "cleaned_text" in df.columns:
            texts = df["cleaned_text"].to_list()
            lengths = [len(t.split()) for t in texts]
            print(f"\n  {name}:")
            print(f"    Avg word count: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
            print(f"    Min/Max: {min(lengths)} / {max(lengths)}")


def check_sample_analysis(X_train, X_dev, X_test, y_train, y_dev, y_test):
    """Analyze which test samples the model gets wrong."""
    print("\n" + "=" * 80)
    print("6. ERROR ANALYSIS")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4})
    clf.fit(X_train_s, y_train)

    dev_pred = clf.predict(X_dev_s)
    test_pred = clf.predict(X_test_s)

    dev_proba = clf.predict_proba(X_dev_s)[:, 1]
    test_proba = clf.predict_proba(X_test_s)[:, 1]

    # Analyze prediction confidence distribution
    print(f"\n  Prediction probability distribution:")
    print(f"    Dev positive class proba:  mean={dev_proba.mean():.3f}, std={dev_proba.std():.3f}")
    print(f"    Test positive class proba: mean={test_proba.mean():.3f}, std={test_proba.std():.3f}")

    # Error breakdown
    dev_tp = ((dev_pred == 1) & (y_dev == 1)).sum()
    dev_fp = ((dev_pred == 1) & (y_dev == 0)).sum()
    dev_fn = ((dev_pred == 0) & (y_dev == 1)).sum()
    dev_tn = ((dev_pred == 0) & (y_dev == 0)).sum()

    test_tp = ((test_pred == 1) & (y_test == 1)).sum()
    test_fp = ((test_pred == 1) & (y_test == 0)).sum()
    test_fn = ((test_pred == 0) & (y_test == 1)).sum()
    test_tn = ((test_pred == 0) & (y_test == 0)).sum()

    print(f"\n  Confusion matrix comparison:")
    print(f"    {'Metric':<20} {'Dev':<10} {'Test':<10} {'Ratio':<10}")
    print(f"    {'-'*50}")
    print(f"    {'True Positives':<20} {dev_tp:<10} {test_tp:<10} {test_tp/max(dev_tp,1):.2f}")
    print(f"    {'False Positives':<20} {dev_fp:<10} {test_fp:<10} {test_fp/max(dev_fp,1):.2f}")
    print(f"    {'False Negatives':<20} {dev_fn:<10} {test_fn:<10} {test_fn/max(dev_fn,1):.2f}")
    print(f"    {'True Negatives':<20} {dev_tn:<10} {test_tn:<10} {test_tn/max(dev_tn,1):.2f}")

    dev_precision = dev_tp / max(dev_tp + dev_fp, 1)
    dev_recall = dev_tp / max(dev_tp + dev_fn, 1)
    test_precision = test_tp / max(test_tp + test_fp, 1)
    test_recall = test_tp / max(test_tp + test_fn, 1)

    print(f"\n  Precision/Recall breakdown:")
    print(f"    Dev:  P={dev_precision:.3f}, R={dev_recall:.3f}")
    print(f"    Test: P={test_precision:.3f}, R={test_recall:.3f}")

    if test_recall < dev_recall - 0.1:
        print(f"\n  ⚠️ Test RECALL is much lower - model missing positive cases in test")
    if test_precision < dev_precision - 0.1:
        print(f"\n  ⚠️ Test PRECISION is much lower - more false positives in test")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("DIAGNOSING DEV-TEST PERFORMANCE GAP")
    print("=" * 80)
    print("\nInvestigating why Dev F1 >> Test F1")

    X_train, X_dev, X_test, y_train, y_dev, y_test, df_train, df_dev, df_test = load_all_data()

    check_class_balance(y_train, y_dev, y_test)
    check_embedding_distribution(X_train, X_dev, X_test)
    check_cross_validation(X_train, y_train, X_dev, y_dev, X_test, y_test)
    check_train_dev_test_combined(X_train, X_dev, X_test, y_train, y_dev, y_test)
    check_text_analysis(df_train, df_dev, df_test)
    check_sample_analysis(X_train, X_dev, X_test, y_train, y_dev, y_test)

    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print("""
    Common causes of Dev >> Test gap:

    1. DISTRIBUTION SHIFT: Dev and test come from different sources/times
       → Check if KS tests show significant differences
       → Solution: Use more diverse training data

    2. CLASS IMBALANCE DIFFERENCE: Different positive rates
       → Check class balance analysis
       → Solution: Stratified sampling, rebalancing

    3. OVERFITTING TO DEV: Model memorizes dev patterns
       → Check if CV score is much lower than dev score
       → Solution: More regularization, early stopping

    4. DATA LEAKAGE: Dev has information not in test
       → Check if train+dev improves test significantly
       → Solution: Review data pipeline

    5. SMALL TEST SET VARIANCE: 341 samples = high variance
       → Expected variance: ~±0.05 F1 for this size
       → Solution: Bootstrap confidence intervals
    """)


if __name__ == "__main__":
    main()
