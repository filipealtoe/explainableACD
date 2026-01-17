#!/usr/bin/env python3
"""
Classifier Training for Checkworthiness Detection

Trains and evaluates classifiers on different feature variants:
1. text_only: 35 text features (baseline)
2. text_embed_full: text + full embeddings (419 features)
3. text_embed_pca64: text + PCA-64 (99 features)
4. text_embed_pca128: text + PCA-128 (163 features)

Classifiers:
- XGBoost (gradient boosting)
- LogisticRegression (linear baseline)
- RandomForest (ensemble baseline)

Usage:
    python experiments/scripts/train_classifier.py
    python experiments/scripts/train_classifier.py --variant text_only
    python experiments/scripts/train_classifier.py --classifier xgboost
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: uv add xgboost")

warnings.filterwarnings("ignore")


# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_classifier"

# Metadata columns to exclude from features
METADATA_COLS = ["Sentence_id", "class_label", "cleaned_text", "original_text"]


# =============================================================================
# Data Loading
# =============================================================================

def load_variant(variant: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/dev/test for a specific variant."""
    train = pl.read_parquet(DATA_DIR / f"train_{variant}.parquet")
    dev = pl.read_parquet(DATA_DIR / f"dev_{variant}.parquet")
    test = pl.read_parquet(DATA_DIR / f"test_{variant}.parquet")
    return train, dev, test


def prepare_data(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from DataFrame."""
    # Get feature columns (everything except metadata)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]

    # Extract features
    X = df.select(feature_cols).to_numpy()

    # Extract labels (convert Yes/No to 1/0)
    y = (df["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X, y


# =============================================================================
# Classifiers
# =============================================================================

def get_classifier(name: str, n_features: int):
    """Get classifier by name."""
    if name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
    elif name == "logistic":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, split_name: str) -> dict:
    """Evaluate predictions and return metrics."""
    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Print metrics in a nice format."""
    print(f"  {metrics['split']:6} | "
          f"Acc: {metrics['accuracy']:.3f} | "
          f"P: {metrics['precision']:.3f} | "
          f"R: {metrics['recall']:.3f} | "
          f"F1: {metrics['f1']:.3f}")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, split_name: str) -> None:
    """Print confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  {split_name} Confusion Matrix:")
    print(f"              Pred No  Pred Yes")
    print(f"    True No     {cm[0,0]:5}     {cm[0,1]:5}")
    print(f"    True Yes    {cm[1,0]:5}     {cm[1,1]:5}")


def get_feature_importances(clf, feature_names: list[str], top_n: int = 15) -> list[tuple[str, float]]:
    """Get top feature importances from classifier."""
    if hasattr(clf, "feature_importances_"):
        # Tree-based models
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        # Linear models
        importances = np.abs(clf.coef_[0])
    else:
        return []

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], importances[i]) for i in indices]


# =============================================================================
# Main Training Loop
# =============================================================================

def train_and_evaluate(
    variant: str,
    classifier_name: str,
    verbose: bool = True,
) -> dict:
    """Train and evaluate a classifier on a specific variant."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"Variant: {variant} | Classifier: {classifier_name}")
        print(f"{'='*70}")

    # Load data
    train_df, dev_df, test_df = load_variant(variant)

    # Prepare data
    X_train, y_train = prepare_data(train_df)
    X_dev, y_dev = prepare_data(dev_df)
    X_test, y_test = prepare_data(test_df)

    # Get feature names
    feature_names = [c for c in train_df.columns if c not in METADATA_COLS]
    n_features = len(feature_names)

    if verbose:
        print(f"\n  Train: {len(y_train)} samples, {n_features} features")
        print(f"  Dev:   {len(y_dev)} samples")
        print(f"  Test:  {len(y_test)} samples")
        print(f"  Train class balance: {y_train.sum()}/{len(y_train)} ({100*y_train.mean():.1f}% positive)")

    # Scale features for logistic regression
    if classifier_name == "logistic":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)

    # Train
    if verbose:
        print(f"\n  Training {classifier_name}...")

    clf = get_classifier(classifier_name, n_features)
    clf.fit(X_train, y_train)

    # Predict
    y_pred_train = clf.predict(X_train)
    y_pred_dev = clf.predict(X_dev)
    y_pred_test = clf.predict(X_test)

    # Evaluate
    if verbose:
        print(f"\n  Results:")

    metrics_train = evaluate(y_train, y_pred_train, "Train")
    metrics_dev = evaluate(y_dev, y_pred_dev, "Dev")
    metrics_test = evaluate(y_test, y_pred_test, "Test")

    if verbose:
        print_metrics(metrics_train)
        print_metrics(metrics_dev)
        print_metrics(metrics_test)

        # Confusion matrix for test
        print_confusion_matrix(y_test, y_pred_test, "Test")

    # Feature importances
    if verbose:
        importances = get_feature_importances(clf, feature_names, top_n=15)
        if importances:
            print(f"\n  Top 15 Feature Importances:")
            for name, imp in importances:
                bar = "█" * int(imp * 50 / importances[0][1])
                print(f"    {name:35} {imp:.4f} {bar}")

    return {
        "variant": variant,
        "classifier": classifier_name,
        "n_features": n_features,
        "train": metrics_train,
        "dev": metrics_dev,
        "test": metrics_test,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train classifiers on CT24 feature variants")
    parser.add_argument(
        "--variant",
        choices=["text_only", "text_embed_full", "text_embed_pca64", "text_embed_pca128", "all"],
        default="all",
        help="Feature variant to use (default: all)",
    )
    parser.add_argument(
        "--classifier",
        choices=["xgboost", "logistic", "random_forest", "all"],
        default="all",
        help="Classifier to use (default: all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary table",
    )
    args = parser.parse_args()

    # Determine variants and classifiers to run
    variants = ["text_only", "text_embed_pca64", "text_embed_pca128", "text_embed_full"] if args.variant == "all" else [args.variant]

    if args.classifier == "all":
        classifiers = ["logistic", "random_forest"]
        if HAS_XGBOOST:
            classifiers.append("xgboost")
    else:
        classifiers = [args.classifier]

    print("="*70)
    print("CT24 CLASSIFIER TRAINING")
    print("="*70)
    print(f"Variants: {variants}")
    print(f"Classifiers: {classifiers}")

    # Run all combinations
    all_results = []

    for variant in variants:
        for clf_name in classifiers:
            try:
                results = train_and_evaluate(
                    variant=variant,
                    classifier_name=clf_name,
                    verbose=not args.quiet,
                )
                all_results.append(results)
            except Exception as e:
                print(f"  ERROR: {e}")

    # Summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE (Train / Dev / Test)")
    print("="*120)

    # Header
    print(f"\n{'Variant':<20} {'Classifier':<12} {'#Feat':<6} | "
          f"{'Train F1':<9} {'Train Acc':<9} | "
          f"{'Dev F1':<8} {'Dev Acc':<8} | "
          f"{'Test F1':<8} {'Test Acc':<8} {'Test P':<8} {'Test R':<8}")
    print("-"*130)

    # Sort by test F1
    all_results.sort(key=lambda x: x["test"]["f1"], reverse=True)

    for r in all_results:
        print(f"{r['variant']:<20} {r['classifier']:<12} {r['n_features']:<6} | "
              f"{r['train']['f1']:<9.3f} {r['train']['accuracy']:<9.3f} | "
              f"{r['dev']['f1']:<8.3f} {r['dev']['accuracy']:<8.3f} | "
              f"{r['test']['f1']:<8.3f} {r['test']['accuracy']:<8.3f} "
              f"{r['test']['precision']:<8.3f} {r['test']['recall']:<8.3f}")

    # Best result
    best = all_results[0]
    print(f"\n✅ Best: {best['variant']} + {best['classifier']} → Test F1: {best['test']['f1']:.3f}")

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
