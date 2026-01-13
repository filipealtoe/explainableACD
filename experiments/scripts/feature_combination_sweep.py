#!/usr/bin/env python3
"""
Feature Combination Sweep for Checkworthiness Classification

Tests various combinations of:
- Embedding models: bge-small, bge-base, bge-large, e5-large, nomic, mpnet
- Dimensionality reduction: none, pca-64, pca-128
- Text features: none, top7, top15, all35
- Classifiers: LogisticRegression, RandomForest, XGBoost

Usage:
    python experiments/scripts/feature_combination_sweep.py
    python experiments/scripts/feature_combination_sweep.py --models bge-large --quick
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from itertools import product
import time

import numpy as np
import polars as pl
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

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

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed")

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "embedding_cache"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "feature_sweep"

EMBEDDING_MODELS = {
    # Local models (sentence-transformers)
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-large": "intfloat/e5-large-v2",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    # OpenAI models (pre-generated, load from cache)
    "openai-small_512": "cache:openai-small_512",
    "openai-small_1536": "cache:openai-small_1536",
    "openai-large_256": "cache:openai-large_256",
    "openai-large_1024": "cache:openai-large_1024",
    "openai-large_3072": "cache:openai-large_3072",
}

# Text feature groups based on importance analysis
TEXT_FEATURES_TOP7 = [
    "number_count", "has_precise_number", "has_number", "has_large_scale",
    "avg_word_length", "word_count", "alpha_ratio"
]

TEXT_FEATURES_TOP15 = TEXT_FEATURES_TOP7 + [
    "has_first_person_stance", "has_desire_intent", "has_delta",
    "has_number_and_time", "has_future_modal", "has_increase_decrease",
    "has_official_source", "has_voted"
]

TEXT_FEATURES_ALL = [
    "has_number", "has_precise_number", "has_large_scale", "number_count", "has_range", "has_delta",
    "has_specific_year", "has_relative_time", "has_temporal_anchor",
    "has_source_attribution", "has_evidence_noun", "has_official_source", "has_said_claimed",
    "has_comparative", "has_superlative", "has_ranking",
    "has_increase_decrease", "has_voted", "has_negation_claim",
    "has_first_person_stance", "has_desire_intent", "has_future_modal", "has_hedge", "has_vague_quantifier",
    "has_rhetorical_filler", "has_fact_assertion", "is_question", "has_transcript_artifact",
    "word_count", "avg_word_length", "alpha_ratio",
    "has_number_and_time", "has_number_and_comparative", "has_change_and_time", "has_source_and_number",
]

TEXT_FEATURE_SETS = {
    "none": [],
    "top7": TEXT_FEATURES_TOP7,
    "top15": TEXT_FEATURES_TOP15,
    "all35": TEXT_FEATURES_ALL,
}

REDUCTION_CONFIGS = {
    "none": None,
    "pca-64": ("pca", 64),
    "pca-128": ("pca", 128),
}

METADATA_COLS = ["Sentence_id", "class_label", "cleaned_text", "original_text"]


# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/dev/test feature datasets."""
    train = pl.read_parquet(DATA_DIR / "CT24_train_features.parquet")
    dev = pl.read_parquet(DATA_DIR / "CT24_dev_features.parquet")
    test = pl.read_parquet(DATA_DIR / "CT24_test_features.parquet")
    return train, dev, test


def get_text_features(df: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Extract text features from DataFrame."""
    if not feature_names:
        return np.empty((len(df), 0))

    features = []
    for col in feature_names:
        if col in df.columns:
            series = df[col]
            if series.dtype == pl.Boolean:
                features.append(series.cast(pl.Int8).to_numpy())
            else:
                features.append(series.to_numpy())

    if not features:
        return np.empty((len(df), 0))
    return np.column_stack(features)


def get_labels(df: pl.DataFrame) -> np.ndarray:
    """Extract labels from DataFrame."""
    return (df["class_label"] == "Yes").cast(pl.Int8).to_numpy()


# =============================================================================
# Embeddings
# =============================================================================

def get_or_generate_embeddings(
    model_name: str,
    texts_train: list[str],
    texts_dev: list[str],
    texts_test: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get embeddings from cache or generate them."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = EMBEDDING_MODELS[model_name]

    # Handle pre-generated cache files (e.g., OpenAI embeddings)
    if model_path.startswith("cache:"):
        cache_name = model_path.replace("cache:", "")
        cache_file = CACHE_DIR / f"{cache_name}_embeddings.npz"
        if not cache_file.exists():
            raise FileNotFoundError(f"Pre-generated embeddings not found: {cache_file}")
        data = np.load(cache_file)
        return data["train"], data["dev"], data["test"]

    # Check for existing cache
    cache_file = CACHE_DIR / f"{model_name}_embeddings.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        return data["train"], data["dev"], data["test"]

    # Generate new embeddings
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers required for embedding generation")

    print(f"  Generating embeddings for {model_name}...")

    if model_name == "nomic":
        model = SentenceTransformer(model_path, trust_remote_code=True)
    else:
        model = SentenceTransformer(model_path)

    embed_train = model.encode(texts_train, show_progress_bar=True, normalize_embeddings=True)
    embed_dev = model.encode(texts_dev, show_progress_bar=False, normalize_embeddings=True)
    embed_test = model.encode(texts_test, show_progress_bar=False, normalize_embeddings=True)

    np.savez(cache_file, train=embed_train, dev=embed_dev, test=embed_test)

    return embed_train, embed_dev, embed_test


def apply_reduction(
    embed_train: np.ndarray,
    embed_dev: np.ndarray,
    embed_test: np.ndarray,
    reduction_config: tuple | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply dimensionality reduction."""
    if reduction_config is None:
        return embed_train, embed_dev, embed_test

    method, n_components = reduction_config

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "kpca":
        reducer = KernelPCA(n_components=n_components, kernel="rbf", random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    embed_train_r = reducer.fit_transform(embed_train)
    embed_dev_r = reducer.transform(embed_dev)
    embed_test_r = reducer.transform(embed_test)

    return embed_train_r, embed_dev_r, embed_test_r


# =============================================================================
# Classifiers
# =============================================================================

def get_classifier(name: str):
    """Get classifier by name."""
    if name == "logistic":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced")
    elif name == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
    elif name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="logloss")
    elif name == "catboost":
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not installed")
        return CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1, random_state=42,
            auto_class_weights="Balanced", verbose=False
        )
    elif name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        return LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
            class_weight="balanced", verbose=-1, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_combination(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_name: str,
) -> dict:
    """Evaluate a feature combination with a classifier."""

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    # Train and predict
    clf = get_classifier(classifier_name)
    clf.fit(X_train_s, y_train)

    y_pred_train = clf.predict(X_train_s)
    y_pred_dev = clf.predict(X_dev_s)
    y_pred_test = clf.predict(X_test_s)

    return {
        "train_f1": f1_score(y_train, y_pred_train),
        "train_acc": accuracy_score(y_train, y_pred_train),
        "dev_f1": f1_score(y_dev, y_pred_dev),
        "dev_acc": accuracy_score(y_dev, y_pred_dev),
        "test_f1": f1_score(y_test, y_pred_test),
        "test_acc": accuracy_score(y_test, y_pred_test),
        "test_p": precision_score(y_test, y_pred_test),
        "test_r": recall_score(y_test, y_pred_test),
    }


# =============================================================================
# Main Sweep
# =============================================================================

def run_sweep(
    models: list[str],
    reductions: list[str],
    text_sets: list[str],
    classifiers: list[str],
) -> pl.DataFrame:
    """Run the full feature combination sweep."""

    print("Loading data...")
    train_df, dev_df, test_df = load_data()

    texts_train = train_df["cleaned_text"].to_list()
    texts_dev = dev_df["cleaned_text"].to_list()
    texts_test = test_df["cleaned_text"].to_list()

    y_train = get_labels(train_df)
    y_dev = get_labels(dev_df)
    y_test = get_labels(test_df)

    results = []
    total_combos = len(models) * len(reductions) * len(text_sets) * len(classifiers)
    combo_idx = 0

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Get embeddings
        embed_train, embed_dev, embed_test = get_or_generate_embeddings(
            model_name, texts_train, texts_dev, texts_test
        )
        embed_dim = embed_train.shape[1]

        for reduction_name in reductions:
            reduction_config = REDUCTION_CONFIGS[reduction_name]

            # Apply reduction
            if reduction_config:
                embed_train_r, embed_dev_r, embed_test_r = apply_reduction(
                    embed_train, embed_dev, embed_test, reduction_config
                )
            else:
                embed_train_r, embed_dev_r, embed_test_r = embed_train, embed_dev, embed_test

            for text_set_name in text_sets:
                text_features = TEXT_FEATURE_SETS[text_set_name]

                # Get text features
                X_text_train = get_text_features(train_df, text_features)
                X_text_dev = get_text_features(dev_df, text_features)
                X_text_test = get_text_features(test_df, text_features)

                # Combine features
                if X_text_train.shape[1] > 0:
                    X_train = np.hstack([embed_train_r, X_text_train])
                    X_dev = np.hstack([embed_dev_r, X_text_dev])
                    X_test = np.hstack([embed_test_r, X_text_test])
                else:
                    X_train, X_dev, X_test = embed_train_r, embed_dev_r, embed_test_r

                n_features = X_train.shape[1]

                for clf_name in classifiers:
                    combo_idx += 1

                    try:
                        metrics = evaluate_combination(
                            X_train, y_train, X_dev, y_dev, X_test, y_test, clf_name
                        )

                        result = {
                            "model": model_name,
                            "reduction": reduction_name,
                            "text_features": text_set_name,
                            "classifier": clf_name,
                            "embed_dim": embed_dim,
                            "n_features": n_features,
                            **metrics,
                        }
                        results.append(result)

                        print(f"  [{combo_idx}/{total_combos}] {model_name}+{reduction_name}+{text_set_name}+{clf_name}: "
                              f"Test F1={metrics['test_f1']:.3f} Acc={metrics['test_acc']:.3f}")

                    except Exception as e:
                        print(f"  [{combo_idx}/{total_combos}] ERROR: {e}")

    return pl.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Feature combination sweep")
    parser.add_argument("--models", nargs="+", default=["bge-large"],
                        choices=list(EMBEDDING_MODELS.keys()),
                        help="Embedding models to test")
    parser.add_argument("--openai-only", action="store_true",
                        help="Run only OpenAI embedding models")
    parser.add_argument("--reductions", nargs="+", default=["none", "pca-128"],
                        choices=list(REDUCTION_CONFIGS.keys()),
                        help="Reduction methods to test")
    parser.add_argument("--text-sets", nargs="+", default=["none", "top7", "all35"],
                        choices=list(TEXT_FEATURE_SETS.keys()),
                        help="Text feature sets to test")
    parser.add_argument("--classifiers", nargs="+", default=["logistic", "rf", "xgboost", "catboost", "lightgbm"],
                        choices=["logistic", "rf", "xgboost", "catboost", "lightgbm"],
                        help="Classifiers to test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with minimal combinations")
    args = parser.parse_args()

    if args.quick:
        args.models = ["bge-large"]
        args.reductions = ["none", "pca-128"]
        args.text_sets = ["none", "top7"]
        args.classifiers = ["logistic"]

    if args.openai_only:
        args.models = [
            "openai-small_512", "openai-small_1536",
            "openai-large_256", "openai-large_1024", "openai-large_3072"
        ]
        # No reduction needed for OpenAI (already sized correctly)
        args.reductions = ["none"]

    # Check classifier availability
    if "xgboost" in args.classifiers and not HAS_XGBOOST:
        args.classifiers.remove("xgboost")
        print("Warning: XGBoost not available, skipping")
    if "catboost" in args.classifiers and not HAS_CATBOOST:
        args.classifiers.remove("catboost")
        print("Warning: CatBoost not available, skipping")
    if "lightgbm" in args.classifiers and not HAS_LIGHTGBM:
        args.classifiers.remove("lightgbm")
        print("Warning: LightGBM not available, skipping")

    print("="*70)
    print("FEATURE COMBINATION SWEEP")
    print("="*70)
    print(f"Models: {args.models}")
    print(f"Reductions: {args.reductions}")
    print(f"Text sets: {args.text_sets}")
    print(f"Classifiers: {args.classifiers}")
    print()

    start_time = time.time()
    results_df = run_sweep(args.models, args.reductions, args.text_sets, args.classifiers)
    elapsed = time.time() - start_time

    # Sort by test F1
    results_df = results_df.sort("test_f1", descending=True)

    # Print results
    print("\n" + "="*120)
    print("RESULTS SUMMARY (sorted by Test F1)")
    print("="*120)

    print(f"\n{'Model':<12} {'Reduce':<10} {'Text':<8} {'Clf':<10} {'#Feat':<6} | "
          f"{'Dev F1':<8} {'Dev Acc':<8} | {'Test F1':<8} {'Test Acc':<8} {'Test P':<8} {'Test R':<8}")
    print("-"*120)

    for row in results_df.iter_rows(named=True):
        print(f"{row['model']:<12} {row['reduction']:<10} {row['text_features']:<8} {row['classifier']:<10} "
              f"{row['n_features']:<6} | "
              f"{row['dev_f1']:<8.3f} {row['dev_acc']:<8.3f} | "
              f"{row['test_f1']:<8.3f} {row['test_acc']:<8.3f} {row['test_p']:<8.3f} {row['test_r']:<8.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "feature_sweep_results.csv"
    results_df.write_csv(results_file)
    print(f"\nResults saved to: {results_file}")

    # Best result
    best = results_df.row(0, named=True)
    print(f"\n✅ Best: {best['model']} + {best['reduction']} + {best['text_features']} + {best['classifier']}")
    print(f"   Test F1: {best['test_f1']:.3f} | Test Acc: {best['test_acc']:.3f}")

    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
