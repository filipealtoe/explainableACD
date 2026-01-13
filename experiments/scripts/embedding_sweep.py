#!/usr/bin/env python3
"""
Embedding Model & Dimensionality Reduction Sweep

Tests multiple embedding models and reduction techniques to find the best combination.

Embedding Models:
- BAAI/bge-small-en-v1.5 (384 dims)
- BAAI/bge-base-en-v1.5 (768 dims)
- BAAI/bge-large-en-v1.5 (1024 dims)
- intfloat/e5-large-v2 (1024 dims)
- nomic-ai/nomic-embed-text-v1.5 (768 dims)
- sentence-transformers/all-mpnet-base-v2 (768 dims)

Dimensionality Reduction:
- None (raw embeddings)
- PCA (64, 128)
- UMAP (64, 128)
- Kernel PCA (64, 128)

Classifier: Logistic Regression (best from previous experiments)

Usage:
    python experiments/scripts/embedding_sweep.py
    python experiments/scripts/embedding_sweep.py --models bge-small bge-base
    python experiments/scripts/embedding_sweep.py --reductions pca umap
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Optional: UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed. Install with: uv add umap-learn")

# Sentence transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed.")
    print("Install with: uv add sentence-transformers")
    exit(1)

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "experiments" / "results" / "embedding_sweep"

# Embedding models to test
EMBEDDING_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-large": "intfloat/e5-large-v2",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

# Dimensionality reduction configs
REDUCTION_CONFIGS = {
    "none": {"method": "none", "dims": None},
    "pca-64": {"method": "pca", "dims": 64},
    "pca-128": {"method": "pca", "dims": 128},
    "umap-64": {"method": "umap", "dims": 64},
    "umap-128": {"method": "umap", "dims": 128},
    "kpca-64": {"method": "kpca", "dims": 64},
    "kpca-128": {"method": "kpca", "dims": 128},
}

# Feature columns from text extraction
TEXT_FEATURE_COLS = [
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

METADATA_COLS = ["Sentence_id", "class_label", "cleaned_text", "original_text"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentResult:
    model_name: str
    model_dims: int
    reduction: str
    final_dims: int
    train_f1: float
    train_acc: float
    dev_f1: float
    dev_acc: float
    test_f1: float
    test_acc: float
    test_precision: float
    test_recall: float
    embed_time: float
    reduce_time: float
    train_time: float


# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load feature datasets."""
    train = pl.read_parquet(INPUT_DIR / "CT24_train_features.parquet")
    dev = pl.read_parquet(INPUT_DIR / "CT24_dev_features.parquet")
    test = pl.read_parquet(INPUT_DIR / "CT24_test_features.parquet")
    return train, dev, test


def extract_text_features(df: pl.DataFrame) -> np.ndarray:
    """Extract text features as numpy array."""
    available_cols = [c for c in TEXT_FEATURE_COLS if c in df.columns]
    feature_data = []
    for col in available_cols:
        series = df[col]
        if series.dtype == pl.Boolean:
            feature_data.append(series.cast(pl.Int8).to_numpy())
        else:
            feature_data.append(series.to_numpy())
    return np.column_stack(feature_data)


def extract_labels(df: pl.DataFrame) -> np.ndarray:
    """Extract labels as numpy array."""
    return (df["class_label"] == "Yes").cast(pl.Int8).to_numpy()


# =============================================================================
# Embedding Generation
# =============================================================================

def generate_embeddings(
    texts: list[str],
    model_name: str,
    model_path: str,
) -> tuple[np.ndarray, int, float]:
    """Generate embeddings and return (embeddings, dims, time_seconds)."""
    start = time.time()

    # Handle nomic model which needs trust_remote_code
    if "nomic" in model_path:
        model = SentenceTransformer(model_path, trust_remote_code=True)
    else:
        model = SentenceTransformer(model_path)

    # E5 models need instruction prefix
    if "e5" in model_path.lower():
        texts = [f"query: {t}" for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start
    dims = embeddings.shape[1]

    return embeddings, dims, elapsed


# =============================================================================
# Dimensionality Reduction
# =============================================================================

def apply_reduction(
    train_embed: np.ndarray,
    dev_embed: np.ndarray,
    test_embed: np.ndarray,
    method: str,
    dims: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Apply dimensionality reduction. Fit on train, transform all."""
    start = time.time()

    if method == "none" or dims is None:
        elapsed = time.time() - start
        return train_embed, dev_embed, test_embed, train_embed.shape[1], elapsed

    if method == "pca":
        reducer = PCA(n_components=dims, random_state=42)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP not installed")
        reducer = UMAP(n_components=dims, random_state=42, n_neighbors=15, min_dist=0.1)
    elif method == "kpca":
        reducer = KernelPCA(n_components=dims, kernel="rbf", random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    # Fit on train
    train_reduced = reducer.fit_transform(train_embed)

    # Transform dev/test
    if method == "umap":
        # UMAP uses transform after fit
        dev_reduced = reducer.transform(dev_embed)
        test_reduced = reducer.transform(test_embed)
    else:
        dev_reduced = reducer.transform(dev_embed)
        test_reduced = reducer.transform(test_embed)

    elapsed = time.time() - start
    final_dims = train_reduced.shape[1]

    return train_reduced, dev_reduced, test_reduced, final_dims, elapsed


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, float]:
    """Train logistic regression and evaluate."""
    start = time.time()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    # Train
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_dev = clf.predict(X_dev_scaled)
    y_pred_test = clf.predict(X_test_scaled)

    elapsed = time.time() - start

    metrics = {
        "train_f1": f1_score(y_train, y_pred_train),
        "train_acc": accuracy_score(y_train, y_pred_train),
        "dev_f1": f1_score(y_dev, y_pred_dev),
        "dev_acc": accuracy_score(y_dev, y_pred_dev),
        "test_f1": f1_score(y_test, y_pred_test),
        "test_acc": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test),
    }

    return metrics, elapsed


# =============================================================================
# Main Sweep
# =============================================================================

def run_experiment(
    model_name: str,
    model_path: str,
    reduction_name: str,
    reduction_config: dict,
    train_df: pl.DataFrame,
    dev_df: pl.DataFrame,
    test_df: pl.DataFrame,
    text_features_train: np.ndarray,
    text_features_dev: np.ndarray,
    text_features_test: np.ndarray,
    y_train: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    cached_embeddings: dict,
) -> ExperimentResult | None:
    """Run a single experiment."""

    try:
        # Get or generate embeddings
        cache_key = model_name
        if cache_key not in cached_embeddings:
            print(f"    Generating embeddings for {model_name}...")

            train_texts = train_df["cleaned_text"].to_list()
            dev_texts = dev_df["cleaned_text"].to_list()
            test_texts = test_df["cleaned_text"].to_list()

            train_embed, dims, t1 = generate_embeddings(train_texts, model_name, model_path)
            dev_embed, _, t2 = generate_embeddings(dev_texts, model_name, model_path)
            test_embed, _, t3 = generate_embeddings(test_texts, model_name, model_path)

            embed_time = t1 + t2 + t3
            cached_embeddings[cache_key] = {
                "train": train_embed,
                "dev": dev_embed,
                "test": test_embed,
                "dims": dims,
                "time": embed_time,
            }

        cached = cached_embeddings[cache_key]
        train_embed = cached["train"]
        dev_embed = cached["dev"]
        test_embed = cached["test"]
        model_dims = cached["dims"]
        embed_time = cached["time"]

        # Apply dimensionality reduction
        method = reduction_config["method"]
        dims = reduction_config["dims"]

        train_reduced, dev_reduced, test_reduced, final_embed_dims, reduce_time = apply_reduction(
            train_embed, dev_embed, test_embed, method, dims
        )

        # Combine with text features
        X_train = np.hstack([text_features_train, train_reduced])
        X_dev = np.hstack([text_features_dev, dev_reduced])
        X_test = np.hstack([text_features_test, test_reduced])

        final_dims = X_train.shape[1]

        # Train and evaluate
        metrics, train_time = train_and_evaluate(
            X_train, y_train, X_dev, y_dev, X_test, y_test
        )

        return ExperimentResult(
            model_name=model_name,
            model_dims=model_dims,
            reduction=reduction_name,
            final_dims=final_dims,
            train_f1=metrics["train_f1"],
            train_acc=metrics["train_acc"],
            dev_f1=metrics["dev_f1"],
            dev_acc=metrics["dev_acc"],
            test_f1=metrics["test_f1"],
            test_acc=metrics["test_acc"],
            test_precision=metrics["test_precision"],
            test_recall=metrics["test_recall"],
            embed_time=embed_time,
            reduce_time=reduce_time,
            train_time=train_time,
        )

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Embedding model and reduction sweep")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(EMBEDDING_MODELS.keys()) + ["all"],
        default=["all"],
        help="Embedding models to test (default: all)",
    )
    parser.add_argument(
        "--reductions",
        nargs="+",
        choices=list(REDUCTION_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Reduction methods to test (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Determine models and reductions to test
    if "all" in args.models:
        models_to_test = list(EMBEDDING_MODELS.keys())
    else:
        models_to_test = args.models

    if "all" in args.reductions:
        reductions_to_test = list(REDUCTION_CONFIGS.keys())
        # Skip UMAP if not installed
        if not HAS_UMAP:
            reductions_to_test = [r for r in reductions_to_test if "umap" not in r]
    else:
        reductions_to_test = args.reductions

    print("="*80)
    print("EMBEDDING MODEL & DIMENSIONALITY REDUCTION SWEEP")
    print("="*80)
    print(f"\nModels: {models_to_test}")
    print(f"Reductions: {reductions_to_test}")
    print(f"Total experiments: {len(models_to_test) * len(reductions_to_test)}")

    # Load data
    print("\n" + "="*80)
    print("Loading data...")
    print("="*80)

    train_df, dev_df, test_df = load_data()
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # Extract text features
    text_features_train = extract_text_features(train_df)
    text_features_dev = extract_text_features(dev_df)
    text_features_test = extract_text_features(test_df)
    print(f"  Text features: {text_features_train.shape[1]}")

    # Extract labels
    y_train = extract_labels(train_df)
    y_dev = extract_labels(dev_df)
    y_test = extract_labels(test_df)

    # Run experiments
    print("\n" + "="*80)
    print("Running experiments...")
    print("="*80)

    results = []
    cached_embeddings = {}

    total = len(models_to_test) * len(reductions_to_test)
    current = 0

    for model_name in models_to_test:
        model_path = EMBEDDING_MODELS[model_name]
        print(f"\n  Model: {model_name} ({model_path})")

        for reduction_name in reductions_to_test:
            current += 1
            reduction_config = REDUCTION_CONFIGS[reduction_name]

            print(f"    [{current}/{total}] {model_name} + {reduction_name}...", end=" ", flush=True)

            result = run_experiment(
                model_name=model_name,
                model_path=model_path,
                reduction_name=reduction_name,
                reduction_config=reduction_config,
                train_df=train_df,
                dev_df=dev_df,
                test_df=test_df,
                text_features_train=text_features_train,
                text_features_dev=text_features_dev,
                text_features_test=text_features_test,
                y_train=y_train,
                y_dev=y_dev,
                y_test=y_test,
                cached_embeddings=cached_embeddings,
            )

            if result:
                results.append(result)
                print(f"Test F1: {result.test_f1:.3f}")
            else:
                print("FAILED")

    # Sort by test F1
    results.sort(key=lambda x: x.test_f1, reverse=True)

    # Print summary table
    print("\n" + "="*120)
    print("RESULTS SUMMARY (sorted by Test F1)")
    print("="*120)

    print(f"\n{'Model':<12} {'Reduction':<10} {'EmbDim':<7} {'Final':<6} | "
          f"{'Train F1':<9} {'Train Acc':<9} | "
          f"{'Dev F1':<8} {'Dev Acc':<8} | "
          f"{'Test F1':<8} {'Test Acc':<8} {'Test P':<8} {'Test R':<8}")
    print("-"*130)

    for r in results:
        print(f"{r.model_name:<12} {r.reduction:<10} {r.model_dims:<7} {r.final_dims:<6} | "
              f"{r.train_f1:<9.3f} {r.train_acc:<9.3f} | "
              f"{r.dev_f1:<8.3f} {r.dev_acc:<8.3f} | "
              f"{r.test_f1:<8.3f} {r.test_acc:<8.3f} {r.test_precision:<8.3f} {r.test_recall:<8.3f}")

    # Best result
    if results:
        best = results[0]
        print(f"\n✅ Best: {best.model_name} + {best.reduction} → Test F1: {best.test_f1:.3f}")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "sweep_results.csv"

    # Convert to DataFrame and save
    results_data = [
        {
            "model": r.model_name,
            "reduction": r.reduction,
            "embed_dims": r.model_dims,
            "final_dims": r.final_dims,
            "train_f1": r.train_f1,
            "train_acc": r.train_acc,
            "dev_f1": r.dev_f1,
            "dev_acc": r.dev_acc,
            "test_f1": r.test_f1,
            "test_acc": r.test_acc,
            "test_precision": r.test_precision,
            "test_recall": r.test_recall,
            "embed_time": r.embed_time,
            "reduce_time": r.reduce_time,
            "train_time": r.train_time,
        }
        for r in results
    ]

    pl.DataFrame(results_data).write_csv(results_path)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "="*80)
    print("Sweep complete!")
    print("="*80)


if __name__ == "__main__":
    main()
