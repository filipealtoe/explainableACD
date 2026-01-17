#!/usr/bin/env python3
"""Test if normalized claim embeddings improve classifier performance.

Compares:
1. Original text embeddings (baseline)
2. Normalized claim embeddings (fallback to original if no claim)

Combined with v4 LLM features + PCA64 + Logistic Regression.

Baseline to beat: F1=0.7607, Acc=0.8856

Usage:
    python experiments/scripts/test_normalized_claim_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# =============================================================================
# Configuration
# =============================================================================

# Embedding model (same as v4 features)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

# PCA settings
PCA_COMPONENTS = 64

# Data paths
V4_FEATURES_DIR = Path("data/processed/CT24_llm_features_v4")
NORMALIZED_CLAIMS_DIR = Path("data/processed/CT24_normalized_claims")
RAW_DATA_DIR = Path("data/raw/CT24_checkworthy_english")

# Feature columns from v4 (excluding sentence_id)
V4_FEATURE_COLS = [
    # Checkability features
    "check_score", "check_p_yes", "check_p_no", "check_entropy", "check_entropy_norm",
    "check_margin_p", "check_margin_logit", "check_score_p_residual",
    # Verifiability features
    "verif_score", "verif_p_yes", "verif_p_no", "verif_entropy", "verif_entropy_norm",
    "verif_margin_p", "verif_margin_logit", "verif_score_p_residual",
    # Harm features
    "harm_score", "harm_p_yes", "harm_p_no", "harm_entropy", "harm_entropy_norm",
    "harm_margin_p", "harm_margin_logit", "harm_score_p_residual",
    "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
    # Cross-module features
    "score_variance", "score_max_diff", "check_minus_verif", "check_minus_harm", "verif_minus_harm",
    "yes_vote_count", "check_verif_agree", "check_harm_agree", "verif_harm_agree",
]


# =============================================================================
# Data Loading
# =============================================================================

def load_v4_features(split: str) -> pl.DataFrame:
    """Load v4 LLM features for a split."""
    files = {
        "train": "train_llm_features.parquet",
        "dev": "dev_llm_features.parquet",
        "test": "test_llm_features.parquet",
    }
    return pl.read_parquet(V4_FEATURES_DIR / files[split])


def load_normalized_claims(split: str) -> pl.DataFrame:
    """Load normalized claims for a split."""
    # Find the latest file for this split
    pattern = f"CT24_{split}_normalized_*.parquet"
    files = sorted(NORMALIZED_CLAIMS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No normalized claims found for {split}")
    return pl.read_parquet(files[-1])


def load_raw_data(split: str) -> pl.DataFrame:
    """Load raw data with original text and labels."""
    files = {
        "train": "CT24_checkworthy_english_train.tsv",
        "dev": "CT24_checkworthy_english_dev.tsv",
        "test": "CT24_checkworthy_english_test_gold.tsv",
    }
    df = pl.read_csv(RAW_DATA_DIR / files[split], separator="\t")
    df = df.rename({
        "Sentence_id": "sentence_id",
        "Text": "text",
        "class_label": "label",
    })
    df = df.with_columns(pl.col("sentence_id").cast(pl.Utf8))
    return df


def prepare_text_for_embedding(split: str, use_normalized: bool = True) -> tuple[list[str], list[str], np.ndarray]:
    """Prepare texts for embedding generation.

    Args:
        split: Dataset split
        use_normalized: If True, use normalized_claim (fallback to original text)

    Returns:
        Tuple of (texts, sentence_ids, labels as 0/1)
    """
    raw_df = load_raw_data(split)

    if use_normalized:
        norm_df = load_normalized_claims(split)
        # Join to get normalized claims
        merged = raw_df.join(
            norm_df.select(["sentence_id", "normalized_claim", "has_claim"]),
            on="sentence_id",
            how="left"
        )
        # Use normalized_claim if available, else original text
        texts = []
        for row in merged.to_dicts():
            if row.get("has_claim") and row.get("normalized_claim"):
                texts.append(row["normalized_claim"])
            else:
                texts.append(row["text"])
    else:
        texts = raw_df["text"].to_list()

    sentence_ids = raw_df["sentence_id"].to_list()
    labels = np.array([1 if l == "Yes" else 0 for l in raw_df["label"].to_list()])

    return texts, sentence_ids, labels


# =============================================================================
# Embedding Generation
# =============================================================================

def generate_embeddings(texts: list[str], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for texts."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # BGE models benefit from normalization
    )
    return embeddings


# =============================================================================
# Feature Preparation
# =============================================================================

def prepare_features(
    split: str,
    embeddings: np.ndarray,
    sentence_ids: list[str],
    pca: PCA | None = None,
    fit_pca: bool = False,
) -> tuple[np.ndarray, np.ndarray, PCA | None]:
    """Prepare features by combining embeddings with v4 LLM features.

    Args:
        split: Dataset split
        embeddings: Text embeddings
        sentence_ids: Sentence IDs (for joining)
        pca: Pre-fitted PCA object (None if not using PCA or fitting new)
        fit_pca: If True, fit new PCA on this split

    Returns:
        Tuple of (combined_features, labels, pca_object)
    """
    # Apply PCA if specified
    if fit_pca:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"    PCA fit: {embeddings.shape[1]} -> {PCA_COMPONENTS} dims "
              f"(explained variance: {pca.explained_variance_ratio_.sum():.2%})")
    elif pca is not None:
        embeddings_reduced = pca.transform(embeddings)
    else:
        embeddings_reduced = embeddings

    # Load v4 features
    v4_df = load_v4_features(split)

    # Create DataFrame with embeddings
    embed_df = pl.DataFrame({
        "sentence_id": sentence_ids,
        **{f"embed_{i}": embeddings_reduced[:, i] for i in range(embeddings_reduced.shape[1])}
    })

    # Join with v4 features
    combined = v4_df.join(embed_df, on="sentence_id", how="inner")

    # Extract feature columns
    embed_cols = [f"embed_{i}" for i in range(embeddings_reduced.shape[1])]
    feature_cols = V4_FEATURE_COLS + embed_cols

    # Filter to existing columns (some v4 features might be missing)
    available_cols = [c for c in feature_cols if c in combined.columns]

    X = combined.select(available_cols).to_numpy()

    # Get labels from raw data
    raw_df = load_raw_data(split)
    # Align labels with combined data
    label_map = dict(zip(raw_df["sentence_id"].to_list(), raw_df["label"].to_list()))
    y = np.array([1 if label_map.get(sid) == "Yes" else 0 for sid in combined["sentence_id"].to_list()])

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    return X, y, pca


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_dev: np.ndarray, y_dev: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> dict:
    """Train logistic regression and evaluate on all splits."""

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    results = {}
    for split_name, X, y in [("train", X_train, y_train), ("dev", X_dev, y_dev), ("test", X_test, y_test)]:
        y_pred = clf.predict(X)
        results[f"{split_name}_f1"] = f1_score(y, y_pred)
        results[f"{split_name}_acc"] = accuracy_score(y, y_pred)
        results[f"{split_name}_precision"] = precision_score(y, y_pred)
        results[f"{split_name}_recall"] = recall_score(y, y_pred)

    return results


def print_results(results: dict, name: str, baseline: dict | None = None):
    """Print results with optional comparison to baseline."""
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")

    for split in ["train", "dev", "test"]:
        f1 = results[f"{split}_f1"]
        acc = results[f"{split}_acc"]
        prec = results[f"{split}_precision"]
        rec = results[f"{split}_recall"]

        print(f"\n  {split.upper()}:")
        print(f"    F1:        {f1:.4f}", end="")
        if baseline and f"{split}_f1" in baseline:
            diff = f1 - baseline[f"{split}_f1"]
            print(f"  ({'+' if diff >= 0 else ''}{diff:.4f})", end="")
        print()

        print(f"    Accuracy:  {acc:.4f}", end="")
        if baseline and f"{split}_acc" in baseline:
            diff = acc - baseline[f"{split}_acc"]
            print(f"  ({'+' if diff >= 0 else ''}{diff:.4f})", end="")
        print()

        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("NORMALIZED CLAIM EMBEDDING EXPERIMENT")
    print("=" * 60)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"PCA components: {PCA_COMPONENTS}")
    print(f"Classifier: LogisticRegression (balanced)")
    print("=" * 60)

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Baseline: User reported F1=0.7607, Acc=0.8856 (likely dev set)
    baseline = {
        "dev_f1": 0.7607,
        "dev_acc": 0.8856,
    }

    # ==========================================================================
    # Experiment 1: Original text embeddings (baseline reproduction)
    # ==========================================================================
    print("\n" + "#" * 60)
    print("# Experiment 1: Original Text Embeddings")
    print("#" * 60)

    # Generate embeddings for original text
    print("\nGenerating embeddings for original text...")

    print("  Train split:")
    train_texts_orig, train_ids, _ = prepare_text_for_embedding("train", use_normalized=False)
    train_embeds_orig = generate_embeddings(train_texts_orig, model)

    print("  Dev split:")
    dev_texts_orig, dev_ids, _ = prepare_text_for_embedding("dev", use_normalized=False)
    dev_embeds_orig = generate_embeddings(dev_texts_orig, model)

    print("  Test split:")
    test_texts_orig, test_ids, _ = prepare_text_for_embedding("test", use_normalized=False)
    test_embeds_orig = generate_embeddings(test_texts_orig, model)

    # Prepare features with PCA
    print("\nPreparing features with PCA64...")
    X_train_orig, y_train, pca = prepare_features("train", train_embeds_orig, train_ids, fit_pca=True)
    X_dev_orig, y_dev, _ = prepare_features("dev", dev_embeds_orig, dev_ids, pca=pca)
    X_test_orig, y_test, _ = prepare_features("test", test_embeds_orig, test_ids, pca=pca)

    print(f"\nFeature dimensions: {X_train_orig.shape[1]}")
    print(f"  Train: {X_train_orig.shape[0]} samples")
    print(f"  Dev: {X_dev_orig.shape[0]} samples")
    print(f"  Test: {X_test_orig.shape[0]} samples")

    # Train and evaluate
    print("\nTraining classifier...")
    results_orig = train_and_evaluate(X_train_orig, y_train, X_dev_orig, y_dev, X_test_orig, y_test)
    print_results(results_orig, "Original Text + v4 LLM + PCA64", baseline)

    # ==========================================================================
    # Experiment 2: Normalized claim embeddings
    # ==========================================================================
    print("\n" + "#" * 60)
    print("# Experiment 2: Normalized Claim Embeddings")
    print("#" * 60)

    # Generate embeddings for normalized claims
    print("\nGenerating embeddings for normalized claims (fallback to original)...")

    print("  Train split:")
    train_texts_norm, train_ids_norm, _ = prepare_text_for_embedding("train", use_normalized=True)
    train_embeds_norm = generate_embeddings(train_texts_norm, model)

    print("  Dev split:")
    dev_texts_norm, dev_ids_norm, _ = prepare_text_for_embedding("dev", use_normalized=True)
    dev_embeds_norm = generate_embeddings(dev_texts_norm, model)

    print("  Test split:")
    test_texts_norm, test_ids_norm, _ = prepare_text_for_embedding("test", use_normalized=True)
    test_embeds_norm = generate_embeddings(test_texts_norm, model)

    # Count how many normalized claims are used
    for split_name, texts_norm, texts_orig in [
        ("train", train_texts_norm, train_texts_orig),
        ("dev", dev_texts_norm, dev_texts_orig),
        ("test", test_texts_norm, test_texts_orig),
    ]:
        n_normalized = sum(1 for n, o in zip(texts_norm, texts_orig) if n != o)
        print(f"    {split_name}: {n_normalized}/{len(texts_norm)} using normalized claims")

    # Prepare features with PCA (fit new PCA on normalized embeddings)
    print("\nPreparing features with PCA64...")
    X_train_norm, y_train_norm, pca_norm = prepare_features("train", train_embeds_norm, train_ids_norm, fit_pca=True)
    X_dev_norm, y_dev_norm, _ = prepare_features("dev", dev_embeds_norm, dev_ids_norm, pca=pca_norm)
    X_test_norm, y_test_norm, _ = prepare_features("test", test_embeds_norm, test_ids_norm, pca=pca_norm)

    # Train and evaluate
    print("\nTraining classifier...")
    results_norm = train_and_evaluate(X_train_norm, y_train_norm, X_dev_norm, y_dev_norm, X_test_norm, y_test_norm)
    print_results(results_norm, "Normalized Claims + v4 LLM + PCA64", baseline)

    # ==========================================================================
    # Summary Comparison
    # ==========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"\n{'Method':<40} {'Dev F1':<12} {'Test F1':<12} {'Test Acc':<12}")
    print("-" * 76)
    print(f"{'Baseline (reported)':<40} {baseline['dev_f1']:<12.4f} {'N/A':<12} {baseline['dev_acc']:<12.4f}")
    print(f"{'Original Text + v4 + PCA64':<40} {results_orig['dev_f1']:<12.4f} {results_orig['test_f1']:<12.4f} {results_orig['test_acc']:<12.4f}")
    print(f"{'Normalized Claims + v4 + PCA64':<40} {results_norm['dev_f1']:<12.4f} {results_norm['test_f1']:<12.4f} {results_norm['test_acc']:<12.4f}")

    # Highlight winner
    if results_norm["dev_f1"] > results_orig["dev_f1"]:
        print(f"\n✓ Normalized claims IMPROVE Dev F1 by {results_norm['dev_f1'] - results_orig['dev_f1']:.4f}")
    else:
        print(f"\n✗ Normalized claims DECREASE Dev F1 by {results_orig['dev_f1'] - results_norm['dev_f1']:.4f}")

    if results_norm["test_f1"] > results_orig["test_f1"]:
        print(f"✓ Normalized claims IMPROVE Test F1 by {results_norm['test_f1'] - results_orig['test_f1']:.4f}")
    else:
        print(f"✗ Normalized claims DECREASE Test F1 by {results_orig['test_f1'] - results_norm['test_f1']:.4f}")


if __name__ == "__main__":
    main()
