#!/usr/bin/env python3
"""
Embedding Generation for Checkworthiness Classification

Generates embeddings using bge-small-en-v1.5 and creates multiple feature set variants:
1. text_only: 35 text features (baseline)
2. text_embed_full: text features + raw embeddings (35 + 384 = 419)
3. text_embed_pca64: text features + PCA-64 embeddings (35 + 64 = 99)
4. text_embed_pca128: text features + PCA-128 embeddings (35 + 128 = 163)

IMPORTANT: PCA is fit on train set only, then applied to dev/test.

Usage:
    python experiments/scripts/generate_embeddings.py
    python experiments/scripts/generate_embeddings.py --model BAAI/bge-base-en-v1.5
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed.")
    print("Install with: uv add sentence-transformers")
    exit(1)


# =============================================================================
# Paths
# =============================================================================

INPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_classifier"


# =============================================================================
# Feature Columns
# =============================================================================

# All boolean and numeric features from text extraction
TEXT_FEATURE_COLS = [
    # Group A: Quantification
    "has_number", "has_precise_number", "has_large_scale", "number_count", "has_range", "has_delta",
    # Group B: Temporal
    "has_specific_year", "has_relative_time", "has_temporal_anchor",
    # Group C: Source/Evidence
    "has_source_attribution", "has_evidence_noun", "has_official_source", "has_said_claimed",
    # Group D: Comparative
    "has_comparative", "has_superlative", "has_ranking",
    # Group E: Action/Change
    "has_increase_decrease", "has_voted", "has_negation_claim",
    # Group F: Opinion/Hedge
    "has_first_person_stance", "has_desire_intent", "has_future_modal", "has_hedge", "has_vague_quantifier",
    # Group G: Rhetorical
    "has_rhetorical_filler", "has_fact_assertion", "is_question", "has_transcript_artifact",
    # Group H: Metadata
    "word_count", "avg_word_length", "alpha_ratio",
    # Group I: Interactions
    "has_number_and_time", "has_number_and_comparative", "has_change_and_time", "has_source_and_number",
]

# Columns to keep for reference (not features)
METADATA_COLS = ["Sentence_id", "class_label", "cleaned_text", "original_text"]


# =============================================================================
# Embedding Generation
# =============================================================================

def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    return embeddings


# =============================================================================
# Dataset Creation
# =============================================================================

def create_feature_matrix(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Extract feature matrix from DataFrame."""
    # Convert boolean to int for sklearn
    feature_data = []
    for col in feature_cols:
        if col in df.columns:
            series = df[col]
            if series.dtype == pl.Boolean:
                feature_data.append(series.cast(pl.Int8).to_numpy())
            else:
                feature_data.append(series.to_numpy())
        else:
            print(f"  Warning: column {col} not found, using zeros")
            feature_data.append(np.zeros(len(df)))

    return np.column_stack(feature_data)


def create_dataset_variants(
    df: pl.DataFrame,
    embeddings: np.ndarray,
    pca_64: PCA | None,
    pca_128: PCA | None,
    feature_cols: list[str],
) -> dict[str, pl.DataFrame]:
    """Create all dataset variants with different feature combinations."""

    # Extract text features
    text_features = create_feature_matrix(df, feature_cols)

    # Get metadata
    metadata_df = df.select([c for c in METADATA_COLS if c in df.columns])

    variants = {}

    # 1. Text only (baseline)
    text_col_names = [f"feat_{col}" for col in feature_cols]
    text_feature_dict = {name: text_features[:, i] for i, name in enumerate(text_col_names)}
    variants["text_only"] = pl.concat([
        metadata_df,
        pl.DataFrame(text_feature_dict),
    ], how="horizontal")

    # 2. Text + Full embeddings
    embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
    embed_dict = {col: embeddings[:, i] for i, col in enumerate(embed_cols)}
    variants["text_embed_full"] = pl.concat([
        metadata_df,
        pl.DataFrame(text_feature_dict),
        pl.DataFrame(embed_dict),
    ], how="horizontal")

    # 3. Text + PCA-64 embeddings
    if pca_64 is not None:
        embed_pca64 = pca_64.transform(embeddings)
        pca64_cols = [f"pca64_{i}" for i in range(embed_pca64.shape[1])]
        pca64_dict = {col: embed_pca64[:, i] for i, col in enumerate(pca64_cols)}
        variants["text_embed_pca64"] = pl.concat([
            metadata_df,
            pl.DataFrame(text_feature_dict),
            pl.DataFrame(pca64_dict),
        ], how="horizontal")

    # 4. Text + PCA-128 embeddings
    if pca_128 is not None:
        embed_pca128 = pca_128.transform(embeddings)
        pca128_cols = [f"pca128_{i}" for i in range(embed_pca128.shape[1])]
        pca128_dict = {col: embed_pca128[:, i] for i, col in enumerate(pca128_cols)}
        variants["text_embed_pca128"] = pl.concat([
            metadata_df,
            pl.DataFrame(text_feature_dict),
            pl.DataFrame(pca128_dict),
        ], how="horizontal")

    return variants


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and create classifier datasets")
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence transformer model to use (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load embedding model
    print(f"\n{'='*60}")
    print(f"Loading embedding model: {args.model}")
    print(f"{'='*60}")
    model = SentenceTransformer(args.model)
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embed_dim}")

    # Load datasets
    print(f"\n{'='*60}")
    print("Loading feature datasets")
    print(f"{'='*60}")

    datasets = {}
    for split in ["train", "dev", "test"]:
        path = INPUT_DIR / f"CT24_{split}_features.parquet"
        if not path.exists():
            print(f"  ERROR: {path} not found")
            return
        datasets[split] = pl.read_parquet(path)
        print(f"  {split}: {len(datasets[split])} rows")

    # Generate embeddings for all splits
    print(f"\n{'='*60}")
    print("Generating embeddings")
    print(f"{'='*60}")

    embeddings = {}
    for split, df in datasets.items():
        print(f"\n  {split} ({len(df)} texts):")
        texts = df["cleaned_text"].to_list()
        embeddings[split] = generate_embeddings(texts, model, args.batch_size)
        print(f"    Shape: {embeddings[split].shape}")

    # Save raw embeddings
    print(f"\n{'='*60}")
    print("Saving raw embeddings")
    print(f"{'='*60}")

    for split, embed in embeddings.items():
        embed_path = args.output_dir / f"embeddings_{split}.npy"
        np.save(embed_path, embed)
        print(f"  Saved: {embed_path}")

    # Fit PCA on train only
    print(f"\n{'='*60}")
    print("Fitting PCA on train embeddings")
    print(f"{'='*60}")

    train_embed = embeddings["train"]

    # PCA-64
    print("\n  Fitting PCA-64...")
    pca_64 = PCA(n_components=64, random_state=42)
    pca_64.fit(train_embed)
    explained_64 = sum(pca_64.explained_variance_ratio_) * 100
    print(f"    Explained variance: {explained_64:.1f}%")

    # PCA-128
    print("\n  Fitting PCA-128...")
    pca_128 = PCA(n_components=128, random_state=42)
    pca_128.fit(train_embed)
    explained_128 = sum(pca_128.explained_variance_ratio_) * 100
    print(f"    Explained variance: {explained_128:.1f}%")

    # Save PCA models
    pca_path = args.output_dir / "pca_models.pkl"
    with open(pca_path, "wb") as f:
        pickle.dump({"pca_64": pca_64, "pca_128": pca_128}, f)
    print(f"\n  Saved PCA models: {pca_path}")

    # Create all dataset variants
    print(f"\n{'='*60}")
    print("Creating dataset variants")
    print(f"{'='*60}")

    # Get actual feature columns present in data
    available_features = [c for c in TEXT_FEATURE_COLS if c in datasets["train"].columns]
    print(f"\n  Using {len(available_features)} text features")

    for split in ["train", "dev", "test"]:
        print(f"\n  Processing {split}...")

        variants = create_dataset_variants(
            datasets[split],
            embeddings[split],
            pca_64,
            pca_128,
            available_features,
        )

        for variant_name, variant_df in variants.items():
            # Save parquet
            out_path = args.output_dir / f"{split}_{variant_name}.parquet"
            variant_df.write_parquet(out_path)

            # Count feature columns (exclude metadata)
            feature_count = len([c for c in variant_df.columns if c not in METADATA_COLS])
            print(f"    {variant_name}: {feature_count} features -> {out_path.name}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"""
Dataset variants created:
  1. text_only:        {len(available_features)} text features (baseline)
  2. text_embed_full:  {len(available_features)} text + {embed_dim} embed = {len(available_features) + embed_dim} features
  3. text_embed_pca64: {len(available_features)} text + 64 PCA = {len(available_features) + 64} features
  4. text_embed_pca128:{len(available_features)} text + 128 PCA = {len(available_features) + 128} features

PCA explained variance:
  - PCA-64:  {explained_64:.1f}%
  - PCA-128: {explained_128:.1f}%

Output directory: {args.output_dir}
""")

    print("="*60)
    print("Embedding generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
