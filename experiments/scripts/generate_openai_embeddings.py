#!/usr/bin/env python3
"""
Generate OpenAI Embeddings for CT24 Dataset

Generates embeddings using OpenAI's text-embedding-3-small and text-embedding-3-large
models for train/dev/test splits.

Usage:
    python experiments/scripts/generate_openai_embeddings.py
    python experiments/scripts/generate_openai_embeddings.py --model text-embedding-3-small
    python experiments/scripts/generate_openai_embeddings.py --model text-embedding-3-large --dimensions 1024
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment from .env file
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH, override=True)

# Verify API key is loaded
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(f"OPENAI_API_KEY not found. Check {ENV_PATH}")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "embedding_cache"

MODELS = {
    "text-embedding-3-small": {"max_dims": 1536, "cost_per_m": 0.02},
    "text-embedding-3-large": {"max_dims": 3072, "cost_per_m": 0.13},
}

BATCH_SIZE = 2000  # OpenAI allows up to 2048 texts per request


# =============================================================================
# Embedding Generation
# =============================================================================

def generate_embeddings(
    client: OpenAI,
    texts: list[str],
    model: str,
    dimensions: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI API."""

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Generating embeddings"):
        batch = texts[i:i + batch_size]

        # Build request params
        params = {
            "input": batch,
            "model": model,
        }
        if dimensions is not None:
            params["dimensions"] = dimensions

        # Call API with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(**params)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise

        # Extract embeddings
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def estimate_cost(n_texts: int, avg_tokens: int, model: str) -> float:
    """Estimate cost for embedding generation."""
    total_tokens = n_texts * avg_tokens
    cost_per_m = MODELS[model]["cost_per_m"]
    return (total_tokens / 1_000_000) * cost_per_m


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate OpenAI embeddings")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="text-embedding-3-small",
        help="OpenAI embedding model to use",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Output dimensions (optional, uses model default if not specified)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="Splits to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost without generating embeddings",
    )
    args = parser.parse_args()

    # Validate dimensions
    max_dims = MODELS[args.model]["max_dims"]
    if args.dimensions is not None and args.dimensions > max_dims:
        print(f"Error: {args.model} supports max {max_dims} dimensions")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize client with explicit API key
    client = OpenAI(api_key=OPENAI_API_KEY)

    print("=" * 70)
    print(f"OpenAI Embedding Generation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dimensions: {args.dimensions or 'default (' + str(max_dims) + ')'}")
    print(f"Splits: {args.splits}")
    print()

    # Load data and estimate
    split_data = {}
    total_texts = 0

    for split in args.splits:
        path = DATA_DIR / f"CT24_{split}_features.parquet"
        if not path.exists():
            print(f"Error: {path} not found")
            return
        df = pl.read_parquet(path)
        texts = df["cleaned_text"].to_list()
        split_data[split] = texts
        total_texts += len(texts)
        print(f"  {split}: {len(texts)} texts")

    # Estimate cost (rough estimate: ~5 tokens per text on average for short claims)
    avg_tokens = 15  # Conservative estimate for claim text
    estimated_cost = estimate_cost(total_texts * 3, avg_tokens, args.model)  # x3 for 3 splits typical
    actual_cost = estimate_cost(total_texts, avg_tokens, args.model)

    print()
    print(f"Total texts: {total_texts}")
    print(f"Estimated cost: ~${actual_cost:.3f}")

    if args.dry_run:
        print("\n[Dry run - no embeddings generated]")
        return

    print()

    # Generate embeddings for each split
    dims_suffix = f"_{args.dimensions}" if args.dimensions else ""
    model_short = args.model.replace("text-embedding-3-", "openai-")

    for split in args.splits:
        texts = split_data[split]

        # Check cache
        cache_file = OUTPUT_DIR / f"{model_short}{dims_suffix}_{split}.npy"
        if cache_file.exists():
            print(f"  {split}: Already exists at {cache_file.name}, skipping")
            continue

        print(f"\nProcessing {split} ({len(texts)} texts)...")
        start_time = time.time()

        embeddings = generate_embeddings(
            client=client,
            texts=texts,
            model=args.model,
            dimensions=args.dimensions,
        )

        elapsed = time.time() - start_time

        # Save
        np.save(cache_file, embeddings)
        print(f"  Saved: {cache_file.name} ({embeddings.shape})")
        print(f"  Time: {elapsed:.1f}s ({len(texts)/elapsed:.1f} texts/s)")

    # Also save combined npz for compatibility with feature_combination_sweep.py
    print("\nCreating combined cache file...")

    combined_file = OUTPUT_DIR / f"{model_short}{dims_suffix}_embeddings.npz"

    embeddings_dict = {}
    for split in ["train", "dev", "test"]:
        split_file = OUTPUT_DIR / f"{model_short}{dims_suffix}_{split}.npy"
        if split_file.exists():
            embeddings_dict[split] = np.load(split_file)

    if len(embeddings_dict) == 3:
        np.savez(combined_file, **embeddings_dict)
        print(f"Saved: {combined_file.name}")

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)

    # Print dimensions for reference
    if embeddings_dict:
        sample = list(embeddings_dict.values())[0]
        print(f"\nEmbedding dimensions: {sample.shape[1]}")
        print(f"\nTo use in feature_combination_sweep.py, add to EMBEDDING_MODELS:")
        print(f'    "{model_short}{dims_suffix}": "openai:{args.model}",')


if __name__ == "__main__":
    main()
