#!/usr/bin/env python3
"""
Generate Embeddings from Multiple Models.

Supports:
1. Fine-tuned DeBERTa models (CLS token)
2. Sentence-transformers models (mean pooling)
3. Pre-trained DeBERTa/BERT models

Usage:
    # From fine-tuned DeBERTa
    python generate_embeddings_multimodel.py --model finetuned --model-path ~/deberta_model

    # Sentence-transformers
    python generate_embeddings_multimodel.py --model bge-large-en-v1.5
    python generate_embeddings_multimodel.py --model all-mpnet-base-v2
    python generate_embeddings_multimodel.py --model all-MiniLM-L6-v2

    # Pre-trained DeBERTa (no fine-tuning)
    python generate_embeddings_multimodel.py --model deberta-v2-xxlarge
    python generate_embeddings_multimodel.py --model deberta-v3-large
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

# =============================================================================
# Model Registry
# =============================================================================

EMBEDDING_MODELS = {
    # Sentence-transformers (best for semantic similarity)
    "bge-large-en-v1.5": {
        "type": "sentence-transformers",
        "hf_name": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
    },
    "bge-base-en-v1.5": {
        "type": "sentence-transformers",
        "hf_name": "BAAI/bge-base-en-v1.5",
        "dim": 768,
    },
    "all-mpnet-base-v2": {
        "type": "sentence-transformers",
        "hf_name": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
    },
    "all-MiniLM-L6-v2": {
        "type": "sentence-transformers",
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
    },
    "gte-large": {
        "type": "sentence-transformers",
        "hf_name": "thenlper/gte-large",
        "dim": 1024,
    },
    "e5-large-v2": {
        "type": "sentence-transformers",
        "hf_name": "intfloat/e5-large-v2",
        "dim": 1024,
    },
    # Pre-trained DeBERTa (CLS token, not fine-tuned)
    "deberta-v2-xxlarge": {
        "type": "deberta",
        "hf_name": "microsoft/deberta-v2-xxlarge",
        "dim": 1536,
    },
    "deberta-v2-xlarge": {
        "type": "deberta",
        "hf_name": "microsoft/deberta-v2-xlarge",
        "dim": 1024,
    },
    "deberta-v3-large": {
        "type": "deberta",
        "hf_name": "microsoft/deberta-v3-large",
        "dim": 1024,
    },
    "deberta-v3-base": {
        "type": "deberta",
        "hf_name": "microsoft/deberta-v3-base",
        "dim": 768,
    },
    # Fine-tuned model (path provided separately)
    "finetuned": {
        "type": "finetuned",
        "hf_name": None,  # Will be set from --model-path
        "dim": None,  # Auto-detect
    },
}


# =============================================================================
# Embedding Extractors
# =============================================================================

def load_sentence_transformer(model_name: str, device: torch.device):
    """Load a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=str(device))
    return model


def extract_sentence_transformer_embeddings(
    model,
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract embeddings using sentence-transformers."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def load_deberta_model(model_name: str, device: torch.device):
    """Load a DeBERTa model for CLS embedding extraction."""
    from transformers import AutoModel, AutoTokenizer, DebertaV2Tokenizer

    if "deberta" in model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return model, tokenizer


def load_finetuned_model(model_path: str, device: torch.device):
    """Load a fine-tuned classification model for CLS extraction."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Tokenizer

    # Try to detect tokenizer type
    try:
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    return model, tokenizer


def extract_deberta_embeddings(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
    is_finetuned: bool = False,
) -> np.ndarray:
    """Extract CLS embeddings from DeBERTa."""
    all_embeddings = []

    # Get base model for fine-tuned models
    if is_finetuned:
        if hasattr(model, "deberta"):
            base_model = model.deberta
        elif hasattr(model, "base_model"):
            base_model = model.base_model
        else:
            base_model = model
    else:
        base_model = model

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = base_model(**inputs)

            # CLS token is first token
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    embeddings = np.vstack(all_embeddings)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return embeddings


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: Path, split: str) -> tuple[list[str], list[int]]:
    """Load texts and sentence IDs for a split."""
    clean_dir = data_dir / "CT24_clean"
    features_dir = data_dir / "CT24_features"

    # Try multiple paths
    for base_dir in [clean_dir, features_dir]:
        for suffix in ["_clean", "_features", ""]:
            for ext in [".parquet", ".tsv"]:
                path = base_dir / f"CT24_{split}{suffix}{ext}"
                if path.exists():
                    if ext == ".parquet":
                        df = pl.read_parquet(path)
                    else:
                        df = pl.read_csv(path, separator="\t")

                    texts = df["Text"].to_list()
                    sentence_ids = df["Sentence_id"].to_list()
                    return texts, sentence_ids

    raise FileNotFoundError(f"Data not found for split: {split}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from multiple models")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(EMBEDDING_MODELS.keys()),
                        help="Embedding model to use")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to fine-tuned model (required for 'finetuned' model)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Data directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-MODEL EMBEDDING GENERATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_info = EMBEDDING_MODELS[args.model]
    print(f"\nModel: {args.model}")
    print(f"Type: {model_info['type']}")

    # Load model
    if model_info["type"] == "sentence-transformers":
        print(f"Loading: {model_info['hf_name']}")
        model = load_sentence_transformer(model_info["hf_name"], device)
        tokenizer = None
        emb_dim = model_info["dim"]

    elif model_info["type"] == "deberta":
        print(f"Loading: {model_info['hf_name']}")
        model, tokenizer = load_deberta_model(model_info["hf_name"], device)
        emb_dim = model_info["dim"]

    elif model_info["type"] == "finetuned":
        if not args.model_path:
            raise ValueError("--model-path required for 'finetuned' model")
        print(f"Loading fine-tuned model from: {args.model_path}")
        model, tokenizer = load_finetuned_model(str(args.model_path), device)
        # Auto-detect embedding dim
        if hasattr(model, "deberta"):
            emb_dim = model.deberta.config.hidden_size
        elif hasattr(model, "config"):
            emb_dim = model.config.hidden_size
        else:
            emb_dim = 1024  # Default

    print(f"Embedding dimension: {emb_dim}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in args.splits:
        print(f"\n📊 Processing {split}...")

        try:
            texts, sentence_ids = load_data(args.data_dir, split)
            print(f"   Samples: {len(texts)}")

            # Extract embeddings
            if model_info["type"] == "sentence-transformers":
                embeddings = extract_sentence_transformer_embeddings(
                    model, texts, args.batch_size
                )
            else:
                is_finetuned = model_info["type"] == "finetuned"
                embeddings = extract_deberta_embeddings(
                    model, tokenizer, texts, device,
                    args.batch_size, args.max_length, is_finetuned
                )

            print(f"   Embeddings shape: {embeddings.shape}")

            # Save
            model_name_safe = args.model.replace("/", "_")
            output_path = args.output_dir / f"CT24_{split}_{model_name_safe}_embeddings.npy"
            np.save(output_path, embeddings)
            print(f"   💾 Saved: {output_path}")

            # Save sentence IDs for alignment
            ids_path = args.output_dir / f"CT24_{split}_sentence_ids.npy"
            if not ids_path.exists():
                np.save(ids_path, np.array(sentence_ids))

        except FileNotFoundError as e:
            print(f"   ⚠️ Skipped: {e}")

    print(f"\n✅ Done! Embeddings saved to {args.output_dir}")

    # Print model summary
    print(f"\n📋 Available embedding models:")
    for key, info in EMBEDDING_MODELS.items():
        marker = "→" if key == args.model else " "
        dim = info["dim"] if info["dim"] else "auto"
        print(f"   {marker} {key}: {info['type']}, dim={dim}")


if __name__ == "__main__":
    main()
