#!/usr/bin/env python3
"""
Generate DeBERTa CLS embeddings for CT24 splits.

Extracts the [CLS] token embedding from the fine-tuned DeBERTa model
for use in downstream classifiers.

Usage:
    python generate_deberta_embeddings.py --model-path ~/deberta_model --data-dir ~/data --output-dir ~/data/CT24_embeddings
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_data(data_dir: Path, split: str) -> tuple[list[str], list[int]]:
    """Load texts and sentence IDs for a split."""
    clean_dir = data_dir / "CT24_clean"

    parquet_path = clean_dir / f"CT24_{split}_clean.parquet"
    if parquet_path.exists():
        df = pl.read_parquet(parquet_path)
    else:
        tsv_path = clean_dir / f"CT24_{split}_clean.tsv"
        if tsv_path.exists():
            df = pl.read_csv(tsv_path, separator="\t")
        else:
            raise FileNotFoundError(f"Data not found for split: {split} at {clean_dir}")

    texts = df["Text"].to_list()
    sentence_ids = df["Sentence_id"].to_list()

    return texts, sentence_ids


def extract_embeddings(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract CLS embeddings from DeBERTa."""
    model.eval()
    all_embeddings = []

    # Access the base DeBERTa model (before classification head)
    if hasattr(model, "deberta"):
        base_model = model.deberta
    elif hasattr(model, "base_model"):
        base_model = model.base_model
    else:
        raise ValueError("Cannot find base model in DeBERTa")

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get hidden states
            outputs = base_model(**inputs)

            # CLS token is the first token
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Generate DeBERTa embeddings")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to fine-tuned DeBERTa model")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Base data directory containing CT24_clean")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                        help="Splits to process")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATE DEBERTA CLS EMBEDDINGS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\n🤖 Loading DeBERTa from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)

    # Get embedding dimension
    if hasattr(model, "deberta"):
        emb_dim = model.deberta.config.hidden_size
    else:
        emb_dim = model.config.hidden_size
    print(f"   Embedding dimension: {emb_dim}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in args.splits:
        print(f"\n📊 Processing {split}...")

        try:
            texts, sentence_ids = load_data(args.data_dir, split)
            print(f"   Samples: {len(texts)}")

            # Extract embeddings
            embeddings = extract_embeddings(
                model, tokenizer, texts, device, args.batch_size
            )
            print(f"   Embeddings shape: {embeddings.shape}")

            # Save embeddings
            output_path = args.output_dir / f"CT24_{split}_deberta_embeddings.npy"
            np.save(output_path, embeddings)
            print(f"   💾 Saved to: {output_path}")

            # Also save sentence IDs for alignment verification
            ids_path = args.output_dir / f"CT24_{split}_sentence_ids.npy"
            np.save(ids_path, np.array(sentence_ids))

        except FileNotFoundError as e:
            print(f"   ⚠️ Skipped: {e}")

    print(f"\n✅ Done! Embeddings saved to {args.output_dir}")


if __name__ == "__main__":
    main()
