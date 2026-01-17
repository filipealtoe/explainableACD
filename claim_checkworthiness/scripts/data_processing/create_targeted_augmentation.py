#!/usr/bin/env python3
"""
Create targeted augmentation dataset based on DeBERTa errors.

Strategy:
1. Run DeBERTa on dev and dev-test to find misclassified examples
2. Compute embeddings for all error cases
3. For each synthetic sample, compute cosine similarity to error cases
4. Select synthetic samples most similar to error cases
5. Create augmented training set with only these targeted samples

This is "hard example mining" - we augment specifically where the model struggles.

Usage:
    python experiments/scripts/create_targeted_augmentation.py
    python experiments/scripts/create_targeted_augmentation.py --top-k 500 --similarity-threshold 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths
DEBERTA_MODEL = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
CLEAN_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_clean"
SYNTHETIC_DIR = Path(__file__).parent.parent / "results" / "synthetic_data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_targeted_augmented"


def load_data(split: str) -> pl.DataFrame:
    """Load cleaned data for a split."""
    parquet_path = CLEAN_DATA_DIR / f"CT24_{split}_clean.parquet"
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)

    tsv_path = CLEAN_DATA_DIR / f"CT24_{split}_clean.tsv"
    if tsv_path.exists():
        return pl.read_csv(tsv_path, separator="\t")

    raise FileNotFoundError(f"Data not found for split: {split}")


def get_deberta_predictions(model, tokenizer, texts: list[str], device, batch_size: int = 16) -> tuple[list[int], list[float]]:
    """Get DeBERTa predictions and probabilities."""
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="DeBERTa inference"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # P(Yes)

    return all_preds, all_probs


def get_deberta_embeddings(model, tokenizer, texts: list[str], device, batch_size: int = 16) -> np.ndarray:
    """Get DeBERTa CLS embeddings for texts."""
    model.eval()
    all_embeddings = []

    # Access the base model (before classification head)
    if hasattr(model, "deberta"):
        base_model = model.deberta
    else:
        base_model = model.base_model

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get hidden states from base model
            outputs = base_model(**inputs)

            # CLS token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def find_errors(df: pl.DataFrame, predictions: list[int]) -> tuple[list[int], list[int]]:
    """Find indices of false positives and false negatives."""
    labels = [1 if l == "Yes" else 0 for l in df["class_label"].to_list()]

    false_positives = []  # Predicted Yes, actual No
    false_negatives = []  # Predicted No, actual Yes

    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred == 1 and label == 0:
            false_positives.append(i)
        elif pred == 0 and label == 1:
            false_negatives.append(i)

    return false_positives, false_negatives


def main():
    parser = argparse.ArgumentParser(description="Create targeted augmentation dataset")
    parser.add_argument("--synthetic-path", type=Path, default=None,
                        help="Path to validated synthetic data (auto-detect if not specified)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Select top-k synthetic samples per error type (default: all above threshold)")
    parser.add_argument("--similarity-threshold", type=float, default=0.65,
                        help="Minimum cosine similarity to include synthetic sample")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-augment-ratio", type=float, default=0.5,
                        help="Maximum ratio of synthetic samples to original (default: 50%)")
    args = parser.parse_args()

    print("=" * 70)
    print("TARGETED AUGMENTATION: Hard Example Mining")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ==========================================================================
    # Step 1: Load DeBERTa model
    # ==========================================================================
    print(f"\n🤖 Loading DeBERTa from: {DEBERTA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL).to(device)

    # ==========================================================================
    # Step 2: Find DeBERTa errors on dev and dev-test
    # ==========================================================================
    print("\n📊 Finding DeBERTa errors on dev and dev-test...")

    all_error_texts = []
    all_error_labels = []
    error_sources = []

    for split in ["dev", "dev-test"]:
        try:
            df = load_data(split)
            texts = df["Text"].to_list()

            preds, probs = get_deberta_predictions(model, tokenizer, texts, device, args.batch_size)
            fp_indices, fn_indices = find_errors(df, preds)

            # Collect error texts
            for idx in fp_indices:
                all_error_texts.append(texts[idx])
                all_error_labels.append("No")  # FP: actual label is No
                error_sources.append(f"{split}_FP")

            for idx in fn_indices:
                all_error_texts.append(texts[idx])
                all_error_labels.append("Yes")  # FN: actual label is Yes
                error_sources.append(f"{split}_FN")

            print(f"   {split}: {len(fp_indices)} FP, {len(fn_indices)} FN")

        except FileNotFoundError:
            print(f"   {split}: Skipped (not found)")

    print(f"\n   Total errors: {len(all_error_texts)}")
    fp_count = sum(1 for s in error_sources if "FP" in s)
    fn_count = sum(1 for s in error_sources if "FN" in s)
    print(f"   False Positives: {fp_count}")
    print(f"   False Negatives: {fn_count}")

    if not all_error_texts:
        print("\n⚠️ No errors found! Model is perfect on dev/dev-test (unlikely).")
        return

    # ==========================================================================
    # Step 3: Load synthetic data
    # ==========================================================================
    print("\n📦 Loading synthetic data...")

    if args.synthetic_path:
        synthetic_path = args.synthetic_path
    else:
        # Auto-detect latest validated synthetic file
        synthetic_files = list(SYNTHETIC_DIR.glob("*_validated.parquet"))
        if not synthetic_files:
            print("❌ No validated synthetic data found!")
            print(f"   Expected in: {SYNTHETIC_DIR}")
            return
        synthetic_path = max(synthetic_files, key=lambda p: p.stat().st_mtime)

    print(f"   Using: {synthetic_path.name}")
    synthetic_df = pl.read_parquet(synthetic_path)
    print(f"   Total synthetic samples: {len(synthetic_df)}")

    # Filter to hard samples if available
    if "is_hard" in synthetic_df.columns:
        hard_synthetic = synthetic_df.filter(pl.col("is_hard") == True)
        print(f"   Hard samples: {len(hard_synthetic)}")
        synthetic_df = hard_synthetic

    synthetic_texts = synthetic_df["Text"].to_list()
    synthetic_labels = synthetic_df["class_label"].to_list()

    # ==========================================================================
    # Step 4: Compute embeddings
    # ==========================================================================
    print("\n🔢 Computing embeddings...")

    error_embeddings = get_deberta_embeddings(model, tokenizer, all_error_texts, device, args.batch_size)
    synthetic_embeddings = get_deberta_embeddings(model, tokenizer, synthetic_texts, device, args.batch_size)

    print(f"   Error embeddings: {error_embeddings.shape}")
    print(f"   Synthetic embeddings: {synthetic_embeddings.shape}")

    # ==========================================================================
    # Step 5: Compute similarities and select synthetic samples
    # ==========================================================================
    print(f"\n🎯 Computing cosine similarities (threshold={args.similarity_threshold})...")

    # Compute similarity matrix: synthetic x errors
    similarities = cosine_similarity(synthetic_embeddings, error_embeddings)

    # For each synthetic sample, get max similarity to any error
    max_similarities = similarities.max(axis=1)

    # Also track which error type each synthetic is most similar to
    most_similar_error_idx = similarities.argmax(axis=1)

    # Select synthetic samples above threshold
    selected_indices = np.where(max_similarities >= args.similarity_threshold)[0]

    print(f"   Samples above threshold: {len(selected_indices)}")

    # Optionally limit by top-k
    if args.top_k and len(selected_indices) > args.top_k:
        # Sort by similarity and take top-k
        sorted_indices = selected_indices[np.argsort(-max_similarities[selected_indices])]
        selected_indices = sorted_indices[:args.top_k]
        print(f"   Limited to top-k: {len(selected_indices)}")

    # Apply max augment ratio
    train_df = load_data("train")
    max_synthetic = int(len(train_df) * args.max_augment_ratio)
    if len(selected_indices) > max_synthetic:
        sorted_indices = selected_indices[np.argsort(-max_similarities[selected_indices])]
        selected_indices = sorted_indices[:max_synthetic]
        print(f"   Limited by max ratio ({args.max_augment_ratio}): {len(selected_indices)}")

    if len(selected_indices) == 0:
        print("\n⚠️ No synthetic samples passed the similarity threshold!")
        print("   Try lowering --similarity-threshold")
        return

    # Analyze selected samples
    selected_similarities = max_similarities[selected_indices]
    print(f"\n   Selected {len(selected_indices)} synthetic samples:")
    print(f"   Similarity range: [{selected_similarities.min():.3f}, {selected_similarities.max():.3f}]")
    print(f"   Similarity mean: {selected_similarities.mean():.3f}")

    # Count by target error type
    selected_error_types = [error_sources[most_similar_error_idx[i]] for i in selected_indices]
    from collections import Counter
    type_counts = Counter(selected_error_types)
    print(f"\n   Targeting error types:")
    for error_type, count in sorted(type_counts.items()):
        print(f"      {error_type}: {count}")

    # ==========================================================================
    # Step 6: Create augmented training set
    # ==========================================================================
    print("\n📝 Creating targeted augmented training set...")

    # Prepare selected synthetic samples
    selected_texts = [synthetic_texts[i] for i in selected_indices]
    selected_labels = [synthetic_labels[i] for i in selected_indices]
    selected_sims = [float(max_similarities[i]) for i in selected_indices]
    selected_targets = [error_sources[most_similar_error_idx[i]] for i in selected_indices]

    # Generate new sentence IDs
    max_id = train_df["Sentence_id"].max()
    if isinstance(max_id, str):
        max_id = int(max_id)
    new_ids = list(range(max_id + 1, max_id + 1 + len(selected_indices)))

    # Create synthetic dataframe
    synthetic_for_merge = pl.DataFrame({
        "Sentence_id": new_ids,
        "Text": selected_texts,
        "class_label": selected_labels,
        "source": ["synthetic_targeted"] * len(selected_indices),
        "target_error_type": selected_targets,
        "similarity_to_error": selected_sims,
    })

    # Create original with source column
    original_with_source = train_df.select(["Sentence_id", "Text", "class_label"]).with_columns([
        pl.lit("original").alias("source"),
        pl.lit(None).cast(pl.Utf8).alias("target_error_type"),
        pl.lit(None).cast(pl.Float64).alias("similarity_to_error"),
    ])

    # Combine and shuffle
    combined_df = pl.concat([original_with_source, synthetic_for_merge])
    combined_df = combined_df.sample(fraction=1.0, shuffle=True, seed=42)

    print(f"\n   Original training: {len(train_df)}")
    print(f"   Targeted synthetic: {len(selected_indices)}")
    print(f"   Combined: {len(combined_df)}")

    # Class distribution
    yes_count = combined_df.filter(pl.col("class_label") == "Yes").height
    no_count = combined_df.filter(pl.col("class_label") == "No").height
    print(f"\n   Class distribution:")
    print(f"      Yes: {yes_count} ({100*yes_count/len(combined_df):.1f}%)")
    print(f"      No: {no_count} ({100*no_count/len(combined_df):.1f}%)")

    # ==========================================================================
    # Step 7: Save outputs
    # ==========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full augmented dataset
    output_path = OUTPUT_DIR / "CT24_train_targeted_augmented.parquet"
    combined_df.write_parquet(output_path)
    print(f"\n💾 Saved to: {output_path}")

    # Save TSV version (compatible format)
    tsv_df = combined_df.select(["Sentence_id", "Text", "class_label"])
    tsv_path = OUTPUT_DIR / "CT24_train_targeted_augmented.tsv"
    tsv_df.write_csv(tsv_path, separator="\t")
    print(f"💾 TSV copy: {tsv_path}")

    # Save metadata
    meta = {
        "original_samples": len(train_df),
        "synthetic_samples_added": len(selected_indices),
        "total_samples": len(combined_df),
        "similarity_threshold": args.similarity_threshold,
        "top_k": args.top_k,
        "max_augment_ratio": args.max_augment_ratio,
        "error_sources": {
            "dev_FP": sum(1 for s in error_sources if s == "dev_FP"),
            "dev_FN": sum(1 for s in error_sources if s == "dev_FN"),
            "dev-test_FP": sum(1 for s in error_sources if s == "dev-test_FP"),
            "dev-test_FN": sum(1 for s in error_sources if s == "dev-test_FN"),
        },
        "targeted_counts": dict(type_counts),
        "similarity_stats": {
            "min": float(selected_similarities.min()),
            "max": float(selected_similarities.max()),
            "mean": float(selected_similarities.mean()),
            "std": float(selected_similarities.std()),
        },
        "synthetic_source": str(synthetic_path),
    }

    meta_path = OUTPUT_DIR / "targeted_augmentation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Metadata: {meta_path}")

    # ==========================================================================
    # Step 8: Show examples
    # ==========================================================================
    print(f"\n{'='*70}")
    print("EXAMPLE TARGETED SYNTHETIC SAMPLES")
    print("=" * 70)

    # Show top 5 most similar synthetic samples
    top_5_idx = np.argsort(-max_similarities[selected_indices])[:5]
    for rank, idx in enumerate(top_5_idx, 1):
        orig_idx = selected_indices[idx]
        error_idx = most_similar_error_idx[orig_idx]

        print(f"\n{rank}. Synthetic (similarity={max_similarities[orig_idx]:.3f}, label={synthetic_labels[orig_idx]}):")
        print(f"   \"{synthetic_texts[orig_idx][:100]}...\"")
        print(f"   → Targets {error_sources[error_idx]}:")
        print(f"   \"{all_error_texts[error_idx][:100]}...\"")

    print(f"\n✅ Done! Use {output_path} for training.")
    print(f"\nNext step:")
    print(f"   python experiments/scripts/finetune_deberta_augmented.py \\")
    print(f"       --train-path {output_path}")


if __name__ == "__main__":
    main()
