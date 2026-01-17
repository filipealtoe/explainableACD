#!/usr/bin/env python3
"""
Validate that synthetic samples are actually "hard" for our DeBERTa baseline.

A sample is "hard" if DeBERTa predicts it incorrectly:
- Hard Negative (label=No): DeBERTa predicts Yes (FP)
- Hard Positive (label=Yes): DeBERTa predicts No (FN)

Samples where DeBERTa is already correct are filtered out as "too easy".

Usage:
    python experiments/scripts/validate_synthetic_hardness.py path/to/synthetic.parquet
    python experiments/scripts/validate_synthetic_hardness.py path/to/synthetic.parquet --threshold 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

# Paths
MODEL_PATH = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"


def run_inference(model, tokenizer, texts: list[str], device, batch_size: int = 16) -> np.ndarray:
    """Run DeBERTa inference and return probabilities."""
    probs = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs)

    return np.array(probs)


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic data hardness")
    parser.add_argument("input_file", type=Path, help="Path to synthetic data (parquet or csv)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC DATA HARDNESS VALIDATION")
    print("=" * 70)

    # Load synthetic data
    print(f"\n📄 Loading: {args.input_file}")
    if args.input_file.suffix == ".parquet":
        df = pl.read_parquet(args.input_file)
    else:
        df = pl.read_csv(args.input_file)

    print(f"   Total samples: {len(df)}")
    print(f"   Label distribution:")
    for row in df.group_by("class_label").len().sort("class_label").iter_rows(named=True):
        print(f"      {row['class_label']}: {row['len']}")

    # Load model
    print(f"\n🤖 Loading DeBERTa from: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Device: {device}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

    # Run inference
    texts = df["Text"].to_list()
    probs = run_inference(model, tokenizer, texts, device, args.batch_size)
    preds = (probs >= args.threshold).astype(int)

    # Add predictions to dataframe
    df = df.with_columns([
        pl.Series("deberta_prob", probs),
        pl.Series("deberta_pred", ["Yes" if p == 1 else "No" for p in preds]),
    ])

    # Determine hardness
    # Hard = DeBERTa prediction != ground truth label
    df = df.with_columns([
        (pl.col("class_label") != pl.col("deberta_pred")).alias("is_hard"),
    ])

    # Stats
    n_hard = df.filter(pl.col("is_hard")).height
    n_easy = df.filter(~pl.col("is_hard")).height

    print(f"\n📊 Hardness Analysis (threshold={args.threshold}):")
    print(f"   Hard samples (DeBERTa wrong): {n_hard} ({100*n_hard/len(df):.1f}%)")
    print(f"   Easy samples (DeBERTa right): {n_easy} ({100*n_easy/len(df):.1f}%)")

    # Breakdown by label
    print(f"\n   By label:")
    for label in ["No", "Yes"]:
        subset = df.filter(pl.col("class_label") == label)
        hard_count = subset.filter(pl.col("is_hard")).height
        easy_count = subset.filter(~pl.col("is_hard")).height
        print(f"      {label}: {hard_count} hard, {easy_count} easy ({100*hard_count/len(subset):.1f}% hard)")

    # Breakdown by category
    print(f"\n   By category:")
    category_stats = (
        df.group_by("category")
        .agg([
            pl.len().alias("total"),
            pl.col("is_hard").sum().alias("hard"),
        ])
        .with_columns([
            (100 * pl.col("hard") / pl.col("total")).alias("hard_pct"),
        ])
        .sort("hard_pct", descending=True)
    )
    for row in category_stats.iter_rows(named=True):
        print(f"      {row['category']}: {row['hard']}/{row['total']} ({row['hard_pct']:.0f}% hard)")

    # Keep all samples with flags (no filtering)
    print(f"\n📦 Keeping all {len(df)} samples with hardness flags")

    # Save validated data (all samples, with is_hard flag)
    output_stem = args.input_file.stem
    output_dir = args.input_file.parent

    validated_path = output_dir / f"{output_stem}_validated.parquet"
    df.write_parquet(validated_path)
    print(f"\n💾 Validated data: {validated_path}")

    # Also save CSV for inspection
    csv_path = validated_path.with_suffix(".csv")
    df.write_csv(csv_path)
    print(f"💾 CSV copy: {csv_path}")

    # Save validation report
    report = {
        "input_file": str(args.input_file),
        "threshold": args.threshold,
        "total_samples": len(df),
        "hard_samples": n_hard,
        "easy_samples": n_easy,
        "hard_ratio": n_hard / len(df),
        "by_label": {
            label: {
                "total": int(df.filter(pl.col("class_label") == label).height),
                "hard": int(df.filter((pl.col("class_label") == label) & pl.col("is_hard")).height),
            }
            for label in ["No", "Yes"]
        },
        "by_category": {
            row["category"]: {"total": row["total"], "hard": row["hard"], "hard_pct": row["hard_pct"]}
            for row in category_stats.iter_rows(named=True)
        },
    }

    import json
    report_path = output_dir / f"{output_stem}_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📝 Report: {report_path}")

    # Show examples of hard vs easy
    print(f"\n" + "=" * 70)
    print("EXAMPLE HARD SAMPLES (correctly labeled, DeBERTa wrong)")
    print("=" * 70)

    hard_examples = df.filter(pl.col("is_hard")).sample(min(5, n_hard), seed=42)
    for row in hard_examples.iter_rows(named=True):
        print(f"\n[{row['category']}] label={row['class_label']}, DeBERTa={row['deberta_pred']} (p={row['deberta_prob']:.3f})")
        print(f"   {row['Text']}")

    print(f"\n" + "=" * 70)
    print("EXAMPLE EASY SAMPLES (DeBERTa already correct - will discard)")
    print("=" * 70)

    easy_examples = df.filter(~pl.col("is_hard")).sample(min(5, n_easy), seed=42)
    for row in easy_examples.iter_rows(named=True):
        print(f"\n[{row['category']}] label={row['class_label']}, DeBERTa={row['deberta_pred']} (p={row['deberta_prob']:.3f})")
        print(f"   {row['Text']}")

    # Final summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"   Total samples:  {len(df)}")
    print(f"   Hard (is_hard=True):  {n_hard} ({100*n_hard/len(df):.1f}%) - DeBERTa predicts wrong")
    print(f"   Easy (is_hard=False): {n_easy} ({100*n_easy/len(df):.1f}%) - DeBERTa predicts right")

    if n_hard < len(df) * 0.3:
        print(f"\n⚠️  WARNING: Only {100*n_hard/len(df):.0f}% of samples are hard.")
        print("   Consider adjusting prompts to generate more challenging examples.")

    print(f"\n💡 Use is_hard column to filter during training if desired.")


if __name__ == "__main__":
    main()
