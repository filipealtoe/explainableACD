#!/usr/bin/env python3
"""
Analyze DeBERTa errors on dev and dev-test to inform synthetic data generation.

Outputs:
- Error counts (FP vs FN)
- Text patterns in errors
- Suggested synthetic data distribution

Usage:
    python experiments/scripts/analyze_errors_for_augmentation.py
"""

from pathlib import Path
import numpy as np
import polars as pl
import torch
from collections import Counter
import re
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

# Paths
MODEL_PATH = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "error_analysis"

THRESHOLD = 0.50


def get_predictions(model, tokenizer, texts, device):
    """Run inference and return probabilities."""
    probs = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Inference"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            p = torch.softmax(model(**inputs).logits, dim=-1)[0, 1].item()
            probs.append(p)
    return np.array(probs)


def analyze_text_patterns(texts):
    """Analyze linguistic patterns in a set of texts."""
    patterns = {
        "has_question": 0,
        "has_numbers": 0,
        "has_percentage": 0,
        "has_dollar": 0,
        "has_quote": 0,
        "has_said_claimed": 0,
        "has_i_think_believe": 0,
        "has_many_people": 0,
        "is_short": 0,  # < 8 words
        "has_superlative": 0,  # best, worst, most, etc.
        "has_vague_quantifier": 0,  # many, some, several
    }

    for text in texts:
        text_lower = text.lower()
        if "?" in text:
            patterns["has_question"] += 1
        if re.search(r'\d', text):
            patterns["has_numbers"] += 1
        if "%" in text or "percent" in text_lower:
            patterns["has_percentage"] += 1
        if "$" in text or "dollar" in text_lower:
            patterns["has_dollar"] += 1
        if '"' in text or "'" in text:
            patterns["has_quote"] += 1
        if any(w in text_lower for w in ["said", "claimed", "stated", "according to"]):
            patterns["has_said_claimed"] += 1
        if any(w in text_lower for w in ["i think", "i believe", "i feel", "in my opinion"]):
            patterns["has_i_think_believe"] += 1
        if any(w in text_lower for w in ["many people", "a lot of people", "everyone knows"]):
            patterns["has_many_people"] += 1
        if len(text.split()) < 8:
            patterns["is_short"] += 1
        if any(w in text_lower for w in ["best", "worst", "most", "least", "greatest", "largest"]):
            patterns["has_superlative"] += 1
        if any(w in text_lower for w in ["many", "some", "several", "few", "various"]):
            patterns["has_vague_quantifier"] += 1

    # Convert to percentages
    n = len(texts)
    return {k: (v, f"{100*v/n:.1f}%") for k, v in patterns.items()}


def categorize_errors(texts, probs):
    """Try to categorize errors into types."""
    categories = {
        "vague_underspecified": [],
        "opinion_as_fact": [],
        "numbers_but_unverifiable": [],
        "political_rhetoric": [],
        "reported_speech": [],
        "short_fragment": [],
        "other": [],
    }

    for text, prob in zip(texts, probs):
        text_lower = text.lower()

        # Categorization heuristics
        if len(text.split()) < 8:
            categories["short_fragment"].append((text, prob))
        elif any(w in text_lower for w in ["said", "claimed", "stated", "according to", "told"]):
            categories["reported_speech"].append((text, prob))
        elif any(w in text_lower for w in ["i think", "i believe", "in my opinion", "i feel"]):
            categories["opinion_as_fact"].append((text, prob))
        elif any(w in text_lower for w in ["many people", "everyone", "nobody", "a lot of"]):
            categories["vague_underspecified"].append((text, prob))
        elif any(w in text_lower for w in ["best", "worst", "tremendous", "disaster", "incredible", "amazing"]):
            categories["political_rhetoric"].append((text, prob))
        elif re.search(r'\d', text) and any(w in text_lower for w in ["many", "some", "about", "around"]):
            categories["numbers_but_unverifiable"].append((text, prob))
        else:
            categories["other"].append((text, prob))

    return categories


def main():
    print("=" * 70)
    print("ERROR ANALYSIS FOR SYNTHETIC DATA GENERATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_fp_texts, all_fn_texts = [], []
    all_fp_probs, all_fn_probs = [], []

    # Analyze each split
    for split in ["dev", "dev-test"]:
        print(f"\n{'='*70}")
        print(f"ANALYZING {split.upper()}")
        print("=" * 70)

        # Load data
        processed_path = PROCESSED_DIR / f"CT24_{split}_features.parquet"
        raw_path = RAW_DIR / f"CT24_checkworthy_english_{split}.tsv"

        if processed_path.exists():
            df = pl.read_parquet(processed_path)
        elif raw_path.exists():
            df = pl.read_csv(raw_path, separator="\t")
        else:
            print(f"Skipping {split} - no data found")
            continue

        texts = df["Text"].to_list()
        y_true = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])

        # Get predictions
        probs = get_predictions(model, tokenizer, texts, device)
        y_pred = (probs >= THRESHOLD).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        f1 = f1_score(y_true, y_pred)

        print(f"\nResults @ threshold {THRESHOLD}:")
        print(f"  F1: {f1:.4f}")
        print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"  FP rate: {100*fp/(fp+tn):.1f}%")
        print(f"  FN rate: {100*fn/(fn+tp):.1f}%")

        # Extract error samples
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        fp_texts = [texts[i] for i in range(len(texts)) if fp_mask[i]]
        fn_texts = [texts[i] for i in range(len(texts)) if fn_mask[i]]
        fp_probs_list = probs[fp_mask]
        fn_probs_list = probs[fn_mask]

        all_fp_texts.extend(fp_texts)
        all_fn_texts.extend(fn_texts)
        all_fp_probs.extend(fp_probs_list)
        all_fn_probs.extend(fn_probs_list)

        # Pattern analysis
        print(f"\n--- FALSE POSITIVES ({len(fp_texts)}) - Patterns ---")
        if fp_texts:
            fp_patterns = analyze_text_patterns(fp_texts)
            for k, (count, pct) in fp_patterns.items():
                if count > 0:
                    print(f"  {k}: {count} ({pct})")

        print(f"\n--- FALSE NEGATIVES ({len(fn_texts)}) - Patterns ---")
        if fn_texts:
            fn_patterns = analyze_text_patterns(fn_texts)
            for k, (count, pct) in fn_patterns.items():
                if count > 0:
                    print(f"  {k}: {count} ({pct})")

    # Combined analysis
    print(f"\n{'='*70}")
    print("COMBINED ERROR ANALYSIS (dev + dev-test)")
    print("=" * 70)

    total_fp = len(all_fp_texts)
    total_fn = len(all_fn_texts)
    total_errors = total_fp + total_fn

    print(f"\nTotal errors: {total_errors}")
    print(f"  False Positives: {total_fp} ({100*total_fp/total_errors:.1f}%)")
    print(f"  False Negatives: {total_fn} ({100*total_fn/total_errors:.1f}%)")

    # Print ALL False Positives
    print(f"\n{'='*70}")
    print(f"ALL FALSE POSITIVES ({total_fp}) - Claims we incorrectly flagged as checkworthy")
    print("=" * 70)
    for i, (text, prob) in enumerate(sorted(zip(all_fp_texts, all_fp_probs), key=lambda x: -x[1]), 1):
        print(f"\n[FP-{i}] prob={prob:.3f}")
        print(f"  {text}")

    # Print ALL False Negatives
    print(f"\n{'='*70}")
    print(f"ALL FALSE NEGATIVES ({total_fn}) - Checkworthy claims we MISSED")
    print("=" * 70)
    for i, (text, prob) in enumerate(sorted(zip(all_fn_texts, all_fn_probs), key=lambda x: -x[1]), 1):
        print(f"\n[FN-{i}] prob={prob:.3f}")
        print(f"  {text}")

    # Categorize FP errors
    print(f"\n{'='*70}")
    print("FP CATEGORIES (for synthetic data generation)")
    print("=" * 70)
    fp_categories = categorize_errors(all_fp_texts, all_fp_probs)
    for cat, items in fp_categories.items():
        if items:
            print(f"\n  {cat}: {len(items)} samples")

    # Categorize FN errors
    print(f"\n--- FN CATEGORIES ---")
    fn_categories = categorize_errors(all_fn_texts, all_fn_probs)
    for cat, items in fn_categories.items():
        if items:
            print(f"\n  {cat}: {len(items)} samples")

    # Recommended distribution
    print(f"\n{'='*70}")
    print("RECOMMENDED SYNTHETIC DATA DISTRIBUTION")
    print("=" * 70)

    total_synthetic = 5600  # 20% of ~28k total

    # Base distribution on error ratio
    fp_ratio = total_fp / total_errors if total_errors > 0 else 0.5
    fn_ratio = total_fn / total_errors if total_errors > 0 else 0.5

    synthetic_no = int(total_synthetic * fp_ratio)  # To fix FPs, add hard negatives
    synthetic_yes = int(total_synthetic * fn_ratio)  # To fix FNs, add hard positives

    print(f"\nBased on error distribution ({100*fp_ratio:.0f}% FP, {100*fn_ratio:.0f}% FN):")
    print(f"  Synthetic label=No:  {synthetic_no} samples (to reduce FPs)")
    print(f"  Synthetic label=Yes: {synthetic_yes} samples (to reduce FNs)")

    # Category-based distribution for FPs
    print(f"\n  Suggested FP category breakdown ({synthetic_no} No samples):")
    fp_cat_counts = {k: len(v) for k, v in fp_categories.items() if v}
    fp_total = sum(fp_cat_counts.values())
    for cat, count in sorted(fp_cat_counts.items(), key=lambda x: -x[1]):
        pct = count / fp_total if fp_total > 0 else 0
        suggested = int(synthetic_no * pct)
        print(f"    {cat}: {suggested} ({100*pct:.0f}%)")

    # Save error samples for reference
    fp_df = pl.DataFrame({"text": all_fp_texts, "prob": all_fp_probs, "error_type": ["FP"] * len(all_fp_texts)})
    fn_df = pl.DataFrame({"text": all_fn_texts, "prob": all_fn_probs, "error_type": ["FN"] * len(all_fn_texts)})
    errors_df = pl.concat([fp_df, fn_df])
    errors_df.write_csv(OUTPUT_DIR / "all_errors_for_augmentation.csv")
    print(f"\nErrors saved to: {OUTPUT_DIR / 'all_errors_for_augmentation.csv'}")


if __name__ == "__main__":
    main()
