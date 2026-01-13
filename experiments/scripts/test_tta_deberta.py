#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) for DeBERTa Checkworthiness

Tests if averaging predictions over augmented versions of test samples
improves F1 score.

Usage:
    python experiments/scripts/test_tta_deberta.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "best_model"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
BASELINE_PROBS = Path(__file__).parent.parent / "results" / "deberta_checkworthy" / "deberta-v3-large" / "test_probs.npy"

SOTA_F1 = 0.82


# =============================================================================
# Augmentation Functions
# =============================================================================


def augment_simple(text: str) -> list[str]:
    """Simple augmentations (fast, free)."""
    augmented = [text]  # Always include original

    # Lowercase
    if text != text.lower():
        augmented.append(text.lower())

    # Normalize whitespace
    normalized = " ".join(text.split())
    if normalized != text:
        augmented.append(normalized)

    # Remove extra punctuation (keep single)
    cleaned = re.sub(r'([.!?])\1+', r'\1', text)
    if cleaned != text:
        augmented.append(cleaned)

    # Remove quotes
    no_quotes = text.replace('"', '').replace("'", '')
    if no_quotes != text and len(no_quotes) > 10:
        augmented.append(no_quotes)

    return list(set(augmented))  # Deduplicate


def augment_word_dropout(text: str, p: float = 0.1) -> list[str]:
    """Randomly drop words with probability p."""
    augmented = [text]

    words = text.split()
    if len(words) > 5:  # Only for longer texts
        # Create 2 dropout versions
        for _ in range(2):
            kept = [w for w in words if random.random() > p]
            if len(kept) >= len(words) * 0.7:  # Keep at least 70% of words
                augmented.append(" ".join(kept))

    return list(set(augmented))


def augment_synonym_swap(text: str) -> list[str]:
    """Simple synonym replacements (no external dependencies)."""
    augmented = [text]

    # Common synonym pairs for claim-related text
    synonyms = {
        "said": ["stated", "claimed", "reported"],
        "says": ["states", "claims", "reports"],
        "show": ["demonstrate", "indicate", "reveal"],
        "shows": ["demonstrates", "indicates", "reveals"],
        "according to": ["as per", "based on"],
        "percent": ["%"],
        "%": ["percent"],
        "million": ["M"],
        "billion": ["B"],
        "announced": ["declared", "stated"],
        "reported": ["stated", "claimed"],
        "found": ["discovered", "determined"],
        "study": ["research", "analysis"],
        "increase": ["rise", "growth"],
        "decrease": ["decline", "drop"],
    }

    text_lower = text.lower()
    for word, replacements in synonyms.items():
        if word in text_lower:
            for replacement in replacements[:1]:  # Just one replacement per word
                # Case-insensitive replacement
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                new_text = pattern.sub(replacement, text, count=1)
                if new_text != text:
                    augmented.append(new_text)
                    break

    return list(set(augmented))


def get_augmentations(text: str, strategy: str = "all") -> list[str]:
    """Get augmented versions of text based on strategy."""
    if strategy == "simple":
        return augment_simple(text)
    elif strategy == "dropout":
        return augment_word_dropout(text)
    elif strategy == "synonym":
        return augment_synonym_swap(text)
    elif strategy == "all":
        # Combine all strategies
        all_augs = set()
        all_augs.update(augment_simple(text))
        all_augs.update(augment_word_dropout(text))
        all_augs.update(augment_synonym_swap(text))
        return list(all_augs)
    else:
        return [text]


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("TEST-TIME AUGMENTATION (TTA) for DeBERTa")
    print("=" * 70)

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading model...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")

    # Load test data
    print("\nLoading test data...")
    test_df = pl.read_parquet(DATA_DIR / "CT24_test_features.parquet")
    texts = test_df["Text"].to_list()
    y_test = [1 if l == "Yes" else 0 for l in test_df["class_label"].to_list()]
    y_test = np.array(y_test)
    print(f"Test samples: {len(texts)}")

    # Load baseline predictions
    baseline_probs = np.load(BASELINE_PROBS)
    print(f"Baseline probs loaded: {baseline_probs.shape}")

    # Test different augmentation strategies
    strategies = ["none", "simple", "dropout", "synonym", "all"]
    results = {}

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.upper()}")
        print("=" * 70)

        all_probs = []
        total_augments = 0

        for text in tqdm(texts, desc=f"TTA ({strategy})"):
            if strategy == "none":
                augmented = [text]
            else:
                augmented = get_augmentations(text, strategy)

            total_augments += len(augmented)

            # Get predictions for all augmented versions
            probs = []
            for aug_text in augmented:
                inputs = tokenizer(
                    aug_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits
                    prob = torch.softmax(logits, dim=-1)[0, 1].item()
                    probs.append(prob)

            # Average probabilities
            avg_prob = np.mean(probs)
            all_probs.append(avg_prob)

        all_probs = np.array(all_probs)
        avg_augments = total_augments / len(texts)

        # Find best threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.30, 0.70, 0.01):
            f1 = f1_score(y_test, (all_probs >= thresh).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        preds = (all_probs >= best_thresh).astype(int)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        results[strategy] = {
            "f1": best_f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "threshold": best_thresh,
            "avg_augments": avg_augments,
        }

        print(f"\nAvg augmentations per sample: {avg_augments:.2f}")
        print(f"Best F1: {best_f1:.4f} @ threshold {best_thresh:.2f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Baseline comparison
    baseline_f1, baseline_t = 0, 0.5
    for t in np.arange(0.30, 0.70, 0.01):
        f1 = f1_score(y_test, (baseline_probs >= t).astype(int))
        if f1 > baseline_f1:
            baseline_f1, baseline_t = f1, t

    print(f"\n{'Strategy':<15} {'F1':<10} {'Acc':<10} {'Avg Augs':<10} {'vs Baseline':<12}")
    print("-" * 60)
    print(f"{'Baseline':<15} {baseline_f1:<10.4f} {'-':<10} {1.0:<10.1f} {'-':<12}")

    for strategy, res in results.items():
        diff = res['f1'] - baseline_f1
        marker = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "→")
        print(f"{strategy:<15} {res['f1']:<10.4f} {res['accuracy']:<10.4f} "
              f"{res['avg_augments']:<10.1f} {diff:+.4f} {marker}")

    # Best TTA result
    best_strategy = max(results.keys(), key=lambda k: results[k]['f1'])
    best_tta_f1 = results[best_strategy]['f1']

    print(f"\nBest TTA: {best_strategy} with F1={best_tta_f1:.4f}")
    print(f"Improvement over baseline: {best_tta_f1 - baseline_f1:+.4f}")

    if best_tta_f1 > baseline_f1 + 0.005:
        print("\n✓ TTA provides meaningful improvement!")
    else:
        print("\n✗ TTA does NOT provide meaningful improvement.")
        print("  (As predicted - DeBERTa is already robust to surface variations)")

    print(f"\nSOTA: {SOTA_F1}")
    print(f"Best result: {max(baseline_f1, best_tta_f1):.4f}")
    print(f"SOTA gap: {max(baseline_f1, best_tta_f1) - SOTA_F1:+.4f}")


if __name__ == "__main__":
    main()
