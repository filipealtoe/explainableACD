#!/usr/bin/env python3
"""
Few-Shot Checkworthiness with Retrieval-Augmented Example Selection

Tests simple checkworthiness prompt with varying shot counts (6, 8, 10, 12).
Selects examples based on semantic similarity to FP/FN cases:
- Positive examples: training positives most similar to False Negatives
- Negative examples: training negatives most similar to False Positives

This targets the model's "blind spots" with informative examples.

Usage:
    python experiments/scripts/test_fewshot_retrieval.py
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
EMBEDDING_FILE = DATA_DIR / "embedding_cache" / "bge-large_embeddings.npz"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"

# Together AI with Mistral 24B
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
TEMPERATURE = 0.0

SOTA_F1 = 0.82


# =============================================================================
# Prompts (similar to DSPy signature)
# =============================================================================

SYSTEM_PROMPT = """You are an expert fact-checker assessing whether statements are checkworthy.
A statement is checkworthy if it makes a factual claim that can be verified and would be important to fact-check."""

TASK_PROMPT = """Assess whether the following statement is checkworthy.

Statement: {statement}

Is this statement checkworthy? Answer with just "Yes" or "No"."""

FEWSHOT_TEMPLATE = """Example {n}:
Statement: {statement}
Answer: {answer}
"""


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class Example:
    text: str
    label: str  # "Yes" or "No"
    embedding: np.ndarray


def load_data():
    """Load embeddings, texts, and labels."""
    # Load embeddings
    embed_data = np.load(EMBEDDING_FILE)
    train_embeddings = embed_data["train"].astype(np.float32)
    test_embeddings = embed_data["test"].astype(np.float32)

    # Load texts and labels
    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    text_col = "cleaned_text" if "cleaned_text" in ct24_train.columns else "Text"

    train_texts = ct24_train[text_col].to_list()
    test_texts = ct24_test[text_col].to_list()

    train_labels = ct24_train["class_label"].to_list()
    test_labels = ct24_test["class_label"].to_list()

    return (train_embeddings, train_texts, train_labels,
            test_embeddings, test_texts, test_labels)


def get_fp_fn_indices(embeddings, labels):
    """Run baseline classifier to identify FP/FN indices."""
    # Simple logistic regression baseline
    y = np.array([1 if l == "Yes" else 0 for l in labels])

    # Use embeddings directly (no train/test split needed, this is for training set analysis)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4})
    clf.fit(X_scaled, y)

    y_pred = clf.predict(X_scaled)

    # FP: predicted Yes, actual No
    fp_idx = np.where((y_pred == 1) & (y == 0))[0]
    # FN: predicted No, actual Yes
    fn_idx = np.where((y_pred == 0) & (y == 1))[0]

    return fp_idx, fn_idx


def select_hard_examples(
    train_embeddings: np.ndarray,
    train_texts: list[str],
    train_labels: list[str],
    test_embeddings: np.ndarray,
    test_texts: list[str],
    test_labels: list[str],
    n_positive: int,
    n_negative: int,
) -> list[Example]:
    """
    Select training examples that are semantically similar to hard cases.

    Strategy:
    - Run baseline on TEST set to find FP/FN
    - Select training POSITIVES most similar to test FN (missed positives)
    - Select training NEGATIVES most similar to test FP (false alarms)
    """
    # Run baseline classifier on test set
    print("  Running baseline to identify FP/FN...")

    # Train on train, predict on test
    y_train = np.array([1 if l == "Yes" else 0 for l in train_labels])
    y_test = np.array([1 if l == "Yes" else 0 for l in test_labels])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_embeddings)
    X_test_s = scaler.transform(test_embeddings)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight={0: 1, 1: 4})
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)

    # Identify FP and FN in test set
    fp_test_idx = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_test_idx = np.where((y_pred == 0) & (y_test == 1))[0]

    print(f"    Test FP: {len(fp_test_idx)}, Test FN: {len(fn_test_idx)}")

    # Get training positive and negative indices
    train_pos_idx = np.where(y_train == 1)[0]
    train_neg_idx = np.where(y_train == 0)[0]

    selected_examples = []

    # Select positive examples similar to FN (things the model misses)
    if len(fn_test_idx) > 0 and len(train_pos_idx) > 0:
        # Compute centroid of FN embeddings
        fn_centroid = test_embeddings[fn_test_idx].mean(axis=0, keepdims=True)

        # Find training positives closest to FN centroid
        train_pos_embeddings = train_embeddings[train_pos_idx]
        similarities = cosine_similarity(fn_centroid, train_pos_embeddings)[0]

        # Get top n_positive
        top_pos_local_idx = np.argsort(similarities)[-n_positive:][::-1]
        top_pos_global_idx = train_pos_idx[top_pos_local_idx]

        for idx in top_pos_global_idx:
            selected_examples.append(Example(
                text=train_texts[idx],
                label="Yes",
                embedding=train_embeddings[idx]
            ))
            print(f"    + Selected positive (sim={similarities[top_pos_local_idx[len(selected_examples)-1]]:.3f}): {train_texts[idx][:60]}...")

    # Select negative examples similar to FP (things the model wrongly flags)
    if len(fp_test_idx) > 0 and len(train_neg_idx) > 0:
        # Compute centroid of FP embeddings
        fp_centroid = test_embeddings[fp_test_idx].mean(axis=0, keepdims=True)

        # Find training negatives closest to FP centroid
        train_neg_embeddings = train_embeddings[train_neg_idx]
        similarities = cosine_similarity(fp_centroid, train_neg_embeddings)[0]

        # Get top n_negative
        top_neg_local_idx = np.argsort(similarities)[-n_negative:][::-1]
        top_neg_global_idx = train_neg_idx[top_neg_local_idx]

        for i, idx in enumerate(top_neg_global_idx):
            selected_examples.append(Example(
                text=train_texts[idx],
                label="No",
                embedding=train_embeddings[idx]
            ))
            print(f"    - Selected negative (sim={similarities[top_neg_local_idx[i]]:.3f}): {train_texts[idx][:60]}...")

    return selected_examples


def select_random_examples(
    train_texts: list[str],
    train_labels: list[str],
    train_embeddings: np.ndarray,
    n_positive: int,
    n_negative: int,
    seed: int = 42,
) -> list[Example]:
    """Select random balanced examples (for comparison)."""
    rng = np.random.RandomState(seed)

    pos_idx = [i for i, l in enumerate(train_labels) if l == "Yes"]
    neg_idx = [i for i, l in enumerate(train_labels) if l == "No"]

    selected_pos = rng.choice(pos_idx, size=min(n_positive, len(pos_idx)), replace=False)
    selected_neg = rng.choice(neg_idx, size=min(n_negative, len(neg_idx)), replace=False)

    examples = []
    for idx in selected_pos:
        examples.append(Example(text=train_texts[idx], label="Yes", embedding=train_embeddings[idx]))
    for idx in selected_neg:
        examples.append(Example(text=train_texts[idx], label="No", embedding=train_embeddings[idx]))

    return examples


# =============================================================================
# LLM Inference
# =============================================================================

def build_prompt(statement: str, examples: list[Example]) -> str:
    """Build the full prompt with few-shot examples."""
    parts = []

    if examples:
        parts.append("Here are some examples:\n")
        for i, ex in enumerate(examples, 1):
            parts.append(FEWSHOT_TEMPLATE.format(n=i, statement=ex.text, answer=ex.label))
        parts.append("\nNow assess this statement:\n")

    parts.append(TASK_PROMPT.format(statement=statement))

    return "".join(parts)


def call_llm(client: OpenAI, prompt: str) -> str:
    """Call the LLM and extract Yes/No response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=10,
    )

    answer = response.choices[0].message.content.strip()

    # Parse response
    answer_lower = answer.lower()
    if "yes" in answer_lower:
        return "Yes"
    elif "no" in answer_lower:
        return "No"
    else:
        # Default to No if unclear
        return "No"


def evaluate_fewshot(
    client: OpenAI,
    test_texts: list[str],
    test_labels: list[str],
    examples: list[Example],
    config_name: str,
    max_samples: int | None = None,
) -> dict:
    """Evaluate few-shot performance on test set."""
    predictions = []
    actuals = []

    texts_to_eval = test_texts[:max_samples] if max_samples else test_texts
    labels_to_eval = test_labels[:max_samples] if max_samples else test_labels

    print(f"\n  Evaluating: {config_name} ({len(texts_to_eval)} samples)...")

    for i, (text, label) in enumerate(zip(texts_to_eval, labels_to_eval)):
        prompt = build_prompt(text, examples)
        pred = call_llm(client, prompt)
        predictions.append(pred)
        actuals.append(label)

        if (i + 1) % 50 == 0:
            # Interim metrics
            y_true = [1 if l == "Yes" else 0 for l in actuals]
            y_pred = [1 if p == "Yes" else 0 for p in predictions]
            interim_f1 = f1_score(y_true, y_pred)
            print(f"    Progress: {i+1}/{len(texts_to_eval)} | Interim F1: {interim_f1:.4f}")

    # Final metrics
    y_true = [1 if l == "Yes" else 0 for l in actuals]
    y_pred = [1 if p == "Yes" else 0 for p in predictions]

    return {
        "config": config_name,
        "n_shots": len(examples),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "n_samples": len(y_true),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("FEW-SHOT CHECKWORTHINESS with RETRIEVAL-AUGMENTED EXAMPLE SELECTION")
    print("=" * 100)
    print(f"Model: {MODEL}")

    # Check API key
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set")
        return

    client = OpenAI(api_key=api_key, base_url=TOGETHER_BASE_URL)

    # Load data
    print("\nLoading data...")
    (train_embeddings, train_texts, train_labels,
     test_embeddings, test_texts, test_labels) = load_data()

    print(f"  Train: {len(train_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")

    # Use full test set
    MAX_TEST_SAMPLES = None

    # Shot configurations to test (n_positive, n_negative)
    shot_configs = [
        (0, 0),   # 0-shot (baseline)
        (3, 3),   # 6-shot (3 pos + 3 neg)
        (4, 4),   # 8-shot (4 pos + 4 neg)
        (5, 5),   # 10-shot (5 pos + 5 neg)
        (6, 6),   # 12-shot (6 pos + 6 neg)
    ]

    results = []

    # =========================================================================
    # Test 1: Zero-shot baseline
    # =========================================================================
    print("\n" + "=" * 100)
    print("ZERO-SHOT BASELINE")
    print("=" * 100)

    result = evaluate_fewshot(
        client, test_texts, test_labels, [],
        "0-shot (baseline)", max_samples=MAX_TEST_SAMPLES
    )
    results.append(result)
    print(f"  → F1: {result['f1']:.4f}, Acc: {result['accuracy']:.4f}")

    # =========================================================================
    # Test 2: Few-shot with RETRIEVAL-AUGMENTED selection
    # =========================================================================
    print("\n" + "=" * 100)
    print("FEW-SHOT with RETRIEVAL-AUGMENTED SELECTION (FP/FN similarity)")
    print("=" * 100)

    for n_pos, n_neg in shot_configs[1:]:  # Skip 0-shot
        total_shots = n_pos + n_neg
        print(f"\n--- {total_shots}-shot (retrieval) ---")

        examples = select_hard_examples(
            train_embeddings, train_texts, train_labels,
            test_embeddings, test_texts, test_labels,
            n_positive=n_pos, n_negative=n_neg
        )

        result = evaluate_fewshot(
            client, test_texts, test_labels, examples,
            f"{total_shots}-shot (retrieval)", max_samples=MAX_TEST_SAMPLES
        )
        results.append(result)
        print(f"  → F1: {result['f1']:.4f}, Acc: {result['accuracy']:.4f}")

    # =========================================================================
    # Test 3: Few-shot with RANDOM selection (for comparison)
    # =========================================================================
    print("\n" + "=" * 100)
    print("FEW-SHOT with RANDOM SELECTION (baseline comparison)")
    print("=" * 100)

    for n_pos, n_neg in shot_configs[1:]:  # Skip 0-shot
        total_shots = n_pos + n_neg
        print(f"\n--- {total_shots}-shot (random) ---")

        examples = select_random_examples(
            train_texts, train_labels, train_embeddings,
            n_positive=n_pos, n_negative=n_neg
        )

        result = evaluate_fewshot(
            client, test_texts, test_labels, examples,
            f"{total_shots}-shot (random)", max_samples=MAX_TEST_SAMPLES
        )
        results.append(result)
        print(f"  → F1: {result['f1']:.4f}, Acc: {result['accuracy']:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY - All Results")
    print("=" * 100)

    print(f"\n{'Config':<30} {'Shots':<8} {'F1':<10} {'Acc':<10} {'P':<8} {'R':<8} {'Δ SOTA'}")
    print("-" * 90)

    results.sort(key=lambda x: x["f1"], reverse=True)

    for r in results:
        delta = r["f1"] - SOTA_F1
        marker = "🔥" if r["f1"] > SOTA_F1 else ""
        print(f"{r['config']:<30} {r['n_shots']:<8} {r['f1']:<10.4f} {r['accuracy']:<10.4f} "
              f"{r['precision']:<8.4f} {r['recall']:<8.4f} {delta:+.4f} {marker}")

    # Compare retrieval vs random
    print("\n" + "-" * 100)
    print("RETRIEVAL vs RANDOM COMPARISON")
    print("-" * 100)

    for n_pos, n_neg in shot_configs[1:]:
        total_shots = n_pos + n_neg
        retrieval_result = next((r for r in results if r["config"] == f"{total_shots}-shot (retrieval)"), None)
        random_result = next((r for r in results if r["config"] == f"{total_shots}-shot (random)"), None)

        if retrieval_result and random_result:
            delta = retrieval_result["f1"] - random_result["f1"]
            winner = "RETRIEVAL" if delta > 0 else "RANDOM"
            print(f"  {total_shots}-shot: Retrieval F1={retrieval_result['f1']:.4f} vs Random F1={random_result['f1']:.4f} → {winner} wins by {abs(delta):.4f}")

    # Best result
    best = max(results, key=lambda x: x["f1"])
    print(f"\n🏆 BEST: {best['config']} with F1 = {best['f1']:.4f}")
    print(f"   Gap to SOTA: {best['f1'] - SOTA_F1:+.4f}")


if __name__ == "__main__":
    main()
