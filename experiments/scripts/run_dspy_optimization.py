#!/usr/bin/env python3
"""
DSPy Prompt Optimization for Checkworthiness

Uses DSPy optimizers (MIPROv2, BootstrapFewShot) to automatically optimize
prompts for the checkworthiness task.

This runs independently of the v4 prompts experiment, allowing parallel comparison.

Usage:
    python experiments/scripts/run_dspy_optimization.py --optimizer mipro
    python experiments/scripts/run_dspy_optimization.py --optimizer bootstrap
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

import polars as pl
import dspy
from dotenv import load_dotenv

# Load environment
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "experiments" / "results" / "dspy_optimization"

# Model config
TOGETHER_API_BASE = "https://api.together.xyz/v1"
MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"

# Optimization settings
MAX_BOOTSTRAPPED_DEMOS = 4  # Few-shot examples to find
MAX_LABELED_DEMOS = 8  # Max labeled examples for optimization
NUM_CANDIDATES = 10  # Number of prompt candidates to try
NUM_TRIALS = 15  # Number of optimization trials


# =============================================================================
# Data Loading
# =============================================================================

def load_ct24_data(split: str, max_samples: int | None = None) -> list[dspy.Example]:
    """Load CT24 data as DSPy examples."""
    data_file = DATA_DIR / f"CT24_{split}_features.parquet"
    df = pl.read_parquet(data_file)

    if max_samples:
        df = df.head(max_samples)

    examples = []
    for row in df.iter_rows(named=True):
        ex = dspy.Example(
            claim=row["cleaned_text"],
            label=row["class_label"],  # "Yes" or "No"
        ).with_inputs("claim")
        examples.append(ex)

    return examples


# =============================================================================
# DSPy Signatures (simplified for optimization)
# =============================================================================

class CheckworthinessSignature(dspy.Signature):
    """Determine if a claim is checkworthy (should be fact-checked).

    A claim is checkworthy if it is:
    1. A factual assertion (not opinion/prediction)
    2. Verifiable with public evidence
    3. Has potential for harm if false

    Answer Yes or No.
    """

    claim: str = dspy.InputField(desc="The claim to assess")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for your decision")
    answer: str = dspy.OutputField(desc="Yes or No")


class CheckworthinessModule(dspy.Module):
    """Single-module checkworthiness predictor for optimization."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(CheckworthinessSignature)

    def forward(self, claim: str):
        return self.predictor(claim=claim)


# =============================================================================
# Metrics
# =============================================================================

def checkworthiness_metric(example, prediction, trace=None) -> float:
    """
    F1-oriented metric for optimization (Bootstrap/MIPROv2 compatible).

    Since F1 is a batch metric but DSPy needs per-example scores, we use
    weighted scoring that prioritizes True Positives (the key to high F1):

    - True Positive (pred=Yes, label=Yes):  3.0  ← Most valuable for F1
    - True Negative (pred=No, label=No):    1.0  ← Easy to get, less valuable
    - False Negative (pred=No, label=Yes):  0.0  ← Hurts recall
    - False Positive (pred=Yes, label=No):  0.0  ← Hurts precision

    This weighting encourages the optimizer to find prompts that correctly
    identify checkworthy claims (Yes) rather than just predicting No.
    """
    pred_answer = prediction.answer.strip().lower()

    # Normalize to Yes/No
    if pred_answer in ["yes", "true", "1"]:
        pred_label = "Yes"
    elif pred_answer in ["no", "false", "0"]:
        pred_label = "No"
    else:
        if "yes" in pred_answer:
            pred_label = "Yes"
        elif "no" in pred_answer:
            pred_label = "No"
        else:
            pred_label = "No"

    true_label = example.label

    # Weighted scoring for F1 optimization
    if pred_label == "Yes" and true_label == "Yes":
        return 3.0  # True Positive - most valuable
    elif pred_label == "No" and true_label == "No":
        return 1.0  # True Negative - correct but easy
    else:
        return 0.0  # False Positive or False Negative


def checkworthiness_metric_gepa(
    gold,
    pred,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    """
    GEPA-compatible metric with feedback for reflection.

    Returns ScoreWithFeedback to help GEPA understand WHY predictions fail.
    """
    from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

    pred_answer = pred.answer.strip().lower()

    # Normalize to Yes/No
    if pred_answer in ["yes", "true", "1"]:
        pred_label = "Yes"
    elif pred_answer in ["no", "false", "0"]:
        pred_label = "No"
    else:
        if "yes" in pred_answer:
            pred_label = "Yes"
        elif "no" in pred_answer:
            pred_label = "No"
        else:
            pred_label = "No"

    true_label = gold.label
    claim_preview = gold.claim[:100] + "..." if len(gold.claim) > 100 else gold.claim
    reasoning = getattr(pred, "reasoning", "")[:200] if hasattr(pred, "reasoning") else ""

    # Determine score and feedback
    if pred_label == "Yes" and true_label == "Yes":
        score = 3.0
        feedback = f"CORRECT (True Positive): Correctly identified checkworthy claim."
    elif pred_label == "No" and true_label == "No":
        score = 1.0
        feedback = f"CORRECT (True Negative): Correctly rejected non-checkworthy claim."
    elif pred_label == "Yes" and true_label == "No":
        score = 0.0
        feedback = (
            f"WRONG (False Positive): Predicted Yes but claim is NOT checkworthy. "
            f"Claim: '{claim_preview}'. "
            f"Model reasoning: '{reasoning}'. "
            f"The model may be confusing opinions, predictions, or rhetorical statements with factual claims."
        )
    else:  # pred_label == "No" and true_label == "Yes"
        score = 0.0
        feedback = (
            f"WRONG (False Negative): Predicted No but claim IS checkworthy. "
            f"Claim: '{claim_preview}'. "
            f"Model reasoning: '{reasoning}'. "
            f"The model may be too conservative or missing verifiable factual assertions."
        )

    return ScoreWithFeedback(score=score, feedback=feedback)


# =============================================================================
# Optimization
# =============================================================================

def run_bootstrap_optimization(
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
    seed: int = 42,
    max_demos: int = 4,
    num_candidate_sets: int = 10,
) -> dspy.Module:
    """
    Run BootstrapFewShotWithRandomSearch - ACTUAL optimization.

    This generates multiple candidate demo sets and picks the best one
    based on dev set performance. This is what optimization should be.
    """
    import random

    print("\n" + "=" * 70)
    print("Running BootstrapFewShot WITH RANDOM SEARCH")
    print("=" * 70)

    random.seed(seed)
    print(f"  Training examples: {len(trainset)}")
    print(f"  Dev examples for evaluation: {len(devset)}")
    print(f"  Max demos per set: {max_demos}")
    print(f"  Candidate sets to try: {num_candidate_sets}")

    from dspy.teleprompt import BootstrapFewShotWithRandomSearch

    optimizer = BootstrapFewShotWithRandomSearch(
        metric=checkworthiness_metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_demos,
        num_candidate_programs=num_candidate_sets,  # Try this many different demo sets
        num_threads=4,
    )

    module = CheckworthinessModule()
    optimized = optimizer.compile(module, trainset=trainset, valset=devset)

    return optimized


def run_gepa_optimization(
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
    auto: str = "medium",
    api_key: str | None = None,
) -> dspy.Module:
    """
    Run GEPA (Gradient-free Efficient Prompt Adaptation).

    Uses a reflection LLM to analyze failures and iteratively improve instructions.
    """
    print("\n" + "=" * 70)
    print("Running GEPA Optimization")
    print("=" * 70)
    print(f"  Training examples: {len(trainset)}")
    print(f"  Dev examples: {len(devset)}")
    print(f"  Auto mode: {auto} (light/medium/heavy)")

    # GEPA requires a reflection LM to analyze failures and propose improvements
    # Using the same model but with higher temperature for creative instruction proposals
    reflection_lm = dspy.LM(
        model=f"together_ai/{MODEL_NAME}",
        api_key=api_key,
        temperature=0.7,  # Higher temp for creative reflection
        max_tokens=2048,
    )
    print(f"  Reflection LM: {MODEL_NAME} (temp=0.7)")

    optimizer = dspy.GEPA(
        metric=checkworthiness_metric_gepa,
        reflection_lm=reflection_lm,
        auto=auto,
        track_stats=True,
    )

    module = CheckworthinessModule()
    optimized = optimizer.compile(module, trainset=trainset, valset=devset)

    return optimized


def run_mipro_optimization(
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
) -> dspy.Module:
    """Run MIPROv2 optimization (more sophisticated)."""
    print("\n" + "=" * 70)
    print("Running MIPROv2 Optimization")
    print("=" * 70)

    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=checkworthiness_metric,
        auto=None,  # Disable auto to use manual settings
        num_candidates=NUM_CANDIDATES,
        init_temperature=1.0,
        verbose=True,
    )

    module = CheckworthinessModule()
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        num_trials=NUM_TRIALS,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MAX_LABELED_DEMOS,
    )

    return optimized


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_module(
    module: dspy.Module,
    dataset: list[dspy.Example],
    name: str = "eval",
) -> dict:
    """Evaluate a module on a dataset with F1, Precision, Recall, Accuracy."""
    from tqdm import tqdm

    print(f"\nEvaluating on {name} ({len(dataset)} samples)...")

    y_true = []
    y_pred = []
    predictions = []

    for ex in tqdm(dataset, desc=f"  {name}", ncols=80):
        try:
            pred = module(claim=ex.claim)
            pred_answer = pred.answer.strip().lower()

            if "yes" in pred_answer:
                pred_label = "Yes"
            elif "no" in pred_answer:
                pred_label = "No"
            else:
                pred_label = "No"

            y_true.append(1 if ex.label == "Yes" else 0)
            y_pred.append(1 if pred_label == "Yes" else 0)

            predictions.append({
                "claim": ex.claim[:50],
                "label": ex.label,
                "pred": pred_label,
                "correct": pred_label == ex.label,
                "reasoning": pred.reasoning[:100] if hasattr(pred, "reasoning") else "",
            })

        except Exception as e:
            print(f"  Error: {e}")
            y_true.append(1 if ex.label == "Yes" else 0)
            y_pred.append(0)  # Default to No on error

    # Compute metrics
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "total": len(y_true),
        "predictions": predictions,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DSPy Prompt Optimization")
    parser.add_argument(
        "--optimizer",
        choices=["bootstrap", "mipro", "gepa"],
        default="bootstrap",
        help="Optimization method",
    )
    parser.add_argument(
        "--gepa-mode",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="GEPA auto mode (light=fast, heavy=thorough)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=500,
        help="Max training samples for optimization",
    )
    parser.add_argument(
        "--dev-samples",
        type=int,
        default=200,
        help="Max dev samples for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling trainset",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=4,
        help="Max few-shot demos per candidate set",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        help="Number of candidate demo sets to try (more = better search, slower)",
    )
    args = parser.parse_args()

    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure DSPy with Together AI
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set")
        return

    lm = dspy.LM(
        model=f"together_ai/{MODEL_NAME}",
        api_key=api_key,
        max_tokens=512,
        temperature=0.0,
    )
    dspy.configure(lm=lm)

    print("=" * 70)
    print("DSPy Prompt Optimization")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Train samples: {args.train_samples}")
    print(f"Dev samples: {args.dev_samples}")
    print(f"Seed: {args.seed}")
    print(f"Max demos: {args.max_demos}")
    if args.optimizer == "bootstrap":
        print(f"Candidate sets: {args.num_candidates}")
    elif args.optimizer == "gepa":
        print(f"GEPA mode: {args.gepa_mode}")

    # Load data
    print("\nLoading data...")
    trainset = load_ct24_data("train", args.train_samples)
    devset = load_ct24_data("dev", args.dev_samples)
    testset = load_ct24_data("test")  # Full test set

    print(f"  Train: {len(trainset)}")
    print(f"  Dev: {len(devset)}")
    print(f"  Test: {len(testset)}")

    # Evaluate baseline (no optimization) on TEST set for fair comparison
    print("\n" + "=" * 70)
    print("Baseline (no optimization)")
    print("=" * 70)

    baseline_module = CheckworthinessModule()
    baseline_test = evaluate_module(baseline_module, testset, "test_baseline")

    # Run optimization
    if args.optimizer == "bootstrap":
        optimized = run_bootstrap_optimization(
            trainset, devset,
            seed=args.seed,
            max_demos=args.max_demos,
            num_candidate_sets=args.num_candidates,
        )
    elif args.optimizer == "mipro":
        optimized = run_mipro_optimization(trainset, devset)
    elif args.optimizer == "gepa":
        optimized = run_gepa_optimization(trainset, devset, auto=args.gepa_mode, api_key=api_key)

    # Evaluate optimized model
    print("\n" + "=" * 70)
    print("Optimized Model Evaluation")
    print("=" * 70)

    opt_dev = evaluate_module(optimized, devset, "dev")
    opt_test = evaluate_module(optimized, testset, "test")

    # Save results
    results = {
        "timestamp": timestamp,
        "optimizer": args.optimizer,
        "model": MODEL_NAME,
        "train_samples": len(trainset),
        "dev_samples": len(devset),
        "test_samples": len(testset),
        "baseline": {
            "test_acc": baseline_test["accuracy"],
            "test_f1": baseline_test["f1"],
            "test_precision": baseline_test["precision"],
            "test_recall": baseline_test["recall"],
        },
        "optimized": {
            "dev_acc": opt_dev["accuracy"],
            "dev_f1": opt_dev["f1"],
            "dev_precision": opt_dev["precision"],
            "dev_recall": opt_dev["recall"],
            "test_acc": opt_test["accuracy"],
            "test_f1": opt_test["f1"],
            "test_precision": opt_test["precision"],
            "test_recall": opt_test["recall"],
        },
        "improvement": {
            "test_acc": opt_test["accuracy"] - baseline_test["accuracy"],
            "test_f1": opt_test["f1"] - baseline_test["f1"],
        },
    }

    results_file = OUTPUT_DIR / f"dspy_{args.optimizer}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save optimized module
    module_file = OUTPUT_DIR / f"optimized_module_{args.optimizer}_{timestamp}.json"
    optimized.save(str(module_file))
    print(f"Optimized module saved to: {module_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Metric':<20} {'Baseline Test':<15} {'Optimized Dev':<15} {'Optimized Test':<15}")
    print(f"  {'-'*65}")
    print(f"  {'Accuracy':<20} {baseline_test['accuracy']:<15.4f} {opt_dev['accuracy']:<15.4f} {opt_test['accuracy']:<15.4f}")
    print(f"  {'F1':<20} {baseline_test['f1']:<15.4f} {opt_dev['f1']:<15.4f} {opt_test['f1']:<15.4f}")
    print(f"  {'Precision':<20} {baseline_test['precision']:<15.4f} {opt_dev['precision']:<15.4f} {opt_test['precision']:<15.4f}")
    print(f"  {'Recall':<20} {baseline_test['recall']:<15.4f} {opt_dev['recall']:<15.4f} {opt_test['recall']:<15.4f}")
    print(f"\n  Improvement (test): Acc {results['improvement']['test_acc']:+.4f}, F1 {results['improvement']['test_f1']:+.4f}")
    print(f"\n  SOTA Target: F1 = 0.82, Acc = 0.905")
    print(f"  Gap to SOTA: F1 {opt_test['f1'] - 0.82:+.4f}, Acc {opt_test['accuracy'] - 0.905:+.4f}")


if __name__ == "__main__":
    main()
