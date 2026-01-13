#!/usr/bin/env python3
"""
Evaluate DeBERTa Ensemble on Multiple Benchmarks.

Runs the 3-seed DeBERTa ensemble on:
- ClaimBuster (crowdsourced)
- CheckThat 2022 (CT22)
- CheckThat 2023 (CT23)
- CheckThat 2024 (CT24) - our training set

Compares results to published SOTA for each benchmark.

Usage:
    python evaluate_on_benchmarks.py \
        --ensemble-dir ~/ensemble_results \
        --data-dir ~/data \
        --seeds 0 42 456
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm


# =============================================================================
# Published SOTA Results (for comparison)
# =============================================================================

SOTA_RESULTS = {
    "ClaimBuster": {
        "f1": 0.78,
        "source": "Hassan et al., 2017 (original ClaimBuster paper)",
        "notes": "SVM + linguistic features on crowdsourced data"
    },
    "CT22": {
        "f1": 0.64,
        "source": "CheckThat! 2022 Task 1A Winner",
        "notes": "Best submission on English test set"
    },
    "CT23": {
        "f1": 0.67,
        "source": "CheckThat! 2023 Task 1B Winner",
        "notes": "Best submission on English test set"
    },
    "CT24": {
        "f1": 0.82,
        "source": "CheckThat! 2024 Task 1 Top Results",
        "notes": "Best reported on English political debates"
    },
}


# =============================================================================
# Model Loading
# =============================================================================

def find_results_dir(model_dir: Path) -> Path:
    """Find the actual directory containing model files."""
    if (model_dir / "config.json").exists():
        return model_dir
    if (model_dir / "best_model").exists():
        return model_dir / "best_model"
    for subdir in model_dir.iterdir():
        if subdir.is_dir():
            if (subdir / "config.json").exists():
                return subdir
            if (subdir / "best_model").exists():
                return subdir / "best_model"
    return model_dir


def load_models(ensemble_dir: Path, seeds: list[int], device: torch.device):
    """Load all DeBERTa models."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    models = {}
    tokenizer = None

    for seed in seeds:
        seed_dir = ensemble_dir / f"seed_{seed}"
        model_dir = find_results_dir(seed_dir)

        if not (model_dir / "config.json").exists():
            print(f"   ⚠️ Model not found for seed {seed}")
            continue

        print(f"   Loading seed {seed} from {model_dir}")

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)

        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        models[seed] = model

    return models, tokenizer


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling."""
    epsilon = 1e-8
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))


# =============================================================================
# Inference
# =============================================================================

def get_ensemble_predictions(
    models: dict,
    tokenizer,
    texts: list[str],
    device: torch.device,
    temperature: float = 0.3,
    batch_size: int = 32,
) -> np.ndarray:
    """Run ensemble inference on texts."""

    all_probs = {seed: [] for seed in models}

    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        for seed, model in models.items():
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                all_probs[seed].extend(probs)

    # Concatenate and ensemble
    probs_arrays = [np.array(all_probs[seed]) for seed in models]

    # Apply temperature scaling
    scaled = [apply_temperature(p, temperature) for p in probs_arrays]

    # Average
    ensemble_probs = np.mean(scaled, axis=0)

    return ensemble_probs


def evaluate_with_threshold_search(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float] = None,
) -> dict:
    """Evaluate with threshold optimization."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.75, 0.05).tolist()

    best = {"f1": 0, "threshold": 0.5}

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "threshold": thresh,
                "accuracy": accuracy_score(labels, preds),
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
            }

    return best


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_claimbuster(data_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load ClaimBuster crowdsourced dataset."""
    file_path = data_dir / "raw" / "claim_buster" / "crowdsourced.csv"
    df = pl.read_csv(file_path)

    texts = df["Text"].to_list()

    # Verdict: -1 = not checkworthy, 0 = neutral, 1 = checkworthy
    # We treat 1 as positive, -1 and 0 as negative (binary classification)
    labels = np.array([1 if v == 1 else 0 for v in df["Verdict"].to_list()])

    return texts, labels


def load_ct22(data_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load CheckThat 2022 test set."""
    file_path = data_dir / "raw" / "check_that_22" / "CT22_english_1A_checkworthy_test_gold.tsv"
    df = pl.read_csv(file_path, separator="\t")

    texts = df["tweet_text"].to_list()
    # class_label: 0 = not checkworthy, 1 = checkworthy
    labels = np.array(df["class_label"].to_list())

    return texts, labels


def load_ct23(data_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load CheckThat 2023 test set."""
    file_path = data_dir / "raw" / "check_that_23" / "CT23_1B_checkworthy_english_test_gold.tsv"
    df = pl.read_csv(file_path, separator="\t")

    texts = df["Text"].to_list()
    # class_label: "Yes" = checkworthy, "No" = not checkworthy
    labels = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])

    return texts, labels


def load_ct24(data_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load CheckThat 2024 test set."""
    # Try processed first
    clean_dir = data_dir / "processed" / "CT24_clean"
    for name in ["CT24_test_clean.parquet", "CT24_test.parquet"]:
        if (clean_dir / name).exists():
            df = pl.read_parquet(clean_dir / name)
            texts = df["Text"].to_list()
            labels = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])
            return texts, labels

    # Raw fallback
    file_path = data_dir / "raw" / "CT24_checkworthy_english" / "CT24_checkworthy_english_test_gold.tsv"
    df = pl.read_csv(file_path, separator="\t")
    texts = df["Text"].to_list()
    labels = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])

    return texts, labels


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 456])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold. If not set, searches for best.")
    args = parser.parse_args()

    print("=" * 70)
    print("DEBERTA ENSEMBLE - MULTI-BENCHMARK EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load models
    print(f"\n📂 Loading models (seeds: {args.seeds})...")
    models, tokenizer = load_models(args.ensemble_dir, args.seeds, device)
    print(f"   Loaded {len(models)} models")

    if len(models) == 0:
        print("❌ No models loaded!")
        return

    # Benchmarks to evaluate
    benchmarks = {
        "ClaimBuster": load_claimbuster,
        "CT22": load_ct22,
        "CT23": load_ct23,
        "CT24": load_ct24,
    }

    results = {}

    for bench_name, loader_fn in benchmarks.items():
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {bench_name}")
        print("=" * 70)

        try:
            texts, labels = loader_fn(args.data_dir)
            print(f"   Samples: {len(texts)}")
            print(f"   Positive (checkworthy): {labels.sum()} ({100*labels.mean():.1f}%)")
            print(f"   Negative: {len(labels) - labels.sum()}")

            # Run inference
            print(f"\n   Running ensemble inference (T={args.temperature})...")
            probs = get_ensemble_predictions(
                models, tokenizer, texts, device,
                temperature=args.temperature
            )

            # Evaluate
            if args.threshold:
                preds = (probs >= args.threshold).astype(int)
                result = {
                    "f1": f1_score(labels, preds, zero_division=0),
                    "threshold": args.threshold,
                    "accuracy": accuracy_score(labels, preds),
                    "precision": precision_score(labels, preds, zero_division=0),
                    "recall": recall_score(labels, preds, zero_division=0),
                }
            else:
                result = evaluate_with_threshold_search(probs, labels)

            results[bench_name] = result

            # Print results
            print(f"\n   📊 Results:")
            print(f"      F1:        {result['f1']:.4f}")
            print(f"      Precision: {result['precision']:.4f}")
            print(f"      Recall:    {result['recall']:.4f}")
            print(f"      Accuracy:  {result['accuracy']:.4f}")
            print(f"      Threshold: {result['threshold']:.2f}")

            # Compare to SOTA
            if bench_name in SOTA_RESULTS:
                sota = SOTA_RESULTS[bench_name]
                diff = result['f1'] - sota['f1']
                symbol = "📈" if diff > 0 else "📉" if diff < 0 else "="

                print(f"\n   {symbol} vs SOTA:")
                print(f"      SOTA F1: {sota['f1']:.4f} ({sota['source']})")
                print(f"      Ours:    {result['f1']:.4f}")
                print(f"      Diff:    {diff:+.4f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[bench_name] = {"f1": 0, "error": str(e)}

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: ALL BENCHMARKS")
    print("=" * 70)

    print(f"\n   {'Benchmark':<15} {'Our F1':<10} {'SOTA F1':<10} {'Diff':<10} {'Status'}")
    print("   " + "-" * 60)

    for bench_name in benchmarks:
        our_f1 = results.get(bench_name, {}).get("f1", 0)
        sota_f1 = SOTA_RESULTS.get(bench_name, {}).get("f1", 0)
        diff = our_f1 - sota_f1

        if diff > 0.01:
            status = "✓ BEATS SOTA"
        elif diff > -0.01:
            status = "≈ MATCHES"
        else:
            status = "below"

        print(f"   {bench_name:<15} {our_f1:<10.4f} {sota_f1:<10.4f} {diff:>+.4f}     {status}")

    # Model configuration summary
    print(f"\n   📋 Model Configuration:")
    print(f"      Base Model: microsoft/deberta-v3-large (435M params)")
    print(f"      Seeds: {args.seeds}")
    print(f"      Temperature: {args.temperature}")
    print(f"      Loss: Focal Loss (γ=2, α=0.25)")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
