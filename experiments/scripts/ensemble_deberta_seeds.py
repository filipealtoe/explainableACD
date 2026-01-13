#!/usr/bin/env python3
"""
Seed Ensemble Training and Inference for DeBERTa.

Trains multiple DeBERTa models with different random seeds, then ensembles
their predictions for improved robustness and performance.

Strategy:
1. Train N models with different seeds (default: 3)
2. Each model uses the best config (focal + LLRD + R-Drop + FGM + cosine)
3. Ensemble predictions via:
   - Soft voting (average probabilities) ← usually best
   - Hard voting (majority vote)
   - Max confidence voting
4. Evaluate ensemble on dev-test and test

Usage:
    # Train 3 models and ensemble
    python ensemble_deberta_seeds.py --data-dir ~/data --output-dir ~/ensemble_results

    # Use existing trained models (skip training)
    python ensemble_deberta_seeds.py --data-dir ~/data --output-dir ~/ensemble_results \
        --model-dirs ~/model1 ~/model2 ~/model3

    # Train with specific seeds
    python ensemble_deberta_seeds.py --data-dir ~/data --output-dir ~/ensemble_results \
        --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

# Best config from our experiments (F1=0.8242)
BEST_CONFIG = {
    "model": "deberta-v3-large",
    "focal_loss": True,
    "llrd": True,
    "rdrop": True,
    "fgm": True,
    "cosine_schedule": True,
    "eval_on_devtest": True,
    "epochs": 5,
    "batch_size": 8,
    "grad_accum": 4,
    "lr": 2e-5,
}

DEFAULT_SEEDS = [42, 123, 456]


# =============================================================================
# Path Utilities
# =============================================================================

def find_results_dir(model_dir: Path) -> Path:
    """
    Find the actual directory containing results.json and probs.

    The finetune script creates structure: seed_42/deberta-v3-large/
    But we pass seed_42/ to the ensemble script.
    """
    # Check if results.json is directly in model_dir
    if (model_dir / "results.json").exists():
        return model_dir

    # Check for model-named subdirectory (e.g., deberta-v3-large/)
    if model_dir.exists():
        subdirs = [d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        for subdir in subdirs:
            if (subdir / "results.json").exists():
                return subdir

    # Return original path as fallback
    return model_dir


# =============================================================================
# Training
# =============================================================================

def train_single_model(
    seed: int,
    data_dir: Path,
    output_dir: Path,
    config: dict,
) -> Path:
    """Train a single model with given seed."""
    model_dir = output_dir / f"seed_{seed}"

    # Check if already trained (results.json might be in a subdirectory)
    results_dir = find_results_dir(model_dir) if model_dir.exists() else model_dir
    if (results_dir / "results.json").exists():
        print(f"   ✓ Seed {seed} already trained, skipping...")
        return model_dir

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "finetune_deberta_multimodel.py"),
        "--model", config["model"],
        "--data-dir", str(data_dir),
        "--output-dir", str(model_dir),
        "--seed", str(seed),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--grad-accum", str(config["grad_accum"]),
        "--lr", str(config["lr"]),
    ]

    if config.get("focal_loss"):
        cmd.append("--focal-loss")
    if config.get("llrd"):
        cmd.append("--llrd")
    if config.get("rdrop"):
        cmd.append("--rdrop")
    if config.get("fgm"):
        cmd.append("--fgm")
    if config.get("cosine_schedule"):
        cmd.append("--cosine-schedule")
    if config.get("eval_on_devtest"):
        cmd.append("--eval-on-devtest")

    print(f"\n{'='*70}")
    print(f"TRAINING SEED {seed}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed for seed {seed}")

    return model_dir


def load_saved_probs(model_dir: Path) -> dict[str, np.ndarray]:
    """Load saved probability arrays from a trained model."""
    # Find the actual results directory
    results_dir = find_results_dir(model_dir)

    probs = {}
    for split in ["dev", "dev_test", "test"]:
        prob_file = results_dir / f"{split}_probs.npy"
        if prob_file.exists():
            probs[split] = np.load(prob_file)

    # Debug: show what we found
    if probs:
        print(f"      Found saved probs: {list(probs.keys())} in {results_dir.name}/")
    else:
        print(f"      No saved probs found in {results_dir}")
        # List .npy files for debugging
        npy_files = list(results_dir.glob("*.npy"))
        if npy_files:
            print(f"      Available .npy files: {[f.name for f in npy_files]}")

    return probs, results_dir  # Return results_dir for model loading


# =============================================================================
# Inference
# =============================================================================

def find_model_dir(base_dir: Path) -> Path | None:
    """Find the directory containing model files (config.json)."""
    # Check common locations
    candidates = [
        base_dir / "best_model",
        base_dir / "checkpoint-best",
        base_dir,
    ]

    # Also check for any checkpoint directories
    for ckpt_dir in sorted(base_dir.glob("checkpoint-*")):
        candidates.append(ckpt_dir)

    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate

    return None


def load_model_and_tokenizer(model_dir: Path, device: torch.device):
    """Load a trained model and tokenizer."""
    # Find the actual model directory
    actual_model_dir = find_model_dir(model_dir)

    if actual_model_dir is None:
        # List contents for debugging
        contents = list(model_dir.iterdir()) if model_dir.exists() else []
        raise FileNotFoundError(
            f"No model found in {model_dir}\n"
            f"Contents: {[c.name for c in contents]}\n"
            f"Looking for config.json in: best_model/, checkpoint-*/, or root"
        )

    print(f"      Loading model from: {actual_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(actual_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(actual_model_dir).to(device)
    model.eval()

    return model, tokenizer


def get_predictions(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Get prediction probabilities for texts."""
    all_probs = []

    with torch.no_grad():
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

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


# =============================================================================
# Ensemble Methods
# =============================================================================

@dataclass
class EnsembleResult:
    """Result of ensemble prediction."""
    method: str
    threshold: float
    f1: float
    accuracy: float
    precision: float
    recall: float
    probs: np.ndarray | None = None


def ensemble_soft_voting(probs_list: list[np.ndarray]) -> np.ndarray:
    """Average probabilities across models (soft voting)."""
    return np.mean(probs_list, axis=0)


def ensemble_weighted_voting(
    probs_list: list[np.ndarray],
    weights: list[float] | None = None,
    f1_scores: list[float] | None = None,
) -> np.ndarray:
    """
    Weighted average of probabilities.

    If f1_scores provided, weights are computed as normalized F1 scores.
    Otherwise uses provided weights or uniform weights.
    """
    if f1_scores is not None:
        # Weight by F1 score (higher F1 = more weight)
        weights = np.array(f1_scores)
        weights = weights / weights.sum()  # Normalize
    elif weights is None:
        weights = np.ones(len(probs_list)) / len(probs_list)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()

    probs_stack = np.stack(probs_list)
    return np.average(probs_stack, axis=0, weights=weights)


def ensemble_hard_voting(probs_list: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Majority voting on predictions."""
    votes = np.array([(p >= threshold).astype(int) for p in probs_list])
    # Return fraction of models voting positive (can be thresholded later)
    return votes.mean(axis=0)


def ensemble_max_confidence(probs_list: list[np.ndarray]) -> np.ndarray:
    """Use prediction from most confident model per sample."""
    probs_stack = np.stack(probs_list)  # (n_models, n_samples)

    # Confidence = distance from 0.5
    confidences = np.abs(probs_stack - 0.5)
    most_confident_idx = np.argmax(confidences, axis=0)

    # Select probability from most confident model
    return probs_stack[most_confident_idx, np.arange(len(most_confident_idx))]


def ensemble_median(probs_list: list[np.ndarray]) -> np.ndarray:
    """Median probability (robust to outliers)."""
    return np.median(probs_list, axis=0)


def ensemble_geometric_mean(probs_list: list[np.ndarray], epsilon: float = 1e-8) -> np.ndarray:
    """Geometric mean of probabilities (multiplicative ensemble)."""
    probs_stack = np.stack(probs_list)
    # Clip to avoid log(0)
    probs_stack = np.clip(probs_stack, epsilon, 1 - epsilon)
    log_probs = np.log(probs_stack)
    return np.exp(np.mean(log_probs, axis=0))


def ensemble_rank_average(probs_list: list[np.ndarray]) -> np.ndarray:
    """
    Average ranks instead of raw probabilities.

    More robust to different probability scales across models.
    """
    from scipy.stats import rankdata

    n_samples = len(probs_list[0])
    ranks = np.zeros((len(probs_list), n_samples))

    for i, probs in enumerate(probs_list):
        # Higher prob = higher rank
        ranks[i] = rankdata(probs) / n_samples

    return np.mean(ranks, axis=0)


def ensemble_temperature_scaled(
    probs_list: list[np.ndarray],
    temperature: float = 1.5,
) -> np.ndarray:
    """
    Apply temperature scaling before averaging.

    Temperature > 1: Softer predictions (less confident)
    Temperature < 1: Sharper predictions (more confident)
    """
    epsilon = 1e-8
    scaled_probs = []

    for probs in probs_list:
        # Convert to logits, scale, convert back
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        logits = np.log(probs_clipped / (1 - probs_clipped))  # inverse sigmoid
        scaled_logits = logits / temperature
        scaled = 1 / (1 + np.exp(-scaled_logits))  # sigmoid
        scaled_probs.append(scaled)

    return np.mean(scaled_probs, axis=0)


def evaluate_ensemble(
    probs: np.ndarray,
    labels: list[int],
    method_name: str,
    thresholds: list[float] | None = None,
) -> list[EnsembleResult]:
    """Evaluate ensemble predictions at different thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.75, 0.05)

    results = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        results.append(EnsembleResult(
            method=method_name,
            threshold=float(thresh),
            f1=float(f1_score(labels, preds)),
            accuracy=float(accuracy_score(labels, preds)),
            precision=float(precision_score(labels, preds, zero_division=0)),
            recall=float(recall_score(labels, preds, zero_division=0)),
            probs=probs if thresh == 0.5 else None,  # Only store once
        ))

    return results


# =============================================================================
# Data Loading
# =============================================================================

def load_split_data(data_dir: Path, split: str) -> tuple[list[str], list[int]]:
    """Load texts and labels for a split."""
    clean_dir = data_dir / "processed" / "CT24_clean"

    # Try different paths
    for suffix in ["_clean", ""]:
        for ext in [".parquet", ".tsv"]:
            path = clean_dir / f"CT24_{split}{suffix}{ext}"
            if path.exists():
                if ext == ".parquet":
                    df = pl.read_parquet(path)
                else:
                    df = pl.read_csv(path, separator="\t")

                texts = df["Text"].to_list()
                labels = [1 if l == "Yes" else 0 for l in df["class_label"].to_list()]
                return texts, labels

    raise FileNotFoundError(f"Data not found for split: {split}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Seed ensemble training and inference")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Data directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for ensemble results")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help=f"Random seeds to use (default: {DEFAULT_SEEDS})")
    parser.add_argument("--model-dirs", type=Path, nargs="+", default=None,
                        help="Use existing trained models (skip training)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use saved probabilities")
    args = parser.parse_args()

    print("=" * 70)
    print("SEED ENSEMBLE: DeBERTa Checkworthiness")
    print("=" * 70)
    print(f"\nSeeds: {args.seeds}")
    print(f"Ensemble size: {len(args.seeds)} models")
    print(f"Best config: {BEST_CONFIG}")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Train models or load existing
    # =========================================================================

    if args.model_dirs:
        model_dirs = args.model_dirs
        print(f"\n📦 Using {len(model_dirs)} provided models")
    else:
        print(f"\n🏋️ Training {len(args.seeds)} models...")
        model_dirs = []
        for seed in args.seeds:
            model_dir = train_single_model(
                seed=seed,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                config=BEST_CONFIG,
            )
            model_dirs.append(model_dir)

    # =========================================================================
    # Step 2: Load or compute predictions
    # =========================================================================

    print("\n📊 Loading predictions from each model...")

    all_probs = {"dev_test": [], "test": []}
    individual_results = []

    for i, model_dir in enumerate(model_dirs):
        print(f"\n   Model {i+1}: {model_dir.name}")

        # Try to load saved probabilities first
        saved_probs, results_dir = load_saved_probs(model_dir)

        if saved_probs and "test" in saved_probs:
            print(f"      ✓ Using saved probabilities")
            if "dev_test" in saved_probs:
                all_probs["dev_test"].append(saved_probs["dev_test"])
            all_probs["test"].append(saved_probs["test"])

            # Load individual results from the actual results directory
            results_file = results_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    individual_results.append(json.load(f))
        else:
            # Need to run inference - use results_dir for model loading
            print(f"      🔄 Running inference...")
            model, tokenizer = load_model_and_tokenizer(results_dir, device)

            for split in ["dev_test", "test"]:
                try:
                    texts, labels = load_split_data(args.data_dir, split.replace("_", "-"))
                    probs = get_predictions(model, tokenizer, texts, device, args.batch_size)
                    all_probs[split].append(probs)
                except FileNotFoundError:
                    print(f"      ⚠️ {split} not found, skipping")

            # Clean up GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================================================================
    # Step 3: Load labels
    # =========================================================================

    print("\n📋 Loading labels...")

    labels = {}
    for split in ["dev-test", "test"]:
        try:
            _, split_labels = load_split_data(args.data_dir, split)
            labels[split.replace("-", "_")] = split_labels
            print(f"   {split}: {len(split_labels)} samples")
        except FileNotFoundError:
            print(f"   {split}: not found")

    # =========================================================================
    # Step 4: Compute ensembles
    # =========================================================================

    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)

    ensemble_results = {}

    for split in ["dev_test", "test"]:
        if split not in labels or not all_probs.get(split):
            continue

        split_labels = labels[split]
        probs_list = all_probs[split]

        print(f"\n📈 {split.upper().replace('_', '-')} ({len(probs_list)} models)")
        print("-" * 50)

        # Individual model results
        print(f"\n   Individual models:")
        individual_f1s = []
        for i, probs in enumerate(probs_list):
            preds = (probs >= 0.5).astype(int)
            f1 = f1_score(split_labels, preds)
            individual_f1s.append(f1)
            print(f"      Model {i+1}: F1={f1:.4f}")

        # Ensemble methods - now with more techniques!
        methods = {
            # Basic averaging
            "soft_voting": ensemble_soft_voting(probs_list),
            "weighted_by_f1": ensemble_weighted_voting(probs_list, f1_scores=individual_f1s),
            # Voting methods
            "hard_voting": ensemble_hard_voting(probs_list),
            "max_confidence": ensemble_max_confidence(probs_list),
            # Robust aggregation
            "median": ensemble_median(probs_list),
            "geometric_mean": ensemble_geometric_mean(probs_list),
            "rank_average": ensemble_rank_average(probs_list),
            # Temperature scaling (grid search)
            "temp_0.3": ensemble_temperature_scaled(probs_list, temperature=0.3),
            "temp_0.4": ensemble_temperature_scaled(probs_list, temperature=0.4),
            "temp_0.5": ensemble_temperature_scaled(probs_list, temperature=0.5),
            "temp_0.6": ensemble_temperature_scaled(probs_list, temperature=0.6),
            "temp_0.7": ensemble_temperature_scaled(probs_list, temperature=0.7),
            "temp_1.5": ensemble_temperature_scaled(probs_list, temperature=1.5),
        }

        print(f"\n   Ensemble methods (at threshold=0.5):")

        split_results = {}
        for method_name, ensemble_probs in methods.items():
            results = evaluate_ensemble(ensemble_probs, split_labels, method_name)
            best = max(results, key=lambda x: x.f1)
            at_05 = next((r for r in results if abs(r.threshold - 0.5) < 0.01), best)

            split_results[method_name] = {
                "best": {"threshold": best.threshold, "f1": best.f1, "accuracy": best.accuracy,
                         "precision": best.precision, "recall": best.recall},
                "at_0.5": {"f1": at_05.f1, "accuracy": at_05.accuracy,
                           "precision": at_05.precision, "recall": at_05.recall},
                "all": [{"threshold": r.threshold, "f1": r.f1, "accuracy": r.accuracy} for r in results],
            }

            improvement = ""
            if individual_results:
                # Compare to average individual F1
                avg_individual = np.mean([r.get("test", {}).get("best", {}).get("f1", 0)
                                          for r in individual_results])
                if avg_individual > 0:
                    delta = best.f1 - avg_individual
                    improvement = f" ({delta:+.4f} vs avg individual)"

            print(f"      {method_name:<15}: F1={at_05.f1:.4f} (best={best.f1:.4f} @ {best.threshold:.2f}){improvement}")

        ensemble_results[split] = split_results

        # Save ensemble probabilities
        for method_name, ensemble_probs in methods.items():
            np.save(args.output_dir / f"{split}_{method_name}_probs.npy", ensemble_probs)

    # =========================================================================
    # Step 5: Threshold calibration for ensemble
    # =========================================================================

    if "dev_test" in ensemble_results and "test" in ensemble_results:
        print(f"\n🎯 THRESHOLD CALIBRATION (dev-test → test)")
        print("-" * 50)

        for method_name in methods.keys():
            # Get dev-test optimal threshold
            dev_test_best = ensemble_results["dev_test"][method_name]["best"]
            calibrated_thresh = dev_test_best["threshold"]

            # Apply to test
            test_probs = methods[method_name] if "test" in labels else None
            if test_probs is not None and "test" in labels:
                # Reload test probs for this method
                test_method_probs = np.load(args.output_dir / f"test_{method_name}_probs.npy")
                calibrated_preds = (test_method_probs >= calibrated_thresh).astype(int)
                calibrated_f1 = f1_score(labels["test"], calibrated_preds)
                test_oracle_f1 = ensemble_results["test"][method_name]["best"]["f1"]

                print(f"   {method_name:<15}: calibrated F1={calibrated_f1:.4f} "
                      f"(oracle={test_oracle_f1:.4f}, thresh={calibrated_thresh:.2f})")

                ensemble_results["test"][method_name]["calibrated"] = {
                    "threshold": calibrated_thresh,
                    "f1": float(calibrated_f1),
                }

    # =========================================================================
    # Step 6: Summary and save
    # =========================================================================

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    if "test" in ensemble_results:
        # Find the best ensemble method
        best_method = None
        best_f1 = 0
        for method_name, method_results in ensemble_results["test"].items():
            method_f1 = method_results["best"]["f1"]
            if method_f1 > best_f1:
                best_f1 = method_f1
                best_method = method_name

        print(f"\n   🏆 Best Ensemble Method: {best_method}")
        print(f"      F1={best_f1:.4f} @ threshold={ensemble_results['test'][best_method]['best']['threshold']:.2f}")

        # Show all methods ranked
        print(f"\n   📊 All Methods Ranked:")
        sorted_methods = sorted(
            ensemble_results["test"].items(),
            key=lambda x: x[1]["best"]["f1"],
            reverse=True
        )
        for rank, (method_name, method_results) in enumerate(sorted_methods, 1):
            f1 = method_results["best"]["f1"]
            thresh = method_results["best"]["threshold"]
            print(f"      {rank}. {method_name:<18} F1={f1:.4f} @ {thresh:.2f}")

        if individual_results:
            best_individual = max([r.get("test", {}).get("best", {}).get("f1", 0)
                                   for r in individual_results])
            print(f"\n   📈 Best Individual Model:      F1={best_individual:.4f}")
            print(f"   📈 Ensemble Gain:              {best_f1 - best_individual:+.4f}")

    # Save results
    output = {
        "config": BEST_CONFIG,
        "seeds": args.seeds,
        "model_dirs": [str(d) for d in model_dirs],
        "ensemble_results": ensemble_results,
        "individual_results": individual_results,
    }

    with open(args.output_dir / "ensemble_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: {args.output_dir / 'ensemble_results.json'}")
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
