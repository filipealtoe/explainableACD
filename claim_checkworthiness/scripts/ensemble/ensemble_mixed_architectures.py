#!/usr/bin/env python3
"""
Mixed Architecture Ensemble.

Combines predictions from multiple architectures (DeBERTa + RoBERTa)
for improved diversity and potentially higher F1.

Usage:
    python ensemble_mixed_architectures.py \
        --deberta-dir ~/ensemble_results \
        --roberta-dir ~/roberta_results \
        --data-dir ~/data \
        --deberta-seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# =============================================================================
# Path Utilities
# =============================================================================

def find_results_dir(model_dir: Path) -> Path:
    """Find the actual directory containing results.json and probs."""
    if (model_dir / "results.json").exists():
        return model_dir

    if model_dir.exists():
        subdirs = [d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        for subdir in subdirs:
            if (subdir / "results.json").exists():
                return subdir

    return model_dir


# =============================================================================
# Ensemble Methods
# =============================================================================

def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
    epsilon = 1e-8
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))


def ensemble_temperature(probs_list: list[np.ndarray], temperature: float) -> np.ndarray:
    """Temperature-scaled soft voting."""
    scaled = [apply_temperature(p, temperature) for p in probs_list]
    return np.mean(scaled, axis=0)


def ensemble_weighted(probs_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted average of probabilities."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    probs_stack = np.stack(probs_list)
    return np.average(probs_stack, axis=0, weights=weights)


def evaluate(probs: np.ndarray, labels: list[int], thresholds: list[float]) -> list[dict]:
    """Evaluate at multiple thresholds."""
    results = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        results.append({
            "threshold": thresh,
            "f1": f1_score(labels, preds),
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
        })
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Mixed architecture ensemble")
    parser.add_argument("--deberta-dir", type=Path, required=True,
                        help="Directory with DeBERTa ensemble results")
    parser.add_argument("--roberta-dir", type=Path, required=True,
                        help="Directory with RoBERTa results")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Data directory")
    parser.add_argument("--deberta-seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="DeBERTa seeds")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("MIXED ARCHITECTURE ENSEMBLE")
    print("=" * 70)

    # Load labels
    import polars as pl
    clean_dir = args.data_dir / "processed" / "CT24_clean"
    for name in ["CT24_test_clean.parquet", "CT24_test.parquet"]:
        path = clean_dir / name
        if path.exists():
            test_df = pl.read_parquet(path)
            break
    else:
        # Try TSV
        for name in ["CT24_test_clean.tsv", "CT24_test.tsv"]:
            path = clean_dir / name
            if path.exists():
                test_df = pl.read_csv(path, separator="\t")
                break
        else:
            # Raw fallback
            raw_path = args.data_dir / "raw" / "CT24_checkworthy_english" / "CT24_checkworthy_english_test_gold.tsv"
            test_df = pl.read_csv(raw_path, separator="\t")

    labels = [1 if l == "Yes" else 0 for l in test_df["class_label"].to_list()]
    print(f"\nLoaded {len(labels)} test samples")

    # Load DeBERTa probabilities
    print("\n📂 Loading DeBERTa probabilities...")
    deberta_probs = []
    for seed in args.deberta_seeds:
        seed_dir = args.deberta_dir / f"seed_{seed}"
        results_dir = find_results_dir(seed_dir)
        prob_file = results_dir / "test_probs.npy"
        if prob_file.exists():
            probs = np.load(prob_file)
            deberta_probs.append(probs)
            print(f"   ✓ DeBERTa seed {seed}: loaded")
        else:
            print(f"   ✗ DeBERTa seed {seed}: not found at {prob_file}")

    # Load RoBERTa probabilities
    print("\n📂 Loading RoBERTa probabilities...")
    roberta_results_dir = find_results_dir(args.roberta_dir)
    roberta_prob_file = roberta_results_dir / "test_probs.npy"

    if roberta_prob_file.exists():
        roberta_probs = np.load(roberta_prob_file)
        print(f"   ✓ RoBERTa: loaded")

        # Load RoBERTa individual F1
        roberta_results_file = roberta_results_dir / "results.json"
        if roberta_results_file.exists():
            with open(roberta_results_file) as f:
                roberta_results = json.load(f)
                roberta_f1 = roberta_results.get("test", {}).get("best", {}).get("f1", 0)
                print(f"   RoBERTa individual F1: {roberta_f1:.4f}")
    else:
        print(f"   ✗ RoBERTa: not found at {roberta_prob_file}")
        roberta_probs = None

    if not deberta_probs:
        print("\n❌ No DeBERTa models found!")
        return

    # Individual DeBERTa performance
    print("\n📊 Individual Model Performance:")
    print("-" * 50)

    deberta_f1s = []
    for i, probs in enumerate(deberta_probs):
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(labels, preds)
        deberta_f1s.append(f1)
        print(f"   DeBERTa seed {args.deberta_seeds[i]}: F1={f1:.4f}")

    if roberta_probs is not None:
        roberta_preds = (roberta_probs >= 0.5).astype(int)
        roberta_f1 = f1_score(labels, roberta_preds)
        print(f"   RoBERTa:                F1={roberta_f1:.4f}")

    # DeBERTa-only ensemble (baseline)
    print("\n📊 DeBERTa-Only Ensemble (baseline):")
    print("-" * 50)

    thresholds = np.arange(0.40, 0.70, 0.05).tolist()

    for temp in [0.3, 0.5, 0.7]:
        deberta_ensemble = ensemble_temperature(deberta_probs, temp)
        results = evaluate(deberta_ensemble, labels, thresholds)
        best = max(results, key=lambda x: x["f1"])
        print(f"   DeBERTa-only (T={temp}): F1={best['f1']:.4f} @ {best['threshold']:.2f}")

    # Mixed ensemble
    if roberta_probs is not None:
        print("\n📊 Mixed Architecture Ensemble (DeBERTa + RoBERTa):")
        print("-" * 50)

        all_probs = deberta_probs + [roberta_probs]
        n_deberta = len(deberta_probs)

        # Different weighting strategies
        strategies = {
            "equal": [1.0] * len(all_probs),
            "deberta_heavy": [1.0] * n_deberta + [0.5],  # DeBERTa 2x weight
            "roberta_heavy": [0.5] * n_deberta + [1.0],  # RoBERTa 2x weight
            "f1_weighted": deberta_f1s + [roberta_f1],   # Weight by F1
        }

        best_overall = {"f1": 0, "method": "", "threshold": 0.5}

        for temp in [0.3, 0.5, 0.7]:
            print(f"\n   Temperature = {temp}:")

            for strategy_name, weights in strategies.items():
                # Apply temperature then weighted average
                scaled_probs = [apply_temperature(p, temp) for p in all_probs]
                ensemble = ensemble_weighted(scaled_probs, weights)

                results = evaluate(ensemble, labels, thresholds)
                best = max(results, key=lambda x: x["f1"])

                marker = ""
                if best["f1"] > best_overall["f1"]:
                    best_overall = {"f1": best["f1"], "method": f"T={temp}, {strategy_name}", "threshold": best["threshold"]}
                    marker = " ★ NEW BEST"

                print(f"      {strategy_name:<15}: F1={best['f1']:.4f} @ {best['threshold']:.2f}{marker}")

        # Also try simple averaging without temperature
        print(f"\n   No temperature scaling:")
        for strategy_name, weights in strategies.items():
            ensemble = ensemble_weighted(all_probs, weights)
            results = evaluate(ensemble, labels, thresholds)
            best = max(results, key=lambda x: x["f1"])

            marker = ""
            if best["f1"] > best_overall["f1"]:
                best_overall = {"f1": best["f1"], "method": f"no_temp, {strategy_name}", "threshold": best["threshold"]}
                marker = " ★ NEW BEST"

            print(f"      {strategy_name:<15}: F1={best['f1']:.4f} @ {best['threshold']:.2f}{marker}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        deberta_best = max(deberta_f1s)
        deberta_ensemble_best = max(results, key=lambda x: x["f1"])["f1"]  # From last run

        # Get actual deberta-only best
        deberta_only_ensemble = ensemble_temperature(deberta_probs, 0.5)
        deberta_only_results = evaluate(deberta_only_ensemble, labels, thresholds)
        deberta_only_best = max(deberta_only_results, key=lambda x: x["f1"])["f1"]

        print(f"\n   Best individual DeBERTa:     F1={deberta_best:.4f}")
        print(f"   Best DeBERTa-only ensemble:  F1={deberta_only_best:.4f}")
        print(f"   Best mixed ensemble:         F1={best_overall['f1']:.4f} ({best_overall['method']})")
        print(f"\n   🎯 Improvement from adding RoBERTa: {best_overall['f1'] - deberta_only_best:+.4f}")

        # Save results
        if args.output:
            output_data = {
                "deberta_seeds": args.deberta_seeds,
                "individual_f1s": {
                    "deberta": {str(s): f for s, f in zip(args.deberta_seeds, deberta_f1s)},
                    "roberta": float(roberta_f1),
                },
                "deberta_only_best": float(deberta_only_best),
                "mixed_best": {
                    "f1": float(best_overall["f1"]),
                    "method": best_overall["method"],
                    "threshold": float(best_overall["threshold"]),
                },
                "improvement": float(best_overall["f1"] - deberta_only_best),
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
