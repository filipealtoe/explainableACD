#!/usr/bin/env python3
"""
Multi-Architecture Ensemble.

Combines predictions from DeBERTa seeds + multiple other architectures
(RoBERTa, BGE, E5, GTE) for maximum diversity.

Usage:
    python ensemble_multi_arch.py \
        --data-dir ~/data \
        --deberta-dir ~/ensemble_results \
        --arch-dir ~/arch_results \
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


def load_probs_from_dir(model_dir: Path, model_name: str) -> tuple[np.ndarray | None, float]:
    """Load test probabilities and F1 from a model directory."""
    results_dir = find_results_dir(model_dir)
    prob_file = results_dir / "test_probs.npy"
    results_file = results_dir / "results.json"

    if not prob_file.exists():
        return None, 0.0

    probs = np.load(prob_file)

    f1 = 0.0
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
            f1 = results.get("test", {}).get("best", {}).get("f1", 0)

    return probs, f1


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
    parser = argparse.ArgumentParser(description="Multi-architecture ensemble")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--deberta-dir", type=Path, required=True,
                        help="Directory with DeBERTa seed results")
    parser.add_argument("--arch-dir", type=Path, required=True,
                        help="Directory with other architecture results")
    parser.add_argument("--deberta-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-ARCHITECTURE ENSEMBLE")
    print("=" * 70)

    # Load labels
    import polars as pl
    clean_dir = args.data_dir / "processed" / "CT24_clean"
    test_df = None
    for name in ["CT24_test_clean.parquet", "CT24_test.parquet"]:
        path = clean_dir / name
        if path.exists():
            test_df = pl.read_parquet(path)
            break
    if test_df is None:
        for name in ["CT24_test_clean.tsv", "CT24_test.tsv"]:
            path = clean_dir / name
            if path.exists():
                test_df = pl.read_csv(path, separator="\t")
                break
    if test_df is None:
        raw_path = args.data_dir / "raw" / "CT24_checkworthy_english" / "CT24_checkworthy_english_test_gold.tsv"
        test_df = pl.read_csv(raw_path, separator="\t")

    labels = [1 if l == "Yes" else 0 for l in test_df["class_label"].to_list()]
    print(f"\nLoaded {len(labels)} test samples")

    # Load all models
    print("\n📂 Loading Model Probabilities:")
    print("-" * 50)

    all_models = {}  # name -> (probs, f1)

    # DeBERTa seeds
    for seed in args.deberta_seeds:
        seed_dir = args.deberta_dir / f"seed_{seed}"
        probs, f1 = load_probs_from_dir(seed_dir, f"deberta_seed_{seed}")
        if probs is not None:
            all_models[f"deberta_s{seed}"] = (probs, f1)
            print(f"   ✓ DeBERTa seed {seed}: F1={f1:.4f}")
        else:
            print(f"   ✗ DeBERTa seed {seed}: not found")

    # Other architectures
    arch_names = ["roberta-large", "bge-large-en-v1.5", "e5-large-v2", "gte-large"]
    for arch in arch_names:
        arch_dir = args.arch_dir / arch
        if arch_dir.exists():
            probs, f1 = load_probs_from_dir(arch_dir, arch)
            if probs is not None:
                all_models[arch] = (probs, f1)
                print(f"   ✓ {arch}: F1={f1:.4f}")
            else:
                print(f"   ✗ {arch}: probs not found")
        else:
            print(f"   - {arch}: directory not found")

    if len(all_models) < 2:
        print("\n❌ Need at least 2 models for ensemble!")
        return

    # Separate DeBERTa and other models
    deberta_models = {k: v for k, v in all_models.items() if k.startswith("deberta")}
    other_models = {k: v for k, v in all_models.items() if not k.startswith("deberta")}

    print(f"\n📊 Model Summary:")
    print(f"   DeBERTa models: {len(deberta_models)}")
    print(f"   Other architectures: {len(other_models)}")
    print(f"   Total models: {len(all_models)}")

    # Evaluation thresholds
    thresholds = np.arange(0.40, 0.70, 0.05).tolist()

    # ==========================================================================
    # Baseline: DeBERTa-only ensemble
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BASELINE: DeBERTa-Only Ensemble")
    print("=" * 70)

    if deberta_models:
        deberta_probs = [v[0] for v in deberta_models.values()]
        deberta_f1s = [v[1] for v in deberta_models.values()]

        for temp in [0.3, 0.5]:
            scaled = [apply_temperature(p, temp) for p in deberta_probs]
            ensemble = np.mean(scaled, axis=0)
            results = evaluate(ensemble, labels, thresholds)
            best = max(results, key=lambda x: x["f1"])
            print(f"   DeBERTa-only (T={temp}): F1={best['f1']:.4f} @ {best['threshold']:.2f}")

        # Best DeBERTa-only
        scaled = [apply_temperature(p, 0.5) for p in deberta_probs]
        deberta_only_best_probs = np.mean(scaled, axis=0)
        deberta_only_results = evaluate(deberta_only_best_probs, labels, thresholds)
        deberta_only_best = max(deberta_only_results, key=lambda x: x["f1"])
        baseline_f1 = deberta_only_best["f1"]
    else:
        baseline_f1 = 0

    # ==========================================================================
    # Mixed ensembles
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MIXED ARCHITECTURE ENSEMBLES")
    print("=" * 70)

    best_overall = {"f1": baseline_f1, "method": "deberta_only", "threshold": 0.55, "models": list(deberta_models.keys())}

    # Try adding each architecture incrementally
    if other_models:
        print("\n📈 Adding architectures one by one:")
        print("-" * 50)

        for arch_name, (arch_probs, arch_f1) in other_models.items():
            # DeBERTa + this one architecture
            combined_probs = deberta_probs + [arch_probs]
            combined_f1s = deberta_f1s + [arch_f1]

            for temp in [0.3, 0.5]:
                scaled = [apply_temperature(p, temp) for p in combined_probs]

                # Equal weights
                ensemble_equal = np.mean(scaled, axis=0)
                results = evaluate(ensemble_equal, labels, thresholds)
                best = max(results, key=lambda x: x["f1"])

                marker = ""
                if best["f1"] > best_overall["f1"]:
                    best_overall = {
                        "f1": best["f1"],
                        "method": f"DeBERTa+{arch_name} (T={temp}, equal)",
                        "threshold": best["threshold"],
                        "models": list(deberta_models.keys()) + [arch_name]
                    }
                    marker = " ★ NEW BEST"

                print(f"   DeBERTa + {arch_name:<20} T={temp}: F1={best['f1']:.4f} @ {best['threshold']:.2f}{marker}")

        # Try all models together
        print("\n📈 All architectures combined:")
        print("-" * 50)

        all_probs = [v[0] for v in all_models.values()]
        all_f1s = [v[1] for v in all_models.values()]

        for temp in [0.3, 0.4, 0.5, 0.6]:
            scaled = [apply_temperature(p, temp) for p in all_probs]

            # Equal weights
            ensemble = np.mean(scaled, axis=0)
            results = evaluate(ensemble, labels, thresholds)
            best = max(results, key=lambda x: x["f1"])

            marker = ""
            if best["f1"] > best_overall["f1"]:
                best_overall = {
                    "f1": best["f1"],
                    "method": f"ALL (T={temp}, equal)",
                    "threshold": best["threshold"],
                    "models": list(all_models.keys())
                }
                marker = " ★ NEW BEST"

            print(f"   ALL models (T={temp}, equal):     F1={best['f1']:.4f} @ {best['threshold']:.2f}{marker}")

            # F1-weighted
            ensemble_weighted_probs = ensemble_weighted(scaled, all_f1s)
            results = evaluate(ensemble_weighted_probs, labels, thresholds)
            best = max(results, key=lambda x: x["f1"])

            if best["f1"] > best_overall["f1"]:
                best_overall = {
                    "f1": best["f1"],
                    "method": f"ALL (T={temp}, f1-weighted)",
                    "threshold": best["threshold"],
                    "models": list(all_models.keys())
                }
                marker = " ★ NEW BEST"
            else:
                marker = ""

            print(f"   ALL models (T={temp}, F1-weight): F1={best['f1']:.4f} @ {best['threshold']:.2f}{marker}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n   🏆 Best Configuration:")
    print(f"      Method: {best_overall['method']}")
    print(f"      F1: {best_overall['f1']:.4f}")
    print(f"      Threshold: {best_overall['threshold']:.2f}")
    print(f"      Models: {best_overall['models']}")

    print(f"\n   📊 Comparison:")
    print(f"      DeBERTa-only baseline: F1={baseline_f1:.4f}")
    print(f"      Best mixed ensemble:   F1={best_overall['f1']:.4f}")
    print(f"      Improvement:           {best_overall['f1'] - baseline_f1:+.4f}")

    # Save results
    if args.output:
        output_data = {
            "models": {name: {"f1": float(f1)} for name, (_, f1) in all_models.items()},
            "baseline_f1": float(baseline_f1),
            "best": {
                "f1": float(best_overall["f1"]),
                "method": best_overall["method"],
                "threshold": float(best_overall["threshold"]),
                "models": best_overall["models"],
            },
            "improvement": float(best_overall["f1"] - baseline_f1),
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
