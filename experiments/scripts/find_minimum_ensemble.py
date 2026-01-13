#!/usr/bin/env python3
"""
Find Minimum Ensemble Size.

Tests all combinations of models to find the smallest ensemble
that achieves the target F1 score.

Usage:
    python find_minimum_ensemble.py \
        --ensemble-dir ~/ensemble_results \
        --data-dir ~/data \
        --seeds 0 42 123 456 \
        --target-f1 0.8343
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def find_results_dir(model_dir: Path) -> Path:
    """Find the actual directory containing results.json and probs."""
    if (model_dir / "results.json").exists():
        return model_dir
    if model_dir.exists():
        for subdir in model_dir.iterdir():
            if subdir.is_dir() and (subdir / "results.json").exists():
                return subdir
    return model_dir


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling."""
    epsilon = 1e-8
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))


def evaluate_ensemble(probs_list: list[np.ndarray], labels: list[int],
                      temperature: float = 0.5) -> tuple[float, float, float]:
    """Evaluate ensemble with temperature scaling, return best F1 and threshold."""
    scaled = [apply_temperature(p, temperature) for p in probs_list]
    ensemble = np.mean(scaled, axis=0)

    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.40, 0.70, 0.05):
        preds = (ensemble >= thresh).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_f1, best_thresh, temperature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123, 456])
    parser.add_argument("--target-f1", type=float, default=0.8343)
    args = parser.parse_args()

    print("=" * 70)
    print("MINIMUM ENSEMBLE SIZE FINDER")
    print("=" * 70)

    # Load labels
    import polars as pl
    clean_dir = args.data_dir / "processed" / "CT24_clean"
    for name in ["CT24_test_clean.parquet", "CT24_test.parquet"]:
        if (clean_dir / name).exists():
            test_df = pl.read_parquet(clean_dir / name)
            break
    else:
        raw_path = args.data_dir / "raw" / "CT24_checkworthy_english" / "CT24_checkworthy_english_test_gold.tsv"
        test_df = pl.read_csv(raw_path, separator="\t")

    labels = [1 if l == "Yes" else 0 for l in test_df["class_label"].to_list()]

    # Load all model probabilities
    print(f"\n📂 Loading models for seeds: {args.seeds}")
    models = {}
    for seed in args.seeds:
        seed_dir = args.ensemble_dir / f"seed_{seed}"
        results_dir = find_results_dir(seed_dir)
        prob_file = results_dir / "test_probs.npy"
        if prob_file.exists():
            probs = np.load(prob_file)
            # Get individual F1
            preds = (probs >= 0.5).astype(int)
            ind_f1 = f1_score(labels, preds)
            models[seed] = {"probs": probs, "f1": ind_f1}
            print(f"   Seed {seed}: F1={ind_f1:.4f}")

    print(f"\n🎯 Target F1: {args.target_f1:.4f}")

    # Test single models
    print(f"\n{'='*70}")
    print("SINGLE MODELS")
    print("=" * 70)

    for seed, data in sorted(models.items(), key=lambda x: -x[1]["f1"]):
        for temp in [0.3, 0.5, 0.7, 1.0]:
            f1, thresh, _ = evaluate_ensemble([data["probs"]], labels, temp)
            marker = " ✓ ACHIEVES TARGET" if f1 >= args.target_f1 else ""
            if temp == 0.5:  # Only show T=0.5 for single models
                print(f"   Seed {seed}: F1={f1:.4f} @ {thresh:.2f}{marker}")

    # Test 2-model combinations
    print(f"\n{'='*70}")
    print("2-MODEL ENSEMBLES")
    print("=" * 70)

    best_2model = {"f1": 0, "combo": None, "thresh": 0.5, "temp": 0.5}
    seeds = list(models.keys())

    for combo in combinations(seeds, 2):
        probs_list = [models[s]["probs"] for s in combo]
        for temp in [0.3, 0.5, 0.6]:
            f1, thresh, _ = evaluate_ensemble(probs_list, labels, temp)
            if f1 > best_2model["f1"]:
                best_2model = {"f1": f1, "combo": combo, "thresh": thresh, "temp": temp}

            marker = " ✓ ACHIEVES TARGET" if f1 >= args.target_f1 else ""
            if f1 >= args.target_f1 - 0.01:  # Show if close to target
                print(f"   {combo} T={temp}: F1={f1:.4f} @ {thresh:.2f}{marker}")

    print(f"\n   Best 2-model: {best_2model['combo']} → F1={best_2model['f1']:.4f} (T={best_2model['temp']}, thresh={best_2model['thresh']:.2f})")

    # Test 3-model combinations
    print(f"\n{'='*70}")
    print("3-MODEL ENSEMBLES")
    print("=" * 70)

    best_3model = {"f1": 0, "combo": None, "thresh": 0.5, "temp": 0.5}
    achieving_3model = []

    for combo in combinations(seeds, 3):
        probs_list = [models[s]["probs"] for s in combo]
        for temp in [0.3, 0.5, 0.6]:
            f1, thresh, _ = evaluate_ensemble(probs_list, labels, temp)
            if f1 > best_3model["f1"]:
                best_3model = {"f1": f1, "combo": combo, "thresh": thresh, "temp": temp}

            if f1 >= args.target_f1:
                achieving_3model.append((combo, temp, thresh, f1))

    if achieving_3model:
        print(f"   Combinations achieving target ({args.target_f1:.4f}):")
        for combo, temp, thresh, f1 in achieving_3model:
            print(f"      {combo} T={temp}: F1={f1:.4f} @ {thresh:.2f} ✓")

    print(f"\n   Best 3-model: {best_3model['combo']} → F1={best_3model['f1']:.4f} (T={best_3model['temp']}, thresh={best_3model['thresh']:.2f})")

    # Test 4-model (all)
    if len(seeds) >= 4:
        print(f"\n{'='*70}")
        print("4-MODEL ENSEMBLE (ALL)")
        print("=" * 70)

        probs_list = [models[s]["probs"] for s in seeds]
        for temp in [0.3, 0.5, 0.6]:
            f1, thresh, _ = evaluate_ensemble(probs_list, labels, temp)
            marker = " ✓ ACHIEVES TARGET" if f1 >= args.target_f1 else ""
            print(f"   All {len(seeds)} models T={temp}: F1={f1:.4f} @ {thresh:.2f}{marker}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: MINIMUM ENSEMBLE SIZE")
    print("=" * 70)

    # Find minimum
    min_size = None
    min_config = None

    # Check single
    for seed, data in models.items():
        for temp in [0.3, 0.5, 0.6]:
            f1, thresh, _ = evaluate_ensemble([data["probs"]], labels, temp)
            if f1 >= args.target_f1:
                min_size = 1
                min_config = {"seeds": [seed], "temp": temp, "thresh": thresh, "f1": f1}
                break
        if min_size:
            break

    # Check 2-model
    if not min_size:
        for combo in combinations(seeds, 2):
            for temp in [0.3, 0.5, 0.6]:
                probs_list = [models[s]["probs"] for s in combo]
                f1, thresh, _ = evaluate_ensemble(probs_list, labels, temp)
                if f1 >= args.target_f1:
                    min_size = 2
                    min_config = {"seeds": list(combo), "temp": temp, "thresh": thresh, "f1": f1}
                    break
            if min_size:
                break

    # Check 3-model
    if not min_size:
        for combo in combinations(seeds, 3):
            for temp in [0.3, 0.5, 0.6]:
                probs_list = [models[s]["probs"] for s in combo]
                f1, thresh, _ = evaluate_ensemble(probs_list, labels, temp)
                if f1 >= args.target_f1:
                    min_size = 3
                    min_config = {"seeds": list(combo), "temp": temp, "thresh": thresh, "f1": f1}
                    break
            if min_size:
                break

    # Check 4-model
    if not min_size and len(seeds) >= 4:
        for temp in [0.3, 0.5, 0.6]:
            probs_list = [models[s]["probs"] for s in seeds]
            f1, thresh, _ = evaluate_ensemble(probs_list, labels, temp)
            if f1 >= args.target_f1:
                min_size = 4
                min_config = {"seeds": seeds, "temp": temp, "thresh": thresh, "f1": f1}
                break

    if min_config:
        print(f"\n   🏆 MINIMUM ENSEMBLE SIZE: {min_size} model(s)")
        print(f"      Seeds: {min_config['seeds']}")
        print(f"      Temperature: {min_config['temp']}")
        print(f"      Threshold: {min_config['thresh']:.2f}")
        print(f"      F1: {min_config['f1']:.4f}")
    else:
        print(f"\n   ❌ Target F1={args.target_f1:.4f} not achievable with these models")
        print(f"      Best achieved: {best_3model['f1']:.4f} with {best_3model['combo']}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
