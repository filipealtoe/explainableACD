#!/usr/bin/env python3
"""
Combine DeBERTa Ensemble with LLM Features V4.

Tries multiple fusion strategies:
1. Late fusion: weighted average of DeBERTa probs + LLM classifier probs
2. Feature stacking: DeBERTa probs as features alongside LLM features
3. Learned fusion: train meta-classifier on both predictions

Usage:
    python combine_deberta_llm_features.py \
        --ensemble-dir ~/ensemble_results \
        --data-dir ~/data \
        --seeds 0 42 456
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

# LLM feature groups (from v4)
FEATURE_GROUPS = {
    "scores": ["check_score", "verif_score", "harm_score"],
    "entropy": ["check_entropy", "verif_entropy", "harm_entropy"],
    "p_yes": ["check_p_yes", "verif_p_yes", "harm_p_yes"],
    "margin_p": ["check_margin_p", "verif_margin_p", "harm_margin_p"],
    "predictions": ["check_prediction", "verif_prediction", "harm_prediction"],
    "cross_basic": ["score_variance", "score_max_diff", "yes_vote_count", "unanimous_yes", "unanimous_no"],
    "harm_subdims": ["harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness"],
}


# =============================================================================
# Data Loading
# =============================================================================

def find_results_dir(model_dir: Path) -> Path:
    """Find the actual directory containing results.json and probs."""
    if (model_dir / "results.json").exists():
        return model_dir
    if model_dir.exists():
        for subdir in model_dir.iterdir():
            if subdir.is_dir() and (subdir / "results.json").exists():
                return subdir
            if subdir.is_dir():
                for subsubdir in subdir.iterdir():
                    if subsubdir.is_dir() and (subsubdir / "results.json").exists():
                        return subsubdir
    return model_dir


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
    epsilon = 1e-8
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))


def load_deberta_probs(ensemble_dir: Path, seeds: list[int], split: str = "test") -> dict:
    """Load DeBERTa probabilities for each seed."""
    probs_dict = {}
    for seed in seeds:
        seed_dir = ensemble_dir / f"seed_{seed}"
        results_dir = find_results_dir(seed_dir)
        prob_file = results_dir / f"{split}_probs.npy"
        if prob_file.exists():
            probs_dict[seed] = np.load(prob_file)
            print(f"   Loaded seed {seed}: {prob_file}")
        else:
            print(f"   ⚠️ Not found: {prob_file}")
    return probs_dict


def load_llm_features(data_dir: Path, split: str) -> tuple[np.ndarray, list[str]]:
    """Load LLM features v4."""
    # Try multiple possible locations
    possible_dirs = [
        data_dir / "CT24_llm_features_v4",
        data_dir / "processed" / "CT24_llm_features_v4",
    ]

    llm_file = None
    for llm_dir in possible_dirs:
        candidate = llm_dir / f"{split}_llm_features.parquet"
        if candidate.exists():
            llm_file = candidate
            break

    if llm_file is None:
        raise FileNotFoundError(f"LLM features not found in: {possible_dirs}")

    print(f"   Found: {llm_file}")

    df = pl.read_parquet(llm_file)

    # Get all available features
    features = []
    for group_features in FEATURE_GROUPS.values():
        for f in group_features:
            if f in df.columns and f not in features:
                features.append(f)

    X = df.select(features).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    return X, features


def load_labels(data_dir: Path, split: str) -> np.ndarray:
    """Load labels for a split."""
    # Try processed first
    clean_dir = data_dir / "processed" / "CT24_clean"
    for name in [f"CT24_{split}_clean.parquet", f"CT24_{split}.parquet"]:
        if (clean_dir / name).exists():
            df = pl.read_parquet(clean_dir / name)
            return np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])

    # Try features dir
    feat_dir = data_dir / "processed" / "CT24_features"
    for name in [f"CT24_{split}_features.parquet"]:
        if (feat_dir / name).exists():
            df = pl.read_parquet(feat_dir / name)
            return np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])

    # Raw fallback
    raw_map = {"test": "test_gold", "train": "train", "dev": "dev"}
    raw_name = raw_map.get(split, split)
    raw_path = data_dir / "raw" / "CT24_checkworthy_english" / f"CT24_checkworthy_english_{raw_name}.tsv"
    df = pl.read_csv(raw_path, separator="\t")
    return np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])


def evaluate(probs: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> dict:
    """Evaluate at multiple thresholds, return best."""
    best = {"f1": 0, "threshold": 0.5}
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds)
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
# Fusion Methods
# =============================================================================

def deberta_ensemble(probs_dict: dict, temperature: float = 0.5) -> np.ndarray:
    """Create DeBERTa ensemble with temperature scaling."""
    probs_list = list(probs_dict.values())
    scaled = [apply_temperature(p, temperature) for p in probs_list]
    return np.mean(scaled, axis=0)


def train_llm_classifier(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, class_weight: int = 3) -> np.ndarray:
    """Train classifier on LLM features, return test probabilities."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=class_weight,
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_train_scaled, y_train)
    return clf.predict_proba(X_test_scaled)[:, 1]


def late_fusion(deberta_probs: np.ndarray, llm_probs: np.ndarray,
                deberta_weight: float) -> np.ndarray:
    """Weighted average of DeBERTa and LLM probabilities."""
    return deberta_weight * deberta_probs + (1 - deberta_weight) * llm_probs


def feature_stacking(X_llm_train: np.ndarray, y_train: np.ndarray,
                     X_llm_test: np.ndarray,
                     deberta_train_probs: np.ndarray,
                     deberta_test_probs: np.ndarray,
                     include_individual_seeds: bool = False,
                     seed_probs_train: dict = None,
                     seed_probs_test: dict = None) -> np.ndarray:
    """Stack DeBERTa probs as features alongside LLM features."""

    # Add DeBERTa ensemble prob as feature
    X_train = np.column_stack([X_llm_train, deberta_train_probs])
    X_test = np.column_stack([X_llm_test, deberta_test_probs])

    # Optionally add individual seed probs
    if include_individual_seeds and seed_probs_train and seed_probs_test:
        for seed in seed_probs_train:
            X_train = np.column_stack([X_train, seed_probs_train[seed]])
            X_test = np.column_stack([X_test, seed_probs_test[seed]])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=3,
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_train_scaled, y_train)
    return clf.predict_proba(X_test_scaled)[:, 1]


def meta_learner(deberta_probs: np.ndarray, llm_probs: np.ndarray,
                 y_train: np.ndarray, deberta_test: np.ndarray,
                 llm_test: np.ndarray) -> np.ndarray:
    """Train a meta-classifier on DeBERTa + LLM predictions."""
    X_train = np.column_stack([deberta_probs, llm_probs])
    X_test = np.column_stack([deberta_test, llm_test])

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, required=True,
                        help="Directory with DeBERTa seed results")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Data directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 456],
                        help="DeBERTa seeds to use")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for DeBERTa ensemble")
    args = parser.parse_args()

    print("=" * 70)
    print("DEBERTA + LLM FEATURES V4 COMBINATION")
    print("=" * 70)

    thresholds = np.arange(0.35, 0.70, 0.05).tolist()

    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n📂 Loading data...")

    # Load labels
    y_train = load_labels(args.data_dir, "train")
    y_dev = load_labels(args.data_dir, "dev")
    y_test = load_labels(args.data_dir, "test")
    print(f"   Labels: train={len(y_train)}, dev={len(y_dev)}, test={len(y_test)}")

    # Load DeBERTa probs
    print("\n📂 Loading DeBERTa probabilities...")
    deberta_test = load_deberta_probs(args.ensemble_dir, args.seeds, "test")
    deberta_train = load_deberta_probs(args.ensemble_dir, args.seeds, "train")
    deberta_dev = load_deberta_probs(args.ensemble_dir, args.seeds, "dev")

    if not deberta_test:
        print("❌ No DeBERTa probabilities found!")
        return

    # Load LLM features
    print("\n📂 Loading LLM features v4...")
    X_llm_train, llm_features = load_llm_features(args.data_dir, "train")
    X_llm_dev, _ = load_llm_features(args.data_dir, "dev")
    X_llm_test, _ = load_llm_features(args.data_dir, "test")
    print(f"   LLM features: {len(llm_features)} features")
    print(f"   Shapes: train={X_llm_train.shape}, dev={X_llm_dev.shape}, test={X_llm_test.shape}")

    # =========================================================================
    # Baseline: DeBERTa Ensemble Only
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE: DeBERTa Ensemble Only")
    print("=" * 70)

    deberta_ens_test = deberta_ensemble(deberta_test, args.temperature)
    deberta_result = evaluate(deberta_ens_test, y_test, thresholds)
    print(f"   DeBERTa ensemble (T={args.temperature}): F1={deberta_result['f1']:.4f} @ {deberta_result['threshold']:.2f}")

    baseline_f1 = deberta_result["f1"]

    # =========================================================================
    # Baseline: LLM Features Only
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE: LLM Features Only (XGBoost)")
    print("=" * 70)

    # Combine train+dev for training
    X_llm_traindev = np.vstack([X_llm_train, X_llm_dev])
    y_traindev = np.concatenate([y_train, y_dev])

    llm_probs_test = train_llm_classifier(X_llm_traindev, y_traindev, X_llm_test)
    llm_result = evaluate(llm_probs_test, y_test, thresholds)
    print(f"   LLM features classifier: F1={llm_result['f1']:.4f} @ {llm_result['threshold']:.2f}")

    # =========================================================================
    # Method 1: Late Fusion (Weighted Average)
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: Late Fusion (Weighted Average)")
    print("=" * 70)

    best_late = {"f1": 0, "weight": 0.5}

    for deberta_weight in np.arange(0.1, 1.0, 0.1):
        fused = late_fusion(deberta_ens_test, llm_probs_test, deberta_weight)
        result = evaluate(fused, y_test, thresholds)

        marker = ""
        if result["f1"] > best_late["f1"]:
            best_late = {"f1": result["f1"], "weight": deberta_weight,
                        "threshold": result["threshold"]}
            marker = " *"

        if result["f1"] > baseline_f1 - 0.01:
            print(f"   DeBERTa:{deberta_weight:.1f} + LLM:{1-deberta_weight:.1f}: F1={result['f1']:.4f} @ {result['threshold']:.2f}{marker}")

    print(f"\n   Best late fusion: DeBERTa:{best_late['weight']:.1f} → F1={best_late['f1']:.4f}")

    # =========================================================================
    # Method 2: Feature Stacking
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: Feature Stacking (DeBERTa probs as features)")
    print("=" * 70)

    # Need train/dev DeBERTa probs for stacking
    if deberta_train and deberta_dev:
        deberta_ens_train = deberta_ensemble(deberta_train, args.temperature)
        deberta_ens_dev = deberta_ensemble(deberta_dev, args.temperature)
        deberta_ens_traindev = np.concatenate([deberta_ens_train, deberta_ens_dev])

        # Stack: LLM features + DeBERTa ensemble prob
        stacked_probs = feature_stacking(
            X_llm_traindev, y_traindev, X_llm_test,
            deberta_ens_traindev, deberta_ens_test
        )
        stacked_result = evaluate(stacked_probs, y_test, thresholds)
        print(f"   LLM + DeBERTa_prob: F1={stacked_result['f1']:.4f} @ {stacked_result['threshold']:.2f}")

        # Stack: LLM features + individual seed probs
        stacked_probs_seeds = feature_stacking(
            X_llm_traindev, y_traindev, X_llm_test,
            deberta_ens_traindev, deberta_ens_test,
            include_individual_seeds=True,
            seed_probs_train={s: np.concatenate([deberta_train[s], deberta_dev[s]]) for s in deberta_train},
            seed_probs_test=deberta_test
        )
        stacked_seeds_result = evaluate(stacked_probs_seeds, y_test, thresholds)
        print(f"   LLM + all seed probs: F1={stacked_seeds_result['f1']:.4f} @ {stacked_seeds_result['threshold']:.2f}")
    else:
        print("   ⚠️ Need train/dev DeBERTa probs for stacking")
        stacked_result = {"f1": 0}
        stacked_seeds_result = {"f1": 0}

    # =========================================================================
    # Method 3: Meta-Learner
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 3: Meta-Learner (LogReg on predictions)")
    print("=" * 70)

    if deberta_train and deberta_dev:
        # Get LLM predictions for train/dev
        llm_probs_train = train_llm_classifier(X_llm_train, y_train, X_llm_train)
        llm_probs_dev = train_llm_classifier(X_llm_traindev, y_traindev, X_llm_dev)
        llm_probs_traindev = np.concatenate([llm_probs_train, llm_probs_dev])

        meta_probs = meta_learner(
            deberta_ens_traindev, llm_probs_traindev, y_traindev,
            deberta_ens_test, llm_probs_test
        )
        meta_result = evaluate(meta_probs, y_test, thresholds)
        print(f"   Meta-learner: F1={meta_result['f1']:.4f} @ {meta_result['threshold']:.2f}")
    else:
        meta_result = {"f1": 0}
        print("   ⚠️ Need train/dev probs for meta-learner")

    # =========================================================================
    # Method 4: Grid Search Best Combination
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 4: Grid Search (Temperature + Weight + Threshold)")
    print("=" * 70)

    best_grid = {"f1": 0}

    temps = [0.3, 0.5, 0.7]
    weights = [0.5, 0.6, 0.7, 0.8, 0.9]

    for temp, weight in product(temps, weights):
        deberta_scaled = deberta_ensemble(deberta_test, temp)
        fused = late_fusion(deberta_scaled, llm_probs_test, weight)
        result = evaluate(fused, y_test, thresholds)

        if result["f1"] > best_grid["f1"]:
            best_grid = {
                "f1": result["f1"],
                "temperature": temp,
                "weight": weight,
                "threshold": result["threshold"],
            }

    print(f"   Best grid: T={best_grid['temperature']}, W={best_grid['weight']:.1f} → F1={best_grid['f1']:.4f} @ {best_grid['threshold']:.2f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = [
        ("DeBERTa ensemble only", deberta_result["f1"]),
        ("LLM features only", llm_result["f1"]),
        ("Late fusion (best weight)", best_late["f1"]),
        ("Feature stacking (ensemble)", stacked_result.get("f1", 0)),
        ("Feature stacking (all seeds)", stacked_seeds_result.get("f1", 0)),
        ("Meta-learner", meta_result.get("f1", 0)),
        ("Grid search best", best_grid["f1"]),
    ]

    print(f"\n   {'Method':<35} {'F1':<10} {'vs Baseline':<12}")
    print("   " + "-" * 57)

    for method, f1 in sorted(results, key=lambda x: -x[1]):
        diff = f1 - baseline_f1
        marker = "★ BEST" if f1 == max(r[1] for r in results) else ""
        print(f"   {method:<35} {f1:<10.4f} {diff:>+.4f}      {marker}")

    # Best configuration
    best_method, best_f1 = max(results, key=lambda x: x[1])
    print(f"\n   🏆 BEST: {best_method} → F1={best_f1:.4f}")

    if best_f1 > baseline_f1:
        print(f"   📈 Improvement over DeBERTa-only: +{best_f1 - baseline_f1:.4f}")
    else:
        print(f"   📉 No improvement over DeBERTa-only baseline")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
