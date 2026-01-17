#!/usr/bin/env python3
"""
Fast Grid Search with Parallel Processing

Parallelized version for quick results.

Usage:
    python experiments/scripts/fast_grid_search.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"
EMBEDDING_CACHE_DIR = DATA_DIR / "embedding_cache"

SOTA_F1 = 0.82
SOTA_ACC = 0.905

# =============================================================================
# Feature Definitions (subset of best performers)
# =============================================================================

LLM_SETS = {
    "none": [],
    "scores_3": ["check_score", "verif_score", "harm_score"],
    "interpretable_11": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction",
        "score_variance", "yes_vote_count", "unanimous_yes",
        "harm_spurs_action", "harm_believability",
    ],
    "top_15": [
        "check_score", "check_margin_logit", "check_logit_p_false",
        "check_p_false", "check_margin_p", "check_prediction",
        "verif_score", "verif_logit_p_true", "verif_margin_logit",
        "score_variance", "yes_vote_count", "check_p_uncertain_dominant",
        "harm_spurs_action", "verif_is_argmax_match", "harm_is_argmax_match",
    ],
    "balanced_18": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction",
        "check_margin_p", "verif_margin_p", "harm_margin_p",
        "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
        "score_variance", "yes_vote_count", "unanimous_yes",
        "harm_spurs_action", "harm_believability", "harm_social_fragmentation",
    ],
}

TEXT_SETS = {
    "none": [],
    "top7": ["number_count", "has_precise_number", "has_number", "has_large_scale",
             "avg_word_length", "word_count", "alpha_ratio"],
    "top15": ["number_count", "has_precise_number", "has_number", "has_large_scale",
              "avg_word_length", "word_count", "alpha_ratio",
              "has_first_person_stance", "has_desire_intent", "has_delta",
              "has_number_and_time", "has_future_modal", "has_increase_decrease",
              "has_official_source", "has_voted"],
}

EMBEDDINGS = {
    "none": None,
    "bge-large": "bge-large_embeddings.npz",
    "bge-base": "bge-base_embeddings.npz",
    "openai-large-1024": "openai-large_1024_embeddings.npz",
    "openai-small-512": "openai-small_512_embeddings.npz",
}

# =============================================================================
# Data Loading (called once in main process)
# =============================================================================

def load_all_data():
    """Load all data into memory."""
    # LLM features
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    # CT24 features (text features + labels)
    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    # Labels
    y_train = (ct24_train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (ct24_dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (ct24_test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    # Embeddings
    emb_cache = {}
    for name, fname in EMBEDDINGS.items():
        if fname and (EMBEDDING_CACHE_DIR / fname).exists():
            data = np.load(EMBEDDING_CACHE_DIR / fname)
            emb_cache[name] = (data["train"], data["dev"], data["test"])

    return {
        "llm": (llm_train, llm_dev, llm_test),
        "ct24": (ct24_train, ct24_dev, ct24_test),
        "labels": (y_train, y_dev, y_test),
        "embeddings": emb_cache,
    }


def extract_features(data, llm_set, text_set, emb_name):
    """Extract and combine features for a specific configuration."""
    llm_train, llm_dev, llm_test = data["llm"]
    ct24_train, ct24_dev, ct24_test = data["ct24"]

    parts_train, parts_dev, parts_test = [], [], []

    # LLM features
    if llm_set != "none":
        cols = [c for c in LLM_SETS[llm_set] if c in llm_train.columns]
        if cols:
            parts_train.append(llm_train.select(cols).to_numpy().astype(np.float32))
            parts_dev.append(llm_dev.select(cols).to_numpy().astype(np.float32))
            parts_test.append(llm_test.select(cols).to_numpy().astype(np.float32))

    # Text features
    if text_set != "none":
        cols = [c for c in TEXT_SETS[text_set] if c in ct24_train.columns]
        if cols:
            parts_train.append(ct24_train.select(cols).cast(pl.Float32).to_numpy())
            parts_dev.append(ct24_dev.select(cols).cast(pl.Float32).to_numpy())
            parts_test.append(ct24_test.select(cols).cast(pl.Float32).to_numpy())

    # Embeddings
    if emb_name != "none" and emb_name in data["embeddings"]:
        emb_tr, emb_dv, emb_te = data["embeddings"][emb_name]
        parts_train.append(emb_tr)
        parts_dev.append(emb_dv)
        parts_test.append(emb_te)

    if not parts_train:
        return None, None, None

    X_train = np.hstack(parts_train)
    X_dev = np.hstack(parts_dev)
    X_test = np.hstack(parts_test)

    return X_train, X_dev, X_test


def run_experiment(config):
    """Run a single experiment. Returns result dict."""
    llm_set, text_set, emb_name, clf_name, data = config

    # Skip invalid
    if llm_set == "none" and text_set == "none" and emb_name == "none":
        return None

    X_train, X_dev, X_test = extract_features(data, llm_set, text_set, emb_name)
    if X_train is None:
        return None

    y_train, y_dev, y_test = data["labels"]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    # Classifier (fast configs)
    if clf_name == "lightgbm":
        clf = LGBMClassifier(n_estimators=100, max_depth=8, learning_rate=0.1,
                             random_state=42, class_weight="balanced", verbose=-1, n_jobs=1)
    elif clf_name == "catboost":
        clf = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1,
                                 random_state=42, auto_class_weights="Balanced", verbose=False)
    else:
        return None

    try:
        clf.fit(X_train_s, y_train)
        y_pred_test = clf.predict(X_test_s)
        y_pred_dev = clf.predict(X_dev_s)

        return {
            "llm": llm_set,
            "text": text_set,
            "emb": emb_name,
            "clf": clf_name,
            "dims": X_train.shape[1],
            "dev_f1": f1_score(y_dev, y_pred_dev),
            "test_f1": f1_score(y_test, y_pred_test),
            "test_acc": accuracy_score(y_test, y_pred_test),
            "test_p": precision_score(y_test, y_pred_test, zero_division=0),
            "test_r": recall_score(y_test, y_pred_test, zero_division=0),
        }
    except Exception as e:
        return {"error": str(e), "llm": llm_set, "text": text_set, "emb": emb_name}


def main():
    print("=" * 100)
    print("FAST GRID SEARCH (Optimized)")
    print("=" * 100)
    print(f"SOTA: F1={SOTA_F1}, Acc={SOTA_ACC}")

    # Load data once
    print("\nLoading data...")
    data = load_all_data()
    print(f"  Train: {len(data['labels'][0])}")
    print(f"  Embeddings loaded: {list(data['embeddings'].keys())}")

    # Generate configs
    classifiers = []
    if HAS_LIGHTGBM:
        classifiers.append("lightgbm")
    if HAS_CATBOOST:
        classifiers.append("catboost")

    configs = []
    for llm in LLM_SETS.keys():
        for txt in TEXT_SETS.keys():
            for emb in EMBEDDINGS.keys():
                if llm == "none" and txt == "none" and emb == "none":
                    continue
                for clf in classifiers:
                    configs.append((llm, txt, emb, clf, data))

    print(f"\nTotal experiments: {len(configs)}")
    print("Running...\n")

    # Run sequentially (parallel has overhead for small jobs)
    results = []
    for i, cfg in enumerate(configs):
        result = run_experiment(cfg)
        if result and "error" not in result:
            results.append(result)
            if (i + 1) % 20 == 0 or result["test_f1"] >= 0.75:
                marker = "🔥" if result["test_f1"] >= 0.75 else ""
                print(f"  [{i+1}/{len(configs)}] {result['llm']}+{result['text']}+{result['emb']}+{result['clf']}: "
                      f"F1={result['test_f1']:.3f} {marker}")

    # Sort and display
    results.sort(key=lambda x: x["test_f1"], reverse=True)

    print("\n" + "=" * 100)
    print("TOP 30 RESULTS")
    print("=" * 100)

    print(f"\n{'#':<3} {'LLM':<14} {'Text':<6} {'Embed':<18} {'Clf':<10} {'Dims':<6} {'DevF1':<7} {'F1':<7} {'Acc':<7} {'P':<6} {'R':<6}")
    print("-" * 100)

    for i, r in enumerate(results[:30], 1):
        sota = "✅" if r["test_f1"] >= SOTA_F1 else ""
        print(f"{i:<3} {r['llm']:<14} {r['text']:<6} {r['emb']:<18} {r['clf']:<10} "
              f"{r['dims']:<6} {r['dev_f1']:<7.3f} {r['test_f1']:<7.3f} {r['test_acc']:<7.3f} "
              f"{r['test_p']:<6.3f} {r['test_r']:<6.3f} {sota}")

    # Best
    if results:
        best = results[0]
        print(f"\n{'='*100}")
        print(f"🏆 BEST: {best['llm']} + {best['text']} + {best['emb']} + {best['clf']}")
        print(f"   F1: {best['test_f1']:.4f} (Δ SOTA: {best['test_f1']-SOTA_F1:+.4f})")
        print(f"   Acc: {best['test_acc']:.4f} (Δ SOTA: {best['test_acc']-SOTA_ACC:+.4f})")

        if best["test_f1"] >= SOTA_F1:
            print("\n🎉 SOTA F1 ACHIEVED!")
        else:
            print(f"\n❌ Need F1 +{SOTA_F1-best['test_f1']:.3f} to beat SOTA")


if __name__ == "__main__":
    main()
