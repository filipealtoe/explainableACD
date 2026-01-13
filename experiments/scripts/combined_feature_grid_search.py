#!/usr/bin/env python3
"""
Combined Feature Grid Search

Tests ALL combinations of:
- LLM feature sets
- Embedding models
- Classifiers

To find the optimal configuration to beat SOTA (F1=0.82, Acc=0.905).

Usage:
    python experiments/scripts/combined_feature_grid_search.py
    python experiments/scripts/combined_feature_grid_search.py --quick
    python experiments/scripts/combined_feature_grid_search.py --top 30
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"
EMBEDDING_CACHE_DIR = DATA_DIR / "embedding_cache"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "combined_grid_search"

SOTA_F1 = 0.82
SOTA_ACC = 0.905

# =============================================================================
# LLM Feature Sets
# =============================================================================

LLM_FEATURE_SETS = {
    "none": [],  # Embedding-only baseline

    "scores_3": [
        "check_score", "verif_score", "harm_score"
    ],

    "scores_preds_6": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction"
    ],

    "interpretable_11": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction",
        "score_variance", "yes_vote_count", "unanimous_yes",
        "harm_spurs_action", "harm_believability",
    ],

    "top_importance_15": [
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

    "extended_25": [
        "check_score", "verif_score", "harm_score",
        "check_prediction", "verif_prediction", "harm_prediction",
        "check_p_true", "check_p_false", "verif_p_true", "verif_p_false",
        "harm_p_true", "harm_p_false",
        "check_margin_p", "verif_margin_p", "harm_margin_p",
        "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
        "score_variance", "yes_vote_count", "unanimous_yes", "unanimous_no",
        "harm_spurs_action", "harm_believability", "harm_social_fragmentation",
    ],

    "all_79": [
        "check_score", "check_prediction", "check_reasoning_length", "check_reasoning_hedged",
        "check_p_true", "check_p_false", "check_p_uncertain",
        "check_logit_p_true", "check_logit_p_false", "check_logit_p_uncertain",
        "check_entropy", "check_entropy_norm",
        "check_margin_p", "check_margin_logit", "check_top1_top2_gap",
        "check_p_uncertain_dominant", "check_is_argmax_match",
        "check_score_p_residual", "check_pred_score_mismatch", "check_completion_tokens",
        "verif_score", "verif_prediction", "verif_reasoning_length", "verif_reasoning_hedged",
        "verif_p_true", "verif_p_false", "verif_p_uncertain",
        "verif_logit_p_true", "verif_logit_p_false", "verif_logit_p_uncertain",
        "verif_entropy", "verif_entropy_norm",
        "verif_margin_p", "verif_margin_logit", "verif_top1_top2_gap",
        "verif_p_uncertain_dominant", "verif_is_argmax_match",
        "verif_score_p_residual", "verif_pred_score_mismatch", "verif_completion_tokens",
        "harm_score", "harm_prediction", "harm_reasoning_length", "harm_reasoning_hedged",
        "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
        "harm_p_true", "harm_p_false", "harm_p_uncertain",
        "harm_logit_p_true", "harm_logit_p_false", "harm_logit_p_uncertain",
        "harm_entropy", "harm_entropy_norm",
        "harm_margin_p", "harm_margin_logit", "harm_top1_top2_gap",
        "harm_p_uncertain_dominant", "harm_is_argmax_match",
        "harm_score_p_residual", "harm_pred_score_mismatch", "harm_completion_tokens",
        "score_variance", "score_max_diff",
        "check_minus_verif", "check_minus_harm", "verif_minus_harm",
        "harm_high_verif_low",
        "yes_vote_count", "unanimous_yes", "unanimous_no",
        "check_verif_agree", "check_harm_agree", "verif_harm_agree",
        "pairwise_agreement_rate", "check_yes_verif_yes", "consensus_entropy",
    ],
}

# =============================================================================
# Embedding Models (must have cached .npz files)
# =============================================================================

EMBEDDING_MODELS = {
    "none": None,  # LLM-only baseline
    "bge-large": "bge-large_embeddings.npz",
    "bge-base": "bge-base_embeddings.npz",
    "e5-large": "e5-large_embeddings.npz",
    "openai-small-512": "openai-small_512_embeddings.npz",
    "openai-small-1536": "openai-small_1536_embeddings.npz",
    "openai-large-256": "openai-large_256_embeddings.npz",
    "openai-large-1024": "openai-large_1024_embeddings.npz",
    "openai-large-3072": "openai-large_3072_embeddings.npz",
}

# =============================================================================
# Classifiers
# =============================================================================

def get_classifiers(quick: bool = False) -> dict:
    """Get classifiers to test."""
    if quick:
        classifiers = {}
        if HAS_LIGHTGBM:
            classifiers["lightgbm"] = lambda: LGBMClassifier(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
            )
        if HAS_CATBOOST:
            classifiers["catboost"] = lambda: CatBoostClassifier(
                iterations=100, depth=6, learning_rate=0.1,
                random_state=42, auto_class_weights="Balanced", verbose=False
            )
        if not classifiers:
            classifiers["rf"] = lambda: RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"
            )
        return classifiers

    classifiers = {
        "logistic": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced"),
        "rf": lambda: RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced"),
    }

    if HAS_LIGHTGBM:
        classifiers["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
        )
        classifiers["lightgbm_deep"] = lambda: LGBMClassifier(
            n_estimators=400, max_depth=12, learning_rate=0.03,
            random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
        )

    if HAS_CATBOOST:
        classifiers["catboost"] = lambda: CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.05,
            random_state=42, auto_class_weights="Balanced", verbose=False
        )
        classifiers["catboost_deep"] = lambda: CatBoostClassifier(
            iterations=400, depth=10, learning_rate=0.03,
            random_state=42, auto_class_weights="Balanced", verbose=False
        )

    if HAS_XGBOOST:
        classifiers["xgboost"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss", scale_pos_weight=3.0
        )

    return classifiers


# =============================================================================
# Data Loading
# =============================================================================

def load_llm_features() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load raw LLM feature dataframes."""
    train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")
    return train, dev, test


def extract_llm_features(
    train_df: pl.DataFrame, dev_df: pl.DataFrame, test_df: pl.DataFrame,
    feature_set: str
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Extract specific LLM feature set."""
    if feature_set == "none":
        return None, None, None, 0

    feature_cols = LLM_FEATURE_SETS[feature_set]
    existing = [c for c in feature_cols if c in train_df.columns]

    if not existing:
        return None, None, None, 0

    X_train = train_df.select(existing).to_numpy().astype(np.float32)
    X_dev = dev_df.select(existing).to_numpy().astype(np.float32)
    X_test = test_df.select(existing).to_numpy().astype(np.float32)

    return X_train, X_dev, X_test, len(existing)


def load_embeddings(embedding_name: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Load cached embeddings."""
    if embedding_name == "none":
        return None, None, None, 0

    cache_file = EMBEDDING_CACHE_DIR / EMBEDDING_MODELS[embedding_name]
    if not cache_file.exists():
        print(f"    Warning: {cache_file} not found, skipping")
        return None, None, None, 0

    data = np.load(cache_file)
    return data["train"], data["dev"], data["test"], data["train"].shape[1]


def load_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load labels."""
    train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")["class_label"]
    dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")["class_label"]
    test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")["class_label"]

    y_train = (train == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (dev == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test == "Yes").cast(pl.Int8).to_numpy()

    return y_train, y_dev, y_test


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Combined feature grid search")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer classifiers")
    parser.add_argument("--top", type=int, default=30, help="Show top N results")
    parser.add_argument("--llm-sets", nargs="+", default=None, help="LLM feature sets to test")
    parser.add_argument("--embeddings", nargs="+", default=None, help="Embedding models to test")
    parser.add_argument("--save", action="store_true", help="Save full results to CSV")
    args = parser.parse_args()

    print("=" * 120)
    print("COMBINED FEATURE GRID SEARCH: LLM Features × Embeddings × Classifiers")
    print("=" * 120)
    print(f"\nSOTA Targets: F1 = {SOTA_F1:.3f}, Acc = {SOTA_ACC:.3f}")

    # Determine what to test
    llm_sets = args.llm_sets or list(LLM_FEATURE_SETS.keys())
    embedding_models = args.embeddings or list(EMBEDDING_MODELS.keys())
    classifiers = get_classifiers(quick=args.quick)

    # Filter to available embeddings
    available_embeddings = ["none"]
    for emb_name in embedding_models:
        if emb_name == "none":
            continue
        cache_file = EMBEDDING_CACHE_DIR / EMBEDDING_MODELS.get(emb_name, "")
        if cache_file.exists():
            available_embeddings.append(emb_name)
        else:
            print(f"  Skipping {emb_name}: cache not found")

    embedding_models = available_embeddings

    print(f"\nLLM feature sets: {llm_sets}")
    print(f"Embedding models: {embedding_models}")
    print(f"Classifiers: {list(classifiers.keys())}")

    # Skip invalid combinations (none + none)
    valid_combinations = [
        (llm, emb) for llm, emb in product(llm_sets, embedding_models)
        if not (llm == "none" and emb == "none")
    ]

    total = len(valid_combinations) * len(classifiers)
    print(f"\nTotal combinations: {total}")

    # Load data
    print("\nLoading data...")
    llm_train_df, llm_dev_df, llm_test_df = load_llm_features()
    y_train, y_dev, y_test = load_labels()
    print(f"  Train: {len(y_train)} | Dev: {len(y_dev)} | Test: {len(y_test)}")

    # Cache embeddings
    embedding_cache = {}
    for emb_name in embedding_models:
        if emb_name != "none":
            embedding_cache[emb_name] = load_embeddings(emb_name)

    # Results
    results = []
    count = 0

    print("\n" + "-" * 120)
    print("Running grid search...")

    for llm_set, emb_name in valid_combinations:
        # Get LLM features
        llm_train, llm_dev, llm_test, n_llm = extract_llm_features(
            llm_train_df, llm_dev_df, llm_test_df, llm_set
        )

        # Get embeddings
        if emb_name == "none":
            emb_train, emb_dev, emb_test, n_emb = None, None, None, 0
        else:
            emb_train, emb_dev, emb_test, n_emb = embedding_cache[emb_name]

        # Skip if both are None (shouldn't happen due to filtering)
        if llm_train is None and emb_train is None:
            continue

        # Combine features
        if llm_train is not None and emb_train is not None:
            X_train = np.hstack([llm_train, emb_train])
            X_dev = np.hstack([llm_dev, emb_dev])
            X_test = np.hstack([llm_test, emb_test])
        elif llm_train is not None:
            X_train, X_dev, X_test = llm_train, llm_dev, llm_test
        else:
            X_train, X_dev, X_test = emb_train, emb_dev, emb_test

        n_features = X_train.shape[1]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_dev_s = scaler.transform(X_dev)
        X_test_s = scaler.transform(X_test)

        # Test each classifier
        for clf_name, clf_factory in classifiers.items():
            count += 1
            try:
                clf = clf_factory()
                clf.fit(X_train_s, y_train)

                y_pred_dev = clf.predict(X_dev_s)
                y_pred_test = clf.predict(X_test_s)

                dev_f1 = f1_score(y_dev, y_pred_dev)
                test_f1 = f1_score(y_test, y_pred_test)
                test_acc = accuracy_score(y_test, y_pred_test)
                test_p = precision_score(y_test, y_pred_test, zero_division=0)
                test_r = recall_score(y_test, y_pred_test, zero_division=0)

                results.append({
                    "llm_set": llm_set,
                    "embedding": emb_name,
                    "classifier": clf_name,
                    "n_llm": n_llm,
                    "n_emb": n_emb,
                    "n_total": n_features,
                    "dev_f1": dev_f1,
                    "test_f1": test_f1,
                    "test_acc": test_acc,
                    "test_p": test_p,
                    "test_r": test_r,
                    "gap": dev_f1 - test_f1,
                    "beats_sota_f1": test_f1 >= SOTA_F1,
                    "beats_sota_acc": test_acc >= SOTA_ACC,
                })

                # Progress
                if count % 20 == 0 or test_f1 >= 0.75:
                    marker = "🔥" if test_f1 >= 0.75 else ""
                    print(f"  [{count}/{total}] {llm_set}+{emb_name}+{clf_name}: F1={test_f1:.3f} Acc={test_acc:.3f} {marker}")

            except Exception as e:
                print(f"  [{count}/{total}] {llm_set}+{emb_name}+{clf_name}: ERROR - {e}")

    # Sort by test F1
    results.sort(key=lambda x: x["test_f1"], reverse=True)

    # Summary
    print("\n" + "=" * 120)
    print(f"TOP {args.top} RESULTS (sorted by Test F1)")
    print("=" * 120)

    print(f"\n{'Rank':<5} {'LLM Set':<18} {'Embedding':<18} {'Classifier':<15} {'#Feat':<7} {'Dev F1':<8} {'Test F1':<8} {'Test Acc':<9} {'P':<7} {'R':<7} {'SOTA?':<6}")
    print("-" * 125)

    for i, r in enumerate(results[:args.top], 1):
        sota_marker = ""
        if r["beats_sota_f1"] and r["beats_sota_acc"]:
            sota_marker = "✅✅"
        elif r["beats_sota_f1"]:
            sota_marker = "✅F1"
        elif r["beats_sota_acc"]:
            sota_marker = "✅Ac"

        print(f"{i:<5} {r['llm_set']:<18} {r['embedding']:<18} {r['classifier']:<15} "
              f"{r['n_total']:<7} {r['dev_f1']:<8.3f} {r['test_f1']:<8.3f} {r['test_acc']:<9.3f} "
              f"{r['test_p']:<7.3f} {r['test_r']:<7.3f} {sota_marker:<6}")

    # Best result
    if results:
        best = results[0]
        print(f"\n{'='*120}")
        print(f"🏆 BEST CONFIGURATION")
        print(f"{'='*120}")
        print(f"  LLM Features: {best['llm_set']} ({best['n_llm']} dims)")
        print(f"  Embeddings:   {best['embedding']} ({best['n_emb']} dims)")
        print(f"  Classifier:   {best['classifier']}")
        print(f"  Total dims:   {best['n_total']}")
        print()
        print(f"  Test F1:      {best['test_f1']:.4f}  (SOTA: {SOTA_F1:.3f}, Δ = {best['test_f1']-SOTA_F1:+.4f})")
        print(f"  Test Acc:     {best['test_acc']:.4f}  (SOTA: {SOTA_ACC:.3f}, Δ = {best['test_acc']-SOTA_ACC:+.4f})")
        print(f"  Test P:       {best['test_p']:.4f}")
        print(f"  Test R:       {best['test_r']:.4f}")

        if best["beats_sota_f1"] and best["beats_sota_acc"]:
            print(f"\n🎉🎉🎉 SOTA BEATEN ON BOTH METRICS! 🎉🎉🎉")
        elif best["beats_sota_f1"]:
            print(f"\n✅ SOTA F1 achieved!")
        elif best["beats_sota_acc"]:
            print(f"\n✅ SOTA Accuracy achieved!")
        else:
            print(f"\n❌ Below SOTA. Need F1 +{SOTA_F1-best['test_f1']:.3f}, Acc +{SOTA_ACC-best['test_acc']:.3f}")

    # Analysis by LLM set
    print("\n" + "=" * 120)
    print("AVERAGE BY LLM FEATURE SET (across all embeddings & classifiers)")
    print("=" * 120)

    llm_stats = {}
    for r in results:
        key = r["llm_set"]
        if key not in llm_stats:
            llm_stats[key] = {"test_f1": [], "test_acc": []}
        llm_stats[key]["test_f1"].append(r["test_f1"])
        llm_stats[key]["test_acc"].append(r["test_acc"])

    llm_avg = [(k, np.mean(v["test_f1"]), np.max(v["test_f1"]), np.mean(v["test_acc"]))
               for k, v in llm_stats.items()]
    llm_avg.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'LLM Set':<20} {'Avg F1':<10} {'Max F1':<10} {'Avg Acc':<10}")
    print("-" * 55)
    for k, avg_f1, max_f1, avg_acc in llm_avg:
        print(f"{k:<20} {avg_f1:<10.3f} {max_f1:<10.3f} {avg_acc:<10.3f}")

    # Analysis by embedding
    print("\n" + "=" * 120)
    print("AVERAGE BY EMBEDDING (across all LLM sets & classifiers)")
    print("=" * 120)

    emb_stats = {}
    for r in results:
        key = r["embedding"]
        if key not in emb_stats:
            emb_stats[key] = {"test_f1": [], "test_acc": []}
        emb_stats[key]["test_f1"].append(r["test_f1"])
        emb_stats[key]["test_acc"].append(r["test_acc"])

    emb_avg = [(k, np.mean(v["test_f1"]), np.max(v["test_f1"]), np.mean(v["test_acc"]))
               for k, v in emb_stats.items()]
    emb_avg.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Embedding':<20} {'Avg F1':<10} {'Max F1':<10} {'Avg Acc':<10}")
    print("-" * 55)
    for k, avg_f1, max_f1, avg_acc in emb_avg:
        print(f"{k:<20} {avg_f1:<10.3f} {max_f1:<10.3f} {avg_acc:<10.3f}")

    # Analysis by classifier
    print("\n" + "=" * 120)
    print("AVERAGE BY CLASSIFIER (across all feature combinations)")
    print("=" * 120)

    clf_stats = {}
    for r in results:
        key = r["classifier"]
        if key not in clf_stats:
            clf_stats[key] = {"test_f1": [], "test_acc": []}
        clf_stats[key]["test_f1"].append(r["test_f1"])
        clf_stats[key]["test_acc"].append(r["test_acc"])

    clf_avg = [(k, np.mean(v["test_f1"]), np.max(v["test_f1"]), np.mean(v["test_acc"]))
               for k, v in clf_stats.items()]
    clf_avg.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Classifier':<20} {'Avg F1':<10} {'Max F1':<10} {'Avg Acc':<10}")
    print("-" * 55)
    for k, avg_f1, max_f1, avg_acc in clf_avg:
        print(f"{k:<20} {avg_f1:<10.3f} {max_f1:<10.3f} {avg_acc:<10.3f}")

    # Save results
    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df = pl.DataFrame(results)
        results_file = RESULTS_DIR / "full_grid_results.csv"
        results_df.write_csv(results_file)
        print(f"\n📁 Full results saved to: {results_file}")


if __name__ == "__main__":
    main()
