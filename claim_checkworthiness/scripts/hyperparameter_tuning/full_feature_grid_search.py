#!/usr/bin/env python3
"""
Full Feature Grid Search

Tests ALL combinations of:
- LLM feature sets (checkability, verifiability, harm scores + derived features)
- Text features (linguistic patterns extracted from claims)
- Embedding models (semantic representations)
- Classifiers

To find the optimal configuration to beat SOTA (F1=0.82, Acc=0.905).

Usage:
    python experiments/scripts/full_feature_grid_search.py
    python experiments/scripts/full_feature_grid_search.py --quick
    python experiments/scripts/full_feature_grid_search.py --top 50 --save
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
RESULTS_DIR = Path(__file__).parent.parent / "results" / "full_grid_search"

SOTA_F1 = 0.82
SOTA_ACC = 0.905

# =============================================================================
# LLM Feature Sets
# =============================================================================

LLM_FEATURE_SETS = {
    "none": [],

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
# Text Feature Sets (linguistic patterns from claims)
# =============================================================================

TEXT_FEATURES_TOP7 = [
    "number_count", "has_precise_number", "has_number", "has_large_scale",
    "avg_word_length", "word_count", "alpha_ratio"
]

TEXT_FEATURES_TOP15 = TEXT_FEATURES_TOP7 + [
    "has_first_person_stance", "has_desire_intent", "has_delta",
    "has_number_and_time", "has_future_modal", "has_increase_decrease",
    "has_official_source", "has_voted"
]

TEXT_FEATURES_ALL = [
    "has_number", "has_precise_number", "has_large_scale", "number_count", "has_range", "has_delta",
    "has_specific_year", "has_relative_time", "has_temporal_anchor",
    "has_source_attribution", "has_evidence_noun", "has_official_source", "has_said_claimed",
    "has_comparative", "has_superlative", "has_ranking",
    "has_increase_decrease", "has_voted", "has_negation_claim",
    "has_first_person_stance", "has_desire_intent", "has_future_modal", "has_hedge", "has_vague_quantifier",
    "has_rhetorical_filler", "has_fact_assertion", "is_question", "has_transcript_artifact",
    "word_count", "avg_word_length", "alpha_ratio",
    "has_number_and_time", "has_number_and_comparative", "has_change_and_time", "has_source_and_number",
]

TEXT_FEATURE_SETS = {
    "none": [],
    "top7": TEXT_FEATURES_TOP7,
    "top15": TEXT_FEATURES_TOP15,
    "all35": TEXT_FEATURES_ALL,
}

# =============================================================================
# Embedding Models
# =============================================================================

EMBEDDING_MODELS = {
    "none": None,
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
                n_estimators=150, max_depth=8, learning_rate=0.08,
                random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
            )
        if HAS_CATBOOST:
            classifiers["catboost"] = lambda: CatBoostClassifier(
                iterations=150, depth=6, learning_rate=0.08,
                random_state=42, auto_class_weights="Balanced", verbose=False
            )
        if not classifiers:
            classifiers["rf"] = lambda: RandomForestClassifier(
                n_estimators=150, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced"
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
        classifiers["lgbm_deep"] = lambda: LGBMClassifier(
            n_estimators=400, max_depth=12, learning_rate=0.03,
            random_state=42, class_weight="balanced", verbose=-1, n_jobs=-1
        )

    if HAS_CATBOOST:
        classifiers["catboost"] = lambda: CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.05,
            random_state=42, auto_class_weights="Balanced", verbose=False
        )
        classifiers["cb_deep"] = lambda: CatBoostClassifier(
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
    """Load LLM feature dataframes."""
    train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")
    return train, dev, test


def load_ct24_features() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load CT24 feature dataframes (for text features and labels)."""
    train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")
    return train, dev, test


def extract_llm_features(
    train_df: pl.DataFrame, dev_df: pl.DataFrame, test_df: pl.DataFrame,
    feature_set: str
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Extract LLM features."""
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


def extract_text_features(
    train_df: pl.DataFrame, dev_df: pl.DataFrame, test_df: pl.DataFrame,
    feature_set: str
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Extract text features."""
    if feature_set == "none":
        return None, None, None, 0

    feature_cols = TEXT_FEATURE_SETS[feature_set]
    existing = [c for c in feature_cols if c in train_df.columns]

    if not existing:
        return None, None, None, 0

    def extract(df, cols):
        features = []
        for col in cols:
            if col in df.columns:
                series = df[col]
                if series.dtype == pl.Boolean:
                    features.append(series.cast(pl.Float32).to_numpy())
                else:
                    features.append(series.cast(pl.Float32).to_numpy())
        return np.column_stack(features) if features else None

    X_train = extract(train_df, existing)
    X_dev = extract(dev_df, existing)
    X_test = extract(test_df, existing)

    return X_train, X_dev, X_test, len(existing)


def load_embeddings(embedding_name: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Load cached embeddings."""
    if embedding_name == "none":
        return None, None, None, 0

    cache_file = EMBEDDING_CACHE_DIR / EMBEDDING_MODELS[embedding_name]
    if not cache_file.exists():
        return None, None, None, 0

    data = np.load(cache_file)
    return data["train"], data["dev"], data["test"], data["train"].shape[1]


def load_labels(train_df: pl.DataFrame, dev_df: pl.DataFrame, test_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract labels."""
    y_train = (train_df["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (dev_df["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test_df["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    return y_train, y_dev, y_test


# =============================================================================
# Feature Combination
# =============================================================================

def combine_features(*feature_arrays) -> np.ndarray | None:
    """Combine multiple feature arrays, skipping None values."""
    valid = [arr for arr in feature_arrays if arr is not None and arr.size > 0]
    if not valid:
        return None
    return np.hstack(valid)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full feature grid search")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer classifiers")
    parser.add_argument("--top", type=int, default=40, help="Show top N results")
    parser.add_argument("--save", action="store_true", help="Save full results to CSV")
    parser.add_argument("--llm-sets", nargs="+", default=None, help="LLM sets to test")
    parser.add_argument("--text-sets", nargs="+", default=None, help="Text feature sets to test")
    parser.add_argument("--embeddings", nargs="+", default=None, help="Embedding models to test")
    args = parser.parse_args()

    print("=" * 130)
    print("FULL FEATURE GRID SEARCH: LLM Features × Text Features × Embeddings × Classifiers")
    print("=" * 130)
    print(f"\nSOTA Targets: F1 = {SOTA_F1:.3f}, Acc = {SOTA_ACC:.3f}")

    # Determine what to test
    llm_sets = args.llm_sets or list(LLM_FEATURE_SETS.keys())
    text_sets = args.text_sets or list(TEXT_FEATURE_SETS.keys())
    embedding_names = args.embeddings or list(EMBEDDING_MODELS.keys())
    classifiers = get_classifiers(quick=args.quick)

    # Filter to available embeddings
    available_embeddings = ["none"]
    for emb_name in embedding_names:
        if emb_name == "none":
            continue
        if EMBEDDING_MODELS.get(emb_name):
            cache_file = EMBEDDING_CACHE_DIR / EMBEDDING_MODELS[emb_name]
            if cache_file.exists():
                available_embeddings.append(emb_name)
            else:
                print(f"  Skipping {emb_name}: cache not found at {cache_file}")

    embedding_names = available_embeddings

    print(f"\nLLM feature sets ({len(llm_sets)}): {llm_sets}")
    print(f"Text feature sets ({len(text_sets)}): {text_sets}")
    print(f"Embedding models ({len(embedding_names)}): {embedding_names}")
    print(f"Classifiers ({len(classifiers)}): {list(classifiers.keys())}")

    # Generate valid combinations (at least one feature type must be non-none)
    all_combos = list(product(llm_sets, text_sets, embedding_names))
    valid_combos = [(llm, txt, emb) for llm, txt, emb in all_combos
                    if not (llm == "none" and txt == "none" and emb == "none")]

    total = len(valid_combos) * len(classifiers)
    print(f"\nValid feature combinations: {len(valid_combos)}")
    print(f"Total experiments: {total}")

    # Load data
    print("\nLoading data...")
    llm_train_df, llm_dev_df, llm_test_df = load_llm_features()
    ct24_train_df, ct24_dev_df, ct24_test_df = load_ct24_features()
    y_train, y_dev, y_test = load_labels(ct24_train_df, ct24_dev_df, ct24_test_df)
    print(f"  Train: {len(y_train)} | Dev: {len(y_dev)} | Test: {len(y_test)}")
    print(f"  Positive rate: {100*y_train.mean():.1f}%")

    # Pre-load embeddings
    print("\nPre-loading embeddings...")
    embedding_cache = {}
    for emb_name in embedding_names:
        if emb_name != "none":
            emb_data = load_embeddings(emb_name)
            if emb_data[0] is not None:
                embedding_cache[emb_name] = emb_data
                print(f"  {emb_name}: {emb_data[3]} dims")

    # Pre-extract text features
    print("\nPre-extracting text features...")
    text_cache = {}
    for txt_set in text_sets:
        if txt_set != "none":
            txt_data = extract_text_features(ct24_train_df, ct24_dev_df, ct24_test_df, txt_set)
            if txt_data[0] is not None:
                text_cache[txt_set] = txt_data
                print(f"  {txt_set}: {txt_data[3]} features")

    # Results
    results = []
    count = 0

    print("\n" + "-" * 130)
    print("Running grid search...")

    for llm_set, txt_set, emb_name in valid_combos:
        # Get LLM features
        llm_train, llm_dev, llm_test, n_llm = extract_llm_features(
            llm_train_df, llm_dev_df, llm_test_df, llm_set
        )

        # Get text features
        if txt_set == "none":
            txt_train, txt_dev, txt_test, n_txt = None, None, None, 0
        else:
            txt_train, txt_dev, txt_test, n_txt = text_cache.get(txt_set, (None, None, None, 0))

        # Get embeddings
        if emb_name == "none":
            emb_train, emb_dev, emb_test, n_emb = None, None, None, 0
        else:
            emb_train, emb_dev, emb_test, n_emb = embedding_cache.get(emb_name, (None, None, None, 0))

        # Combine features
        X_train = combine_features(llm_train, txt_train, emb_train)
        X_dev = combine_features(llm_dev, txt_dev, emb_dev)
        X_test = combine_features(llm_test, txt_test, emb_test)

        if X_train is None:
            continue

        n_total = X_train.shape[1]

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
                    "text_set": txt_set,
                    "embedding": emb_name,
                    "classifier": clf_name,
                    "n_llm": n_llm,
                    "n_txt": n_txt,
                    "n_emb": n_emb,
                    "n_total": n_total,
                    "dev_f1": dev_f1,
                    "test_f1": test_f1,
                    "test_acc": test_acc,
                    "test_p": test_p,
                    "test_r": test_r,
                    "gap": dev_f1 - test_f1,
                    "beats_f1": test_f1 >= SOTA_F1,
                    "beats_acc": test_acc >= SOTA_ACC,
                })

                # Progress
                if count % 50 == 0 or test_f1 >= 0.78:
                    marker = "🔥" if test_f1 >= 0.78 else ""
                    print(f"  [{count}/{total}] {llm_set}+{txt_set}+{emb_name}+{clf_name}: "
                          f"F1={test_f1:.3f} Acc={test_acc:.3f} {marker}")

            except Exception as e:
                print(f"  [{count}/{total}] ERROR: {e}")

    # Sort by test F1
    results.sort(key=lambda x: x["test_f1"], reverse=True)

    # Summary
    print("\n" + "=" * 130)
    print(f"TOP {args.top} RESULTS (sorted by Test F1)")
    print("=" * 130)

    header = f"{'#':<4} {'LLM':<16} {'Text':<8} {'Embed':<16} {'Clf':<12} {'Dims':<6} {'DevF1':<7} {'F1':<7} {'Acc':<7} {'P':<6} {'R':<6} {'SOTA':<6}"
    print(f"\n{header}")
    print("-" * 130)

    for i, r in enumerate(results[:args.top], 1):
        sota = ""
        if r["beats_f1"] and r["beats_acc"]:
            sota = "✅✅"
        elif r["beats_f1"]:
            sota = "✅F1"
        elif r["beats_acc"]:
            sota = "✅Ac"

        print(f"{i:<4} {r['llm_set']:<16} {r['text_set']:<8} {r['embedding']:<16} {r['classifier']:<12} "
              f"{r['n_total']:<6} {r['dev_f1']:<7.3f} {r['test_f1']:<7.3f} {r['test_acc']:<7.3f} "
              f"{r['test_p']:<6.3f} {r['test_r']:<6.3f} {sota:<6}")

    # Best result details
    if results:
        best = results[0]
        print(f"\n{'='*130}")
        print(f"🏆 BEST CONFIGURATION")
        print(f"{'='*130}")
        print(f"  LLM Features:  {best['llm_set']} ({best['n_llm']} dims)")
        print(f"  Text Features: {best['text_set']} ({best['n_txt']} dims)")
        print(f"  Embeddings:    {best['embedding']} ({best['n_emb']} dims)")
        print(f"  Classifier:    {best['classifier']}")
        print(f"  Total dims:    {best['n_total']}")
        print()
        print(f"  Test F1:       {best['test_f1']:.4f}  (SOTA: {SOTA_F1}, Δ = {best['test_f1']-SOTA_F1:+.4f})")
        print(f"  Test Acc:      {best['test_acc']:.4f}  (SOTA: {SOTA_ACC}, Δ = {best['test_acc']-SOTA_ACC:+.4f})")
        print(f"  Test P:        {best['test_p']:.4f}")
        print(f"  Test R:        {best['test_r']:.4f}")

        if best["beats_f1"] and best["beats_acc"]:
            print(f"\n🎉🎉🎉 SOTA BEATEN ON BOTH METRICS! 🎉🎉🎉")
        elif best["beats_f1"]:
            print(f"\n✅ SOTA F1 achieved! Acc needs +{SOTA_ACC-best['test_acc']:.3f}")
        elif best["beats_acc"]:
            print(f"\n✅ SOTA Accuracy achieved! F1 needs +{SOTA_F1-best['test_f1']:.3f}")
        else:
            print(f"\n❌ Below SOTA. Need F1 +{SOTA_F1-best['test_f1']:.3f}, Acc +{SOTA_ACC-best['test_acc']:.3f}")

    # === ANALYSIS SECTIONS ===

    # By LLM set
    print("\n" + "=" * 130)
    print("AVERAGE BY LLM FEATURE SET")
    print("=" * 130)

    stats = {}
    for r in results:
        k = r["llm_set"]
        if k not in stats:
            stats[k] = {"f1": [], "acc": []}
        stats[k]["f1"].append(r["test_f1"])
        stats[k]["acc"].append(r["test_acc"])

    avg = [(k, np.mean(v["f1"]), np.max(v["f1"]), np.mean(v["acc"])) for k, v in stats.items()]
    avg.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'LLM Set':<20} {'Avg F1':<10} {'Max F1':<10} {'Avg Acc':<10}")
    print("-" * 55)
    for k, avgf1, maxf1, avgacc in avg:
        print(f"{k:<20} {avgf1:<10.3f} {maxf1:<10.3f} {avgacc:<10.3f}")

    # By text set
    print("\n" + "=" * 130)
    print("AVERAGE BY TEXT FEATURE SET")
    print("=" * 130)

    stats = {}
    for r in results:
        k = r["text_set"]
        if k not in stats:
            stats[k] = {"f1": [], "acc": []}
        stats[k]["f1"].append(r["test_f1"])
        stats[k]["acc"].append(r["test_acc"])

    avg = [(k, np.mean(v["f1"]), np.max(v["f1"]), np.mean(v["acc"])) for k, v in stats.items()]
    avg.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Text Set':<20} {'Avg F1':<10} {'Max F1':<10} {'Avg Acc':<10}")
    print("-" * 55)
    for k, avgf1, maxf1, avgacc in avg:
        print(f"{k:<20} {avgf1:<10.3f} {maxf1:<10.3f} {avgacc:<10.3f}")

    # By embedding
    print("\n" + "=" * 130)
    print("AVERAGE BY EMBEDDING MODEL")
    print("=" * 130)

    stats = {}
    for r in results:
        k = r["embedding"]
        if k not in stats:
            stats[k] = {"f1": [], "acc": []}
        stats[k]["f1"].append(r["test_f1"])
        stats[k]["acc"].append(r["test_acc"])

    avg = [(k, np.mean(v["f1"]), np.max(v["f1"]), np.mean(v["acc"])) for k, v in stats.items()]
    avg.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Embedding':<20} {'Avg F1':<10} {'Max F1':<10} {'Avg Acc':<10}")
    print("-" * 55)
    for k, avgf1, maxf1, avgacc in avg:
        print(f"{k:<20} {avgf1:<10.3f} {maxf1:<10.3f} {avgacc:<10.3f}")

    # By classifier
    print("\n" + "=" * 130)
    print("AVERAGE BY CLASSIFIER")
    print("=" * 130)

    stats = {}
    for r in results:
        k = r["classifier"]
        if k not in stats:
            stats[k] = {"f1": [], "acc": []}
        stats[k]["f1"].append(r["test_f1"])
        stats[k]["acc"].append(r["test_acc"])

    avg = [(k, np.mean(v["f1"]), np.max(v["f1"]), np.mean(v["acc"])) for k, v in stats.items()]
    avg.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Classifier':<20} {'Avg F1':<10} {'Max F1':<10} {'Avg Acc':<10}")
    print("-" * 55)
    for k, avgf1, maxf1, avgacc in avg:
        print(f"{k:<20} {avgf1:<10.3f} {maxf1:<10.3f} {avgacc:<10.3f}")

    # Save results
    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df = pl.DataFrame(results)
        results_file = RESULTS_DIR / "full_grid_results.csv"
        results_df.write_csv(results_file)
        print(f"\n📁 Full results ({len(results)} rows) saved to: {results_file}")


if __name__ == "__main__":
    main()
