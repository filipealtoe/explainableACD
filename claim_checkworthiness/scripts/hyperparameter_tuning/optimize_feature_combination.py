#!/usr/bin/env python3
"""
Optimize Feature Combination for Checkworthiness Classification.

Searches over combinations of:
1. DeBERTa CLS embeddings (1024-dim)
2. PCA-reduced embeddings (various dimensions)
3. LLM confidence features (p_yes, entropy, vote_count)
4. Text pattern features (has_number, is_question, etc.)

Trains multiple classifiers and finds the best combination for F1/Accuracy.

Usage:
    python experiments/scripts/optimize_feature_combination.py
    python experiments/scripts/optimize_feature_combination.py --quick  # Fast mode with fewer combinations
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

# Optional imports
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
# Paths (defaults, can be overridden via CLI)
# =============================================================================
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
EMBEDDINGS_DIR = DATA_DIR / "CT24_embeddings"
FEATURES_DIR = DATA_DIR / "CT24_features"
CLEAN_DIR = DATA_DIR / "CT24_clean"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "feature_optimization"

# Global path overrides (set by CLI args)
_PATH_OVERRIDES = {}

# =============================================================================
# Feature Definitions
# =============================================================================

# LLM confidence features (from checkworthiness assessment v4)
# Full set - core features
LLM_FEATURES_CORE = [
    "check_p_yes", "verif_p_yes", "harm_p_yes",
    "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
    "yes_vote_count",
]

# Extended v4 features
LLM_FEATURES_V4_EXTENDED = [
    # Probabilities
    "check_p_yes", "verif_p_yes", "harm_p_yes",
    "check_p_no", "verif_p_no", "harm_p_no",
    # Entropy (normalized)
    "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm",
    # Margins (confidence)
    "check_margin_p", "verif_margin_p", "harm_margin_p",
    # Cross-module agreement
    "yes_vote_count", "unanimous_yes", "unanimous_no",
    "check_verif_agree", "check_harm_agree", "verif_harm_agree",
    # Score differences
    "score_variance", "score_max_diff",
    "check_minus_verif", "check_minus_harm", "verif_minus_harm",
    # Harm sub-scores
    "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness",
]

# Feature subsets for ablation
LLM_FEATURE_SETS = {
    "llm_core": LLM_FEATURES_CORE,  # 7 core features
    "llm_v4_full": LLM_FEATURES_V4_EXTENDED,  # All v4 features
    "llm_p_yes_only": ["check_p_yes", "verif_p_yes", "harm_p_yes"],
    "llm_entropy_only": ["check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm"],
    "llm_margins": ["check_margin_p", "verif_margin_p", "harm_margin_p"],
    "llm_vote_only": ["yes_vote_count"],
    "llm_p_yes_vote": ["check_p_yes", "verif_p_yes", "harm_p_yes", "yes_vote_count"],
    "llm_p_yes_entropy": ["check_p_yes", "verif_p_yes", "harm_p_yes",
                          "check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm"],
    "llm_agreement": ["yes_vote_count", "unanimous_yes", "unanimous_no",
                      "check_verif_agree", "check_harm_agree", "verif_harm_agree"],
    "llm_check_only": ["check_p_yes", "check_entropy_norm", "check_margin_p"],
    "llm_verif_only": ["verif_p_yes", "verif_entropy_norm", "verif_margin_p"],
    "llm_harm_only": ["harm_p_yes", "harm_entropy_norm", "harm_margin_p",
                      "harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness"],
    "llm_check_verif": ["check_p_yes", "verif_p_yes", "check_entropy_norm", "verif_entropy_norm",
                        "check_margin_p", "verif_margin_p"],
    "llm_no_harm": ["check_p_yes", "verif_p_yes", "check_entropy_norm", "verif_entropy_norm",
                    "check_margin_p", "verif_margin_p", "check_verif_agree"],
}

# Default for backward compatibility
LLM_FEATURES = LLM_FEATURES_CORE
LLM_FEATURES_FULL = LLM_FEATURES_V4_EXTENDED

# Text pattern features (high-lift indicators)
TEXT_FEATURES = [
    # Positive indicators
    "has_percentage", "has_dollar", "has_number", "has_precise_number",
    "has_large_scale", "has_source_attribution", "has_specific_year",
    "has_comparative", "has_voted", "has_increase_decrease",
    # Negative indicators
    "is_question", "has_first_person_stance", "has_future_modal",
    "has_hedge", "has_vague_quantifier",
    # Interactions
    "has_number_and_time", "has_number_and_comparative", "has_source_and_number",
    # Metadata
    "word_count", "avg_word_length",
]


@dataclass
class FeatureConfig:
    """Configuration for a feature combination experiment."""
    name: str
    use_embeddings: bool = False
    pca_dim: int | None = None  # None = no PCA, use full embeddings
    use_llm_features: bool = False
    llm_feature_set: str = "llm_core"  # Key into LLM_FEATURE_SETS
    use_text_features: bool = False
    add_llm_interactions: bool = False  # Add interaction features

    def __str__(self):
        parts = []
        if self.use_embeddings:
            if self.pca_dim:
                parts.append(f"PCA({self.pca_dim})")
            else:
                parts.append("Emb(1024)")
        if self.use_llm_features:
            n_feats = len(LLM_FEATURE_SETS.get(self.llm_feature_set, []))
            parts.append(f"LLM({n_feats})")
        if self.use_text_features:
            parts.append("Text(20)")
        if self.add_llm_interactions:
            parts.append("+interactions")
        return " + ".join(parts) if parts else "None"

    def get_llm_features(self) -> list[str]:
        """Get the LLM feature names for this config."""
        return LLM_FEATURE_SETS.get(self.llm_feature_set, LLM_FEATURES_FULL)


def load_split_data(split: str) -> dict[str, Any]:
    """Load all available data for a split."""
    data = {"split": split}

    # Use overridden paths if set
    clean_dir = _PATH_OVERRIDES.get("clean_dir", CLEAN_DIR)
    embeddings_dir = _PATH_OVERRIDES.get("embeddings_dir", EMBEDDINGS_DIR)
    features_dir = _PATH_OVERRIDES.get("features_dir", FEATURES_DIR)

    # Load labels from clean data
    clean_path = clean_dir / f"CT24_{split}_clean.parquet"
    if not clean_path.exists():
        clean_path = clean_dir / f"CT24_{split}_clean.tsv"
        if clean_path.exists():
            df = pl.read_csv(clean_path, separator="\t")
        else:
            raise FileNotFoundError(f"Clean data not found for {split} at {clean_dir}")
    else:
        df = pl.read_parquet(clean_path)

    data["labels"] = np.array([1 if l == "Yes" else 0 for l in df["class_label"].to_list()])
    data["sentence_ids"] = df["Sentence_id"].to_list()
    data["n_samples"] = len(df)

    # Load embeddings
    embedding_name = _PATH_OVERRIDES.get("embedding_name", "deberta")
    emb_path = embeddings_dir / f"CT24_{split}_{embedding_name}_embeddings.npy"
    if emb_path.exists():
        data["embeddings"] = np.load(emb_path)
    else:
        # Fallback to deberta naming convention
        fallback_path = embeddings_dir / f"CT24_{split}_deberta_embeddings.npy"
        if fallback_path.exists():
            data["embeddings"] = np.load(fallback_path)
        else:
            data["embeddings"] = None

    # Load LLM features from v4 directory first, then fall back to regular features
    llm_features_dir = _PATH_OVERRIDES.get("llm_features_dir", None)
    if llm_features_dir is None:
        # Try CT24_llm_features_v4 first
        llm_v4_dir = clean_dir.parent / "CT24_llm_features_v4"
        if llm_v4_dir.exists():
            llm_features_dir = llm_v4_dir
        else:
            llm_features_dir = features_dir

    # Load LLM features
    llm_path = llm_features_dir / f"{split}_llm_features.parquet"
    if not llm_path.exists():
        llm_path = llm_features_dir / f"CT24_{split}_llm_features.parquet"
    if not llm_path.exists():
        llm_path = features_dir / f"CT24_{split}_features.parquet"

    if llm_path.exists():
        llm_df = pl.read_parquet(llm_path)

        # CRITICAL: Join on sentence_id to align features with labels
        # The sentence_id column might be named differently
        llm_id_col = None
        for id_col in ["sentence_id", "Sentence_id", "id"]:
            if id_col in llm_df.columns:
                llm_id_col = id_col
                break

        if llm_id_col:
            # Create a mapping from sentence_id to row index in clean data
            sentence_ids = data["sentence_ids"]

            # Rename to match if needed
            if llm_id_col != "Sentence_id":
                llm_df = llm_df.rename({llm_id_col: "Sentence_id"})

            # Join with the clean data order (df has the correct order)
            # We need to reorder llm_df to match df's sentence_id order
            llm_df = llm_df.with_columns(pl.col("Sentence_id").cast(pl.Utf8))

            # Create ordering dataframe
            order_df = pl.DataFrame({
                "Sentence_id": [str(sid) for sid in sentence_ids],
                "_order": list(range(len(sentence_ids)))
            })

            # Join and sort to match original order
            llm_df = order_df.join(llm_df, on="Sentence_id", how="left").sort("_order")

            # Verify alignment
            if len(llm_df) != len(sentence_ids):
                print(f"   ⚠️ LLM features alignment issue: {len(llm_df)} vs {len(sentence_ids)} samples")

        # Get all available LLM features (check all feature sets)
        all_llm_features = set()
        for feat_list in LLM_FEATURE_SETS.values():
            all_llm_features.update(feat_list)

        available_llm = [c for c in all_llm_features if c in llm_df.columns]
        if available_llm:
            data["llm_features"] = llm_df.select(available_llm).to_numpy().astype(np.float32)
            data["llm_feature_names"] = available_llm
        else:
            data["llm_features"] = None
            data["llm_feature_names"] = []
    else:
        data["llm_features"] = None
        data["llm_feature_names"] = []

    # Load text features from CT24_features
    feat_path = features_dir / f"CT24_{split}_features.parquet"
    if feat_path.exists():
        feat_df = pl.read_parquet(feat_path)

        # CRITICAL: Join on sentence_id to align features with labels
        feat_id_col = None
        for id_col in ["Sentence_id", "sentence_id", "id"]:
            if id_col in feat_df.columns:
                feat_id_col = id_col
                break

        if feat_id_col:
            sentence_ids = data["sentence_ids"]

            if feat_id_col != "Sentence_id":
                feat_df = feat_df.rename({feat_id_col: "Sentence_id"})

            feat_df = feat_df.with_columns(pl.col("Sentence_id").cast(pl.Utf8))

            order_df = pl.DataFrame({
                "Sentence_id": [str(sid) for sid in sentence_ids],
                "_order": list(range(len(sentence_ids)))
            })

            feat_df = order_df.join(feat_df, on="Sentence_id", how="left").sort("_order")

        # Text features
        available_text = [c for c in TEXT_FEATURES if c in feat_df.columns]
        if available_text:
            data["text_features"] = feat_df.select(available_text).to_numpy().astype(np.float32)
            data["text_feature_names"] = available_text
        else:
            data["text_features"] = None
            data["text_feature_names"] = []
    else:
        data["text_features"] = None
        data["text_feature_names"] = []

    return data


def compute_llm_interactions(llm_features: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Compute interaction features from LLM features."""
    interactions = []

    # Get indices for key features
    p_yes_indices = [i for i, n in enumerate(feature_names) if "p_yes" in n]
    entropy_indices = [i for i, n in enumerate(feature_names) if "entropy" in n]

    # Average p_yes across modules
    if len(p_yes_indices) >= 2:
        avg_p_yes = llm_features[:, p_yes_indices].mean(axis=1, keepdims=True)
        interactions.append(avg_p_yes)

        # Std of p_yes (disagreement between modules)
        std_p_yes = llm_features[:, p_yes_indices].std(axis=1, keepdims=True)
        interactions.append(std_p_yes)

        # Min p_yes (weakest signal)
        min_p_yes = llm_features[:, p_yes_indices].min(axis=1, keepdims=True)
        interactions.append(min_p_yes)

        # Max p_yes (strongest signal)
        max_p_yes = llm_features[:, p_yes_indices].max(axis=1, keepdims=True)
        interactions.append(max_p_yes)

    # Average entropy
    if len(entropy_indices) >= 2:
        avg_entropy = llm_features[:, entropy_indices].mean(axis=1, keepdims=True)
        interactions.append(avg_entropy)

        # Max entropy (most uncertain module)
        max_entropy = llm_features[:, entropy_indices].max(axis=1, keepdims=True)
        interactions.append(max_entropy)

    # p_yes * (1 - entropy) for each module (confidence-weighted probability)
    for p_idx, e_idx in zip(p_yes_indices, entropy_indices):
        if p_idx < llm_features.shape[1] and e_idx < llm_features.shape[1]:
            weighted = llm_features[:, p_idx:p_idx+1] * (1 - llm_features[:, e_idx:e_idx+1])
            interactions.append(weighted)

    if interactions:
        return np.hstack(interactions)
    return np.empty((llm_features.shape[0], 0))


def build_feature_matrix(
    data: dict,
    config: FeatureConfig,
    pca_model: PCA | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, PCA | None, StandardScaler | None]:
    """Build feature matrix based on configuration."""
    feature_parts = []

    # Embeddings (with optional PCA)
    if config.use_embeddings and data["embeddings"] is not None:
        emb = data["embeddings"]

        if config.pca_dim is not None:
            if fit:
                pca_model = PCA(n_components=config.pca_dim, random_state=42)
                emb = pca_model.fit_transform(emb)
            else:
                emb = pca_model.transform(emb)

        feature_parts.append(emb)

    # LLM features (with configurable subset)
    if config.use_llm_features and data["llm_features"] is not None:
        # Get the feature subset for this config
        target_features = config.get_llm_features()
        available_features = data["llm_feature_names"]

        # Select only the features we want
        indices = [available_features.index(f) for f in target_features if f in available_features]
        if indices:
            llm_subset = data["llm_features"][:, indices]
            feature_parts.append(llm_subset)

            # Add interaction features if requested
            if config.add_llm_interactions:
                selected_names = [available_features[i] for i in indices]
                interactions = compute_llm_interactions(llm_subset, selected_names)
                if interactions.shape[1] > 0:
                    feature_parts.append(interactions)

    # Text features
    if config.use_text_features and data["text_features"] is not None:
        feature_parts.append(data["text_features"])

    if not feature_parts:
        raise ValueError("No features selected!")

    # Concatenate
    X = np.hstack(feature_parts)

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Scale
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, pca_model, scaler


def get_classifiers(quick: bool = False) -> dict[str, Any]:
    """Get classifiers to try."""
    classifiers = {
        "LogReg": LogisticRegression(
            class_weight={0: 1, 1: 3},
            max_iter=1000,
            random_state=42,
            C=1.0,
        ),
        "LogReg_C10": LogisticRegression(
            class_weight={0: 1, 1: 3},
            max_iter=1000,
            random_state=42,
            C=10.0,
        ),
    }

    # RandomForest - always include
    classifiers["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight={0: 1, 1: 3},
        random_state=42,
        n_jobs=-1,
    )

    # XGBoost - always include if available
    if HAS_XGBOOST:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=3,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        )

    # CatBoost - always include if available
    if HAS_CATBOOST:
        classifiers["CatBoost"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            class_weights={0: 1, 1: 3},
            random_seed=42,
            verbose=False,
        )

    if HAS_LIGHTGBM:
        classifiers["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            class_weight={0: 1, 1: 3},
            random_state=42,
            verbose=-1,
        )

    return classifiers


def optimize_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
    """Find optimal threshold for F1."""
    best_f1, best_thresh = 0, 0.5

    for thresh in np.arange(0.3, 0.7, 0.01):
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1


def evaluate_config(
    config: FeatureConfig,
    train_data: dict,
    eval_splits: dict[str, dict],
    classifiers: dict,
    optimize_thresh: bool = True,
) -> dict[str, Any]:
    """Evaluate a feature configuration with multiple classifiers."""
    results = {
        "config": config.name,
        "config_str": str(config),
        "classifiers": {},
    }

    # Build training features
    try:
        X_train, pca_model, scaler = build_feature_matrix(
            train_data, config, fit=True
        )
    except ValueError as e:
        results["error"] = str(e)
        return results

    y_train = train_data["labels"]

    # Build eval features
    X_eval = {}
    for split_name, split_data in eval_splits.items():
        X_eval[split_name], _, _ = build_feature_matrix(
            split_data, config, pca_model=pca_model, scaler=scaler, fit=False
        )

    results["feature_dim"] = X_train.shape[1]

    # Train and evaluate each classifier
    for clf_name, clf in classifiers.items():
        clf_results = {"splits": {}}

        try:
            # Train
            clf.fit(X_train, y_train)

            # Evaluate on each split
            for split_name, split_data in eval_splits.items():
                X = X_eval[split_name]
                y_true = split_data["labels"]

                # Get probabilities
                if hasattr(clf, "predict_proba"):
                    y_probs = clf.predict_proba(X)[:, 1]
                else:
                    y_probs = clf.decision_function(X)
                    y_probs = 1 / (1 + np.exp(-y_probs))  # Sigmoid

                # Optimize threshold on dev, apply to others
                if optimize_thresh and split_name == "dev":
                    best_thresh, _ = optimize_threshold(y_true, y_probs)
                    clf_results["threshold"] = best_thresh
                else:
                    best_thresh = clf_results.get("threshold", 0.5)

                y_pred = (y_probs >= best_thresh).astype(int)

                clf_results["splits"][split_name] = {
                    "f1": float(f1_score(y_true, y_pred)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred)),
                    "recall": float(recall_score(y_true, y_pred)),
                }

        except Exception as e:
            clf_results["error"] = str(e)

        results["classifiers"][clf_name] = clf_results

    return results


def generate_feature_configs(quick: bool = False) -> list[FeatureConfig]:
    """Generate feature configurations to try."""
    configs = []

    # PCA dimensions to try (including smaller dims that often work well)
    pca_dims = [32, 64, 128] if quick else [8, 16, 32, 64, 128, 256, 512, None]

    # LLM feature sets to try (updated for v4)
    llm_sets = ["llm_core", "llm_p_yes_only", "llm_v4_full"] if quick else list(LLM_FEATURE_SETS.keys())

    # ==========================================================================
    # 1. Embeddings only (with various PCA)
    # ==========================================================================
    for pca_dim in pca_dims:
        name = f"emb_pca{pca_dim}" if pca_dim else "emb_full"
        configs.append(FeatureConfig(
            name=name,
            use_embeddings=True,
            pca_dim=pca_dim,
        ))

    # ==========================================================================
    # 2. LLM features only (all variations)
    # ==========================================================================
    for llm_set in llm_sets:
        configs.append(FeatureConfig(
            name=llm_set,
            use_llm_features=True,
            llm_feature_set=llm_set,
        ))
        # Also try with interactions for core and v4 full sets
        if llm_set in ["llm_core", "llm_v4_full"]:
            configs.append(FeatureConfig(
                name=f"{llm_set}_interact",
                use_llm_features=True,
                llm_feature_set=llm_set,
                add_llm_interactions=True,
            ))

    # ==========================================================================
    # 3. Text features only
    # ==========================================================================
    configs.append(FeatureConfig(
        name="text_only",
        use_text_features=True,
    ))

    # ==========================================================================
    # 4. LLM + Text features (various LLM subsets)
    # ==========================================================================
    for llm_set in ["llm_core", "llm_v4_full", "llm_p_yes_vote"]:
        configs.append(FeatureConfig(
            name=f"{llm_set}_text",
            use_llm_features=True,
            llm_feature_set=llm_set,
            use_text_features=True,
        ))

    # ==========================================================================
    # 5. Embeddings + LLM (with various PCA and LLM subsets)
    # ==========================================================================
    # Full search: try key PCA dims with key LLM sets
    key_pca_dims = [64, 128, 256] if quick else [64, 128, 256, None]
    key_llm_sets = ["llm_core", "llm_v4_full", "llm_p_yes_vote", "llm_check_verif"]

    for pca_dim in key_pca_dims:
        for llm_set in key_llm_sets:
            name = f"emb_pca{pca_dim}_{llm_set}" if pca_dim else f"emb_full_{llm_set}"
            configs.append(FeatureConfig(
                name=name,
                use_embeddings=True,
                pca_dim=pca_dim,
                use_llm_features=True,
                llm_feature_set=llm_set,
            ))

    # Also try with interactions for best PCA dims
    for pca_dim in [128, 256]:
        configs.append(FeatureConfig(
            name=f"emb_pca{pca_dim}_llm_v4_full_interact",
            use_embeddings=True,
            pca_dim=pca_dim,
            use_llm_features=True,
            llm_feature_set="llm_v4_full",
            add_llm_interactions=True,
        ))

    # ==========================================================================
    # 6. Embeddings + Text (with various PCA)
    # ==========================================================================
    for pca_dim in [64, 128, 256]:
        name = f"emb_pca{pca_dim}_text"
        configs.append(FeatureConfig(
            name=name,
            use_embeddings=True,
            pca_dim=pca_dim,
            use_text_features=True,
        ))

    # ==========================================================================
    # 7. Kitchen sink: Embeddings + LLM + Text
    # ==========================================================================
    for pca_dim in [64, 128, 256]:
        for llm_set in ["llm_core", "llm_v4_full", "llm_p_yes_vote"]:
            name = f"emb_pca{pca_dim}_{llm_set}_text"
            configs.append(FeatureConfig(
                name=name,
                use_embeddings=True,
                pca_dim=pca_dim,
                use_llm_features=True,
                llm_feature_set=llm_set,
                use_text_features=True,
            ))

    # With interactions
    configs.append(FeatureConfig(
        name="emb_pca128_llm_v4_full_interact_text",
        use_embeddings=True,
        pca_dim=128,
        use_llm_features=True,
        llm_feature_set="llm_v4_full",
        add_llm_interactions=True,
        use_text_features=True,
    ))

    return configs


def main():
    parser = argparse.ArgumentParser(description="Optimize feature combinations")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer combinations")
    parser.add_argument("--eval-splits", nargs="+", default=["dev", "dev-test", "test"],
                        help="Splits to evaluate on")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Base data directory containing CT24_clean, CT24_features, CT24_embeddings")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results")
    parser.add_argument("--embedding-name", type=str, default="deberta",
                        help="Embedding model name (e.g., 'deberta', 'bge-large-en-v1.5')")
    args = parser.parse_args()

    # Set path overrides if provided
    if args.data_dir:
        data_dir = Path(args.data_dir)
        _PATH_OVERRIDES["clean_dir"] = data_dir / "CT24_clean"
        _PATH_OVERRIDES["features_dir"] = data_dir / "CT24_features"
        _PATH_OVERRIDES["embeddings_dir"] = data_dir / "CT24_embeddings"
        print(f"Using data directory: {data_dir}")

    # Set embedding name
    _PATH_OVERRIDES["embedding_name"] = args.embedding_name
    print(f"Using embedding model: {args.embedding_name}")

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    print("=" * 80)
    print("FEATURE COMBINATION OPTIMIZATION")
    print("=" * 80)

    # ==========================================================================
    # Load data
    # ==========================================================================
    print("\n📦 Loading data...")

    train_data = load_split_data("train")
    print(f"   Train: {train_data['n_samples']} samples")
    print(f"   Embeddings: {'✓' if train_data['embeddings'] is not None else '✗'}")
    print(f"   LLM features: {len(train_data['llm_feature_names'])} ({train_data['llm_feature_names']})")
    print(f"   Text features: {len(train_data['text_feature_names'])}")

    eval_splits = {}
    for split in args.eval_splits:
        try:
            eval_splits[split] = load_split_data(split)
            print(f"   {split}: {eval_splits[split]['n_samples']} samples")
        except FileNotFoundError:
            print(f"   {split}: ✗ (not found)")

    # ==========================================================================
    # Generate configurations
    # ==========================================================================
    configs = generate_feature_configs(quick=args.quick)
    print(f"\n🔧 Testing {len(configs)} feature configurations...")

    # ==========================================================================
    # Get classifiers
    # ==========================================================================
    classifiers = get_classifiers(quick=args.quick)
    print(f"   With {len(classifiers)} classifiers: {list(classifiers.keys())}")

    # ==========================================================================
    # Run experiments
    # ==========================================================================
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)

    all_results = []

    for config in tqdm(configs, desc="Configurations"):
        result = evaluate_config(config, train_data, eval_splits, classifiers)
        all_results.append(result)

        # Print progress
        if "error" not in result:
            best_clf = max(
                result["classifiers"].items(),
                key=lambda x: x[1].get("splits", {}).get("test", {}).get("f1", 0)
            )
            best_f1 = best_clf[1].get("splits", {}).get("test", {}).get("f1", 0)
            tqdm.write(f"   {config.name}: best={best_clf[0]}, test F1={best_f1:.4f}")

    # ==========================================================================
    # Analyze results
    # ==========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Flatten results for ranking
    flat_results = []
    for result in all_results:
        if "error" in result:
            continue
        for clf_name, clf_result in result["classifiers"].items():
            if "error" in clf_result:
                continue
            for split_name, metrics in clf_result.get("splits", {}).items():
                flat_results.append({
                    "config": result["config"],
                    "config_str": result["config_str"],
                    "feature_dim": result["feature_dim"],
                    "classifier": clf_name,
                    "split": split_name,
                    "threshold": clf_result.get("threshold", 0.5),
                    **metrics,
                })

    results_df = pl.DataFrame(flat_results)

    # Best on each split
    for split in args.eval_splits:
        if split not in [r["split"] for r in flat_results]:
            continue

        print(f"\n📊 TOP 10 on {split.upper()} (by F1):")
        print("-" * 100)

        split_df = results_df.filter(pl.col("split") == split).sort("f1", descending=True).head(10)

        print(f"{'Rank':<5} {'Config':<25} {'Classifier':<15} {'Dim':<6} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'Thresh':<7}")
        print("-" * 100)

        for i, row in enumerate(split_df.iter_rows(named=True), 1):
            print(f"{i:<5} {row['config']:<25} {row['classifier']:<15} {row['feature_dim']:<6} "
                  f"{row['f1']:.4f}  {row['accuracy']:.4f}  {row['precision']:.4f}  {row['recall']:.4f}  {row['threshold']:.2f}")

    # Best overall (test F1, with dev for reference)
    print(f"\n{'=' * 80}")
    print("BEST CONFIGURATIONS (ranked by test F1)")
    print("=" * 80)

    # Collect scores per config+classifier
    combined_scores = {}
    for r in flat_results:
        key = (r["config"], r["classifier"])
        if key not in combined_scores:
            combined_scores[key] = {"config": r["config"], "config_str": r["config_str"],
                                    "classifier": r["classifier"], "feature_dim": r["feature_dim"],
                                    "splits": {}}
        combined_scores[key]["splits"][r["split"]] = r["f1"]

    # Rank by test F1
    rankings = []
    for key, data in combined_scores.items():
        if "test" in data["splits"]:
            rankings.append({
                **data,
                "test_f1": data["splits"]["test"],
                "dev_f1": data["splits"].get("dev", 0),
            })

    rankings.sort(key=lambda x: -x["test_f1"])

    print(f"\n{'Rank':<5} {'Config':<25} {'Classifier':<15} {'Dim':<6} {'Test F1':<10} {'Dev F1':<10}")
    print("-" * 80)

    for i, r in enumerate(rankings[:15], 1):
        print(f"{i:<5} {r['config']:<25} {r['classifier']:<15} {r['feature_dim']:<6} "
              f"{r['test_f1']:.4f}     {r['dev_f1']:.4f}")

    # ==========================================================================
    # Save results
    # ==========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full results
    results_path = OUTPUT_DIR / "optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Full results: {results_path}")

    # Save rankings
    rankings_path = OUTPUT_DIR / "rankings.json"
    with open(rankings_path, "w") as f:
        json.dump(rankings[:20], f, indent=2)
    print(f"💾 Rankings: {rankings_path}")

    # Save as CSV for easy viewing
    results_df.write_csv(OUTPUT_DIR / "all_results.csv")
    print(f"💾 CSV: {OUTPUT_DIR / 'all_results.csv'}")

    # ==========================================================================
    # Winner summary
    # ==========================================================================
    if rankings:
        winner = rankings[0]
        print(f"\n{'=' * 80}")
        print("🏆 WINNER")
        print("=" * 80)
        print(f"   Configuration: {winner['config_str']}")
        print(f"   Classifier: {winner['classifier']}")
        print(f"   Feature dim: {winner['feature_dim']}")
        print(f"   Test F1: {winner['test_f1']:.4f}")
        print(f"   Dev F1: {winner['dev_f1']:.4f}")

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
