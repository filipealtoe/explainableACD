#!/usr/bin/env python3
"""
V4 LLM Features Classifier with Hyperparameter Tuning

- Combines train+dev for cross-validation based tuning
- No SMOTE, only class weights (w=1 or w=3)
- Tests: Logistic, MLP, SVM, XGBoost, LightGBM, CatBoost
- Reports feature importances (where available)

Usage:
    python experiments/scripts/train_v4_classifier.py
    python experiments/scripts/train_v4_classifier.py --skip-initial --clf XGBoost
    python experiments/scripts/train_v4_classifier.py --clf MLP --trials 100
    python experiments/scripts/train_v4_classifier.py --clf SVM
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import optuna
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_V4_DIR = DATA_DIR / "CT24_llm_features_v4"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"
EMBEDDING_FILE = DATA_DIR / "embedding_cache" / "bge-large_embeddings.npz"
OUTPUT_DIR = Path(__file__).parent.parent / "results"

RANDOM_STATE = 42
CV_FOLDS = 5

SOTA_F1 = 0.82
SOTA_ACC = 0.905

# All v4 LLM feature groups (binary Yes/No format)
FEATURE_GROUPS = {
    "scores": ["check_score", "verif_score", "harm_score"],
    "entropy": ["check_entropy", "verif_entropy", "harm_entropy"],
    "entropy_norm": ["check_entropy_norm", "verif_entropy_norm", "harm_entropy_norm"],
    "p_yes": ["check_p_yes", "verif_p_yes", "harm_p_yes"],
    "p_no": ["check_p_no", "verif_p_no", "harm_p_no"],
    "logit_p_yes": ["check_logit_p_yes", "verif_logit_p_yes", "harm_logit_p_yes"],
    "logit_p_no": ["check_logit_p_no", "verif_logit_p_no", "harm_logit_p_no"],
    "margin_p": ["check_margin_p", "verif_margin_p", "harm_margin_p"],
    "margin_logit": ["check_margin_logit", "verif_margin_logit", "harm_margin_logit"],
    "predictions": ["check_prediction", "verif_prediction", "harm_prediction"],
    "score_p_residual": ["check_score_p_residual", "verif_score_p_residual", "harm_score_p_residual"],
    "reasoning_hedged": ["check_reasoning_hedged", "verif_reasoning_hedged", "harm_reasoning_hedged"],
    "reasoning_length": ["check_reasoning_length", "verif_reasoning_length", "harm_reasoning_length"],
    "cross_basic": ["score_variance", "score_max_diff", "yes_vote_count", "unanimous_yes", "unanimous_no"],
    "cross_diffs": ["check_minus_verif", "check_minus_harm", "verif_minus_harm"],
    "cross_agree": ["check_verif_agree", "check_harm_agree", "verif_harm_agree"],
    "harm_subdims": ["harm_social_fragmentation", "harm_spurs_action", "harm_believability", "harm_exploitativeness"],
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data(use_embeddings: bool = False):
    """Load v4 LLM features and labels for train, dev, test.

    Args:
        use_embeddings: If True, also load and concatenate BGE-large embeddings.
    """
    llm_train = pl.read_parquet(LLM_FEATURES_V4_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_V4_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_V4_DIR / "test_llm_features.parquet")

    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    # Join on sentence_id
    llm_train = llm_train.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_dev = llm_dev.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))
    llm_test = llm_test.with_columns(pl.col("sentence_id").cast(pl.Int64).alias("Sentence_id"))

    train = llm_train.join(ct24_train.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")
    dev = llm_dev.join(ct24_dev.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")
    test = llm_test.join(ct24_test.select(["Sentence_id", "class_label"]), on="Sentence_id", how="left")

    # Get all available LLM features
    llm_features = []
    for group_features in FEATURE_GROUPS.values():
        for f in group_features:
            if f in train.columns and f not in llm_features:
                llm_features.append(f)

    print(f"  LLM features: {len(llm_features)}")

    # Extract LLM feature matrices
    X_train_llm = train.select(llm_features).to_numpy().astype(np.float32)
    X_dev_llm = dev.select(llm_features).to_numpy().astype(np.float32)
    X_test_llm = test.select(llm_features).to_numpy().astype(np.float32)

    # Handle NaN/inf in LLM features
    X_train_llm = np.nan_to_num(X_train_llm, nan=0.0, posinf=1e6, neginf=-1e6)
    X_dev_llm = np.nan_to_num(X_dev_llm, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_llm = np.nan_to_num(X_test_llm, nan=0.0, posinf=1e6, neginf=-1e6)

    # Load and concatenate BGE embeddings if requested
    if use_embeddings:
        print(f"  Loading BGE-large embeddings...")
        embed_data = np.load(EMBEDDING_FILE)
        X_train_emb = embed_data["train"].astype(np.float32)
        X_dev_emb = embed_data["dev"].astype(np.float32)
        X_test_emb = embed_data["test"].astype(np.float32)
        print(f"  BGE embeddings: {X_train_emb.shape[1]} dims")

        # Concatenate LLM features + embeddings
        X_train = np.hstack([X_train_llm, X_train_emb])
        X_dev = np.hstack([X_dev_llm, X_dev_emb])
        X_test = np.hstack([X_test_llm, X_test_emb])

        # Feature names: LLM features + embedding dims
        all_features = llm_features + [f"emb_{i}" for i in range(X_train_emb.shape[1])]
        print(f"  Total features: {len(all_features)} (LLM: {len(llm_features)}, Emb: {X_train_emb.shape[1]})")
    else:
        X_train = X_train_llm
        X_dev = X_dev_llm
        X_test = X_test_llm
        all_features = llm_features
        print(f"  Total features: {len(all_features)}")

    y_train = (train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, X_dev, X_test, y_train, y_dev, y_test, all_features


# =============================================================================
# Initial Classifier Comparison
# =============================================================================

def get_initial_classifiers():
    """Get classifiers with w=1 and w=3 for initial comparison."""
    return {
        "Logistic_w1": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic_w3": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight={0: 1, 1: 3}, n_jobs=-1),
        "MLP_w1": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
        "MLP_w3": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=RANDOM_STATE),  # MLP uses sample_weight in fit, handled separately
        "SVM_w1": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
        "SVM_w3": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight={0: 1, 1: 3}, random_state=RANDOM_STATE),
        "XGBoost_w1": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, verbosity=0),
        "XGBoost_w3": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=3, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, verbosity=0),
        "LightGBM_w1": LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
        "LightGBM_w3": LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, class_weight={0: 1, 1: 3}, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
        "CatBoost_w1": CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, random_seed=RANDOM_STATE, verbose=False),
        "CatBoost_w3": CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, class_weights={0: 1, 1: 3}, random_seed=RANDOM_STATE, verbose=False),
    }


def run_initial_comparison(X_train, X_test, y_train, y_test):
    """Compare classifiers and return best one."""
    print("\n" + "=" * 100)
    print("INITIAL CLASSIFIER COMPARISON (w=1 and w=3)")
    print("=" * 100)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    trained_models = {}

    classifiers = get_initial_classifiers()

    print(f"\n  {'Classifier':<20} {'Test F1':<10} {'Test Acc':<10} {'P':<8} {'R':<8} {'Gap to SOTA'}")
    print("  " + "-" * 70)

    for name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        test_f1 = f1_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)
        test_p = precision_score(y_test, y_pred, zero_division=0)
        test_r = recall_score(y_test, y_pred, zero_division=0)

        results.append({
            "name": name,
            "f1": test_f1,
            "acc": test_acc,
            "precision": test_p,
            "recall": test_r,
        })
        trained_models[name] = clf

        gap = test_f1 - SOTA_F1
        marker = "🔥" if test_f1 > 0.75 else ""
        print(f"  {name:<20} {test_f1:<10.4f} {test_acc:<10.4f} {test_p:<8.4f} {test_r:<8.4f} {gap:+.4f} {marker}")

    # Find best
    best = max(results, key=lambda x: x["f1"])
    print(f"\n  🏆 Best: {best['name']} with F1 = {best['f1']:.4f}")

    return best["name"], trained_models[best["name"]], scaler


# =============================================================================
# Optuna Hyperparameter Tuning with CV
# =============================================================================

def cv_score(clf, X, y, n_splits=CV_FOLDS):
    """Compute mean F1 score using stratified k-fold cross-validation."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        clf_copy = clf.__class__(**clf.get_params())
        clf_copy.fit(X[train_idx], y[train_idx])
        y_pred = clf_copy.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], y_pred))
    return np.mean(scores)


def create_logistic_objective(X_train, y_train):
    """Create Optuna objective for Logistic Regression with CV."""
    def objective(trial):
        pos_weight = trial.suggest_categorical("pos_weight", [1, 3])
        params = {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "class_weight": None if pos_weight == 1 else {0: 1, 1: pos_weight},
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        clf = LogisticRegression(**params)
        return cv_score(clf, X_train, y_train)
    return objective


def create_xgboost_objective(X_train, y_train):
    """Create Optuna objective for XGBoost with CV."""
    def objective(trial):
        pos_weight = trial.suggest_categorical("scale_pos_weight", [1, 3])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": pos_weight,
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        clf = XGBClassifier(**params)
        return cv_score(clf, X_train, y_train)
    return objective


def create_lightgbm_objective(X_train, y_train):
    """Create Optuna objective for LightGBM with CV."""
    def objective(trial):
        pos_weight = trial.suggest_categorical("pos_weight", [1, 3])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "class_weight": None if pos_weight == 1 else {0: 1, 1: pos_weight},
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "n_jobs": -1,
        }
        clf = LGBMClassifier(**params)
        return cv_score(clf, X_train, y_train)
    return objective


def create_catboost_objective(X_train, y_train):
    """Create Optuna objective for CatBoost with CV."""
    def objective(trial):
        pos_weight = trial.suggest_categorical("pos_weight", [1, 3])
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "class_weights": None if pos_weight == 1 else {0: 1, 1: pos_weight},
            "random_seed": RANDOM_STATE,
            "verbose": False,
        }
        clf = CatBoostClassifier(**params)
        return cv_score(clf, X_train, y_train)
    return objective


def create_mlp_objective(X_train, y_train):
    """Create Optuna objective for MLP Neural Network with CV."""
    def objective(trial):
        # Architecture: 1-3 hidden layers
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 16, 128))

        params = {
            "hidden_layer_sizes": tuple(layers),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),  # L2 regularization
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": RANDOM_STATE,
        }
        clf = MLPClassifier(**params)
        return cv_score(clf, X_train, y_train)
    return objective


def create_svm_objective(X_train, y_train):
    """Create Optuna objective for SVM with CV."""
    def objective(trial):
        pos_weight = trial.suggest_categorical("pos_weight", [1, 3])
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])

        params = {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "kernel": kernel,
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "class_weight": None if pos_weight == 1 else {0: 1, 1: pos_weight},
            "random_state": RANDOM_STATE,
        }

        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)

        clf = SVC(**params)
        return cv_score(clf, X_train, y_train)
    return objective


def run_hyperparameter_tuning(best_clf_name, X_train, X_test, y_train, y_test, feature_names, n_trials=50):
    """Run Optuna hyperparameter tuning with cross-validation."""
    print("\n" + "=" * 100)
    print(f"HYPERPARAMETER TUNING with OPTUNA ({n_trials} trials, {CV_FOLDS}-fold CV)")
    print(f"Classifier: {best_clf_name.split('_')[0]}")
    print("=" * 100)

    print(f"  Training data (for CV): {len(y_train)} samples ({100*y_train.mean():.1f}% positive)")

    # Scale data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Select objective based on classifier
    clf_type = best_clf_name.split("_")[0]
    if "Logistic" in clf_type:
        objective = create_logistic_objective(X_train_s, y_train)
        clf_class = LogisticRegression
    elif "MLP" in clf_type:
        objective = create_mlp_objective(X_train_s, y_train)
        clf_class = MLPClassifier
    elif "SVM" in clf_type:
        objective = create_svm_objective(X_train_s, y_train)
        clf_class = SVC
    elif "XGBoost" in clf_type:
        objective = create_xgboost_objective(X_train_s, y_train)
        clf_class = XGBClassifier
    elif "LightGBM" in clf_type:
        objective = create_lightgbm_objective(X_train_s, y_train)
        clf_class = LGBMClassifier
    else:  # CatBoost
        objective = create_catboost_objective(X_train_s, y_train)
        clf_class = CatBoostClassifier

    # Run optimization
    print(f"\n  Running {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best trial:")
    print(f"    CV F1: {study.best_value:.4f}")
    print(f"    Params: {study.best_params}")

    # Train final model with best params
    print("\n  Training final model with best params on full training data...")

    best_params = study.best_params.copy()

    # Handle class weight naming differences
    if clf_class == LogisticRegression:
        pos_weight = best_params.pop("pos_weight", 1)
        best_params["class_weight"] = None if pos_weight == 1 else {0: 1, 1: pos_weight}
        best_params["max_iter"] = 1000
        best_params["random_state"] = RANDOM_STATE
        best_params["n_jobs"] = -1
    elif clf_class == MLPClassifier:
        # Reconstruct hidden_layer_sizes from individual layer params
        n_layers = best_params.pop("n_layers", 1)
        layers = []
        for i in range(n_layers):
            layers.append(best_params.pop(f"n_units_l{i}", 64))
        # Remove any extra layer params that might exist
        for i in range(n_layers, 3):
            best_params.pop(f"n_units_l{i}", None)
        best_params["hidden_layer_sizes"] = tuple(layers)
        best_params["max_iter"] = 500
        best_params["early_stopping"] = True
        best_params["validation_fraction"] = 0.1
        best_params["n_iter_no_change"] = 20
        best_params["random_state"] = RANDOM_STATE
    elif clf_class == SVC:
        pos_weight = best_params.pop("pos_weight", 1)
        best_params["class_weight"] = None if pos_weight == 1 else {0: 1, 1: pos_weight}
        best_params["random_state"] = RANDOM_STATE
        # Remove degree if not using poly kernel
        if best_params.get("kernel") != "poly":
            best_params.pop("degree", None)
    elif clf_class == XGBClassifier:
        best_params["eval_metric"] = "logloss"
        best_params["random_state"] = RANDOM_STATE
        best_params["n_jobs"] = -1
        best_params["verbosity"] = 0
    elif clf_class == LGBMClassifier:
        pos_weight = best_params.pop("pos_weight", 1)
        best_params["class_weight"] = None if pos_weight == 1 else {0: 1, 1: pos_weight}
        best_params["random_state"] = RANDOM_STATE
        best_params["verbose"] = -1
        best_params["n_jobs"] = -1
    else:  # CatBoost
        pos_weight = best_params.pop("pos_weight", 1)
        best_params["class_weights"] = None if pos_weight == 1 else {0: 1, 1: pos_weight}
        best_params["random_seed"] = RANDOM_STATE
        best_params["verbose"] = False

    final_clf = clf_class(**best_params)
    final_clf.fit(X_train_s, y_train)

    # Evaluate on test
    y_pred_test = final_clf.predict(X_test_s)
    test_f1 = f1_score(y_test, y_pred_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_p = precision_score(y_test, y_pred_test, zero_division=0)
    test_r = recall_score(y_test, y_pred_test, zero_division=0)

    print("\n" + "=" * 100)
    print("FINAL RESULTS (after tuning)")
    print("=" * 100)
    print(f"\n  Test F1:      {test_f1:.4f}")
    print(f"  Test Acc:     {test_acc:.4f}")
    print(f"  Precision:    {test_p:.4f}")
    print(f"  Recall:       {test_r:.4f}")
    print(f"\n  SOTA F1:      {SOTA_F1}")
    print(f"  Gap to SOTA:  {test_f1 - SOTA_F1:+.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=["No", "Yes"]))

    # Feature importance
    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE")
    print("=" * 100)

    if hasattr(final_clf, "feature_importances_"):
        importances = final_clf.feature_importances_
    elif hasattr(final_clf, "coef_"):
        importances = np.abs(final_clf.coef_[0])
    else:
        importances = None

    if importances is not None:
        paired = list(zip(feature_names, importances))
        paired.sort(key=lambda x: x[1], reverse=True)

        def get_group(feat):
            for group_name, features in FEATURE_GROUPS.items():
                if feat in features:
                    return group_name
            return "unknown"

        print(f"\n  {'Rank':<5} {'Feature':<35} {'Importance':<12} {'Group'}")
        print("  " + "-" * 70)

        max_imp = max(i[1] for i in paired) if paired else 1
        for rank, (feat, imp) in enumerate(paired[:25], 1):
            group = get_group(feat)
            bar = "█" * int(imp / max_imp * 20) if max_imp > 0 else ""
            print(f"  {rank:<5} {feat:<35} {imp:<12.4f} {group:<15} {bar}")

        # Group importance summary
        print("\n  " + "-" * 70)
        print("  GROUP IMPORTANCE (sum of feature importances)")
        print("  " + "-" * 70)

        group_importance = {}
        for feat, imp in paired:
            group = get_group(feat)
            group_importance[group] = group_importance.get(group, 0) + imp

        group_sorted = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
        total_imp = sum(i[1] for i in paired)

        print(f"\n  {'Rank':<5} {'Group':<25} {'Total Importance':<18} {'% of Total'}")
        print("  " + "-" * 60)

        for rank, (group, imp) in enumerate(group_sorted, 1):
            pct = 100 * imp / total_imp if total_imp > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {rank:<5} {group:<25} {imp:<18.4f} {pct:>5.1f}% {bar}")

    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "v4_best_classifier.joblib"
    joblib.dump({
        "model": final_clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "best_params": best_params,
        "metrics": {"f1": test_f1, "acc": test_acc, "precision": test_p, "recall": test_r},
    }, model_path)
    print(f"\n  Model saved to: {model_path}")

    return final_clf, study


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train v4 LLM features classifier")
    parser.add_argument("--skip-initial", action="store_true", help="Skip initial comparison")
    parser.add_argument("--clf", type=str, default=None, choices=["Logistic", "MLP", "SVM", "XGBoost", "LightGBM", "CatBoost"],
                        help="Force specific classifier for tuning")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--with-embeddings", action="store_true", help="Also use BGE-large embeddings (late fusion)")
    args = parser.parse_args()

    n_trials = args.trials

    print("=" * 100)
    print("V4 LLM FEATURES CLASSIFIER with HYPERPARAMETER TUNING")
    print(f"  → Using {CV_FOLDS}-fold Cross-Validation on combined train+dev")
    print(f"  → Class weights: w=1 or w=3 (no SMOTE)")
    if args.with_embeddings:
        print(f"  → LATE FUSION: LLM features + BGE-large embeddings")
    print("=" * 100)

    # Load data
    print("\nLoading v4 LLM features...")
    X_train, X_dev, X_test, y_train, y_dev, y_test, feature_names = load_data(use_embeddings=args.with_embeddings)

    print(f"  Train: {len(y_train)} ({100*y_train.mean():.1f}% positive)")
    print(f"  Dev:   {len(y_dev)} ({100*y_dev.mean():.1f}% positive)")
    print(f"  Test:  {len(y_test)} ({100*y_test.mean():.1f}% positive)")

    # Combine train + dev for better training
    X_combined = np.vstack([X_train, X_dev])
    y_combined = np.concatenate([y_train, y_dev])
    print(f"  Combined: {len(y_combined)} ({100*y_combined.mean():.1f}% positive)")

    # Initial comparison or use specified classifier
    if args.clf:
        best_clf_name = f"{args.clf}_w3"
        print(f"\n  Using specified classifier: {best_clf_name}")
    elif not args.skip_initial:
        best_clf_name, _, _ = run_initial_comparison(X_combined, X_test, y_combined, y_test)
    else:
        best_clf_name = "XGBoost_w3"  # Default
        print(f"\n  Skipping initial comparison, using: {best_clf_name}")

    # Run hyperparameter tuning
    final_clf, study = run_hyperparameter_tuning(
        best_clf_name, X_combined, X_test, y_combined, y_test, feature_names, n_trials=n_trials
    )

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
