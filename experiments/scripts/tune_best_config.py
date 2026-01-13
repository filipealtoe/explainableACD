#!/usr/bin/env python3
"""
Hyperparameter Tuning for Best Configuration

Tunes LightGBM hyperparameters using Optuna with cross-validation.

Best config from grid search:
- LLM: interpretable_11
- Text: none
- Embedding: openai-large-1024

Usage:
    python experiments/scripts/tune_best_config.py
    python experiments/scripts/tune_best_config.py --n-trials 50
    python experiments/scripts/tune_best_config.py --classifier catboost
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
LLM_FEATURES_DIR = DATA_DIR / "CT24_llm_features"
CT24_FEATURES_DIR = DATA_DIR / "CT24_features"
EMBEDDING_CACHE_DIR = DATA_DIR / "embedding_cache"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "tuning"

RANDOM_STATE = 42
CV_FOLDS = 5

SOTA_F1 = 0.82
SOTA_ACC = 0.905

# Best configuration from grid search
LLM_FEATURES = [
    "check_score", "verif_score", "harm_score",
    "check_prediction", "verif_prediction", "harm_prediction",
    "score_variance", "yes_vote_count", "unanimous_yes",
    "harm_spurs_action", "harm_believability",
]

EMBEDDING_FILE = "openai-large_1024_embeddings.npz"


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load and prepare features."""
    # LLM features
    llm_train = pl.read_parquet(LLM_FEATURES_DIR / "train_llm_features.parquet")
    llm_dev = pl.read_parquet(LLM_FEATURES_DIR / "dev_llm_features.parquet")
    llm_test = pl.read_parquet(LLM_FEATURES_DIR / "test_llm_features.parquet")

    existing_cols = [c for c in LLM_FEATURES if c in llm_train.columns]
    X_llm_train = llm_train.select(existing_cols).to_numpy().astype(np.float32)
    X_llm_dev = llm_dev.select(existing_cols).to_numpy().astype(np.float32)
    X_llm_test = llm_test.select(existing_cols).to_numpy().astype(np.float32)

    # Embeddings
    emb_data = np.load(EMBEDDING_CACHE_DIR / EMBEDDING_FILE)
    X_emb_train = emb_data["train"]
    X_emb_dev = emb_data["dev"]
    X_emb_test = emb_data["test"]

    # Combine
    X_train = np.hstack([X_llm_train, X_emb_train])
    X_dev = np.hstack([X_llm_dev, X_emb_dev])
    X_test = np.hstack([X_llm_test, X_emb_test])

    # Labels
    ct24_train = pl.read_parquet(CT24_FEATURES_DIR / "CT24_train_features.parquet")
    ct24_dev = pl.read_parquet(CT24_FEATURES_DIR / "CT24_dev_features.parquet")
    ct24_test = pl.read_parquet(CT24_FEATURES_DIR / "CT24_test_features.parquet")

    y_train = (ct24_train["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_dev = (ct24_dev["class_label"] == "Yes").cast(pl.Int8).to_numpy()
    y_test = (ct24_test["class_label"] == "Yes").cast(pl.Int8).to_numpy()

    return X_train, X_dev, X_test, y_train, y_dev, y_test


# =============================================================================
# Hyperparameter Search Spaces
# =============================================================================

def create_lightgbm(trial) -> LGBMClassifier:
    """Create LightGBM with Optuna-suggested hyperparameters."""
    return LGBMClassifier(
        # Core parameters
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        max_depth=trial.suggest_int("max_depth", 3, 15),
        num_leaves=trial.suggest_int("num_leaves", 16, 256),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

        # Regularization
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),

        # Sampling
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=trial.suggest_int("subsample_freq", 0, 10),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # Class imbalance (24% positive)
        class_weight="balanced",

        # Fixed
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )


def create_catboost(trial) -> CatBoostClassifier:
    """Create CatBoost with Optuna-suggested hyperparameters."""
    return CatBoostClassifier(
        # Core parameters
        iterations=trial.suggest_int("iterations", 100, 800),
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

        # Regularization
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),

        # Sampling
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),

        # Overfitting prevention
        early_stopping_rounds=trial.suggest_int("early_stopping_rounds", 20, 100),

        # Class imbalance
        auto_class_weights="Balanced",

        # Fixed
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )


def create_xgboost(trial) -> XGBClassifier:
    """Create XGBoost with Optuna-suggested hyperparameters."""
    return XGBClassifier(
        # Core parameters
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

        # Regularization
        min_child_weight=trial.suggest_int("min_child_weight", 1, 100),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),

        # Sampling
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),

        # Class imbalance (ratio ~3:1)
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 5.0),

        # Fixed
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )


CLASSIFIER_BUILDERS = {
    "lightgbm": create_lightgbm,
    "catboost": create_catboost,
    "xgboost": create_xgboost,
}


# =============================================================================
# Cross-Validation Objective
# =============================================================================

def cross_val_score(model_fn, X, y, trial) -> float:
    """Run stratified k-fold CV and return mean F1 score."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        model = model_fn(trial)

        # Handle CatBoost early stopping
        if hasattr(model, 'fit') and 'catboost' in str(type(model)).lower():
            model.fit(X_train_cv, y_train_cv, eval_set=(X_val_cv, y_val_cv), verbose=False)
        else:
            model.fit(X_train_cv, y_train_cv)

        y_pred = model.predict(X_val_cv)
        scores.append(f1_score(y_val_cv, y_pred))

    return np.mean(scores)


def create_objective(X_train_scaled, y_train, classifier: str):
    """Create Optuna objective function."""
    model_fn = CLASSIFIER_BUILDERS[classifier]

    def objective(trial):
        return cross_val_score(model_fn, X_train_scaled, y_train, trial)

    return objective


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--classifier", default="lightgbm",
                        choices=["lightgbm", "catboost", "xgboost"],
                        help="Classifier to tune")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds (optional)")
    args = parser.parse_args()

    print("=" * 100)
    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 100)
    print(f"\nClassifier: {args.classifier}")
    print(f"Trials: {args.n_trials}")
    print(f"CV Folds: {CV_FOLDS}")
    print(f"SOTA: F1={SOTA_F1}, Acc={SOTA_ACC}")

    # Load data
    print("\n" + "-" * 100)
    print("Loading data...")
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Dev: {X_dev.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Positive rate: {100*y_train.mean():.1f}%")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    # For tuning, combine train + dev
    X_tune = np.vstack([X_train_s, X_dev_s])
    y_tune = np.concatenate([y_train, y_dev])
    print(f"  Tuning set (train+dev): {len(y_tune)} samples")

    # Run Optuna
    print("\n" + "-" * 100)
    print(f"Running Optuna optimization ({args.n_trials} trials)...")
    print("This may take a while...\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    objective = create_objective(X_tune, y_tune, args.classifier)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Best trial
    print("\n" + "=" * 100)
    print("BEST TRIAL")
    print("=" * 100)
    print(f"\n  CV F1 Score: {study.best_value:.4f}")
    print(f"\n  Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Train final model with best params on full train+dev, evaluate on test
    print("\n" + "-" * 100)
    print("Training final model with best params...")

    best_model = CLASSIFIER_BUILDERS[args.classifier](study.best_trial)
    best_model.fit(X_tune, y_tune)

    # Evaluate on test
    y_pred_test = best_model.predict(X_test_s)

    test_f1 = f1_score(y_test, y_pred_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_p = precision_score(y_test, y_pred_test)
    test_r = recall_score(y_test, y_pred_test)

    print("\n" + "=" * 100)
    print("FINAL TEST RESULTS")
    print("=" * 100)
    print(f"\n  Test F1:        {test_f1:.4f}  (SOTA: {SOTA_F1}, Δ = {test_f1-SOTA_F1:+.4f})")
    print(f"  Test Accuracy:  {test_acc:.4f}  (SOTA: {SOTA_ACC}, Δ = {test_acc-SOTA_ACC:+.4f})")
    print(f"  Test Precision: {test_p:.4f}")
    print(f"  Test Recall:    {test_r:.4f}")

    if test_f1 >= SOTA_F1 and test_acc >= SOTA_ACC:
        print("\n🎉🎉🎉 SOTA BEATEN ON BOTH METRICS! 🎉🎉🎉")
    elif test_f1 >= SOTA_F1:
        print(f"\n✅ SOTA F1 achieved!")
    elif test_acc >= SOTA_ACC:
        print(f"\n✅ SOTA Accuracy achieved!")
    else:
        print(f"\n❌ Below SOTA. Need F1 +{SOTA_F1-test_f1:.3f}, Acc +{SOTA_ACC-test_acc:.3f}")

    # Compare with baseline (no tuning)
    print("\n" + "-" * 100)
    print("BASELINE COMPARISON (default hyperparameters)")

    if args.classifier == "lightgbm":
        baseline = LGBMClassifier(n_estimators=100, max_depth=8, learning_rate=0.1,
                                  class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
    elif args.classifier == "catboost":
        baseline = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1,
                                      auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=False)
    else:
        baseline = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 random_state=RANDOM_STATE, eval_metric="logloss")

    baseline.fit(X_tune, y_tune)
    y_pred_baseline = baseline.predict(X_test_s)

    baseline_f1 = f1_score(y_test, y_pred_baseline)
    baseline_acc = accuracy_score(y_test, y_pred_baseline)

    print(f"\n  Baseline F1:  {baseline_f1:.4f}")
    print(f"  Tuned F1:     {test_f1:.4f}  (improvement: {test_f1-baseline_f1:+.4f})")
    print(f"\n  Baseline Acc: {baseline_acc:.4f}")
    print(f"  Tuned Acc:    {test_acc:.4f}  (improvement: {test_acc-baseline_acc:+.4f})")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "classifier": args.classifier,
        "n_trials": args.n_trials,
        "cv_f1": study.best_value,
        "test_f1": test_f1,
        "test_acc": test_acc,
        "test_p": test_p,
        "test_r": test_r,
        "baseline_f1": baseline_f1,
        "baseline_acc": baseline_acc,
        **study.best_params,
    }

    results_df = pl.DataFrame([results])
    results_file = RESULTS_DIR / f"tuning_{args.classifier}.csv"
    results_df.write_csv(results_file)
    print(f"\n📁 Results saved to: {results_file}")

    # Print best params for copy-paste
    print("\n" + "=" * 100)
    print("COPY-PASTE BEST PARAMS")
    print("=" * 100)

    if args.classifier == "lightgbm":
        print(f"""
LGBMClassifier(
    n_estimators={study.best_params.get('n_estimators')},
    max_depth={study.best_params.get('max_depth')},
    num_leaves={study.best_params.get('num_leaves')},
    learning_rate={study.best_params.get('learning_rate'):.6f},
    min_child_samples={study.best_params.get('min_child_samples')},
    reg_alpha={study.best_params.get('reg_alpha'):.8f},
    reg_lambda={study.best_params.get('reg_lambda'):.8f},
    min_split_gain={study.best_params.get('min_split_gain'):.6f},
    subsample={study.best_params.get('subsample'):.4f},
    subsample_freq={study.best_params.get('subsample_freq')},
    colsample_bytree={study.best_params.get('colsample_bytree'):.4f},
    class_weight="balanced",
    random_state={RANDOM_STATE},
)""")
    elif args.classifier == "catboost":
        print(f"""
CatBoostClassifier(
    iterations={study.best_params.get('iterations')},
    depth={study.best_params.get('depth')},
    learning_rate={study.best_params.get('learning_rate'):.6f},
    l2_leaf_reg={study.best_params.get('l2_leaf_reg'):.8f},
    min_data_in_leaf={study.best_params.get('min_data_in_leaf')},
    subsample={study.best_params.get('subsample'):.4f},
    colsample_bylevel={study.best_params.get('colsample_bylevel'):.4f},
    auto_class_weights="Balanced",
    random_seed={RANDOM_STATE},
)""")


if __name__ == "__main__":
    main()
