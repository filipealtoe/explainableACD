#!/usr/bin/env python3
"""
Binary Classification with Hyperparameter Tuning, Model Saving, and Testing
Models: XGBoost, LightGBM, CatBoost
Input: TSV file

Based on colleague's script with modifications:
- Added class weights for imbalanced data
- Using F1-positive as optimization metric
- Disabled synthetic data mode by default
"""

import argparse
from abc import ABC, abstractmethod

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =========================
# Configuration
# =========================

RANDOM_STATE = 42
N_TRIALS = 30
CV_FOLDS = 5

FEATURE_COLUMNS = [
    "verifiability_cc",
    "verifiability_logprob",
    "checkability_cc",
    "checkability_logprob",
    "harmpot_cc",
    "harmpot_logprob",
]


# =========================
# Dataset Preprocessing
# =========================


def preprocess_tsv_dataset(
    tsv_path: str,
    label_encoder: LabelEncoder | None = None,
    fit_encoder: bool = True,
    use_synthetic_features: bool = False,  # Changed default to False
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Load and preprocess TSV dataset.

    Parameters
    ----------
    tsv_path : str
        Path to TSV dataset
    label_encoder : LabelEncoder or None
        Encoder for labels
    fit_encoder : bool
        Whether to fit a new label encoder
    use_synthetic_features : bool
        If True, generate synthetic features (for testing only)
        If False, use all columns except the last as real features (default)

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Encoded labels
    label_encoder : LabelEncoder
        Fitted label encoder
    """
    df = pd.read_csv(tsv_path, sep="\t")

    if df.shape[1] < 2:
        raise ValueError("TSV file must have at least two columns (features + label).")

    # -------------------------
    # Labels (last column)
    # -------------------------
    y_raw = df.iloc[:, -1]

    if fit_encoder:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        if label_encoder is None:
            raise ValueError("label_encoder must be provided when fit_encoder=False")
        y = label_encoder.transform(y_raw)

    # -------------------------
    # Features
    # -------------------------
    if use_synthetic_features:
        # Original synthetic behavior (for testing)
        np.random.seed(RANDOM_STATE)
        X = pd.DataFrame(np.random.rand(len(df), 6), columns=FEATURE_COLUMNS)
    else:
        # Use all columns except the label
        X = df.iloc[:, :-1].copy()

        # Ensure numeric data
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isnull().any().any():
            raise ValueError(
                "Non-numeric feature columns detected. "
                "Please encode or clean the dataset before training."
            )

    return X, y, label_encoder


def compute_class_weights(y: np.ndarray) -> tuple[float, dict]:
    """
    Compute class weights for imbalanced binary classification.

    Returns
    -------
    scale_pos_weight : float
        Ratio of negative to positive samples (for XGBoost)
    class_weight_dict : dict
        Dictionary mapping class labels to weights
    """
    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()
    scale_pos_weight = n_negative / n_positive

    # For balanced class weights
    total = len(y)
    class_weight_dict = {0: total / (2 * n_negative), 1: total / (2 * n_positive)}

    return scale_pos_weight, class_weight_dict


# =========================
# Model Builders
# =========================


class ModelBuilder(ABC):
    """Abstract base class for model builders."""

    @abstractmethod
    def build(self, trial: optuna.Trial, scale_pos_weight: float = 1.0) -> object:
        """Build a model with hyperparameters from Optuna trial."""
        pass

    @abstractmethod
    def build_with_params(self, params: dict, scale_pos_weight: float = 1.0) -> object:
        """Build a model with explicit parameters."""
        pass


class XGBoostBuilder(ModelBuilder):
    """XGBoost model builder with class weight support."""

    def build(self, trial: optuna.Trial, scale_pos_weight: float = 1.0) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            scale_pos_weight=scale_pos_weight,  # Class weight for imbalance
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    def build_with_params(self, params: dict, scale_pos_weight: float = 1.0) -> XGBClassifier:
        return XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )


class LightGBMBuilder(ModelBuilder):
    """LightGBM model builder with class weight support."""

    def build(self, trial: optuna.Trial, scale_pos_weight: float = 1.0) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            num_leaves=trial.suggest_int("num_leaves", 16, 128),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            class_weight="balanced",  # Automatic class weight computation
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    def build_with_params(self, params: dict, scale_pos_weight: float = 1.0) -> LGBMClassifier:
        return LGBMClassifier(
            **params,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )


class CatBoostBuilder(ModelBuilder):
    """CatBoost model builder with class weight support."""

    def build(self, trial: optuna.Trial, scale_pos_weight: float = 1.0) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 200, 600),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            auto_class_weights="Balanced",  # Automatic class weight computation
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
        )

    def build_with_params(self, params: dict, scale_pos_weight: float = 1.0) -> CatBoostClassifier:
        return CatBoostClassifier(
            **params,
            auto_class_weights="Balanced",
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
        )


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "XGBoost": XGBoostBuilder(),
    "LightGBM": LightGBMBuilder(),
    "CatBoost": CatBoostBuilder(),
}


# =========================
# Cross-Validation
# =========================


def cross_val_metrics(
    model: object, X: pd.DataFrame, y: np.ndarray
) -> tuple[float, float]:
    """
    Perform stratified k-fold cross-validation and return metrics.

    Returns
    -------
    accuracy : float
        Mean accuracy across folds
    f1 : float
        Mean F1 score (positive class) across folds
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    y_true, y_pred = [], []

    for train_idx, val_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[val_idx])

        y_true.extend(y[val_idx])
        y_pred.extend(preds)

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, pos_label=1)


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    builder: ModelBuilder,
    scale_pos_weight: float,
) -> float:
    """Optuna objective function optimizing F1 score."""
    model = builder.build(trial, scale_pos_weight=scale_pos_weight)
    _, f1 = cross_val_metrics(model, X, y)
    return f1


# =========================
# Training Function
# =========================


def train_best_model(
    train_tsv_path: str,
    model_path: str,
    use_synthetic_features: bool = False,
    n_trials: int = N_TRIALS,
) -> tuple[object, LabelEncoder, dict]:
    """
    Train models with hyperparameter tuning and save the best one.

    Returns
    -------
    best_model : trained model
    label_encoder : fitted label encoder
    results : dict with training statistics
    """
    X, y, label_encoder = preprocess_tsv_dataset(
        train_tsv_path, use_synthetic_features=use_synthetic_features
    )

    # Compute class weights for imbalanced data
    scale_pos_weight, class_weight_dict = compute_class_weights(y)
    print(f"\n📊 Class imbalance detected:")
    print(f"   Negative (0): {(y == 0).sum():,} samples")
    print(f"   Positive (1): {(y == 1).sum():,} samples")
    print(f"   Scale pos weight: {scale_pos_weight:.3f}")

    results = []

    for name, builder in MODEL_REGISTRY.items():
        print(f"\n🔍 Optimizing {name}...")

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X, y, builder, scale_pos_weight),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # Build final model with best params
        model = builder.build(study.best_trial, scale_pos_weight=scale_pos_weight)
        acc, f1 = cross_val_metrics(model, X, y)

        results.append({
            "name": name,
            "accuracy": acc,
            "f1": f1,
            "model": model,
            "best_params": study.best_params,
            "best_trial_value": study.best_value,
        })

        print(f"   Best F1: {f1:.4f}, Accuracy: {acc:.4f}")

    # Sort by F1 (primary) then accuracy (secondary)
    results.sort(key=lambda x: (x["f1"], x["accuracy"]), reverse=True)
    best_result = results[0]
    best_model = best_result["model"]

    # Retrain on full data
    print(f"\n🏆 Best Model: {best_result['name']}")
    print(f"   CV F1: {best_result['f1']:.4f}, CV Accuracy: {best_result['accuracy']:.4f}")

    best_model.fit(X, y)
    joblib.dump((best_model, label_encoder), model_path)
    print(f"   Model saved to: {model_path}")

    return best_model, label_encoder, {
        "best_model_name": best_result["name"],
        "best_params": best_result["best_params"],
        "cv_f1": best_result["f1"],
        "cv_accuracy": best_result["accuracy"],
        "scale_pos_weight": scale_pos_weight,
        "all_results": [
            {"name": r["name"], "f1": r["f1"], "accuracy": r["accuracy"]}
            for r in results
        ],
    }


# =========================
# Testing Function
# =========================


def test_model(
    test_tsv_path: str,
    model_path: str,
    use_synthetic_features: bool = False,
    show_plot: bool = True,
) -> dict:
    """
    Load a trained model and evaluate on test data.

    Returns
    -------
    results : dict with test metrics
    """
    model, label_encoder = joblib.load(model_path)

    X_test, y_test, _ = preprocess_tsv_dataset(
        test_tsv_path,
        label_encoder=label_encoder,
        use_synthetic_features=use_synthetic_features,
        fit_encoder=False,
    )

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    print("\n📋 Classification Report:")
    report = classification_report(y_test, preds, target_names=label_encoder.classes_)
    print(report)

    acc = accuracy_score(y_test, preds)
    f1_pos = f1_score(y_test, preds, pos_label=1)
    f1_macro = f1_score(y_test, preds, average="macro")

    print(f"\n📊 Summary Metrics:")
    print(f"   Accuracy:    {acc:.4f}")
    print(f"   F1-positive: {f1_pos:.4f}")
    print(f"   F1-macro:    {f1_macro:.4f}")

    if show_plot:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            preds,
            display_labels=label_encoder.classes_,
            cmap="Blues",
        )
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return {
        "accuracy": acc,
        "f1_positive": f1_pos,
        "f1_macro": f1_macro,
        "predictions": preds,
        "probabilities": proba,
        "y_true": y_test,
    }


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and/or test a binary classifier for CT24 checkworthiness"
    )

    parser.add_argument(
        "--train-data",
        type=str,
        help="Path to training TSV file",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test TSV file",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to save/load the best model",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "both"],
        required=True,
        help="Run mode: train, test, or both",
    )

    parser.add_argument(
        "--synthetic-data",
        type=int,
        default=0,
        help="Use synthetic features (1) or real features (0, default)",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of Optuna trials (default: {N_TRIALS})",
    )

    args = parser.parse_args()
    use_synthetic = bool(args.synthetic_data)

    if args.mode in ("train", "both"):
        if not args.train_data:
            raise ValueError("--train-data is required for training")
        train_best_model(args.train_data, args.model_path, use_synthetic, args.n_trials)

    if args.mode in ("test", "both"):
        if not args.test_data:
            raise ValueError("--test-data is required for testing")
        test_model(args.test_data, args.model_path, use_synthetic)
