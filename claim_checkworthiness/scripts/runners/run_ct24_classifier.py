#!/usr/bin/env python3
"""
CT24 Checkworthiness Classifier - Full Pipeline Wrapper

This script orchestrates:
1. Data preparation (parquet -> TSV)
2. Model training with Optuna hyperparameter tuning
3. Threshold tuning on dev set for optimal F1
4. Final evaluation on test set
5. MLflow experiment tracking
6. CT24 official format export
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")  # Non-interactive backend

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.ct24_classifier_core import (
    preprocess_tsv_dataset,
    train_best_model,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = REPO_ROOT / "data" / "processed" / "CT24_classifier_ready"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "ct24_classifier"
MLFLOW_URI = f"file://{REPO_ROOT}/mlruns"

# Baseline comparisons (from experiments/results/mixed_baselines)
BASELINES = {
    "Mistral-7b (zero-shot)": {"f1": 0.434, "accuracy": 0.53},
    "Mixtral-8x7b (zero-shot)": {"f1": 0.462, "accuracy": 0.79},
    "GPT-3.5-turbo (zero-shot)": {"f1": 0.571, "accuracy": 0.76},
    "Paper fine-tuned": {"f1": 0.799, "accuracy": 0.889},
}


# =============================================================================
# DATA PREPARATION
# =============================================================================


def ensure_data_prepared(data_dir: Path, prepare_if_missing: bool) -> bool:
    """Check if TSV files exist, prepare them if not."""
    required_files = ["train.tsv", "dev.tsv", "test.tsv"]
    missing = [f for f in required_files if not (data_dir / f).exists()]

    if missing and prepare_if_missing:
        print(f"📦 Preparing data (missing: {missing})...")
        prepare_script = REPO_ROOT / "experiments" / "scripts" / "prepare_ct24_for_classifier.py"
        result = subprocess.run([sys.executable, str(prepare_script)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Data preparation failed:\n{result.stderr}")
            return False
        print(result.stdout)
    elif missing:
        print(f"❌ Missing required files in {data_dir}: {missing}")
        return False

    return all((data_dir / f).exists() for f in required_files)


# =============================================================================
# THRESHOLD TUNING
# =============================================================================


def find_optimal_threshold(
    model: object,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict]:
    """
    Find the optimal decision threshold on dev set to maximize F1.

    Returns
    -------
    best_threshold : float
        Optimal threshold value
    threshold_results : dict
        Metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.80, 0.01)

    y_proba = model.predict_proba(X_dev)[:, 1]

    results = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_dev, y_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_dev, y_pred)
        prec = precision_score(y_dev, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_dev, y_pred, pos_label=1, zero_division=0)
        results.append({
            "threshold": thresh,
            "f1": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
        })

    # Find best by F1
    best_result = max(results, key=lambda x: x["f1"])
    best_threshold = best_result["threshold"]

    print(f"\n🎯 Threshold Tuning on Dev Set:")
    print(f"   Best threshold: {best_threshold:.2f}")
    print(f"   Dev F1: {best_result['f1']:.4f}")
    print(f"   Dev Accuracy: {best_result['accuracy']:.4f}")
    print(f"   Dev Precision: {best_result['precision']:.4f}")
    print(f"   Dev Recall: {best_result['recall']:.4f}")

    return best_threshold, {"all_thresholds": results, "best": best_result}


def plot_threshold_curve(threshold_results: dict, save_path: Path) -> None:
    """Plot F1/Precision/Recall vs threshold."""
    results = threshold_results["all_thresholds"]
    thresholds = [r["threshold"] for r in results]
    f1s = [r["f1"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1s, "b-", label="F1", linewidth=2)
    ax.plot(thresholds, precs, "g--", label="Precision", linewidth=1.5)
    ax.plot(thresholds, recs, "r--", label="Recall", linewidth=1.5)

    best = threshold_results["best"]
    ax.axvline(best["threshold"], color="gray", linestyle=":", alpha=0.7)
    ax.scatter([best["threshold"]], [best["f1"]], color="blue", s=100, zorder=5, label=f"Best F1={best['f1']:.3f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning on Dev Set")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved threshold curve: {save_path}")


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_with_threshold(
    model: object,
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float,
    label_encoder: object,
    split_name: str = "test",
) -> dict:
    """Evaluate model with custom threshold."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    f1_pos = f1_score(y, y_pred, pos_label=1)
    f1_macro = f1_score(y, y_pred, average="macro")
    prec = precision_score(y, y_pred, pos_label=1)
    rec = recall_score(y, y_pred, pos_label=1)

    print(f"\n📊 {split_name.upper()} Results (threshold={threshold:.2f}):")
    print(f"   Accuracy:    {acc:.4f}")
    print(f"   F1-positive: {f1_pos:.4f}")
    print(f"   F1-macro:    {f1_macro:.4f}")
    print(f"   Precision:   {prec:.4f}")
    print(f"   Recall:      {rec:.4f}")

    return {
        "accuracy": acc,
        "f1_positive": f1_pos,
        "f1_macro": f1_macro,
        "precision": prec,
        "recall": rec,
        "threshold": threshold,
        "predictions": y_pred,
        "probabilities": y_proba,
        "y_true": y,
    }


def compare_to_baselines(test_results: dict) -> None:
    """Print comparison to baseline models."""
    print("\n" + "=" * 60)
    print("COMPARISON TO BASELINES")
    print("=" * 60)

    our_f1 = test_results["f1_positive"]
    our_acc = test_results["accuracy"]

    print(f"\n{'Model':<30} {'F1':>8} {'Acc':>8} {'vs Our F1':>12}")
    print("-" * 60)

    for name, metrics in BASELINES.items():
        delta = our_f1 - metrics["f1"]
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        marker = "✅" if delta > 0 else "❌"
        print(f"{name:<30} {metrics['f1']:>8.3f} {metrics['accuracy']:>8.3f} {delta_str:>10} {marker}")

    print("-" * 60)
    print(f"{'Our Model (Gradient Boosting)':<30} {our_f1:>8.3f} {our_acc:>8.3f}")


def export_ct24_format(
    predictions: np.ndarray,
    mapping_path: Path,
    output_path: Path,
    label_encoder: object,
) -> None:
    """Export predictions in CT24 official submission format."""
    # Load sample IDs from mapping file
    mapping_df = pd.read_csv(mapping_path, sep="\t")
    sample_ids = mapping_df["sample_id"].values

    # Convert numeric predictions to labels
    labels = label_encoder.inverse_transform(predictions)

    # Create submission dataframe
    submission_df = pd.DataFrame({"Sentence_id": sample_ids, "Label": labels})

    submission_df.to_csv(output_path, sep="\t", index=False)
    print(f"\n📄 CT24 submission format exported: {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_full_pipeline(
    n_trials: int = 30,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    prepare_data: bool | None = None,
) -> dict:
    """
    Run the complete CT24 classifier pipeline.

    Steps:
    1. Prepare data (if needed)
    2. Train models with Optuna
    3. Tune threshold on dev set
    4. Evaluate on test set
    5. Log to MLflow
    6. Export results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = data_dir or DATA_DIR
    results_dir = results_dir or RESULTS_DIR
    if prepare_data is None:
        prepare_data = data_dir == DATA_DIR

    # Setup directories
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CT24 CHECKWORTHINESS CLASSIFIER PIPELINE")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")

    # Step 1: Ensure data is prepared
    if not ensure_data_prepared(data_dir, prepare_data):
        raise RuntimeError("Data preparation failed")

    # Step 2: Train model
    train_path = data_dir / "train.tsv"
    model_path = results_dir / f"best_model_{timestamp}.joblib"

    print("\n" + "=" * 70)
    print("STEP 1: MODEL TRAINING")
    print("=" * 70)

    model, label_encoder, train_results = train_best_model(
        str(train_path),
        str(model_path),
        use_synthetic_features=False,
        n_trials=n_trials,
    )

    # Step 3: Threshold tuning on dev set
    print("\n" + "=" * 70)
    print("STEP 2: THRESHOLD TUNING ON DEV SET")
    print("=" * 70)

    dev_path = data_dir / "dev.tsv"
    X_dev, y_dev, _ = preprocess_tsv_dataset(
        str(dev_path), label_encoder=label_encoder, fit_encoder=False
    )

    best_threshold, threshold_results = find_optimal_threshold(model, X_dev, y_dev)

    # Plot threshold curve
    threshold_plot_path = results_dir / f"threshold_curve_{timestamp}.png"
    plot_threshold_curve(threshold_results, threshold_plot_path)

    # Step 4: Evaluate on test set
    print("\n" + "=" * 70)
    print("STEP 3: TEST SET EVALUATION")
    print("=" * 70)

    test_path = data_dir / "test.tsv"
    X_test, y_test, _ = preprocess_tsv_dataset(
        str(test_path), label_encoder=label_encoder, fit_encoder=False
    )

    test_results = evaluate_with_threshold(
        model, X_test, y_test, best_threshold, label_encoder, "test"
    )

    # Also evaluate with default threshold for comparison
    print("\n   (For comparison with default threshold=0.50):")
    test_default = evaluate_with_threshold(
        model, X_test, y_test, 0.50, label_encoder, "test@0.50"
    )

    # Compare to baselines
    compare_to_baselines(test_results)

    # Step 5: Save confusion matrix
    cm_path = results_dir / f"confusion_matrix_{timestamp}.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        test_results["y_true"],
        test_results["predictions"],
        display_labels=label_encoder.classes_,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Test Set Confusion Matrix (threshold={best_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n   Saved confusion matrix: {cm_path}")

    # Step 6: Export CT24 format
    mapping_path = data_dir / "test.mapping.tsv"
    ct24_path = results_dir / f"predictions_ct24_{timestamp}.tsv"
    export_ct24_format(test_results["predictions"], mapping_path, ct24_path, label_encoder)

    # Step 7: Save metrics summary
    metrics_summary = {
        "timestamp": timestamp,
        "model": train_results["best_model_name"],
        "best_params": train_results["best_params"],
        "cv_f1": train_results["cv_f1"],
        "cv_accuracy": train_results["cv_accuracy"],
        "threshold_tuned": best_threshold,
        "dev_f1": threshold_results["best"]["f1"],
        "dev_accuracy": threshold_results["best"]["accuracy"],
        "test_f1_positive": test_results["f1_positive"],
        "test_accuracy": test_results["accuracy"],
        "test_precision": test_results["precision"],
        "test_recall": test_results["recall"],
        "test_f1_default_threshold": test_default["f1_positive"],
        "baselines_beaten": [
            name for name, m in BASELINES.items() if test_results["f1_positive"] > m["f1"]
        ],
    }

    metrics_path = results_dir / f"metrics_summary_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n   Saved metrics summary: {metrics_path}")

    # Step 8: Log to MLflow
    print("\n" + "=" * 70)
    print("STEP 4: MLFLOW LOGGING")
    print("=" * 70)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ct24_checkworthiness_classifier")

    with mlflow.start_run(run_name=f"{train_results['best_model_name']}_{timestamp}"):
        # Log parameters
        mlflow.log_param("model_type", train_results["best_model_name"])
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("threshold_tuned", best_threshold)
        mlflow.log_param("scale_pos_weight", train_results["scale_pos_weight"])

        for k, v in train_results["best_params"].items():
            mlflow.log_param(f"hp_{k}", v)

        # Log metrics
        mlflow.log_metric("cv_f1", train_results["cv_f1"])
        mlflow.log_metric("cv_accuracy", train_results["cv_accuracy"])
        mlflow.log_metric("dev_f1", threshold_results["best"]["f1"])
        mlflow.log_metric("dev_accuracy", threshold_results["best"]["accuracy"])
        mlflow.log_metric("test_f1_positive", test_results["f1_positive"])
        mlflow.log_metric("test_accuracy", test_results["accuracy"])
        mlflow.log_metric("test_precision", test_results["precision"])
        mlflow.log_metric("test_recall", test_results["recall"])
        mlflow.log_metric("test_f1_default", test_default["f1_positive"])

        # Log artifacts
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(threshold_plot_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(ct24_path))

        print(f"   Logged to MLflow experiment: ct24_checkworthiness_classifier")

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n🎯 Final Results:")
    print(f"   Model: {train_results['best_model_name']}")
    print(f"   Threshold: {best_threshold:.2f}")
    print(f"   Test F1: {test_results['f1_positive']:.4f}")
    print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"\n   Baselines beaten: {len(metrics_summary['baselines_beaten'])}/{len(BASELINES)}")
    for baseline in metrics_summary["baselines_beaten"]:
        print(f"      ✅ {baseline}")

    return metrics_summary


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CT24 Checkworthiness Classifier Pipeline")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for hyperparameter search (default: 30)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer trials (5) for testing",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with train/dev/test TSVs (default: CT24_classifier_ready)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: experiments/results/ct24_classifier)",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare data if missing (only valid for default CT24_classifier_ready)",
    )
    parser.add_argument(
        "--no-prepare-data",
        action="store_true",
        help="Do not attempt data preparation if files are missing",
    )

    args = parser.parse_args()
    n_trials = 5 if args.quick else args.n_trials

    if args.prepare_data and args.no_prepare_data:
        raise ValueError("Choose only one of --prepare-data or --no-prepare-data")

    prepare_data = None
    if args.prepare_data:
        prepare_data = True
    if args.no_prepare_data:
        prepare_data = False

    results = run_full_pipeline(
        n_trials=n_trials,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        prepare_data=prepare_data,
    )
