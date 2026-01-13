#!/usr/bin/env python3
"""Mixed Provider Baselines: Mistral (Together AI) + GPT-4 (OpenAI).

Runs checkworthiness experiments with:
- Mistral models via Together AI: mistral-7b, mixtral-8x7b
- OpenAI models: gpt-4-turbo-2024-04-09, gpt-4.1-mini

PRE-REGISTRATION:
- Primary metric: F1-positive (matching CheckThat! 2024)
- Hypothesis: Zero-shot underperforms fine-tuned by 10-30%
- N=100 stratified samples from CT24_test_gold.tsv (configurable)
- T=0.0 (deterministic)

Usage:
    python experiments/scripts/run_mixed_baselines.py
    python experiments/scripts/run_mixed_baselines.py --n-samples 10 --models mistral-7b,gpt-4.1-mini
    python experiments/scripts/run_mixed_baselines.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkworthiness.config import MODELS, ModelConfig, ModelProvider, TokenUsage
from src.checkworthiness.prompting_baseline import PromptingBaseline

# Load environment variables (override=True ensures .env takes precedence over shell env)
load_dotenv(override=True)

# =============================================================================
# Paper Baselines (Target to Compare Against)
# =============================================================================

PAPER_BASELINES: dict[str, dict[str, float]] = {
    # From CheckThat! 2024 paper, Test partition (fine-tuned models)
    "mistral-7b": {"f1": 0.799, "acc": 0.889, "precision": 0.747, "recall": 0.860},
    "mixtral-8x7b": {"f1": 0.807, "acc": 0.891, "precision": 0.741, "recall": 0.886},
    # GPT-4 baselines not in original paper (proprietary models)
    # These will show as "N/A" in comparison table
}

# Models to run in this experiment
# All models have knowledge cutoff BEFORE January 2024 (CT24 dataset release)
# to avoid data contamination
MIXED_MODELS: list[str] = [
    # Mistral family (Together AI) - ~mid-2023 cutoff
    "mistral-7b",
    "mixtral-8x7b",
    # Llama family (Together AI) - various cutoffs
    "llama-3.1-8b",    # ~mid-2023 cutoff
    "llama-3.1-70b",   # ~mid-2023 cutoff
    "llama-3.2-3b",    # ~mid-2023 cutoff
    # OpenAI models
    "gpt-3.5-turbo",   # Sep 2021 cutoff, $0.50-1.50/1M tokens
    "gpt-4.1-mini",    # Dec 2023 cutoff, $0.40-1.60/1M tokens
    "gpt-4.1",         # Dec 2023 cutoff, $2.00-8.00/1M tokens
    "gpt-5.2",         # Latest GPT-5.2, $3.00-12.00/1M tokens
]

# Allowed providers for this experiment
ALLOWED_PROVIDERS: set[ModelProvider] = {
    ModelProvider.TOGETHER_AI,
    ModelProvider.OPENAI,
}

# Rate limiting
MAX_WORKERS = 10  # Concurrent threads (conservative for stability)
RATE_LIMIT_PER_SEC = 8  # Stay well under 600/min


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    sample_id: str
    text: str
    ground_truth: str  # "Yes" or "No"
    model: str
    prediction: str  # "Yes" or "No"
    average_confidence: float  # 0-100: (checkability + verifiability + harm) / 3, BEFORE threshold

    # Per-module SELF-REPORTED confidences (from model's JSON output)
    checkability_self_conf: float | None
    verifiability_self_conf: float | None
    harm_self_conf: float | None

    # Per-module LOGPROB-DERIVED confidences (p_true * 100)
    checkability_logprob_conf: float | None
    verifiability_logprob_conf: float | None
    harm_logprob_conf: float | None

    # Per-module FINAL confidences (logprob if available, else self-reported)
    checkability_confidence: float
    verifiability_confidence: float
    harm_confidence: float

    # Quality flags (from schema)
    json_parse_failed: bool = False
    logprobs_missing: bool = False

    # Metadata
    error: str | None = None
    latency_ms: float = 0.0

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Logprobs data (serialized as JSON string for parquet storage)
    logprobs_checkability: str | None = None
    logprobs_verifiability: str | None = None
    logprobs_harm: str | None = None


@dataclass
class ExperimentResults:
    """Aggregated experiment results."""

    predictions: list[PredictionResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    models_run: list[str] = field(default_factory=list)
    n_samples: int = 0


# =============================================================================
# Data Loading
# =============================================================================


def load_ct24_test_gold(data_path: Path, n_samples: int | None = None) -> pl.DataFrame:
    """Load CT24 test gold dataset.

    Args:
        data_path: Path to CT24_checkworthy_english_test_gold.tsv
        n_samples: Number of samples to use (stratified). None = all.

    Returns:
        DataFrame with columns: Sentence_id, Text, class_label
    """
    df = pl.read_csv(data_path, separator="\t")

    # Rename columns for consistency
    df = df.rename({"Sentence_id": "sample_id", "Text": "text", "class_label": "ground_truth"})

    # Convert sample_id to string
    df = df.with_columns(pl.col("sample_id").cast(pl.Utf8))

    print(f"Loaded {len(df)} samples from {data_path.name}")

    # Class distribution
    yes_count = df.filter(pl.col("ground_truth") == "Yes").height
    no_count = df.filter(pl.col("ground_truth") == "No").height
    print(f"  Class distribution: {yes_count} Yes, {no_count} No")

    if n_samples is not None and n_samples < len(df):
        # Stratified sampling
        yes_ratio = yes_count / len(df)
        n_yes = int(n_samples * yes_ratio)
        n_no = n_samples - n_yes

        yes_samples = df.filter(pl.col("ground_truth") == "Yes").sample(n=n_yes, seed=42)
        no_samples = df.filter(pl.col("ground_truth") == "No").sample(n=n_no, seed=42)

        df = pl.concat([yes_samples, no_samples]).sort("sample_id")
        print(f"  Stratified to {n_samples} samples: {n_yes} Yes, {n_no} No")

    return df


# =============================================================================
# Prediction Functions
# =============================================================================


def run_single_prediction(
    baseline: PromptingBaseline,
    sample: dict,
    model_name: str,
) -> PredictionResult:
    """Run prediction for a single sample.

    Args:
        baseline: The PromptingBaseline instance
        sample: Dict with sample_id, text, ground_truth
        model_name: Name of the model for logging

    Returns:
        PredictionResult with all fields populated including logprobs
    """
    start_time = time.perf_counter()

    try:
        # Run the 3-module pipeline - returns 5 values including logprobs
        result, total_usage, all_logprobs, reasoning_content, reasoning_logprobs = baseline(sample["text"])

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Serialize logprobs as JSON strings for parquet storage
        logprobs_check = json.dumps(all_logprobs.get("checkability", [])) if all_logprobs else None
        logprobs_verif = json.dumps(all_logprobs.get("verifiability", [])) if all_logprobs else None
        logprobs_harm = json.dumps(all_logprobs.get("harm_potential", [])) if all_logprobs else None

        # Determine if any module had quality issues
        json_parse_failed = (
            result.checkability.json_parse_failed
            or result.verifiability.json_parse_failed
            or result.harm_potential.json_parse_failed
        )
        logprobs_missing = (
            result.checkability.logprobs_missing
            or result.verifiability.logprobs_missing
            or result.harm_potential.logprobs_missing
        )

        return PredictionResult(
            sample_id=sample["sample_id"],
            text=sample["text"],
            ground_truth=sample["ground_truth"],
            model=model_name,
            prediction=result.prediction,
            average_confidence=result.average_confidence,
            # Self-reported confidences (from JSON)
            checkability_self_conf=result.checkability.self_confidence,
            verifiability_self_conf=result.verifiability.self_confidence,
            harm_self_conf=result.harm_potential.self_confidence,
            # Logprob-derived confidences (p_true * 100)
            checkability_logprob_conf=result.checkability.logprob_confidence,
            verifiability_logprob_conf=result.verifiability.logprob_confidence,
            harm_logprob_conf=result.harm_potential.logprob_confidence,
            # Final confidences (logprob if available, else self-reported)
            checkability_confidence=result.checkability.confidence,
            verifiability_confidence=result.verifiability.confidence,
            harm_confidence=result.harm_potential.confidence,
            json_parse_failed=json_parse_failed,
            logprobs_missing=logprobs_missing,
            latency_ms=latency_ms,
            prompt_tokens=total_usage.prompt_tokens,
            completion_tokens=total_usage.completion_tokens,
            total_tokens=total_usage.total_tokens,
            logprobs_checkability=logprobs_check,
            logprobs_verifiability=logprobs_verif,
            logprobs_harm=logprobs_harm,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return PredictionResult(
            sample_id=sample["sample_id"],
            text=sample["text"],
            ground_truth=sample["ground_truth"],
            model=model_name,
            prediction="No",  # Default to No on error
            average_confidence=0.0,
            # All confidences None/0.0 on error
            checkability_self_conf=None,
            verifiability_self_conf=None,
            harm_self_conf=None,
            checkability_logprob_conf=None,
            verifiability_logprob_conf=None,
            harm_logprob_conf=None,
            checkability_confidence=0.0,
            verifiability_confidence=0.0,
            harm_confidence=0.0,
            json_parse_failed=False,
            logprobs_missing=True,
            error=str(e),
            latency_ms=latency_ms,
        )


def run_parallel_predictions(
    baseline: PromptingBaseline,
    samples: list[dict],
    model_name: str,
    max_workers: int = MAX_WORKERS,
) -> list[PredictionResult]:
    """Run predictions in parallel with rate limiting.

    Args:
        baseline: The PromptingBaseline instance
        samples: List of sample dicts
        model_name: Name of the model
        max_workers: Number of concurrent workers

    Returns:
        List of PredictionResult
    """
    results: list[PredictionResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i, sample in enumerate(samples):
            # Rate limiting: stagger submissions
            if i > 0 and i % RATE_LIMIT_PER_SEC == 0:
                time.sleep(1.0)

            future = executor.submit(run_single_prediction, baseline, sample, model_name)
            futures[future] = sample["sample_id"]

        # Collect results with progress bar
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {model_name}",
            leave=False,
        ):
            result = future.result()
            results.append(result)

    return results


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_metrics(predictions: list[PredictionResult]) -> dict[str, float | int]:
    """Compute official CheckThat! 2024 metrics.

    Uses sklearn with pos_label='Yes' to match official scorer.

    Args:
        predictions: List of predictions

    Returns:
        Dict with accuracy, precision, recall, f1, and quality stats
    """
    # Filter out errors
    valid_preds = [p for p in predictions if p.error is None]
    n_errors = len(predictions) - len(valid_preds)

    # Count quality issues
    n_json_parse_failed = sum(1 for p in valid_preds if p.json_parse_failed)
    n_logprobs_missing = sum(1 for p in valid_preds if p.logprobs_missing)

    if not valid_preds:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_valid": 0,
            "n_errors": n_errors,
            "n_json_parse_failed": 0,
            "n_logprobs_missing": 0,
        }

    y_true = [p.ground_truth for p in valid_preds]
    y_pred = [p.prediction for p in valid_preds]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="Yes", average="binary", zero_division=0.0),
        "recall": recall_score(y_true, y_pred, pos_label="Yes", average="binary", zero_division=0.0),
        "f1": f1_score(y_true, y_pred, pos_label="Yes", average="binary", zero_division=0.0),
        "n_valid": len(valid_preds),
        "n_errors": n_errors,
        "n_json_parse_failed": n_json_parse_failed,
        "n_logprobs_missing": n_logprobs_missing,
    }


# =============================================================================
# Results Reporting
# =============================================================================


def print_comparison_table(results_by_model: dict[str, dict]) -> None:
    """Print comparison table: our results vs paper baselines (legacy function)."""
    print_complete_results_table(results_by_model, {})


def print_complete_results_table(
    results_by_model: dict[str, dict],
    confidence_stats: dict[str, dict],
) -> None:
    """Print comprehensive results table with confidence extraction and performance metrics.

    Args:
        results_by_model: Dict of model_name -> metrics dict
        confidence_stats: Dict of model_name -> {self_pct, logprob_pct, errors}
    """
    print("\n" + "=" * 140)
    print("COMPLETE RESULTS: Zero-Shot Checkworthiness Baselines")
    print("=" * 140)
    print()

    # Header
    header = (
        f"{'Model':<15} | {'Self-Rep':<8} | {'Logprob':<8} | {'Errors':<6} | "
        f"{'F1':<6} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | "
        f"{'Paper F1':<8} | {'ΔF1':<7} | {'Paper Acc':<9} | {'ΔAcc'}"
    )
    print(header)
    print("-" * 140)

    for model_name, our_metrics in results_by_model.items():
        paper = PAPER_BASELINES.get(model_name, {})
        conf = confidence_stats.get(model_name, {})

        # Confidence extraction stats
        self_pct = conf.get("self_pct", "N/A")
        logprob_pct = conf.get("logprob_pct", "N/A")
        errors = conf.get("errors", our_metrics.get("n_errors", 0))

        # Performance metrics
        our_f1 = our_metrics.get("f1", 0.0)
        our_acc = our_metrics.get("accuracy", 0.0)
        our_prec = our_metrics.get("precision", 0.0)
        our_rec = our_metrics.get("recall", 0.0)

        # Paper baselines
        paper_f1 = paper.get("f1", 0.0)
        paper_acc = paper.get("acc", 0.0)

        # Deltas
        delta_f1 = our_f1 - paper_f1 if paper_f1 > 0 else 0.0
        delta_acc = our_acc - paper_acc if paper_acc > 0 else 0.0

        # Format strings
        paper_f1_str = f"{paper_f1:.3f}" if paper_f1 > 0 else "N/A"
        paper_acc_str = f"{paper_acc:.3f}" if paper_acc > 0 else "N/A"
        delta_f1_str = f"{delta_f1:+.3f}" if paper_f1 > 0 else "N/A"
        delta_acc_str = f"{delta_acc:+.3f}" if paper_acc > 0 else "N/A"

        print(
            f"{model_name:<15} | {self_pct:<8} | {logprob_pct:<8} | {errors:<6} | "
            f"{our_f1:<6.3f} | {our_acc:<6.3f} | {our_prec:<6.3f} | {our_rec:<6.3f} | "
            f"{paper_f1_str:<8} | {delta_f1_str:<7} | {paper_acc_str:<9} | {delta_acc_str}"
        )

    print("=" * 140)
    print()
    print("Legend:")
    print("  Self-Rep: % of self-reported confidences extracted | Logprob: % of logprob-derived confidences")
    print("  Paper F1/Acc: Fine-tuned baselines from CT24 paper | ΔF1/ΔAcc: Our result - Paper baseline")
    print("  Note: Paper baselines used FINE-TUNED models; ours are ZERO-SHOT prompting.")


def plot_confusion_matrices(
    predictions_by_model: dict[str, list[PredictionResult]],
    output_dir: Path,
    timestamp: str,
) -> Path:
    """Plot confusion matrices for all models in a grid layout.

    Args:
        predictions_by_model: Dict of model_name -> list of PredictionResult
        output_dir: Directory to save the plot
        timestamp: Timestamp for filename

    Returns:
        Path to saved figure
    """
    n_models = len(predictions_by_model)
    if n_models == 0:
        return None

    # Calculate grid dimensions (prefer wider than taller)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Ensure axes is always 2D array for consistent indexing
    if n_models == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Labels for confusion matrix
    labels = ["No", "Yes"]

    for idx, (model_name, preds) in enumerate(predictions_by_model.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        # Extract ground truth and predictions
        y_true = [p.ground_truth for p in preds]
        y_pred = [p.prediction for p in preds]

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Plot as heatmap
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add labels and title
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted",
            ylabel="Actual",
        )
        ax.set_title(f"{model_name}\n(F1={f1_score(y_true, y_pred, pos_label='Yes'):.3f})", fontsize=10)

        # Add text annotations in cells
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold"
                )

    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].axis("off")

    plt.suptitle("Confusion Matrices: Zero-Shot Checkworthiness Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"confusion_matrices_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved confusion matrices to: {fig_path.name}")
    return fig_path


def save_results(
    results: ExperimentResults,
    results_by_model: dict[str, dict],
    output_dir: Path,
) -> tuple[list[Path], Path]:
    """Save experiment results to files - ONE PARQUET PER MODEL.

    Args:
        results: ExperimentResults with all predictions
        results_by_model: Metrics by model
        output_dir: Output directory

    Returns:
        Tuple of (list of prediction_paths per model, metrics_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Group predictions by model and save separate files
    predictions_by_model: dict[str, list[PredictionResult]] = {}
    for pred in results.predictions:
        if pred.model not in predictions_by_model:
            predictions_by_model[pred.model] = []
        predictions_by_model[pred.model].append(pred)

    # Fields that should be numeric (float or None)
    float_fields = {
        "average_confidence",
        "checkability_self_conf",
        "verifiability_self_conf",
        "harm_self_conf",
        "checkability_logprob_conf",
        "verifiability_logprob_conf",
        "harm_logprob_conf",
        "checkability_confidence",
        "verifiability_confidence",
        "harm_confidence",
    }

    def sanitize_float(value: Any, field_name: str) -> float | None:
        """Convert value to float, returning None for invalid values."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                # LLM returned non-numeric string like "uncertain"
                return None
        return None

    prediction_paths: list[Path] = []
    for model_name, preds in predictions_by_model.items():
        predictions_data = [asdict(p) for p in preds]

        # Sanitize float fields to handle LLM outputs like "uncertain"
        # Also ensure error field is string (not inferred as Null)
        for row in predictions_data:
            for field in float_fields:
                if field in row:
                    row[field] = sanitize_float(row[field], field)
            # Ensure error is always string or None (for consistent schema)
            if "error" in row and row["error"] is not None:
                row["error"] = str(row["error"])

        # Create DataFrame with full schema inference to handle mixed None/string in error field
        df = pl.DataFrame(predictions_data, infer_schema_length=None)

        # Sanitize model name for filename
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        pred_path = output_dir / f"predictions_{safe_model_name}_{timestamp}.parquet"
        df.write_parquet(pred_path)
        prediction_paths.append(pred_path)
        print(f"  Saved {len(preds)} predictions to: {pred_path.name}")

    # Save metrics as JSON
    metrics_data = {
        "experiment_metadata": {
            "timestamp": timestamp,
            "n_samples": results.n_samples,
            "models_run": results.models_run,
            "temperature": 0.0,
            "dataset": "CT24_checkworthy_english_test_gold.tsv",
        },
        "results_by_model": results_by_model,
        "paper_baselines": PAPER_BASELINES,
    }
    metrics_path = output_dir / f"mixed_baselines_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    return prediction_paths, metrics_path


def save_official_format(
    predictions: list[PredictionResult],
    output_dir: Path,
    model_name: str,
) -> Path:
    """Save predictions in official CheckThat! 2024 format.

    Format: id\tpred_label\trun_id

    Args:
        predictions: List of predictions
        output_dir: Output directory
        model_name: Model name for filename

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize model name for filename (replace special chars)
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    output_path = output_dir / f"predictions_{safe_model_name}.tsv"

    with open(output_path, "w") as f:
        f.write("id\tpred_label\trun_id\n")
        for pred in sorted(predictions, key=lambda p: p.sample_id):
            f.write(f"{pred.sample_id}\t{pred.prediction}\t1\n")

    return output_path


# =============================================================================
# API Key Validation
# =============================================================================


def check_api_keys(models: list[str]) -> dict[ModelProvider, bool]:
    """Check which API keys are available for the requested models.

    Args:
        models: List of model keys to check

    Returns:
        Dict mapping provider to availability status
    """
    providers_needed: set[ModelProvider] = set()

    for model_name in models:
        if model_name in MODELS:
            providers_needed.add(MODELS[model_name].provider)

    key_status = {}
    for provider in providers_needed:
        if provider == ModelProvider.TOGETHER_AI:
            key_status[provider] = bool(os.getenv("TOGETHER_API_KEY"))
        elif provider == ModelProvider.OPENAI:
            key_status[provider] = bool(os.getenv("OPENAI_API_KEY"))
        else:
            key_status[provider] = False

    return key_status


# =============================================================================
# Main Experiment
# =============================================================================


def run_experiment(
    models: list[str],
    n_samples: int | None,
    output_dir: Path,
    dry_run: bool = False,
) -> ExperimentResults:
    """Run the full experiment.

    Args:
        models: List of model keys to test
        n_samples: Number of samples per model
        output_dir: Output directory
        dry_run: If True, only test with 1 sample

    Returns:
        ExperimentResults
    """
    # Load data (resolve relative to project root)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_path = PROJECT_ROOT / "data/raw/CT24_checkworthy_english/CT24_checkworthy_english_test_gold.tsv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data with optional subsampling
    # dry_run overrides n_samples to 1 for quick testing
    actual_n_samples = 1 if dry_run else n_samples
    df = load_ct24_test_gold(data_path, n_samples=actual_n_samples)
    samples = df.to_dicts()

    # Check API keys
    key_status = check_api_keys(models)
    missing_keys = [p.value for p, available in key_status.items() if not available]
    if missing_keys:
        raise ValueError(f"Missing API keys for providers: {missing_keys}")

    print(f"\n{'=' * 60}")
    print("MIXED PROVIDER BASELINES EXPERIMENT")
    print(f"{'=' * 60}")
    print(f"Models: {models}")
    print(f"Samples: {len(samples)}")
    print(f"Temperature: 0.0")
    print(f"Dry run: {dry_run}")
    print(f"API Keys: {', '.join(p.value for p in key_status.keys())}")
    print(f"{'=' * 60}\n")

    results = ExperimentResults(n_samples=len(samples), models_run=models)
    results_by_model: dict[str, dict] = {}

    for model_name in models:
        print(f"\n--- Running {model_name} ---")

        # Get model config
        if model_name not in MODELS:
            print(f"  ERROR: Model {model_name} not found in config")
            continue

        model_config = MODELS[model_name]

        # Check if provider is allowed for this experiment
        if model_config.provider not in ALLOWED_PROVIDERS:
            print(f"  WARNING: {model_name} provider ({model_config.provider.value}) not in allowed list, skipping")
            continue

        # Create baseline
        try:
            baseline = PromptingBaseline(
                model_config=model_config,
                threshold=50.0,
                temperature=0.0,
            )
        except Exception as e:
            print(f"  ERROR: Failed to create baseline: {e}")
            continue

        # Run predictions
        start_time = time.perf_counter()
        predictions = run_parallel_predictions(baseline, samples, model_name)
        elapsed = time.perf_counter() - start_time

        # Add to results
        results.predictions.extend(predictions)

        # Compute metrics
        metrics = compute_metrics(predictions)
        results_by_model[model_name] = metrics

        print(f"  Completed in {elapsed:.1f}s")
        print(f"  F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}")
        print(f"  Valid: {metrics['n_valid']}, Errors: {metrics['n_errors']}")

        # Save official format predictions
        save_official_format(predictions, output_dir / "official_format", model_name)

    results.end_time = datetime.now()

    # Compute confidence extraction statistics per model
    confidence_stats: dict[str, dict] = {}
    predictions_by_model: dict[str, list[PredictionResult]] = {}
    for pred in results.predictions:
        if pred.model not in predictions_by_model:
            predictions_by_model[pred.model] = []
        predictions_by_model[pred.model].append(pred)

    for model_name, preds in predictions_by_model.items():
        n_rows = len(preds)
        n_possible = n_rows * 3  # 3 modules per row

        # Count non-null self-reported confidences
        self_non_null = sum(
            1 for p in preds
            for val in [p.checkability_self_conf, p.verifiability_self_conf, p.harm_self_conf]
            if val is not None
        )
        # Count non-null logprob confidences
        logprob_non_null = sum(
            1 for p in preds
            for val in [p.checkability_logprob_conf, p.verifiability_logprob_conf, p.harm_logprob_conf]
            if val is not None
        )
        # Count errors
        errors = sum(1 for p in preds if p.error is not None)

        confidence_stats[model_name] = {
            "self_pct": f"{100 * self_non_null / n_possible:.1f}%" if n_possible > 0 else "N/A",
            "logprob_pct": f"{100 * logprob_non_null / n_possible:.1f}%" if n_possible > 0 else "N/A",
            "errors": errors,
        }

    # Print comprehensive results table
    print_complete_results_table(results_by_model, confidence_stats)

    # Save results (separate file per model)
    prediction_paths, metrics_path = save_results(results, results_by_model, output_dir)
    print(f"\nResults saved to:")
    print(f"  Predictions ({len(prediction_paths)} files):")
    for p in prediction_paths:
        print(f"    - {p.name}")
    print(f"  Metrics: {metrics_path}")

    # Plot confusion matrices
    predictions_by_model: dict[str, list[PredictionResult]] = {}
    for pred in results.predictions:
        if pred.model not in predictions_by_model:
            predictions_by_model[pred.model] = []
        predictions_by_model[pred.model].append(pred)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = plot_confusion_matrices(predictions_by_model, output_dir, timestamp)
    if cm_path:
        print(f"  Confusion matrices: {cm_path}")

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Mixed Provider Baselines (Mistral + GPT-4)")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per model (default: 100)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MIXED_MODELS),
        help=f"Comma-separated list of models (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/mixed_baselines",
        help="Output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with 1 sample to test setup",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    output_dir = Path(args.output_dir)

    run_experiment(
        models=models,
        n_samples=args.n_samples,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
