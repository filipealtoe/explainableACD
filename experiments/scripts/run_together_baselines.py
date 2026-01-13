#!/usr/bin/env python3
"""Together AI Baselines Comparison for Checkworthiness.

Compares our zero-shot 3-module pipeline against fine-tuned baselines
from CheckThat! 2024 paper using Together AI's open-weight models.

PRE-REGISTRATION:
- Primary metric: F1-positive (matching CheckThat! 2024)
- Hypothesis: Zero-shot underperforms fine-tuned by 10-30%
- N=100 stratified samples from CT24_test_gold.tsv
- T=0.0 (deterministic)

Usage:
    python experiments/scripts/run_together_baselines.py
    python experiments/scripts/run_together_baselines.py --n-samples 10 --models llama-2-7b
    python experiments/scripts/run_together_baselines.py --dry-run
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

import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkworthiness.config import MODELS, ModelConfig, ModelProvider, TokenUsage
from src.checkworthiness.prompting_baseline import PromptingBaseline

# Load environment variables
load_dotenv()

# =============================================================================
# Paper Baselines (Target to Compare Against)
# =============================================================================

PAPER_BASELINES: dict[str, dict[str, float]] = {
    # From CheckThat! 2024 paper, Test partition (fine-tuned models)
    # NOTE: Llama 2 and Falcon are no longer available on Together AI.
    # We use Llama 3.x as modern equivalents for comparison context.
    "llama-2-7b": {"f1": 0.820, "acc": 0.905, "precision": 0.802, "recall": 0.841},  # Unavailable
    "llama-2-13b": {"f1": 0.812, "acc": 0.897, "precision": 0.772, "recall": 0.856},  # Unavailable
    "llama-3-8b": {"f1": 0.809, "acc": 0.907, "precision": 0.863, "recall": 0.761},
    "mistral-7b": {"f1": 0.799, "acc": 0.889, "precision": 0.747, "recall": 0.860},
    "mixtral-8x7b": {"f1": 0.807, "acc": 0.891, "precision": 0.741, "recall": 0.886},
    "gemma-7b": {"f1": 0.813, "acc": 0.900, "precision": 0.788, "recall": 0.841},  # Using gemma-3n instead
    "falcon-7b": {"f1": 0.793, "acc": 0.891, "precision": 0.780, "recall": 0.806},  # Unavailable
}

# Together AI model keys (must match config.py)
# NOTE: Updated 2024-12-29 - Only "Turbo" models are serverless on Together AI
# Llama 2, Falcon, original Llama 3 base, and original Gemma-7b are NOT available
# Gemma-3n excluded due to data contamination risk (cutoff June 2024 > CT24 release Jan 2024)
TOGETHER_MODELS: list[str] = [
    # Llama 3.1/3.2/3.3 Turbo (serverless variants)
    "llama-3.1-8b",  # Llama 3.1 8B Turbo (closest to paper's llama-3-8b)
    "llama-3.1-70b",  # Llama 3.1 70B Turbo
    "llama-3.3-70b",  # Llama 3.3 70B Turbo (latest)
    "llama-3.2-3b",  # Llama 3.2 3B Turbo (smallest)
    # Mistral family (direct paper comparison)
    "mistral-7b",  # Direct comparison to paper
    "mixtral-8x7b",  # Direct comparison to paper
]

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
    confidence: float  # 0-100

    # Per-module confidences
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
            confidence=result.average_confidence,
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
            confidence=0.0,
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
    """Print comparison table: our results vs paper baselines."""
    print("\n" + "=" * 100)
    print("COMPARISON: Zero-Shot (Ours) vs Fine-Tuned (Paper)")
    print("=" * 100)

    header = f"{'Model':<15} | {'Our F1':>8} | {'Paper F1':>9} | {'Delta':>8} | {'Our Acc':>8} | {'Paper Acc':>9} | {'Delta':>8}"
    print(header)
    print("-" * 100)

    for model_name, our_metrics in results_by_model.items():
        paper = PAPER_BASELINES.get(model_name, {})

        our_f1 = our_metrics.get("f1", 0.0)
        paper_f1 = paper.get("f1", 0.0)
        delta_f1 = our_f1 - paper_f1 if paper_f1 > 0 else 0.0

        our_acc = our_metrics.get("accuracy", 0.0)
        paper_acc = paper.get("acc", 0.0)
        delta_acc = our_acc - paper_acc if paper_acc > 0 else 0.0

        # Color-code deltas
        f1_sign = "+" if delta_f1 >= 0 else ""
        acc_sign = "+" if delta_acc >= 0 else ""

        print(
            f"{model_name:<15} | {our_f1:>8.3f} | {paper_f1:>9.3f} | {f1_sign}{delta_f1:>7.3f} | "
            f"{our_acc:>8.3f} | {paper_acc:>9.3f} | {acc_sign}{delta_acc:>7.3f}"
        )

    print("=" * 100)


def save_results(
    results: ExperimentResults,
    results_by_model: dict[str, dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save experiment results to files.

    Args:
        results: ExperimentResults with all predictions
        results_by_model: Metrics by model
        output_dir: Output directory

    Returns:
        Tuple of (predictions_path, metrics_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save predictions as parquet
    predictions_data = [asdict(p) for p in results.predictions]
    df = pl.DataFrame(predictions_data)
    predictions_path = output_dir / f"together_baselines_predictions_{timestamp}.parquet"
    df.write_parquet(predictions_path)

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
    metrics_path = output_dir / f"together_baselines_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    return predictions_path, metrics_path


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
    output_path = output_dir / f"predictions_{model_name}.tsv"

    with open(output_path, "w") as f:
        f.write("id\tpred_label\trun_id\n")
        for pred in sorted(predictions, key=lambda p: p.sample_id):
            f.write(f"{pred.sample_id}\t{pred.prediction}\t1\n")

    return output_path


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

    # Load full gold test set by default (optionally subsample for dry runs)
    df_full = load_ct24_test_gold(data_path, n_samples=None)
    if dry_run:
        df = df_full.head(1)
    else:
        df = df_full
    samples = df.to_dicts()

    # Check API key
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")

    print(f"\n{'=' * 60}")
    print("TOGETHER AI BASELINES EXPERIMENT")
    print(f"{'=' * 60}")
    print(f"Models: {models}")
    print(f"Samples: {len(samples)}")
    print(f"Temperature: 0.0")
    print(f"Dry run: {dry_run}")
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
        if model_config.provider != ModelProvider.TOGETHER_AI:
            print(f"  WARNING: {model_name} is not a Together AI model, skipping")
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

    # Print comparison table
    print_comparison_table(results_by_model)

    # Save results
    predictions_path, metrics_path = save_results(results, results_by_model, output_dir)
    print(f"\nResults saved to:")
    print(f"  Predictions: {predictions_path}")
    print(f"  Metrics: {metrics_path}")

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Together AI Baselines Comparison")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per model (default: 100)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(TOGETHER_MODELS),
        help=f"Comma-separated list of models (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/together_baselines",
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
