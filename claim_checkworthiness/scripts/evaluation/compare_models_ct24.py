#!/usr/bin/env python3
"""Compare LLM models on CT24 dev set for checkworthiness classification.

This script evaluates multiple LLM models to select the best one for generating
confidence features (logprobs) for the downstream ML classifier.

Metrics computed per model:
1. Prompting metrics (Accuracy, F1, Precision, Recall)
2. Expected Calibration Error (ECE)
3. Entropy distribution per module
4. Feature discriminativeness (AUC-ROC per confidence feature)
5. Logprob extraction rate
6. Token usage and cost

Usage:
    # Quick screen (50 samples, single model)
    python experiments/scripts/compare_models_ct24.py --models gpt-4o-mini --samples 50

    # Compare multiple models
    python experiments/scripts/compare_models_ct24.py \
        --models gpt-4o,gpt-4o-mini,gpt-3.5-turbo,llama-3.3-70b \
        --samples 50

    # Full dev set evaluation
    python experiments/scripts/compare_models_ct24.py --models gpt-4o-mini --samples 1031
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# Path setup
PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # claim_checkworthiness/
REPO_ROOT = Path(__file__).resolve().parents[3]     # explainableACD/
sys.path.insert(0, str(PACKAGE_ROOT))

from src.checkworthiness.config import MODELS, ModelConfig
from src.checkworthiness.prompting_baseline import PromptingBaseline


# =============================================================================
# Rate Limiter for Parallel Requests
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter for async requests."""

    def __init__(self, requests_per_minute: float = 50.0):
        self.rate = requests_per_minute / 60.0  # Convert to per-second
        self.tokens = self.rate * 2  # Start with 2 seconds worth
        self.max_tokens = self.rate * 5  # Max 5 seconds worth
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = REPO_ROOT / "data" / "raw" / "CT24_checkworthy_english"
DATA_PATHS = {
    "train": DATA_DIR / "CT24_checkworthy_english_train.tsv",
    "dev": DATA_DIR / "CT24_checkworthy_english_dev.tsv",
    "test": DATA_DIR / "CT24_checkworthy_english_test_gold.tsv",
}
OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "model_comparison"

# Models to compare (keys from config.py)
AVAILABLE_MODELS = [
    # OpenAI
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    # Together AI (Llama)
    "llama-3.3-70b",
    "llama-3.1-70b",
    "llama-3.1-8b",
]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SampleResult:
    """Result from evaluating one sample with one model."""

    sentence_id: str
    model: str
    text: str
    ground_truth: str  # "Yes" or "No"

    # Prediction
    prediction: str
    average_confidence: float

    # Per-module confidences (logprob-derived, 0-100 scale)
    checkability_confidence: float
    verifiability_confidence: float
    harmpot_confidence: float

    # Ternary probabilities (0-1 scale)
    checkability_p_true: float | None = None
    checkability_p_false: float | None = None
    checkability_p_uncertain: float | None = None
    verifiability_p_true: float | None = None
    verifiability_p_false: float | None = None
    verifiability_p_uncertain: float | None = None
    harmpot_p_true: float | None = None
    harmpot_p_false: float | None = None
    harmpot_p_uncertain: float | None = None

    # Entropy per module (0-1.585 scale)
    checkability_entropy: float | None = None
    verifiability_entropy: float | None = None
    harmpot_entropy: float | None = None

    # Quality flags
    json_parse_failed: bool = False
    logprobs_missing: bool = False

    # Error tracking
    error: str | None = None

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Latency
    latency_ms: float = 0.0


@dataclass
class ModelMetrics:
    """Aggregated metrics for one model."""

    model: str
    n_samples: int
    n_errors: int

    # Prompting metrics
    accuracy: float
    f1_positive: float
    precision: float
    recall: float

    # Calibration
    ece: float  # Expected Calibration Error

    # Entropy stats per module
    checkability_entropy_mean: float
    checkability_entropy_std: float
    verifiability_entropy_mean: float
    verifiability_entropy_std: float
    harmpot_entropy_mean: float
    harmpot_entropy_std: float

    # Entropy-error correlation
    entropy_error_correlation: float

    # Feature discriminativeness (AUC-ROC)
    checkability_auc: float
    verifiability_auc: float
    harmpot_auc: float
    average_confidence_auc: float

    # Logprob extraction rate
    logprob_extraction_rate: float
    json_parse_success_rate: float

    # Cost
    total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    estimated_cost_usd: float

    # Latency
    mean_latency_ms: float


# =============================================================================
# Data Loading
# =============================================================================


def load_ct24_dev(data_path: Path, n_samples: int | None = None, seed: int = 42) -> list[dict]:
    """Load CT24 dev set with optional subsampling."""
    print(f"Loading data from {data_path}")

    df = pl.read_csv(data_path, separator="\t")
    print(f"  Total samples: {len(df)}")

    # Convert to list of dicts
    samples = []
    for row in df.iter_rows(named=True):
        samples.append({
            "sentence_id": str(row["Sentence_id"]),
            "text": row["Text"],
            "label": row["class_label"],
        })

    # Subsample if requested (stratified by label)
    if n_samples is not None and n_samples < len(samples):
        np.random.seed(seed)
        yes_samples = [s for s in samples if s["label"] == "Yes"]
        no_samples = [s for s in samples if s["label"] == "No"]

        # Maintain label ratio
        yes_ratio = len(yes_samples) / len(samples)
        n_yes = int(n_samples * yes_ratio)
        n_no = n_samples - n_yes

        np.random.shuffle(yes_samples)
        np.random.shuffle(no_samples)

        samples = yes_samples[:n_yes] + no_samples[:n_no]
        np.random.shuffle(samples)

        print(f"  Subsampled to {len(samples)} samples (Yes: {n_yes}, No: {n_no})")

    return samples


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Lower is better. 0 = perfectly calibrated.
    """
    if not confidences:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]

        # Samples in this bin (normalize confidence to 0-1)
        in_bin = [
            (c / 100.0, is_correct)
            for c, is_correct in zip(confidences, correct)
            if low <= c / 100.0 < high or (i == n_bins - 1 and c / 100.0 == high)
        ]

        if not in_bin:
            continue

        bin_conf = sum(c for c, _ in in_bin) / len(in_bin)
        bin_acc = sum(1 for _, is_correct in in_bin if is_correct) / len(in_bin)

        ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)

    return ece


def compute_auc_roc(confidence_scores: list[float], labels: list[str]) -> float:
    """Compute AUC-ROC for a confidence feature."""
    y_true = [1 if label == "Yes" else 0 for label in labels]

    # Check if we have both classes
    if len(set(y_true)) < 2:
        return 0.5

    try:
        return roc_auc_score(y_true, confidence_scores)
    except ValueError:
        return 0.5


def compute_model_metrics(results: list[SampleResult], model_config: ModelConfig) -> ModelMetrics:
    """Compute all metrics for a model from sample results."""
    valid = [r for r in results if r.error is None]
    n_errors = len(results) - len(valid)

    if not valid:
        return ModelMetrics(
            model=results[0].model if results else "unknown",
            n_samples=len(results),
            n_errors=n_errors,
            accuracy=0.0,
            f1_positive=0.0,
            precision=0.0,
            recall=0.0,
            ece=0.0,
            checkability_entropy_mean=0.0,
            checkability_entropy_std=0.0,
            verifiability_entropy_mean=0.0,
            verifiability_entropy_std=0.0,
            harmpot_entropy_mean=0.0,
            harmpot_entropy_std=0.0,
            entropy_error_correlation=0.0,
            checkability_auc=0.5,
            verifiability_auc=0.5,
            harmpot_auc=0.5,
            average_confidence_auc=0.5,
            logprob_extraction_rate=0.0,
            json_parse_success_rate=0.0,
            total_tokens=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            estimated_cost_usd=0.0,
            mean_latency_ms=0.0,
        )

    # Prompting metrics
    y_true = [r.ground_truth for r in valid]
    y_pred = [r.prediction for r in valid]

    accuracy = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="Yes", zero_division=0)

    # Calibration (ECE)
    confidences = [r.average_confidence for r in valid]
    correct = [r.prediction == r.ground_truth for r in valid]
    ece = compute_ece(confidences, correct)

    # Entropy stats
    check_entropies = [r.checkability_entropy for r in valid if r.checkability_entropy is not None]
    verif_entropies = [r.verifiability_entropy for r in valid if r.verifiability_entropy is not None]
    harm_entropies = [r.harmpot_entropy for r in valid if r.harmpot_entropy is not None]

    # Entropy-error correlation
    avg_entropies = []
    errors_binary = []
    for r in valid:
        if all(e is not None for e in [r.checkability_entropy, r.verifiability_entropy, r.harmpot_entropy]):
            avg_entropy = (r.checkability_entropy + r.verifiability_entropy + r.harmpot_entropy) / 3
            avg_entropies.append(avg_entropy)
            errors_binary.append(0 if r.prediction == r.ground_truth else 1)

    if len(avg_entropies) >= 3:
        entropy_error_corr = float(np.corrcoef(avg_entropies, errors_binary)[0, 1])
        if np.isnan(entropy_error_corr):
            entropy_error_corr = 0.0
    else:
        entropy_error_corr = 0.0

    # Feature AUC-ROC
    check_auc = compute_auc_roc([r.checkability_confidence for r in valid], y_true)
    verif_auc = compute_auc_roc([r.verifiability_confidence for r in valid], y_true)
    harm_auc = compute_auc_roc([r.harmpot_confidence for r in valid], y_true)
    avg_auc = compute_auc_roc([r.average_confidence for r in valid], y_true)

    # Logprob extraction rate
    n_with_logprobs = sum(1 for r in valid if not r.logprobs_missing)
    logprob_rate = n_with_logprobs / len(valid) if valid else 0

    n_json_ok = sum(1 for r in valid if not r.json_parse_failed)
    json_rate = n_json_ok / len(valid) if valid else 0

    # Token usage and cost
    total_prompt = sum(r.prompt_tokens for r in results)
    total_completion = sum(r.completion_tokens for r in results)
    total_tokens = sum(r.total_tokens for r in results)

    cost = (
        (total_prompt / 1e6) * model_config.cost_per_1m_input
        + (total_completion / 1e6) * model_config.cost_per_1m_output
    )

    # Latency
    latencies = [r.latency_ms for r in valid if r.latency_ms > 0]
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    return ModelMetrics(
        model=results[0].model if results else "unknown",
        n_samples=len(results),
        n_errors=n_errors,
        accuracy=accuracy,
        f1_positive=f1_pos,
        precision=precision,
        recall=recall,
        ece=ece,
        checkability_entropy_mean=float(np.mean(check_entropies)) if check_entropies else 0.0,
        checkability_entropy_std=float(np.std(check_entropies)) if check_entropies else 0.0,
        verifiability_entropy_mean=float(np.mean(verif_entropies)) if verif_entropies else 0.0,
        verifiability_entropy_std=float(np.std(verif_entropies)) if verif_entropies else 0.0,
        harmpot_entropy_mean=float(np.mean(harm_entropies)) if harm_entropies else 0.0,
        harmpot_entropy_std=float(np.std(harm_entropies)) if harm_entropies else 0.0,
        entropy_error_correlation=entropy_error_corr,
        checkability_auc=check_auc,
        verifiability_auc=verif_auc,
        harmpot_auc=harm_auc,
        average_confidence_auc=avg_auc,
        logprob_extraction_rate=logprob_rate,
        json_parse_success_rate=json_rate,
        total_tokens=total_tokens,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        estimated_cost_usd=cost,
        mean_latency_ms=mean_latency,
    )


# =============================================================================
# Model Evaluation
# =============================================================================


def run_single_sample(
    baseline: PromptingBaseline,
    sample: dict,
    model_name: str,
) -> SampleResult:
    """Run prediction for a single sample, extracting all logprob data."""
    start = time.perf_counter()

    try:
        result, usage, all_logprobs, reasoning_content, reasoning_logprobs = baseline(sample["text"])
        latency_ms = (time.perf_counter() - start) * 1000

        return SampleResult(
            sentence_id=sample["sentence_id"],
            model=model_name,
            text=sample["text"],
            ground_truth=sample["label"],
            prediction=result.prediction,
            average_confidence=result.average_confidence,
            checkability_confidence=result.checkability.confidence,
            verifiability_confidence=result.verifiability.confidence,
            harmpot_confidence=result.harm_potential.confidence,
            # Ternary probabilities
            checkability_p_true=result.checkability.p_true,
            checkability_p_false=result.checkability.p_false,
            checkability_p_uncertain=result.checkability.p_uncertain,
            verifiability_p_true=result.verifiability.p_true,
            verifiability_p_false=result.verifiability.p_false,
            verifiability_p_uncertain=result.verifiability.p_uncertain,
            harmpot_p_true=result.harm_potential.p_true,
            harmpot_p_false=result.harm_potential.p_false,
            harmpot_p_uncertain=result.harm_potential.p_uncertain,
            # Entropy
            checkability_entropy=result.checkability.entropy,
            verifiability_entropy=result.verifiability.entropy,
            harmpot_entropy=result.harm_potential.entropy,
            # Quality flags
            json_parse_failed=(
                result.checkability.json_parse_failed
                or result.verifiability.json_parse_failed
                or result.harm_potential.json_parse_failed
            ),
            logprobs_missing=(
                result.checkability.logprobs_missing
                or result.verifiability.logprobs_missing
                or result.harm_potential.logprobs_missing
            ),
            # Token usage
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return SampleResult(
            sentence_id=sample["sentence_id"],
            model=model_name,
            text=sample["text"],
            ground_truth=sample["label"],
            prediction="No",
            average_confidence=0.0,
            checkability_confidence=0.0,
            verifiability_confidence=0.0,
            harmpot_confidence=0.0,
            error=str(e),
            latency_ms=latency_ms,
        )


async def run_sample_with_rate_limit(
    baseline: PromptingBaseline,
    sample: dict,
    model_name: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> SampleResult:
    """Run a single sample with rate limiting and concurrency control."""
    async with semaphore:
        await rate_limiter.acquire()
        # Run sync function in thread pool to avoid blocking
        return await asyncio.to_thread(run_single_sample, baseline, sample, model_name)


async def run_model_evaluation_async(
    model_name: str,
    samples: list[dict],
    checkpoint_path: Path | None,
    parallel: int,
    rate_limit: float,
) -> list[SampleResult]:
    """Run evaluation with parallel requests."""
    # Check for existing checkpoint
    completed_ids: set[str] = set()
    existing_results: list[SampleResult] = []

    if checkpoint_path and checkpoint_path.exists():
        df = pl.read_parquet(checkpoint_path)
        model_df = df.filter(pl.col("model") == model_name)
        completed_ids = set(model_df["sentence_id"].to_list())
        for row in model_df.to_dicts():
            existing_results.append(SampleResult(**row))
        print(f"  Resuming: {len(completed_ids)} samples already completed")

    remaining = [s for s in samples if s["sentence_id"] not in completed_ids]

    if not remaining:
        print(f"  All samples already completed for {model_name}")
        return existing_results

    if model_name not in MODELS:
        print(f"  ERROR: Model '{model_name}' not found in config")
        return existing_results

    model_config = MODELS[model_name]
    if not model_config.get_api_key():
        print(f"  SKIP: Missing API key ({model_config.api_key_env})")
        return existing_results

    baseline = PromptingBaseline(
        model_config=model_config,
        threshold=50.0,
        temperature=0.0,
    )

    results = existing_results.copy()
    semaphore = asyncio.Semaphore(parallel)
    rate_limiter = RateLimiter(requests_per_minute=rate_limit)

    # Process in batches for checkpointing
    batch_size = max(25, parallel * 2)
    pbar = tqdm(total=len(remaining), desc=f"  {model_name}", leave=False)

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        tasks = [
            run_sample_with_rate_limit(baseline, sample, model_name, semaphore, rate_limiter)
            for sample in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        pbar.update(len(batch))

        # Checkpoint after each batch
        if checkpoint_path:
            save_checkpoint(results, checkpoint_path)

    pbar.close()
    return results


def run_model_evaluation(
    model_name: str,
    samples: list[dict],
    checkpoint_path: Path | None = None,
    parallel: int = 1,
    rate_limit: float = 60.0,
) -> list[SampleResult]:
    """Run evaluation for a single model with checkpointing."""
    if parallel > 1:
        # Use async parallel execution
        return asyncio.run(
            run_model_evaluation_async(model_name, samples, checkpoint_path, parallel, rate_limit)
        )

    # Original sequential logic
    completed_ids: set[str] = set()
    existing_results: list[SampleResult] = []

    if checkpoint_path and checkpoint_path.exists():
        df = pl.read_parquet(checkpoint_path)
        model_df = df.filter(pl.col("model") == model_name)
        completed_ids = set(model_df["sentence_id"].to_list())
        for row in model_df.to_dicts():
            existing_results.append(SampleResult(**row))
        print(f"  Resuming: {len(completed_ids)} samples already completed")

    remaining = [s for s in samples if s["sentence_id"] not in completed_ids]

    if not remaining:
        print(f"  All samples already completed for {model_name}")
        return existing_results

    if model_name not in MODELS:
        print(f"  ERROR: Model '{model_name}' not found in config")
        return existing_results

    model_config = MODELS[model_name]
    if not model_config.get_api_key():
        print(f"  SKIP: Missing API key ({model_config.api_key_env})")
        return existing_results

    baseline = PromptingBaseline(
        model_config=model_config,
        threshold=50.0,
        temperature=0.0,
    )

    results = existing_results.copy()

    for sample in tqdm(remaining, desc=f"  {model_name}", leave=False):
        result = run_single_sample(baseline, sample, model_name)
        results.append(result)

        if len(results) % 25 == 0 and checkpoint_path:
            save_checkpoint(results, checkpoint_path)

    if checkpoint_path:
        save_checkpoint(results, checkpoint_path)

    return results


def save_checkpoint(results: list[SampleResult], path: Path) -> None:
    """Save intermediate results as parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use infer_schema_length=None to scan all rows (handles mixed None/str in error field)
    df = pl.DataFrame([asdict(r) for r in results], infer_schema_length=None)
    df.write_parquet(path)


# =============================================================================
# Reporting
# =============================================================================


def generate_comparison_table(metrics: dict[str, ModelMetrics]) -> str:
    """Generate markdown comparison table."""
    lines = [
        "| Model | F1 | Acc | ECE | Check AUC | Verif AUC | Harm AUC | Entropy-Err Corr | Logprob % | Cost |",
        "|-------|---:|----:|----:|----------:|----------:|---------:|-----------------:|----------:|-----:|",
    ]

    for model_name, m in sorted(metrics.items(), key=lambda x: -x[1].f1_positive):
        lines.append(
            f"| {model_name} | {m.f1_positive:.3f} | {m.accuracy:.3f} | {m.ece:.3f} | "
            f"{m.checkability_auc:.3f} | {m.verifiability_auc:.3f} | {m.harmpot_auc:.3f} | "
            f"{m.entropy_error_correlation:.3f} | {m.logprob_extraction_rate:.0%} | ${m.estimated_cost_usd:.2f} |"
        )

    return "\n".join(lines)


def generate_entropy_table(metrics: dict[str, ModelMetrics]) -> str:
    """Generate entropy statistics table."""
    lines = [
        "| Model | Check Entropy | Verif Entropy | Harm Entropy | Entropy-Error Corr |",
        "|-------|--------------:|--------------:|-------------:|-------------------:|",
    ]

    for model_name, m in sorted(metrics.items(), key=lambda x: x[0]):
        lines.append(
            f"| {model_name} | {m.checkability_entropy_mean:.3f} ± {m.checkability_entropy_std:.3f} | "
            f"{m.verifiability_entropy_mean:.3f} ± {m.verifiability_entropy_std:.3f} | "
            f"{m.harmpot_entropy_mean:.3f} ± {m.harmpot_entropy_std:.3f} | "
            f"{m.entropy_error_correlation:.3f} |"
        )

    return "\n".join(lines)


def save_results(
    all_results: list[SampleResult],
    metrics: dict[str, ModelMetrics],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save all results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-sample results as parquet
    results_path = output_dir / f"sample_results_{timestamp}.parquet"
    df = pl.DataFrame([asdict(r) for r in all_results])
    df.write_parquet(results_path)
    print(f"  Saved: {results_path}")

    # Metrics summary as JSON
    metrics_path = output_dir / f"metrics_summary_{timestamp}.json"
    metrics_dict = {k: asdict(v) for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # Comparison table as markdown
    table_path = output_dir / f"comparison_table_{timestamp}.md"
    with open(table_path, "w") as f:
        f.write("# Model Comparison Results\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write("## Main Metrics\n\n")
        f.write(generate_comparison_table(metrics))
        f.write("\n\n## Entropy Statistics\n\n")
        f.write(generate_entropy_table(metrics))
    print(f"  Saved: {table_path}")

    return results_path, metrics_path, table_path


# =============================================================================
# Main
# =============================================================================


def run_comparison(
    models: list[str],
    samples: list[dict],
    output_dir: Path,
    no_resume: bool = False,
    parallel: int = 1,
    rate_limit: float = 60.0,
) -> dict[str, ModelMetrics]:
    """Run full comparison across all models."""
    all_results: list[SampleResult] = []
    metrics_by_model: dict[str, ModelMetrics] = {}

    checkpoint_path = output_dir / "checkpoint.parquet"

    # Delete checkpoint if starting fresh
    if no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Deleted existing checkpoint (--no-resume)")

    if parallel > 1:
        print(f"Parallel mode: {parallel} concurrent requests, {rate_limit:.0f} req/min limit")

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        if model_name not in MODELS:
            print(f"  ERROR: Model not found in config")
            continue

        model_config = MODELS[model_name]
        if not model_config.get_api_key():
            print(f"  SKIP: Missing API key ({model_config.api_key_env})")
            continue

        results = run_model_evaluation(
            model_name, samples, checkpoint_path, parallel=parallel, rate_limit=rate_limit
        )

        if not results:
            continue

        all_results.extend(results)
        metrics = compute_model_metrics(results, model_config)
        metrics_by_model[model_name] = metrics

        # Print summary
        print(f"\n  Results:")
        print(f"    F1={metrics.f1_positive:.3f}, Acc={metrics.accuracy:.3f}, Prec={metrics.precision:.3f}, Rec={metrics.recall:.3f}")
        print(f"    ECE={metrics.ece:.3f} (lower is better)")
        print(f"    AUC: check={metrics.checkability_auc:.3f}, verif={metrics.verifiability_auc:.3f}, harm={metrics.harmpot_auc:.3f}")
        print(f"    Entropy: check={metrics.checkability_entropy_mean:.3f}±{metrics.checkability_entropy_std:.3f}")
        print(f"    Entropy-Error Correlation: {metrics.entropy_error_correlation:.3f}")
        print(f"    Logprob extraction: {metrics.logprob_extraction_rate:.1%}")
        print(f"    Cost: ${metrics.estimated_cost_usd:.4f}")
        print(f"    Errors: {metrics.n_errors}/{metrics.n_samples}")

    # Save final results
    if all_results:
        print(f"\n{'='*60}")
        print("Saving results...")
        save_results(all_results, metrics_by_model, output_dir)

    return metrics_by_model


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM models on CT24 dev set for checkworthiness"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gpt-4o-mini",
        help=f"Comma-separated model names. Available: {', '.join(AVAILABLE_MODELS)}",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate (default: dev)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples (default: all samples in split)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1 = sequential)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=60.0,
        help="Max requests per minute per model (default: 60)",
    )

    args = parser.parse_args()

    # Load environment (override=True ensures .env takes precedence over shell env vars)
    load_dotenv(override=True)

    # Parse models
    models = [m.strip() for m in args.models.split(",")]
    print(f"Models to compare: {models}")
    print(f"Split: {args.split}")

    # Validate models
    for model in models:
        if model not in MODELS:
            print(f"WARNING: Model '{model}' not found in config. Available: {list(MODELS.keys())}")

    # Load data
    data_path = DATA_PATHS[args.split]
    samples = load_ct24_dev(data_path, n_samples=args.samples, seed=args.seed)

    # Run comparison
    output_dir = Path(args.output_dir)
    metrics = run_comparison(
        models, samples, output_dir,
        no_resume=args.no_resume,
        parallel=args.parallel,
        rate_limit=args.rate_limit,
    )

    # Print final comparison table
    if metrics:
        print(f"\n{'='*80}")
        print("FINAL COMPARISON")
        print(f"{'='*80}\n")
        print(generate_comparison_table(metrics))
        print()
        print(generate_entropy_table(metrics))
        print()

        # Recommendation
        best_model = max(metrics.items(), key=lambda x: x[1].f1_positive)
        best_calibrated = min(metrics.items(), key=lambda x: x[1].ece)
        best_entropy_corr = max(metrics.items(), key=lambda x: x[1].entropy_error_correlation)

        print("Recommendations:")
        print(f"  Best F1: {best_model[0]} ({best_model[1].f1_positive:.3f})")
        print(f"  Best calibration (lowest ECE): {best_calibrated[0]} ({best_calibrated[1].ece:.3f})")
        print(f"  Best entropy-error correlation: {best_entropy_corr[0]} ({best_entropy_corr[1].entropy_error_correlation:.3f})")


if __name__ == "__main__":
    main()
