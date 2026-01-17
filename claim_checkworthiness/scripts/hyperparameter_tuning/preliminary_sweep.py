#!/usr/bin/env python3
"""Preliminary Hyperparameter Sweep for Checkworthiness Pipeline.

This script tests multiple models, temperatures, and confidence types
on a small fixed sample before committing to full baseline runs.

Design:
- 10 fixed samples (same across all experiments for comparison)
- 5 models (GPT-4o, GPT-4.1-mini, DeepSeek V3, Grok, Kimi K2)
- 2 temperatures (0.0, 0.7)
- 2 confidence types (self-reported, logprob-based)

Total: 10 × 5 × 2 = 100 predictions per module (300 total API calls)
Estimated cost: ~$2-5
"""

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from checkworthiness.config import MODELS, ExperimentStats, ModelConfig, ModelProvider, TokenUsage
from checkworthiness.prompting_baseline import PromptingBaseline

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
N_SAMPLES = 10
TEMPERATURES = [0.0, 0.7]

# Models to test (using config.py definitions where available)
MODEL_CONFIGS = {
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        max_tokens=512,
        logprobs=True,
        top_logprobs=5,
    ),
    "gpt-4.1-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
        max_tokens=512,
        logprobs=True,
        top_logprobs=5,
    ),
    "deepseek-v3": ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        api_base="https://api.deepseek.com/v1",
        max_tokens=512,
        logprobs=True,
        top_logprobs=5,
    ),
    "grok": ModelConfig(
        provider=ModelProvider.XAI,
        model_name="grok-3-latest",
        api_key_env="XAI_API_KEY",
        api_base="https://api.x.ai/v1",
        max_tokens=512,
        logprobs=True,
        top_logprobs=5,
    ),
    "kimi-k2": ModelConfig(
        provider=ModelProvider.MOONSHOT,
        model_name="kimi-k2-0711-preview",
        api_key_env="MOONSHOT_API_KEY",
        api_base="https://api.moonshot.cn/v1",
        max_tokens=512,
        logprobs=True,
        top_logprobs=5,
    ),
}

# Sample claims for testing (diverse set)
SAMPLE_CLAIMS = [
    # Clearly checkable (factual assertions)
    "The COVID-19 vaccine causes autism.",
    "Joe Biden won the 2020 presidential election with 306 electoral votes.",
    "The unemployment rate in the US dropped to 3.5% in December 2023.",

    # Opinions (not checkable)
    "The new healthcare policy is the best thing for America.",
    "Climate change is the most important issue of our time.",

    # Predictions (not checkable)
    "The stock market will crash in 2025.",
    "AI will replace 50% of jobs by 2030.",

    # Vague claims (hard to check)
    "Things have gotten much worse lately.",
    "Many experts agree that the economy is improving.",

    # Mixed/Edge cases
    "5G towers are being used to spread coronavirus.",
]


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    claim: str
    model: str
    temperature: float
    module: str  # checkability, verifiability, harm_potential

    # Self-reported confidence
    self_confidence: float

    # Logprob-based confidence (if available)
    logprob_confidence: float | None

    # Full result
    reasoning: str
    prediction: bool  # For is_checkable, is_verifiable, is_harmful

    # Metadata
    tokens_used: int
    latency_ms: float
    error: str | None = None


@dataclass
class SweepResults:
    """Aggregated results from the sweep."""

    predictions: list[PredictionResult] = field(default_factory=list)
    model_stats: dict = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def add_prediction(self, result: PredictionResult):
        self.predictions.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert predictions to DataFrame for analysis."""
        records = []
        for p in self.predictions:
            records.append({
                "claim": p.claim[:50] + "...",
                "model": p.model,
                "temperature": p.temperature,
                "module": p.module,
                "self_confidence": p.self_confidence,
                "logprob_confidence": p.logprob_confidence,
                "confidence_diff": (p.logprob_confidence - p.self_confidence) if p.logprob_confidence else None,
                "prediction": p.prediction,
                "tokens": p.tokens_used,
                "latency_ms": p.latency_ms,
                "error": p.error,
            })
        return pd.DataFrame(records)

    def summary(self) -> dict:
        """Generate summary statistics."""
        df = self.to_dataframe()

        # Filter successful predictions
        df_ok = df[df["error"].isna()]

        summary = {
            "total_predictions": len(df),
            "successful": len(df_ok),
            "failed": len(df) - len(df_ok),
            "total_tokens": df_ok["tokens"].sum() if len(df_ok) > 0 else 0,
        }

        # Per-model stats
        if len(df_ok) > 0:
            summary["by_model"] = {}
            for model in df_ok["model"].unique():
                model_df = df_ok[df_ok["model"] == model]
                summary["by_model"][model] = {
                    "n_predictions": len(model_df),
                    "avg_self_confidence": model_df["self_confidence"].mean(),
                    "avg_logprob_confidence": model_df["logprob_confidence"].mean(),
                    "avg_confidence_diff": model_df["confidence_diff"].mean(),
                    "avg_tokens": model_df["tokens"].mean(),
                    "avg_latency_ms": model_df["latency_ms"].mean(),
                }

            # Per-temperature stats
            summary["by_temperature"] = {}
            for temp in df_ok["temperature"].unique():
                temp_df = df_ok[df_ok["temperature"] == temp]
                summary["by_temperature"][temp] = {
                    "n_predictions": len(temp_df),
                    "avg_self_confidence": temp_df["self_confidence"].mean(),
                    "avg_logprob_confidence": temp_df["logprob_confidence"].mean(),
                }

        return summary


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are available."""
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "XAI_API_KEY": bool(os.environ.get("XAI_API_KEY")),
        "MOONSHOT_API_KEY": bool(os.environ.get("MOONSHOT_API_KEY")),
    }


def run_single_prediction(
    baseline: PromptingBaseline,
    claim: str,
    model_name: str,
    temperature: float,
) -> list[PredictionResult]:
    """Run prediction for a single claim and return results for all modules."""
    results = []

    # Test checkability
    try:
        start = time.time()
        check_result, check_usage = baseline.assess_checkability(claim)
        latency = (time.time() - start) * 1000

        results.append(PredictionResult(
            claim=claim,
            model=model_name,
            temperature=temperature,
            module="checkability",
            self_confidence=check_result.confidence,  # Now this IS the logprob confidence
            logprob_confidence=check_result.confidence,
            reasoning=check_result.reasoning[:200],
            prediction=check_result.confidence > 50,
            tokens_used=check_usage.total_tokens,
            latency_ms=latency,
        ))
    except Exception as e:
        results.append(PredictionResult(
            claim=claim,
            model=model_name,
            temperature=temperature,
            module="checkability",
            self_confidence=0,
            logprob_confidence=None,
            reasoning="",
            prediction=False,
            tokens_used=0,
            latency_ms=0,
            error=str(e)[:100],
        ))

    # Test verifiability
    try:
        start = time.time()
        verif_result, verif_usage = baseline.assess_verifiability(claim)
        latency = (time.time() - start) * 1000

        results.append(PredictionResult(
            claim=claim,
            model=model_name,
            temperature=temperature,
            module="verifiability",
            self_confidence=verif_result.confidence,
            logprob_confidence=verif_result.confidence,
            reasoning=verif_result.reasoning[:200],
            prediction=verif_result.confidence > 50,
            tokens_used=verif_usage.total_tokens,
            latency_ms=latency,
        ))
    except Exception as e:
        results.append(PredictionResult(
            claim=claim,
            model=model_name,
            temperature=temperature,
            module="verifiability",
            self_confidence=0,
            logprob_confidence=None,
            reasoning="",
            prediction=False,
            tokens_used=0,
            latency_ms=0,
            error=str(e)[:100],
        ))

    # Test harm potential
    try:
        start = time.time()
        harm_result, harm_usage = baseline.assess_harm_potential(claim)
        latency = (time.time() - start) * 1000

        results.append(PredictionResult(
            claim=claim,
            model=model_name,
            temperature=temperature,
            module="harm_potential",
            self_confidence=harm_result.confidence,
            logprob_confidence=harm_result.confidence,
            reasoning=harm_result.reasoning[:200],
            prediction=harm_result.confidence > 50,
            tokens_used=harm_usage.total_tokens,
            latency_ms=latency,
        ))
    except Exception as e:
        results.append(PredictionResult(
            claim=claim,
            model=model_name,
            temperature=temperature,
            module="harm_potential",
            self_confidence=0,
            logprob_confidence=None,
            reasoning="",
            prediction=False,
            tokens_used=0,
            latency_ms=0,
            error=str(e)[:100],
        ))

    return results


def run_sweep(
    models: list[str] | None = None,
    temperatures: list[float] | None = None,
    n_samples: int = N_SAMPLES,
    verbose: bool = True,
) -> SweepResults:
    """Run the preliminary sweep across models and temperatures."""

    if models is None:
        models = list(MODEL_CONFIGS.keys())
    if temperatures is None:
        temperatures = TEMPERATURES

    # Check API keys
    api_keys = check_api_keys()
    if verbose:
        print("\n" + "=" * 60)
        print("API KEY STATUS")
        print("=" * 60)
        for key, available in api_keys.items():
            status = "✅" if available else "❌"
            print(f"  {status} {key}")

    # Filter models based on available API keys
    available_models = []
    for model_name in models:
        config = MODEL_CONFIGS[model_name]
        if api_keys.get(config.api_key_env, False):
            available_models.append(model_name)
        elif verbose:
            print(f"\n⚠️  Skipping {model_name}: {config.api_key_env} not set")

    if not available_models:
        print("\n❌ No models available! Please set API keys.")
        return SweepResults()

    # Select samples
    random.seed(RANDOM_SEED)
    samples = SAMPLE_CLAIMS[:n_samples]

    if verbose:
        print("\n" + "=" * 60)
        print("PRELIMINARY SWEEP CONFIGURATION")
        print("=" * 60)
        print(f"  Models: {available_models}")
        print(f"  Temperatures: {temperatures}")
        print(f"  Samples: {n_samples}")
        print(f"  Total predictions: {n_samples * len(available_models) * len(temperatures) * 3}")
        print("=" * 60)

    results = SweepResults()

    # Run sweep
    for model_name in available_models:
        config = MODEL_CONFIGS[model_name]

        for temp in temperatures:
            if verbose:
                print(f"\n{'─' * 40}")
                print(f"Model: {model_name} | Temperature: {temp}")
                print("─" * 40)

            # Create baseline with this temperature
            baseline = PromptingBaseline(config, threshold=50.0, temperature=temp)

            for i, claim in enumerate(samples):
                if verbose:
                    print(f"  [{i+1}/{n_samples}] {claim[:40]}...")

                predictions = run_single_prediction(
                    baseline=baseline,
                    claim=claim,
                    model_name=model_name,
                    temperature=temp,
                )

                for pred in predictions:
                    results.add_prediction(pred)

                # Small delay to avoid rate limits
                time.sleep(0.5)

    results.end_time = datetime.now()
    return results


def print_summary(results: SweepResults):
    """Print a formatted summary of the sweep results."""
    summary = results.summary()

    print("\n" + "=" * 60)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nTotal predictions: {summary['total_predictions']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total tokens: {summary['total_tokens']:,}")

    if "by_model" in summary:
        print("\n--- BY MODEL ---")
        for model, stats in summary["by_model"].items():
            print(f"\n  {model}:")
            print(f"    Predictions: {stats['n_predictions']}")
            print(f"    Avg Self Confidence: {stats['avg_self_confidence']:.1f}")
            print(f"    Avg Logprob Confidence: {stats['avg_logprob_confidence']:.1f}")
            print(f"    Avg Confidence Diff: {stats['avg_confidence_diff']:.1f}")
            print(f"    Avg Tokens: {stats['avg_tokens']:.0f}")
            print(f"    Avg Latency: {stats['avg_latency_ms']:.0f}ms")

    if "by_temperature" in summary:
        print("\n--- BY TEMPERATURE ---")
        for temp, stats in summary["by_temperature"].items():
            print(f"\n  Temperature {temp}:")
            print(f"    Predictions: {stats['n_predictions']}")
            print(f"    Avg Self Confidence: {stats['avg_self_confidence']:.1f}")
            print(f"    Avg Logprob Confidence: {stats['avg_logprob_confidence']:.1f}")


def save_results(results: SweepResults, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save DataFrame
    df = results.to_dataframe()
    df.to_csv(output_dir / f"sweep_results_{timestamp}.csv", index=False)

    # Save summary
    with open(output_dir / f"sweep_summary_{timestamp}.json", "w") as f:
        json.dump(results.summary(), f, indent=2)

    print(f"\n✅ Results saved to {output_dir}")


def main():
    """Run the preliminary sweep."""
    print("\n" + "=" * 60)
    print("PRELIMINARY HYPERPARAMETER SWEEP")
    print("Checkworthiness Pipeline")
    print("=" * 60)

    # Run sweep
    results = run_sweep(verbose=True)

    if results.predictions:
        # Print summary
        print_summary(results)

        # Save results
        output_dir = project_root / "experiments" / "results" / "preliminary_sweep"
        save_results(results, output_dir)
    else:
        print("\n❌ No predictions completed. Check API keys and try again.")


if __name__ == "__main__":
    main()
