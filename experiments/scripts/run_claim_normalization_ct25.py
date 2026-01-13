#!/usr/bin/env python3
"""
Claim Normalization Benchmark on CheckThat! 2025 Task 2.

Evaluates LLMs on transforming raw social media posts into normalized claims.
Uses METEOR score as the official evaluation metric.

SOTA: dfkinit2b achieved 0.4569 METEOR on English test set.

Models must have training cutoff before January 2025 to avoid data contamination.

Usage:
    # Quick test (10 samples)
    python run_claim_normalization_ct25.py --model mistral-small-24b --split dev --limit 10

    # Full dev set with parallelization (600 RPM)
    python run_claim_normalization_ct25.py --model mistral-small-24b --split dev --parallel 10

    # Compare models on test set
    python run_claim_normalization_ct25.py --compare-models --split test --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from openai import AsyncOpenAI
from tqdm import tqdm

# Ensure NLTK data is available
import nltk
for resource in ['punkt', 'punkt_tab', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=True)  # Override any existing env vars

from src.checkworthiness.config import MODELS, ModelConfig


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = REPO_ROOT / "data" / "raw" / "check_that_25"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "claim_normalization"

# Models eligible for CT25 (training cutoff < Jan 2025)
# Sorted by cost (ascending) for progressive experimentation
# Minimum 7B parameters for quality claim normalization
ELIGIBLE_MODELS = [
    # Tier 1: Cheapest (< $0.20/M) - start here
    "mistral-small-24b",    # 24B, 2024 cutoff - strong, $0.10-0.30/M
    # "llama-3.1-8b",       # 8B, Dec 2023 cutoff - $0.18/M (often unresponsive)
     "gpt-4o-mini",        # ~8B, Oct 2023 cutoff - $0.15-0.60/M (uncomment OPENAI_API_KEY in .env)
    "mistral-7b-v0.3",      # 7B, 2023 cutoff - $0.20/M
    # Tier 2: Mid-range ($0.20-$1.00/M)
    "qwen-2.5-7b",          # 7B, End 2023 cutoff - $0.30/M
    "mixtral-8x7b",         # 8x7B MoE, 2023 cutoff - $0.60/M
    "qwen-2.5-14b",         # 14B, End 2023 cutoff - $0.80/M
    "llama-3.3-70b",        # 70B, Dec 2023 cutoff - latest Llama, $0.88/M
    "llama-3.1-70b",        # 70B, Dec 2023 cutoff - $0.88/M
    # Tier 3: Premium ($1.00+/M)
    "qwen-2.5-72b",         # 72B, End 2023 cutoff - best on CT24, $1.20/M
    "deepseek-v3",          # MoE, Jul 2024 cutoff - borderline, $1.25/M
     "gpt-4o",             # ~200B, Oct 2023 cutoff - $2.50-10/M (uncomment OPENAI_API_KEY in .env)
    "llama-3.1-405b",       # 405B, Dec 2023 cutoff - largest, $3.50/M
]

# Zero-shot prompt for claim normalization
SYSTEM_PROMPT = """You are a claim normalization specialist for fact-checking systems.

Transform raw social media posts into concise, self-contained factual claims suitable for fact-checking.

<task>
Given a social media post, output a single normalized claim that:
1. Is a clear, declarative factual statement
2. Is self-contained (understandable without the original post)
3. Preserves the core factual assertion
4. Removes noise: emojis, hashtags, @mentions, URLs, duplicated text, "See More" markers
5. Uses neutral, formal language
</task>

<rules>
- If multiple claims exist, select the most central/checkworthy one
- Keep essential qualifiers (dates, numbers, names)
- Do NOT add information not present in the original
- Do NOT include opinions, emotions, or meta-commentary
- Output ONLY the normalized claim, nothing else
</rules>"""

USER_PROMPT_TEMPLATE = """Post: {post}

Normalized claim:"""


# =============================================================================
# Rate Limiter (from compare_models_ct24.py)
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for async requests."""

    def __init__(self, requests_per_minute: float = 600.0):
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
# Data Classes
# =============================================================================

@dataclass
class NormalizationResult:
    """Result of normalizing a single post."""
    idx: int
    post: str
    predicted_claim: str
    gold_claim: str | None
    meteor: float | None
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


# =============================================================================
# Data Loading
# =============================================================================

def load_data(split: str) -> pl.DataFrame:
    """Load CT25 data for given split."""
    file_map = {
        "train": "train-eng.csv",
        "dev": "dev-eng.csv",
        "test": "test-eng-gold.csv",
    }
    if split not in file_map:
        raise ValueError(f"Invalid split: {split}. Choose from: {list(file_map.keys())}")

    path = DATA_DIR / file_map[split]
    df = pl.read_csv(path)

    # Standardize column names
    if "normalized claim" in df.columns:
        df = df.rename({"normalized claim": "gold_claim"})

    # For test set without gold, gold_claim won't exist
    if "gold_claim" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("gold_claim"))

    # Add index for tracking
    df = df.with_row_index("idx")

    return df


# =============================================================================
# METEOR Score
# =============================================================================

def compute_meteor(prediction: str, reference: str) -> float:
    """Compute METEOR score between prediction and reference."""
    try:
        pred_tokens = word_tokenize(prediction.lower())
        ref_tokens = word_tokenize(reference.lower())
        return meteor_score([ref_tokens], pred_tokens)
    except Exception:
        return 0.0


# =============================================================================
# Model Interaction
# =============================================================================

async def normalize_single(
    client: AsyncOpenAI,
    config: ModelConfig,
    idx: int,
    post: str,
    gold_claim: str | None,
    model_name: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    timeout_seconds: float = 30.0,
) -> NormalizationResult:
    """Normalize a single post with rate limiting, concurrency control, and timeout."""
    async with semaphore:
        await rate_limiter.acquire()

        # Truncate very long posts
        post_truncated = post[:4000] if len(post) > 4000 else post
        user_prompt = USER_PROMPT_TEMPLATE.format(post=post_truncated)

        start = time.perf_counter()

        try:
            # Add timeout to API call
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=config.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                ),
                timeout=timeout_seconds,
            )

            latency_ms = (time.perf_counter() - start) * 1000
            predicted = response.choices[0].message.content.strip().strip('"\'')
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            predicted = f"TIMEOUT: No response after {timeout_seconds}s"
            input_tokens = 0
            output_tokens = 0

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            predicted = f"ERROR: {e}"
            input_tokens = 0
            output_tokens = 0

        # Compute METEOR if gold is available
        meteor = compute_meteor(predicted, gold_claim) if gold_claim else None

        return NormalizationResult(
            idx=idx,
            post=post,
            predicted_claim=predicted,
            gold_claim=gold_claim,
            meteor=meteor,
            model=model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


async def run_model_async(
    model_name: str,
    samples: list[dict],
    parallel: int = 10,
    rate_limit: float = 600.0,
    checkpoint_path: Path | None = None,
    timeout_seconds: float = 30.0,
    max_consecutive_failures: int = 10,
) -> list[NormalizationResult]:
    """Run evaluation with parallel requests.

    Args:
        model_name: Model to use
        samples: List of samples to process
        parallel: Max concurrent requests
        rate_limit: Max requests per minute
        checkpoint_path: Path to save incremental results
        timeout_seconds: Timeout per request (default 30s)
        max_consecutive_failures: Skip model after this many consecutive failures
    """

    # Check for existing checkpoint
    completed_idxs: set[int] = set()
    existing_results: list[NormalizationResult] = []

    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                data = json.loads(line)
                completed_idxs.add(data["idx"])
                existing_results.append(NormalizationResult(**data))
        print(f"  Resuming: {len(completed_idxs)} samples already completed")

    remaining = [s for s in samples if s["idx"] not in completed_idxs]

    if not remaining:
        print(f"  All samples already completed for {model_name}")
        return existing_results

    # Get model config
    if model_name not in MODELS:
        print(f"  ERROR: Model '{model_name}' not found in config")
        return existing_results

    config = MODELS[model_name]
    api_key = config.get_api_key()
    if not api_key:
        print(f"  SKIP: Missing API key ({config.api_key_env})")
        return existing_results

    # Initialize client
    client = AsyncOpenAI(api_key=api_key, base_url=config.api_base)

    # Set up concurrency control
    semaphore = asyncio.Semaphore(parallel)
    rate_limiter = RateLimiter(requests_per_minute=rate_limit)

    results = existing_results.copy()
    consecutive_failures = 0

    # Process in batches for checkpointing
    batch_size = max(25, parallel * 2)
    pbar = tqdm(total=len(remaining), desc=f"  {model_name}", leave=False)

    # Ensure checkpoint dir exists
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]

        tasks = [
            normalize_single(
                client, config, s["idx"], s["post"], s.get("gold_claim"),
                model_name, semaphore, rate_limiter, timeout_seconds
            )
            for s in batch
        ]

        batch_results = await asyncio.gather(*tasks)

        # Check for failures in this batch
        batch_failures = sum(1 for r in batch_results if r.predicted_claim.startswith(("ERROR:", "TIMEOUT:")))

        if batch_failures == len(batch_results):
            # Entire batch failed
            consecutive_failures += len(batch_results)
            print(f"\n  ⚠ Batch failed ({batch_failures}/{len(batch_results)}), consecutive failures: {consecutive_failures}")

            if consecutive_failures >= max_consecutive_failures:
                print(f"\n  ❌ SKIPPING {model_name}: {consecutive_failures} consecutive failures (unresponsive)")
                pbar.close()
                return results
        else:
            # At least some succeeded, reset counter
            consecutive_failures = 0

        results.extend(batch_results)
        pbar.update(len(batch))

        # Checkpoint after each batch
        if checkpoint_path:
            with open(checkpoint_path, "a") as f:
                for r in batch_results:
                    f.write(json.dumps({
                        "idx": r.idx,
                        "post": r.post,
                        "predicted_claim": r.predicted_claim,
                        "gold_claim": r.gold_claim,
                        "meteor": r.meteor,
                        "model": r.model,
                        "latency_ms": r.latency_ms,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                    }) + "\n")

    pbar.close()
    return results


def run_model(
    model_name: str,
    samples: list[dict],
    parallel: int = 10,
    rate_limit: float = 600.0,
    checkpoint_path: Path | None = None,
    timeout_seconds: float = 30.0,
) -> list[NormalizationResult]:
    """Run evaluation for a single model."""
    return asyncio.run(
        run_model_async(
            model_name, samples, parallel, rate_limit,
            checkpoint_path, timeout_seconds
        )
    )


async def run_all_models_async(
    models: list[str],
    samples: list[dict],
    parallel: int,
    rate_limit: float,
    results_dir: Path,
    timeout_seconds: float,
    no_resume: bool,
    split: str,
) -> dict[str, dict]:
    """Run all models in the same event loop to avoid cleanup issues."""
    all_stats = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        checkpoint = None if no_resume else results_dir / f"{model}_{split}.jsonl"

        try:
            results = await run_model_async(
                model_name=model,
                samples=samples,
                parallel=parallel,
                rate_limit=rate_limit,
                checkpoint_path=checkpoint,
                timeout_seconds=timeout_seconds,
            )

            if results and model in MODELS:
                stats = compute_stats(results, MODELS[model])
                all_stats[model] = stats

                # Count errors/timeouts
                n_errors = sum(1 for r in results if r.predicted_claim.startswith(("ERROR:", "TIMEOUT:")))
                valid_samples = len(results) - n_errors

                print(f"  ✓ METEOR: {stats['avg_meteor']:.4f} | "
                      f"Valid: {valid_samples}/{len(results)} | "
                      f"Cost: ${stats['total_cost_usd']:.4f}")

                # Save individual model results
                save_model_results(model, stats, split, results_dir)

        except Exception as e:
            print(f"  ❌ ERROR: {e}")

    return all_stats


# =============================================================================
# Results Aggregation
# =============================================================================

def compute_stats(results: list[NormalizationResult], config: ModelConfig) -> dict:
    """Compute aggregate statistics."""
    import statistics

    meteors = [r.meteor for r in results if r.meteor is not None]
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)
    total_latency = sum(r.latency_ms for r in results)

    cost = (total_in / 1_000_000 * config.cost_per_1m_input +
            total_out / 1_000_000 * config.cost_per_1m_output)

    return {
        "n_samples": len(results),
        "avg_meteor": statistics.mean(meteors) if meteors else 0.0,
        "std_meteor": statistics.stdev(meteors) if len(meteors) > 1 else 0.0,
        "median_meteor": statistics.median(meteors) if meteors else 0.0,
        "min_meteor": min(meteors) if meteors else 0.0,
        "max_meteor": max(meteors) if meteors else 0.0,
        "total_tokens": total_in + total_out,
        "total_cost_usd": cost,
        "avg_latency_ms": total_latency / len(results) if results else 0.0,
    }


def print_results(model_name: str, stats: dict, sota: float = 0.4569):
    """Print results for a single model."""
    delta = stats["avg_meteor"] - sota
    delta_str = f"{delta:+.4f}" if delta >= 0 else f"{delta:.4f}"

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Samples:      {stats['n_samples']:,}")
    print(f"  METEOR:       {stats['avg_meteor']:.4f} ± {stats['std_meteor']:.4f}")
    print(f"  Median:       {stats['median_meteor']:.4f}")
    print(f"  Range:        [{stats['min_meteor']:.4f}, {stats['max_meteor']:.4f}]")
    print(f"  vs SOTA:      {delta_str} (SOTA = {sota})")
    print(f"  Cost:         ${stats['total_cost_usd']:.4f}")
    print(f"  Avg Latency:  {stats['avg_latency_ms']:.0f}ms")
    print(f"{'='*60}")


def print_comparison_table(all_stats: dict[str, dict], sota: float = 0.4569):
    """Print comparison table for multiple models."""
    print("\n" + "="*85)
    print("CLAIM NORMALIZATION RESULTS (CheckThat! 2025 Task 2)")
    print("="*85)
    print(f"SOTA (dfkinit2b): {sota:.4f} METEOR")
    print("-"*85)
    print(f"{'Model':<25} {'METEOR':>10} {'± Std':>8} {'Δ SOTA':>10} {'Cost':>10} {'Latency':>10}")
    print("-"*85)

    # Sort by METEOR descending
    for model in sorted(all_stats.keys(), key=lambda m: -all_stats[m]["avg_meteor"]):
        s = all_stats[model]
        delta = s["avg_meteor"] - sota
        delta_str = f"{delta:+.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{model:<25} {s['avg_meteor']:>10.4f} {s['std_meteor']:>8.4f} {delta_str:>10} "
              f"${s['total_cost_usd']:>9.4f} {s['avg_latency_ms']:>8.0f}ms")

    print("-"*85)


def save_model_results(
    model_name: str,
    stats: dict,
    split: str,
    results_dir: Path,
    sota: float = 0.4569,
):
    """Save model results to a summary JSON file."""
    summary = {
        "model": model_name,
        "split": split,
        "sota": sota,
        "delta_sota": stats["avg_meteor"] - sota,
        **stats,
    }

    # Save individual model summary
    summary_path = results_dir / f"{model_name}_{split}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  📁 Saved: {summary_path.name}")

    return summary


def save_comparison_summary(all_stats: dict[str, dict], split: str, results_dir: Path):
    """Save comparison summary to CSV and JSON."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON
    json_path = results_dir / f"comparison_{split}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Save as CSV for easy viewing
    csv_path = results_dir / f"comparison_{split}_{timestamp}.csv"
    with open(csv_path, "w") as f:
        headers = ["model", "meteor", "std", "median", "min", "max", "cost_usd", "latency_ms", "samples"]
        f.write(",".join(headers) + "\n")

        for model in sorted(all_stats.keys(), key=lambda m: -all_stats[m]["avg_meteor"]):
            s = all_stats[model]
            row = [
                model,
                f"{s['avg_meteor']:.4f}",
                f"{s['std_meteor']:.4f}",
                f"{s['median_meteor']:.4f}",
                f"{s['min_meteor']:.4f}",
                f"{s['max_meteor']:.4f}",
                f"{s['total_cost_usd']:.4f}",
                f"{s['avg_latency_ms']:.0f}",
                str(s['n_samples']),
            ]
            f.write(",".join(row) + "\n")

    print(f"\n📊 Comparison saved:")
    print(f"   - {json_path.name}")
    print(f"   - {csv_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Claim Normalization on CT25")
    parser.add_argument("--model", type=str, default="mistral-small-24b",
                        help=f"Model to use. Eligible: {ELIGIBLE_MODELS}")
    parser.add_argument("--split", type=str, default="dev",
                        choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--parallel", type=int, default=10,
                        help="Max concurrent requests (default 10)")
    parser.add_argument("--rate-limit", type=float, default=600.0,
                        help="Max requests per minute (default 600)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Timeout per request in seconds (default 30)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing results")
    parser.add_argument("--compare-models", action="store_true",
                        help="Compare all eligible models")

    args = parser.parse_args()

    # Load data
    df = load_data(args.split)
    if args.limit:
        df = df.head(args.limit)

    samples = df.to_dicts()
    print(f"\nLoaded {len(samples)} samples from {args.split} split")

    # Debug: Show API keys being used
    import os
    print("\n=== API Keys ===")
    openai_key = os.getenv("OPENAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    print(f"OPENAI_API_KEY:   {openai_key[:30]}..." if openai_key else "OPENAI_API_KEY:   NOT SET")
    print(f"TOGETHER_API_KEY: {together_key[:30]}..." if together_key else "TOGETHER_API_KEY: NOT SET")
    print("================\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare_models:
        # Run all models in a single event loop to avoid async cleanup issues
        all_stats = asyncio.run(
            run_all_models_async(
                models=ELIGIBLE_MODELS,
                samples=samples,
                parallel=args.parallel,
                rate_limit=args.rate_limit,
                results_dir=RESULTS_DIR,
                timeout_seconds=args.timeout,
                no_resume=args.no_resume,
                split=args.split,
            )
        )

        # Print and save comparison
        print_comparison_table(all_stats)
        if all_stats:
            save_comparison_summary(all_stats, args.split, RESULTS_DIR)

    else:
        checkpoint = None if args.no_resume else RESULTS_DIR / f"{args.model}_{args.split}.jsonl"

        results = run_model(
            model_name=args.model,
            samples=samples,
            parallel=args.parallel,
            rate_limit=args.rate_limit,
            checkpoint_path=checkpoint,
            timeout_seconds=args.timeout,
        )

        if results and args.model in MODELS:
            stats = compute_stats(results, MODELS[args.model])
            print_results(args.model, stats)
            save_model_results(args.model, stats, args.split, RESULTS_DIR)


if __name__ == "__main__":
    main()
