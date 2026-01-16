#!/usr/bin/env python3
"""
Hyperparameter tuning for contrastive few-shot claim normalization.

Optimized for 1xA10 GPU (24GB VRAM) - maximizes throughput via:
- Pre-computed cluster cache (avoid redundant K-means)
- High parallelism for API calls (GPU not the bottleneck)
- Batch embedding computation
- Smart search order (most promising configs first)

Search strategies:
- grid: Exhaustive grid search (default, thorough but slow)
- binary: Coordinate descent with ternary search (fast, assumes unimodal response)

Tunes the following hyperparameters:
1. topic_clusters: Number of K-means clusters for training post clustering
2. retrieval_threshold: Similarity threshold for retrieval fallback
3. num_positive: Number of positive (good) few-shot examples
4. num_negative: Number of negative (bad) contrastive examples

Usage:
    # Grid search (default, exhaustive)
    python tune_fewshot_hyperparams.py --model deepseek-chat --contrastive-file results.jsonl

    # Binary search (fast, ~10x fewer evaluations)
    python tune_fewshot_hyperparams.py --model deepseek-chat --search binary

    # Specify parameter ranges
    python tune_fewshot_hyperparams.py --model deepseek-chat \\
        --topic-clusters 0,10,20,30 \\
        --retrieval-threshold 0.7,0.8,0.85,0.9 \\
        --num-positive 2,3,4,5 \\
        --num-negative 0,1,2,3 \\
        --contrastive-file results.jsonl

    # Quick test with fewer combinations
    python tune_fewshot_hyperparams.py --model deepseek-chat --limit 100 --max-evals 20
"""

import argparse
import asyncio
import itertools
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

# Import from the main script
from run_claim_normalization_ct25 import (
    MODELS,
    RESULTS_DIR,
    SYSTEM_PROMPT,
    ContrastiveExamples,
    ExampleRetriever,
    RateLimiter,
    compute_meteor,
    load_data,
    normalize_single,
)
from openai import AsyncOpenAI

# =============================================================================
# Configuration - A10 GPU Optimized
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "raw" / "check_that_25"

# A10 optimized defaults - GPU handles embeddings, API is the bottleneck
DEFAULT_PARALLEL = 20  # High parallelism - A10 not bottleneck for API calls
DEFAULT_RATE_LIMIT = 500.0  # Requests per minute (adjust per provider)
DEFAULT_BATCH_SIZE = 128  # Embedding batch size for A10

# Default hyperparameter search spaces
DEFAULT_TOPIC_CLUSTERS = [10, 20, 30]
DEFAULT_RETRIEVAL_THRESHOLD = [0.7, 0.8, 0.85, 0.9, 1.0]
DEFAULT_NUM_POSITIVE = [2, 3, 4, 5]
DEFAULT_NUM_NEGATIVE = [0, 1, 2, 3]


@dataclass
class TuningResult:
    """Result of a single hyperparameter configuration."""
    topic_clusters: int
    retrieval_threshold: float
    num_positive: int
    num_negative: int
    avg_meteor: float
    std_meteor: float
    n_samples: int
    n_retrieval_hits: int
    latency_ms: float
    timestamp: str


@dataclass
class ClusterCache:
    """Pre-computed cluster assignments for all cluster counts.

    A10 optimization: Compute K-means once per cluster count, reuse across configs.
    """
    embeddings: np.ndarray
    cluster_data: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def build(cls, embeddings: np.ndarray, cluster_counts: list[int]) -> "ClusterCache":
        """Pre-compute clusters for all specified counts."""
        from sklearn.cluster import KMeans

        cache = cls(embeddings=embeddings)

        # Filter out 0 (no clustering)
        counts_to_compute = [c for c in cluster_counts if c > 0]

        if counts_to_compute:
            print(f"Pre-computing K-means for {len(counts_to_compute)} cluster configurations...")
            for n_clusters in counts_to_compute:
                print(f"  Computing {n_clusters} clusters...", end=" ", flush=True)
                start = time.perf_counter()
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                centroids = kmeans.cluster_centers_
                cache.cluster_data[n_clusters] = (labels, centroids)
                elapsed = time.perf_counter() - start
                print(f"done ({elapsed:.1f}s)")

        return cache

    def get_retriever_with_clusters(
        self,
        base_retriever: ExampleRetriever,
        n_clusters: int,
    ) -> ExampleRetriever:
        """Get a retriever configured with the specified cluster count."""
        if n_clusters == 0:
            # No clustering - return base retriever with clusters disabled
            return ExampleRetriever(
                posts=base_retriever.posts,
                claims=base_retriever.claims,
                embeddings=base_retriever.embeddings,
                model=base_retriever.model,
                bm25=base_retriever.bm25,
                tokenized_posts=base_retriever.tokenized_posts,
                cluster_labels=None,
                cluster_centroids=None,
                n_clusters=0,
            )

        labels, centroids = self.cluster_data[n_clusters]
        return ExampleRetriever(
            posts=base_retriever.posts,
            claims=base_retriever.claims,
            embeddings=base_retriever.embeddings,
            model=base_retriever.model,
            bm25=base_retriever.bm25,
            tokenized_posts=base_retriever.tokenized_posts,
            cluster_labels=labels,
            cluster_centroids=centroids,
            n_clusters=n_clusters,
        )


# =============================================================================
# Tuning Functions - A10 Optimized
# =============================================================================

async def evaluate_config(
    model_name: str,
    samples: list[dict],
    retriever: ExampleRetriever,
    contrastive: ContrastiveExamples | None,
    topic_clusters: int,
    retrieval_threshold: float,
    num_positive: int,
    num_negative: int,
    cluster_cache: ClusterCache,
    client: AsyncOpenAI,
    config,
    parallel: int = DEFAULT_PARALLEL,
    rate_limit: float = DEFAULT_RATE_LIMIT,
    timeout_seconds: float = 30.0,
) -> TuningResult:
    """Evaluate a single hyperparameter configuration.

    A10 optimized: Reuses client, uses pre-computed clusters from cache.
    """
    # Get retriever with correct cluster configuration from cache
    retriever_for_eval = cluster_cache.get_retriever_with_clusters(retriever, topic_clusters)

    semaphore = asyncio.Semaphore(parallel)
    rate_limiter = RateLimiter(rate_limit)

    start_time = time.perf_counter()
    meteors = []
    retrieval_hits = 0

    tasks = []
    for sample in samples:
        task = normalize_single(
            client=client,
            config=config,
            idx=sample["idx"],
            post=sample["post"],
            gold_claim=sample.get("gold_claim"),
            model_name=model_name,
            semaphore=semaphore,
            rate_limiter=rate_limiter,
            retriever=retriever_for_eval if num_positive > 0 else None,
            num_examples=num_positive,
            timeout_seconds=timeout_seconds,
            retrieval_threshold=retrieval_threshold,
            claim_verify_threshold=0.5,
            retrieval_mode="embedding",
            system_prompt=SYSTEM_PROMPT,
            ensemble=1,
            ensemble_temperature=0.7,
            clusters=None,
            cluster_aware=False,
            contrastive=contrastive if num_negative > 0 else None,
            contrastive_k=num_negative,
            contrastive_mode="similar",
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            continue
        if result.meteor is not None:
            meteors.append(result.meteor)
        if result.model.startswith("retrieval@"):
            retrieval_hits += 1

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    avg_meteor = np.mean(meteors) if meteors else 0.0
    std_meteor = np.std(meteors) if meteors else 0.0

    return TuningResult(
        topic_clusters=topic_clusters,
        retrieval_threshold=retrieval_threshold,
        num_positive=num_positive,
        num_negative=num_negative,
        avg_meteor=avg_meteor,
        std_meteor=std_meteor,
        n_samples=len(meteors),
        n_retrieval_hits=retrieval_hits,
        latency_ms=elapsed_ms,
        timestamp=datetime.now().isoformat(),
    )


def smart_order_combinations(
    combos: list[tuple],
    prior_results: list[TuningResult] | None = None,
) -> list[tuple]:
    """Order combinations to try most promising first.

    A10 optimization: Start with configs likely to perform well based on priors.
    Default heuristic: middle values tend to work best (avoid extremes).
    """
    if prior_results:
        # If we have prior results, use them to inform ordering
        # (Future: could use Bayesian optimization here)
        pass

    # Default: sort by expected performance heuristic
    # - Middle cluster counts (10-20) often work best
    # - Middle thresholds (0.8-0.85) often work best
    # - 3-4 positive examples often work best
    # - 1-2 negative examples often work best
    def score(combo):
        tc, rt, np_, nn = combo
        score = 0
        # Prefer middle cluster counts
        if 10 <= tc <= 20:
            score += 2
        elif tc > 0:
            score += 1
        # Prefer middle thresholds
        if 0.8 <= rt <= 0.9:
            score += 2
        # Prefer 3-4 positive examples
        if 3 <= np_ <= 4:
            score += 2
        # Prefer 1-2 negative examples
        if 1 <= nn <= 2:
            score += 1
        return -score  # Negative for descending sort

    return sorted(combos, key=score)


async def run_tuning(
    model_name: str,
    samples: list[dict],
    train_df: pl.DataFrame,
    contrastive_file: Path | None,
    topic_clusters_list: list[int],
    retrieval_threshold_list: list[float],
    num_positive_list: list[int],
    num_negative_list: list[int],
    parallel: int = DEFAULT_PARALLEL,
    rate_limit: float = DEFAULT_RATE_LIMIT,
    timeout_seconds: float = 30.0,
    max_evals: int | None = None,
) -> list[TuningResult]:
    """Run hyperparameter tuning over all combinations.

    A10 optimized:
    - Pre-computes all cluster configurations once
    - Reuses single API client across all configs
    - Smart ordering to try most promising configs first
    """
    model_config = MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Create single client to reuse (A10 optimization: avoid connection overhead)
    client = AsyncOpenAI(
        api_key=os.getenv(model_config.api_key_env),
        base_url=model_config.api_base,
    )

    # Build base retriever WITHOUT clusters (we'll use cache for clusters)
    print(f"\nBuilding base retriever with {len(train_df)} training examples...")
    retriever = ExampleRetriever.from_dataframe(train_df, n_clusters=0)

    # Pre-compute all cluster configurations (A10 optimization: do once, reuse)
    cluster_cache = ClusterCache.build(retriever.embeddings, topic_clusters_list)

    # Load contrastive examples if provided
    contrastive = None
    if contrastive_file and contrastive_file.exists():
        print(f"Loading contrastive examples from {contrastive_file.name}...")
        contrastive = ContrastiveExamples.from_results_file(contrastive_file)

    # Generate all combinations
    all_combos = list(itertools.product(
        topic_clusters_list,
        retrieval_threshold_list,
        num_positive_list,
        num_negative_list,
    ))

    total_combos = len(all_combos)

    # Limit combinations if requested
    if max_evals and len(all_combos) > max_evals:
        import random
        random.seed(42)
        all_combos = random.sample(all_combos, max_evals)
        print(f"Randomly sampling {max_evals} configurations from {total_combos} total")

    # Smart ordering: try most promising configs first
    all_combos = smart_order_combinations(all_combos)

    print(f"\nTuning {len(all_combos)} hyperparameter configurations...")
    print(f"  topic_clusters: {topic_clusters_list}")
    print(f"  retrieval_threshold: {retrieval_threshold_list}")
    print(f"  num_positive: {num_positive_list}")
    print(f"  num_negative: {num_negative_list}")
    print(f"  parallel: {parallel}, rate_limit: {rate_limit}/min")
    print()

    results = []
    best_meteor = 0.0
    best_config = None

    for i, (tc, rt, np_, nn) in enumerate(all_combos):
        print(f"[{i+1}/{len(all_combos)}] clusters={tc}, threshold={rt}, pos={np_}, neg={nn}", end=" ", flush=True)

        try:
            result = await evaluate_config(
                model_name=model_name,
                samples=samples,
                retriever=retriever,
                contrastive=contrastive,
                topic_clusters=tc,
                retrieval_threshold=rt,
                num_positive=np_,
                num_negative=nn,
                cluster_cache=cluster_cache,
                client=client,
                config=model_config,
                parallel=parallel,
                rate_limit=rate_limit,
                timeout_seconds=timeout_seconds,
            )
            results.append(result)

            # Track best
            marker = ""
            if result.avg_meteor > best_meteor:
                best_meteor = result.avg_meteor
                best_config = result
                marker = " *BEST*"

            print(f"-> METEOR={result.avg_meteor:.4f} (+/-{result.std_meteor:.4f}){marker}")

        except Exception as e:
            print(f"-> ERROR: {e}")
            continue

    return results, best_config


# =============================================================================
# Binary Search (Coordinate Descent with Ternary Search)
# =============================================================================

async def evaluate_single_config(
    model_name: str,
    samples: list[dict],
    retriever: ExampleRetriever,
    contrastive: ContrastiveExamples | None,
    cluster_cache: ClusterCache,
    client: AsyncOpenAI,
    config,
    tc: int,
    rt: float,
    np_: int,
    nn: int,
    parallel: int,
    rate_limit: float,
    timeout_seconds: float,
    cache: dict,
) -> float:
    """Evaluate a single config, using cache to avoid re-evaluation."""
    key = (tc, rt, np_, nn)
    if key in cache:
        return cache[key]

    result = await evaluate_config(
        model_name=model_name,
        samples=samples,
        retriever=retriever,
        contrastive=contrastive,
        topic_clusters=tc,
        retrieval_threshold=rt,
        num_positive=np_,
        num_negative=nn,
        cluster_cache=cluster_cache,
        client=client,
        config=config,
        parallel=parallel,
        rate_limit=rate_limit,
        timeout_seconds=timeout_seconds,
    )
    cache[key] = result.avg_meteor
    return result.avg_meteor


async def ternary_search_param(
    param_values: list,
    fixed_params: dict,
    param_name: str,
    eval_func,
) -> tuple:
    """Ternary search to find optimal value for a single parameter.

    Assumes unimodal response surface (single peak).
    Returns (best_value, best_score).
    """
    values = sorted(param_values)
    n = len(values)

    if n <= 3:
        # For small lists, just evaluate all
        best_val, best_score = None, -1
        for v in values:
            params = {**fixed_params, param_name: v}
            score = await eval_func(**params)
            print(f"    {param_name}={v} -> METEOR={score:.4f}")
            if score > best_score:
                best_score = score
                best_val = v
        return best_val, best_score

    # Ternary search for larger lists
    left, right = 0, n - 1

    while right - left > 2:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3

        params1 = {**fixed_params, param_name: values[mid1]}
        params2 = {**fixed_params, param_name: values[mid2]}

        score1 = await eval_func(**params1)
        score2 = await eval_func(**params2)

        print(f"    {param_name}={values[mid1]} -> {score1:.4f} | {param_name}={values[mid2]} -> {score2:.4f}")

        if score1 < score2:
            left = mid1
        else:
            right = mid2

    # Evaluate remaining candidates
    best_val, best_score = None, -1
    for i in range(left, right + 1):
        params = {**fixed_params, param_name: values[i]}
        score = await eval_func(**params)
        if score > best_score:
            best_score = score
            best_val = values[i]

    return best_val, best_score


async def run_binary_search(
    model_name: str,
    samples: list[dict],
    train_df: pl.DataFrame,
    contrastive_file: Path | None,
    topic_clusters_list: list[int],
    retrieval_threshold_list: list[float],
    num_positive_list: list[int],
    num_negative_list: list[int],
    parallel: int = DEFAULT_PARALLEL,
    rate_limit: float = DEFAULT_RATE_LIMIT,
    timeout_seconds: float = 30.0,
    max_iterations: int = 3,
) -> tuple[list[TuningResult], TuningResult | None]:
    """Run coordinate descent with ternary search.

    Much faster than grid search (~10x fewer evaluations).
    Assumes each parameter has a unimodal effect on METEOR.

    Algorithm:
    1. Start with middle values for all parameters
    2. For each parameter, use ternary search to find optimal value
    3. Repeat until convergence or max_iterations
    """
    model_config = MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Create single client
    client = AsyncOpenAI(
        api_key=os.getenv(model_config.api_key_env),
        base_url=model_config.api_base,
    )

    # Build retriever and cluster cache
    print(f"\nBuilding base retriever with {len(train_df)} training examples...")
    retriever = ExampleRetriever.from_dataframe(train_df, n_clusters=0)
    cluster_cache = ClusterCache.build(retriever.embeddings, topic_clusters_list)

    # Load contrastive examples
    contrastive = None
    if contrastive_file and contrastive_file.exists():
        print(f"Loading contrastive examples from {contrastive_file.name}...")
        contrastive = ContrastiveExamples.from_results_file(contrastive_file)

    # Cache for evaluated configs (avoid re-evaluation)
    eval_cache: dict[tuple, float] = {}
    all_results: list[TuningResult] = []

    # Helper to evaluate and track results
    async def eval_and_track(tc, rt, np_, nn) -> float:
        score = await evaluate_single_config(
            model_name, samples, retriever, contrastive,
            cluster_cache, client, model_config,
            tc, rt, np_, nn, parallel, rate_limit, timeout_seconds,
            eval_cache,
        )
        # Track for results
        result = TuningResult(
            topic_clusters=tc, retrieval_threshold=rt,
            num_positive=np_, num_negative=nn,
            avg_meteor=score, std_meteor=0.0,
            n_samples=len(samples), n_retrieval_hits=0,
            latency_ms=0, timestamp=datetime.now().isoformat(),
        )
        # Avoid duplicates
        if not any(r.topic_clusters == tc and r.retrieval_threshold == rt
                   and r.num_positive == np_ and r.num_negative == nn
                   for r in all_results):
            all_results.append(result)
        return score

    # Initialize with middle values
    def middle(lst):
        return lst[len(lst) // 2]

    best_tc = middle(topic_clusters_list)
    best_rt = middle(retrieval_threshold_list)
    best_np = middle(num_positive_list)
    best_nn = middle(num_negative_list)

    print(f"\n{'='*60}")
    print("BINARY SEARCH (Coordinate Descent)")
    print(f"{'='*60}")
    print(f"Initial: clusters={best_tc}, threshold={best_rt}, pos={best_np}, neg={best_nn}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*60}\n")

    prev_score = -1

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Optimize each parameter in turn
        # 1. Topic clusters
        print(f"\n  Optimizing topic_clusters (current={best_tc}):")
        best_tc, score = await ternary_search_param(
            topic_clusters_list,
            {"rt": best_rt, "np_": best_np, "nn": best_nn},
            "tc",
            lambda tc, rt, np_, nn: eval_and_track(tc, rt, np_, nn),
        )
        print(f"  -> Best: {best_tc} (METEOR={score:.4f})")

        # 2. Retrieval threshold
        print(f"\n  Optimizing retrieval_threshold (current={best_rt}):")
        best_rt, score = await ternary_search_param(
            retrieval_threshold_list,
            {"tc": best_tc, "np_": best_np, "nn": best_nn},
            "rt",
            lambda tc, rt, np_, nn: eval_and_track(tc, rt, np_, nn),
        )
        print(f"  -> Best: {best_rt} (METEOR={score:.4f})")

        # 3. Num positive
        print(f"\n  Optimizing num_positive (current={best_np}):")
        best_np, score = await ternary_search_param(
            num_positive_list,
            {"tc": best_tc, "rt": best_rt, "nn": best_nn},
            "np_",
            lambda tc, rt, np_, nn: eval_and_track(tc, rt, np_, nn),
        )
        print(f"  -> Best: {best_np} (METEOR={score:.4f})")

        # 4. Num negative
        print(f"\n  Optimizing num_negative (current={best_nn}):")
        best_nn, score = await ternary_search_param(
            num_negative_list,
            {"tc": best_tc, "rt": best_rt, "np_": best_np},
            "nn",
            lambda tc, rt, np_, nn: eval_and_track(tc, rt, np_, nn),
        )
        print(f"  -> Best: {best_nn} (METEOR={score:.4f})")

        print(f"\n  Iteration {iteration + 1} result: clusters={best_tc}, threshold={best_rt}, pos={best_np}, neg={best_nn}")
        print(f"  METEOR: {score:.4f}")

        # Check convergence
        if abs(score - prev_score) < 0.001:
            print(f"\n  Converged! (delta < 0.001)")
            break
        prev_score = score

    # Find best result
    best_config = max(all_results, key=lambda r: r.avg_meteor) if all_results else None

    print(f"\n{'='*60}")
    print(f"Binary search completed: {len(eval_cache)} unique configurations evaluated")
    print(f"{'='*60}")

    return all_results, best_config


def print_results_table(results: list[TuningResult]):
    """Print results as a formatted table."""
    if not results:
        print("No results to display.")
        return

    # Sort by METEOR descending
    sorted_results = sorted(results, key=lambda x: x.avg_meteor, reverse=True)

    print("\n" + "=" * 90)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 90)
    print(f"{'Rank':<5} {'Clusters':<10} {'Threshold':<10} {'Positive':<10} {'Negative':<10} {'METEOR':<12} {'Std':<10}")
    print("-" * 90)

    for i, r in enumerate(sorted_results[:20], 1):  # Top 20
        print(f"{i:<5} {r.topic_clusters:<10} {r.retrieval_threshold:<10.2f} {r.num_positive:<10} {r.num_negative:<10} {r.avg_meteor:<12.4f} {r.std_meteor:<10.4f}")

    print("-" * 90)


def save_results(results: list[TuningResult], best: TuningResult, output_path: Path, model_name: str):
    """Save results to JSON file."""
    output = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "best_config": {
            "topic_clusters": best.topic_clusters,
            "retrieval_threshold": best.retrieval_threshold,
            "num_positive": best.num_positive,
            "num_negative": best.num_negative,
            "avg_meteor": best.avg_meteor,
            "std_meteor": best.std_meteor,
        },
        "all_results": [
            {
                "topic_clusters": r.topic_clusters,
                "retrieval_threshold": r.retrieval_threshold,
                "num_positive": r.num_positive,
                "num_negative": r.num_negative,
                "avg_meteor": r.avg_meteor,
                "std_meteor": r.std_meteor,
                "n_samples": r.n_samples,
                "n_retrieval_hits": r.n_retrieval_hits,
                "latency_ms": r.latency_ms,
            }
            for r in sorted(results, key=lambda x: x.avg_meteor, reverse=True)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_list(value: str, cast_type=float) -> list:
    """Parse comma-separated list of values."""
    if not value:
        return []
    return [cast_type(x.strip()) for x in value.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for contrastive few-shot claim normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help=f"Model to use. Available: {list(MODELS.keys())}")
    parser.add_argument("--split", type=str, default="dev",
                        choices=["train", "dev", "test"],
                        help="Data split to evaluate on (default: dev)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for faster tuning)")

    # Hyperparameter ranges (comma-separated)
    parser.add_argument("--topic-clusters", type=str, default="0,10,20,30",
                        help="Comma-separated list of topic cluster counts to try (default: 0,10,20,30)")
    parser.add_argument("--retrieval-threshold", type=str, default="0.7,0.8,0.85,0.9,1.0",
                        help="Comma-separated list of retrieval thresholds (default: 0.7,0.8,0.85,0.9,1.0)")
    parser.add_argument("--num-positive", type=str, default="2,3,4,5",
                        help="Comma-separated list of positive example counts (default: 2,3,4,5)")
    parser.add_argument("--num-negative", type=str, default="0,1,2,3",
                        help="Comma-separated list of negative example counts (default: 0,1,2,3)")

    # Contrastive file
    parser.add_argument("--contrastive-file", type=str, default=None,
                        help="Path to results .jsonl file for contrastive examples")

    # Execution settings - A10 GPU optimized defaults
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                        help=f"Max concurrent requests (default: {DEFAULT_PARALLEL}, A10 optimized)")
    parser.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT,
                        help=f"Max requests per minute (default: {DEFAULT_RATE_LIMIT}, adjust per provider)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Timeout per request in seconds (default: 30)")
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Maximum number of configurations to evaluate (random sample if exceeded)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results (default: tuning_results_<model>_<timestamp>.json)")
    parser.add_argument("--search", type=str, default="grid",
                        choices=["grid", "binary"],
                        help="Search strategy: 'grid' (exhaustive, default) or 'binary' (coordinate descent, ~10x faster)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max iterations for binary search (default: 3)")

    args = parser.parse_args()

    # Parse hyperparameter lists
    topic_clusters_list = parse_list(args.topic_clusters, int)
    retrieval_threshold_list = parse_list(args.retrieval_threshold, float)
    num_positive_list = parse_list(args.num_positive, int)
    num_negative_list = parse_list(args.num_negative, int)

    # Validate
    if not topic_clusters_list:
        topic_clusters_list = DEFAULT_TOPIC_CLUSTERS
    if not retrieval_threshold_list:
        retrieval_threshold_list = DEFAULT_RETRIEVAL_THRESHOLD
    if not num_positive_list:
        num_positive_list = DEFAULT_NUM_POSITIVE
    if not num_negative_list:
        num_negative_list = DEFAULT_NUM_NEGATIVE

    # Check model
    if args.model not in MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {list(MODELS.keys())}")
        return

    # Load data
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")

    df = load_data(args.split)
    if args.limit:
        df = df.head(args.limit)
    samples = df.to_dicts()
    print(f"Samples: {len(samples)}")

    # Load training data for retriever
    train_df = load_data("train")
    if args.split != "dev":
        dev_df = load_data("dev")
        train_df = pl.concat([train_df, dev_df])
    print(f"Training examples for retriever: {len(train_df)}")

    # Count total combinations
    total_combos = len(topic_clusters_list) * len(retrieval_threshold_list) * len(num_positive_list) * len(num_negative_list)
    print(f"Total configurations: {total_combos}")
    if args.max_evals and total_combos > args.max_evals:
        print(f"Will randomly sample {args.max_evals} configurations")

    # Contrastive file
    contrastive_file = Path(args.contrastive_file) if args.contrastive_file else None
    if contrastive_file and not contrastive_file.exists():
        print(f"Warning: Contrastive file not found: {contrastive_file}")
        contrastive_file = None

    print(f"Search strategy: {args.search}")
    print(f"{'='*60}\n")

    # Run tuning with selected strategy
    if args.search == "binary":
        results, best = asyncio.run(run_binary_search(
            model_name=args.model,
            samples=samples,
            train_df=train_df,
            contrastive_file=contrastive_file,
            topic_clusters_list=topic_clusters_list,
            retrieval_threshold_list=retrieval_threshold_list,
            num_positive_list=num_positive_list,
            num_negative_list=num_negative_list,
            parallel=args.parallel,
            rate_limit=args.rate_limit,
            timeout_seconds=args.timeout,
            max_iterations=args.max_iterations,
        ))
    else:
        results, best = asyncio.run(run_tuning(
            model_name=args.model,
            samples=samples,
            train_df=train_df,
            contrastive_file=contrastive_file,
            topic_clusters_list=topic_clusters_list,
            retrieval_threshold_list=retrieval_threshold_list,
            num_positive_list=num_positive_list,
            num_negative_list=num_negative_list,
            parallel=args.parallel,
            rate_limit=args.rate_limit,
            timeout_seconds=args.timeout,
            max_evals=args.max_evals,
        ))

    # Print results
    print_results_table(results)

    # Print best configuration
    if best:
        print("\n" + "=" * 60)
        print("BEST CONFIGURATION")
        print("=" * 60)
        print(f"  topic_clusters:      {best.topic_clusters}")
        print(f"  retrieval_threshold: {best.retrieval_threshold}")
        print(f"  num_positive:        {best.num_positive}")
        print(f"  num_negative:        {best.num_negative}")
        print(f"  METEOR:              {best.avg_meteor:.4f} (+/- {best.std_meteor:.4f})")
        print("=" * 60)

        # Print command to run with best config
        print("\nRun with best configuration:")
        cmd = f"python run_claim_normalization_ct25.py --model {args.model}"
        cmd += f" --topic-clusters {best.topic_clusters}"
        cmd += f" --retrieval-threshold {best.retrieval_threshold}"
        cmd += f" --num-examples {best.num_positive}"
        if best.num_negative > 0 and contrastive_file:
            cmd += f" --contrastive-file {contrastive_file}"
            cmd += f" --contrastive-examples {best.num_negative}"
        print(f"  {cmd}")

    # Save results
    if results:
        output_path = Path(args.output) if args.output else RESULTS_DIR / f"tuning_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results(results, best, output_path, args.model)


if __name__ == "__main__":
    main()
