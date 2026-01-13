#!/usr/bin/env python3
"""Compare mistral-small-24b vs qwen-2.5-72b with few-shot prompting.

This script evaluates both models on test + dev-test splits with:
- Zero-shot (no examples)
- 4-shot (2 positive, 2 negative)
- 8-shot (4 positive, 4 negative)
- 12-shot (6 positive, 6 negative)

Example selection strategies:
- random: Random balanced sampling from training set
- similar: Semantic similarity retrieval (embed query, find similar examples)
- diverse: Maximize diversity in example set

Each model uses its optimized prompt format:
- Mistral 24B: XML-style tags, role-first format
- Qwen 2.5 72B: Direct JSON output, clear instructions

Usage:
    # Full comparison (all models, all shots, all splits)
    python experiments/scripts/compare_fewshot_models.py

    # Quick test (5 samples per config)
    python experiments/scripts/compare_fewshot_models.py --dry-run

    # Specific model only
    python experiments/scripts/compare_fewshot_models.py --model mistral-small-24b

    # Specific shot configuration
    python experiments/scripts/compare_fewshot_models.py --shots 4

    # Use semantic similarity for example selection
    python experiments/scripts/compare_fewshot_models.py --selection similar
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
import polars as pl
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkworthiness.config import MODELS

load_dotenv(override=True)

# =============================================================================
# Constants
# =============================================================================

DATA_DIR = Path("data/raw/CT24_checkworthy_english")
V5_DATA_DIR = Path("data/processed/CT24_v5_normalized")
OUTPUT_DIR = Path("experiments/results/fewshot_comparison")

# Model-specific prompt files
PROMPT_FILES = {
    "mistral-small-24b": Path("prompts/checkworthiness_prompts_zeroshot_v5.yaml"),
    "qwen-2.5-72b": Path("prompts/checkworthiness_qwen.yaml"),
}

# Models to compare
COMPARISON_MODELS = ["mistral-small-24b", "qwen-2.5-72b"]

# Shot configurations (must be even for balanced selection)
SHOT_CONFIGS = [0, 4, 8, 12]

# Example selection strategies
SELECTION_STRATEGIES = ["random", "similar", "diverse"]

# Splits to evaluate
EVAL_SPLITS = ["test", "dev-test"]

# Rate limiting (Together AI: 600 req/min)
DEFAULT_RATE_LIMIT = 0.11
DEFAULT_MAX_WORKERS = 10

# Seed for reproducibility
RANDOM_SEED = 42

# Embedding model for similarity-based selection
EMBEDDING_MODEL = "text-embedding-3-small"


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter for API requests."""

    def __init__(self, min_interval: float = DEFAULT_RATE_LIMIT):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.last_request_time = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()


rate_limiter: RateLimiter | None = None


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PredictionResult:
    """Result for a single prediction."""
    sentence_id: str
    text: str
    true_label: str
    predicted_label: str
    confidence: float | None
    raw_response: str
    error: str | None = None


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model: str
    n_shots: int
    split: str
    selection_strategy: str
    examples: list[dict] = field(default_factory=list)


# =============================================================================
# Data Loading
# =============================================================================

def load_train_data() -> pl.DataFrame:
    """Load training data for few-shot examples."""
    # Try v5 first, fall back to raw
    v5_path = V5_DATA_DIR / "train_v5_text.parquet"
    if v5_path.exists():
        df = pl.read_parquet(v5_path)
        return df.select([
            pl.col("sentence_id"),
            pl.col("text"),
            pl.col("label"),
        ])

    # Fall back to raw TSV
    raw_path = DATA_DIR / "CT24_checkworthy_english_train.tsv"
    df = pl.read_csv(raw_path, separator="\t")
    return df.select([
        pl.col("Sentence_id").alias("sentence_id"),
        pl.col("Text").alias("text"),
        pl.col("class_label").alias("label"),
    ])


def load_eval_split(split: str) -> pl.DataFrame:
    """Load evaluation split (test or dev-test)."""
    if split == "test":
        v5_path = V5_DATA_DIR / "test_v5_text.parquet"
        if v5_path.exists():
            df = pl.read_parquet(v5_path)
            return df.select([
                pl.col("sentence_id"),
                pl.col("text"),
                pl.col("label"),
            ])
        raw_path = DATA_DIR / "CT24_checkworthy_english_test_gold.tsv"
    elif split == "dev-test":
        raw_path = DATA_DIR / "CT24_checkworthy_english_dev-test.tsv"
    else:
        raise ValueError(f"Unknown split: {split}")

    df = pl.read_csv(raw_path, separator="\t")
    return df.select([
        pl.col("Sentence_id").alias("sentence_id"),
        pl.col("Text").alias("text"),
        pl.col("class_label").alias("label"),
    ])


# =============================================================================
# Example Selection Strategies
# =============================================================================

class ExampleSelector:
    """Handles few-shot example selection with different strategies."""

    def __init__(
        self,
        train_df: pl.DataFrame,
        strategy: str = "random",
        seed: int = RANDOM_SEED,
    ):
        self.train_df = train_df
        self.strategy = strategy
        self.seed = seed
        self.embeddings: np.ndarray | None = None
        self.embedding_client: OpenAI | None = None

        # Split by label for balanced selection
        self.pos_df = train_df.filter(pl.col("label") == "Yes")
        self.neg_df = train_df.filter(pl.col("label") == "No")

        random.seed(seed)
        np.random.seed(seed)

    def _get_embeddings(self) -> np.ndarray:
        """Get or compute embeddings for training examples."""
        if self.embeddings is not None:
            return self.embeddings

        # Check for cached embeddings
        cache_path = Path("data/processed/train_embeddings_small.npy")
        if cache_path.exists():
            print("  Loading cached embeddings...")
            self.embeddings = np.load(cache_path)
            return self.embeddings

        # Compute embeddings
        print("  Computing embeddings for training set...")
        self.embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        texts = self.train_df["text"].to_list()
        embeddings = []

        # Batch embedding requests
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding"):
            batch = texts[i:i + batch_size]
            response = self.embedding_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            for item in response.data:
                embeddings.append(item.embedding)

        self.embeddings = np.array(embeddings)

        # Cache for future runs
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, self.embeddings)
        print(f"  Cached embeddings to {cache_path}")

        return self.embeddings

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        if self.embedding_client is None:
            self.embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = self.embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text],
        )
        return np.array(response.data[0].embedding)

    def select_random(self, n_shots: int) -> list[dict]:
        """Random balanced selection."""
        if n_shots == 0:
            return []

        n_per_class = n_shots // 2

        pos_indices = random.sample(range(len(self.pos_df)), min(n_per_class, len(self.pos_df)))
        neg_indices = random.sample(range(len(self.neg_df)), min(n_per_class, len(self.neg_df)))

        examples = []
        for i in range(max(len(pos_indices), len(neg_indices))):
            if i < len(pos_indices):
                row = self.pos_df.row(pos_indices[i], named=True)
                examples.append({"text": row["text"], "label": row["label"]})
            if i < len(neg_indices):
                row = self.neg_df.row(neg_indices[i], named=True)
                examples.append({"text": row["text"], "label": row["label"]})

        return examples[:n_shots]

    def select_similar(self, query_text: str, n_shots: int) -> list[dict]:
        """Select examples most similar to the query."""
        if n_shots == 0:
            return []

        n_per_class = n_shots // 2
        embeddings = self._get_embeddings()
        query_emb = self._embed_query(query_text)

        # Compute cosine similarities
        similarities = np.dot(embeddings, query_emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        # Get indices for each class
        labels = self.train_df["label"].to_list()
        pos_mask = np.array([l == "Yes" for l in labels])
        neg_mask = ~pos_mask

        # Get top-k similar for each class
        pos_sims = np.where(pos_mask, similarities, -np.inf)
        neg_sims = np.where(neg_mask, similarities, -np.inf)

        pos_top_k = np.argsort(pos_sims)[-n_per_class:][::-1]
        neg_top_k = np.argsort(neg_sims)[-n_per_class:][::-1]

        examples = []
        texts = self.train_df["text"].to_list()

        # Interleave positive and negative
        for i in range(n_per_class):
            if i < len(pos_top_k) and pos_sims[pos_top_k[i]] > -np.inf:
                examples.append({"text": texts[pos_top_k[i]], "label": "Yes"})
            if i < len(neg_top_k) and neg_sims[neg_top_k[i]] > -np.inf:
                examples.append({"text": texts[neg_top_k[i]], "label": "No"})

        return examples[:n_shots]

    def select_diverse(self, n_shots: int) -> list[dict]:
        """Select diverse examples using k-means clustering."""
        if n_shots == 0:
            return []

        n_per_class = n_shots // 2
        embeddings = self._get_embeddings()

        labels = self.train_df["label"].to_list()
        texts = self.train_df["text"].to_list()

        examples = []

        for target_label in ["Yes", "No"]:
            # Get embeddings for this class
            mask = np.array([l == target_label for l in labels])
            class_embeddings = embeddings[mask]
            class_texts = [t for t, m in zip(texts, mask) if m]

            if len(class_texts) <= n_per_class:
                # Use all if not enough
                for t in class_texts[:n_per_class]:
                    examples.append({"text": t, "label": target_label})
                continue

            # Use k-means++ initialization for diversity
            selected_indices = []
            remaining = list(range(len(class_embeddings)))

            # First point: random
            first = random.choice(remaining)
            selected_indices.append(first)
            remaining.remove(first)

            # Subsequent points: maximize min distance to selected
            for _ in range(n_per_class - 1):
                if not remaining:
                    break

                selected_embs = class_embeddings[selected_indices]
                max_min_dist = -1
                best_idx = remaining[0]

                for idx in remaining:
                    emb = class_embeddings[idx]
                    dists = np.linalg.norm(selected_embs - emb, axis=1)
                    min_dist = np.min(dists)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = idx

                selected_indices.append(best_idx)
                remaining.remove(best_idx)

            for idx in selected_indices:
                examples.append({"text": class_texts[idx], "label": target_label})

        # Shuffle to interleave labels
        random.shuffle(examples)
        return examples[:n_shots]

    def select(
        self,
        n_shots: int,
        query_text: str | None = None,
    ) -> list[dict]:
        """Select examples using the configured strategy."""
        if self.strategy == "random":
            return self.select_random(n_shots)
        elif self.strategy == "similar":
            if query_text is None:
                raise ValueError("query_text required for 'similar' strategy")
            return self.select_similar(query_text, n_shots)
        elif self.strategy == "diverse":
            return self.select_diverse(n_shots)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


# =============================================================================
# Prompt Building
# =============================================================================

def load_prompts(model: str) -> dict:
    """Load model-specific prompts."""
    prompt_file = PROMPT_FILES.get(model)
    if prompt_file is None or not prompt_file.exists():
        # Fall back to Mistral prompts
        prompt_file = PROMPT_FILES["mistral-small-24b"]

    with open(prompt_file) as f:
        return yaml.safe_load(f)


def build_mistral_prompt(
    claim: str,
    examples: list[dict],
) -> tuple[str, str]:
    """Build Mistral-optimized prompt with XML tags."""
    system_prompt = """You are an expert fact-checker determining if claims are checkworthy.

<definition>
A claim is CHECKWORTHY if it:
1. Makes a factual assertion that can be verified
2. Could cause harm if false and spread widely
3. Would benefit from professional fact-checking
</definition>

<constraints>
- Answer ONLY "Yes" or "No"
- "Yes" = checkworthy (should be fact-checked)
- "No" = not checkworthy (opinion, question, trivial, or unverifiable)
</constraints>"""

    user_parts = []

    if examples:
        user_parts.append("<examples>")
        for i, ex in enumerate(examples, 1):
            user_parts.append(f"<example_{i}>")
            user_parts.append(f"Claim: {ex['text']}")
            user_parts.append(f"Checkworthy: {ex['label']}")
            user_parts.append(f"</example_{i}>")
        user_parts.append("</examples>")
        user_parts.append("")

    user_parts.append(f"<claim>{claim}</claim>")
    user_parts.append("")
    user_parts.append("Is this claim checkworthy? Answer Yes or No:")

    return system_prompt, "\n".join(user_parts)


def build_qwen_prompt(
    claim: str,
    examples: list[dict],
) -> tuple[str, str]:
    """Build Qwen-optimized prompt with direct instructions."""
    system_prompt = """You are a fact-checking expert determining if claims are checkworthy.

A claim is CHECKWORTHY (Yes) if:
1. It makes a verifiable factual assertion
2. It could cause harm if false and widely spread
3. It would benefit from professional fact-checking

A claim is NOT CHECKWORTHY (No) if:
- It's an opinion, question, or command
- It's trivial or unverifiable
- It's clearly hypothetical or satirical

Respond with ONLY "Yes" or "No"."""

    user_parts = []

    if examples:
        user_parts.append("Examples:")
        for i, ex in enumerate(examples, 1):
            user_parts.append(f"{i}. Claim: {ex['text']}")
            user_parts.append(f"   Answer: {ex['label']}")
        user_parts.append("")

    user_parts.append(f"Claim: {claim}")
    user_parts.append("")
    user_parts.append("Is this claim checkworthy?")

    return system_prompt, "\n".join(user_parts)


def build_prompt(
    model: str,
    claim: str,
    examples: list[dict],
) -> tuple[str, str]:
    """Build model-specific prompt."""
    if "mistral" in model.lower():
        return build_mistral_prompt(claim, examples)
    elif "qwen" in model.lower():
        return build_qwen_prompt(claim, examples)
    else:
        # Default to Qwen style (simpler)
        return build_qwen_prompt(claim, examples)


# =============================================================================
# API Calling
# =============================================================================

def get_client(model_name: str) -> OpenAI:
    """Get OpenAI client for the given model."""
    config = MODELS[model_name]
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key: {config.api_key_env}")

    return OpenAI(
        api_key=api_key,
        base_url=config.api_base,
    )


def predict_single(
    client: OpenAI,
    model_name: str,
    sample: dict,
    examples: list[dict],
) -> PredictionResult:
    """Make a single prediction."""
    config = MODELS[model_name]

    try:
        if rate_limiter:
            rate_limiter.wait()

        system_prompt, user_prompt = build_prompt(
            model=model_name,
            claim=sample["text"],
            examples=examples,
        )

        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )

        raw_response = response.choices[0].message.content.strip()

        # Parse Yes/No from response
        response_lower = raw_response.lower()
        if "yes" in response_lower:
            predicted = "Yes"
        elif "no" in response_lower:
            predicted = "No"
        else:
            predicted = "No"  # Default on unclear

        return PredictionResult(
            sentence_id=str(sample["sentence_id"]),
            text=sample["text"],
            true_label=sample["label"],
            predicted_label=predicted,
            confidence=None,
            raw_response=raw_response,
        )

    except Exception as e:
        return PredictionResult(
            sentence_id=str(sample["sentence_id"]),
            text=sample["text"],
            true_label=sample["label"],
            predicted_label="No",
            confidence=None,
            raw_response="",
            error=str(e),
        )


def run_experiment(
    config: ExperimentConfig,
    samples: list[dict],
    example_selector: ExampleSelector,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[PredictionResult]:
    """Run a single experiment configuration."""
    client = get_client(config.model)
    results = []

    desc = f"{config.model.split('-')[0]} {config.n_shots}-shot {config.selection_strategy[:3]} {config.split}"

    # For 0-shot, no examples needed
    if config.n_shots == 0:
        fixed_examples = []
        use_per_sample_examples = False
    # For similarity-based selection, we need per-sample examples
    elif config.selection_strategy == "similar":
        fixed_examples = None
        use_per_sample_examples = True
    else:
        # Pre-select examples (same for all samples)
        fixed_examples = example_selector.select(config.n_shots)
        use_per_sample_examples = False

    def process_sample(sample: dict) -> PredictionResult:
        if use_per_sample_examples:
            examples = example_selector.select(config.n_shots, query_text=sample["text"])
        else:
            examples = fixed_examples

        return predict_single(client, config.model, sample, examples)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}

        with tqdm(total=len(samples), desc=desc, leave=False) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    return results


# =============================================================================
# Evaluation
# =============================================================================

def compute_metrics(results: list[PredictionResult]) -> dict:
    """Compute classification metrics."""
    y_true = [r.true_label for r in results]
    y_pred = [r.predicted_label for r in results]

    n_errors = sum(1 for r in results if r.error)

    return {
        "n_samples": len(results),
        "n_errors": n_errors,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, pos_label="Yes"),
        "precision": precision_score(y_true, y_pred, pos_label="Yes", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="Yes", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["Yes", "No"]).tolist(),
    }


def print_results_table(all_results: dict):
    """Print results as a formatted table."""
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    print(f"\n{'Model':<18} {'Shots':<6} {'Sel':<6} {'Split':<10} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8}")
    print("-" * 90)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["f1"],
        reverse=True,
    )

    for key, metrics in sorted_results:
        model, n_shots, selection, split = key
        model_short = model.replace("-small", "").replace("-2.5", "")
        print(
            f"{model_short:<18} {n_shots:<6} {selection[:3]:<6} {split:<10} "
            f"{metrics['f1']:.4f}   {metrics['accuracy']:.4f}   "
            f"{metrics['precision']:.4f}   {metrics['recall']:.4f}"
        )

    print("-" * 90)

    best_key, best_metrics = sorted_results[0]
    print(f"\n🏆 Best: {best_key[0]} {best_key[1]}-shot {best_key[2]} on {best_key[3]} (F1={best_metrics['f1']:.4f})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare mistral-small-24b vs qwen-2.5-72b with few-shot prompting"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=COMPARISON_MODELS + ["all"],
        default="all",
        help="Model to evaluate (default: all)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        choices=SHOT_CONFIGS + [-1],
        default=-1,
        help="Shot configuration (-1 = all)",
    )
    parser.add_argument(
        "--selection",
        type=str,
        choices=SELECTION_STRATEGIES + ["all"],
        default="random",
        help="Example selection strategy (default: random)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=EVAL_SPLITS + ["all"],
        default="all",
        help="Split to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Parallel workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help=f"Min seconds between requests (default: {DEFAULT_RATE_LIMIT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 5 samples per configuration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for example selection (default: {RANDOM_SEED})",
    )

    args = parser.parse_args()

    # Initialize rate limiter
    global rate_limiter
    rate_limiter = RateLimiter(min_interval=args.rate_limit)

    # Determine what to run
    models = COMPARISON_MODELS if args.model == "all" else [args.model]
    shots = SHOT_CONFIGS if args.shots == -1 else [args.shots]
    selections = SELECTION_STRATEGIES if args.selection == "all" else [args.selection]
    splits = EVAL_SPLITS if args.split == "all" else [args.split]

    print("\n" + "=" * 60)
    print("FEW-SHOT MODEL COMPARISON")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Shots: {shots}")
    print(f"Selection strategies: {selections}")
    print(f"Splits: {splits}")
    print(f"Max workers: {args.max_workers}")
    print(f"Rate limit: {args.rate_limit}s (~{1/args.rate_limit:.1f} req/sec)")
    print(f"Random seed: {args.seed}")
    if args.dry_run:
        print("DRY RUN: 5 samples per config")
    print("=" * 60)

    # Load training data
    print("\nLoading training data...")
    train_df = load_train_data()
    print(f"  Train examples: {len(train_df)}")
    print(f"    Positive (Yes): {len(train_df.filter(pl.col('label') == 'Yes'))}")
    print(f"    Negative (No): {len(train_df.filter(pl.col('label') == 'No'))}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = {}
    all_predictions = []

    for selection in selections:
        print(f"\n{'='*60}")
        print(f"Selection strategy: {selection.upper()}")
        print("=" * 60)

        # Create selector for this strategy
        selector = ExampleSelector(train_df, strategy=selection, seed=args.seed)

        # Show example selection (for non-similar, which is fixed)
        if selection != "similar" and shots and max(shots) > 0:
            sample_examples = selector.select(min(4, max(shots)))
            print(f"\nSample examples ({selection}):")
            for ex in sample_examples[:4]:
                emoji = "✓" if ex["label"] == "Yes" else "✗"
                print(f"  {emoji} [{ex['label']}] {ex['text'][:50]}...")

        for model in models:
            for n_shots in shots:
                for split in splits:
                    print(f"\n{'#' * 60}")
                    print(f"# {model} | {n_shots}-shot | {selection} | {split}")
                    print(f"{'#' * 60}")

                    # Load eval data
                    eval_df = load_eval_split(split)
                    samples = eval_df.to_dicts()

                    if args.dry_run:
                        samples = samples[:5]

                    print(f"Samples: {len(samples)}")

                    # Create config
                    config = ExperimentConfig(
                        model=model,
                        n_shots=n_shots,
                        split=split,
                        selection_strategy=selection,
                    )

                    # Run
                    results = run_experiment(
                        config=config,
                        samples=samples,
                        example_selector=selector,
                        max_workers=args.max_workers,
                    )

                    # Compute metrics
                    metrics = compute_metrics(results)
                    key = (model, n_shots, selection, split)
                    all_results[key] = metrics

                    print(f"Results: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

                    # Store predictions
                    for r in results:
                        all_predictions.append({
                            "model": model,
                            "n_shots": n_shots,
                            "selection": selection,
                            "split": split,
                            "sentence_id": r.sentence_id,
                            "text": r.text,
                            "true_label": r.true_label,
                            "predicted_label": r.predicted_label,
                            "raw_response": r.raw_response,
                            "error": r.error,
                        })

    # Print summary table
    print_results_table(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_path = OUTPUT_DIR / f"fewshot_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json_results = {
            f"{m}_{s}shot_{sel}_{sp}": v
            for (m, s, sel, sp), v in all_results.items()
        }
        json.dump(json_results, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    pred_path = OUTPUT_DIR / f"fewshot_predictions_{timestamp}.parquet"
    pred_df = pl.DataFrame(all_predictions)
    pred_df.write_parquet(pred_path)
    print(f"Predictions saved: {pred_path}")


if __name__ == "__main__":
    main()
