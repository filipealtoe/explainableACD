#!/usr/bin/env python3
"""
Claim Normalization Benchmark on CheckThat! 2025 Task 2.

Evaluates LLMs on transforming raw social media posts into normalized claims.
Uses METEOR score as the official evaluation metric.

SOTA: dfkinit2b achieved 0.4569 METEOR on English test set.

Models must have training cutoff before January 2025 to avoid data contamination.

Usage:
    # LOCAL GPU MODE (recommended for Lambda Labs A10)
    python run_claim_normalization_ct25.py --local --split dev --limit 10
    python run_claim_normalization_ct25.py --local --local-model mistralai/Mistral-7B-Instruct-v0.3 --split dev

    # Quick test with few-shot (default: 3 examples) via API
    python run_claim_normalization_ct25.py --model mistral-small-24b --split dev --limit 10

    # Full dev set with few-shot prompting
    python run_claim_normalization_ct25.py --model mistral-small-24b --split dev --parallel 10

    # Use 5 few-shot examples instead of default 3
    python run_claim_normalization_ct25.py --model gpt-4o-mini --split dev --num-examples 5

    # Zero-shot mode (no examples, for comparison)
    python run_claim_normalization_ct25.py --model mistral-small-24b --split dev --zero-shot

    # Compare models on test set
    python run_claim_normalization_ct25.py --compare-models --split test --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Ensure NLTK data is available
import nltk
for resource in ['punkt', 'punkt_tab', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Path setup - handle both full project and standalone script
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

# Try to load from project structure, fall back to standalone mode
try:
    sys.path.insert(0, str(REPO_ROOT))
    load_dotenv(REPO_ROOT / ".env", override=True)
    from src.checkworthiness.config import MODELS, ModelConfig
    API_MODE_AVAILABLE = True
except ImportError:
    # Standalone mode - API mode not available, only local mode works
    API_MODE_AVAILABLE = False
    MODELS = {}
    ModelConfig = None
    print("Note: Running in standalone mode (local GPU only, no API mode)")


# =============================================================================
# Configuration
# =============================================================================

# Data directory - check multiple possible locations
_possible_data_dirs = [
    REPO_ROOT / "data" / "raw" / "check_that_25",  # Full project structure
    SCRIPT_DIR / "data",                            # data/ next to script
    Path.cwd() / "data",                            # data/ in current directory
    Path.cwd(),                                     # Current directory (files directly here)
]
DATA_DIR = next((d for d in _possible_data_dirs if d.exists()), Path.cwd())
RESULTS_DIR = SCRIPT_DIR / "results" if not (REPO_ROOT / "experiments").exists() else REPO_ROOT / "experiments" / "results" / "claim_normalization"

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

# Embedding model for few-shot retrieval
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


# =============================================================================
# Few-Shot Example Retriever
# =============================================================================

@dataclass
class ExampleRetriever:
    """Retrieves similar examples from training data for few-shot prompting."""

    posts: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    model: SentenceTransformer | None = None

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, embedding_model: str = EMBEDDING_MODEL) -> "ExampleRetriever":
        """Create retriever from training DataFrame."""
        import torch

        # Filter to rows with gold claims
        df_valid = df.filter(pl.col("gold_claim").is_not_null())

        posts = df_valid["post"].to_list()
        claims = df_valid["gold_claim"].to_list()

        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {embedding_model} (device: {device})")
        model = SentenceTransformer(embedding_model, device=device)

        print(f"Embedding {len(posts)} training examples...")
        # E5 models need "query: " prefix for queries and "passage: " for documents
        prefixed_posts = [f"passage: {p[:1000]}" for p in posts]  # Truncate long posts
        # Use larger batch size on GPU for faster embedding
        batch_size = 64 if device == "cuda" else 32
        embeddings = model.encode(
            prefixed_posts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=batch_size
        )

        return cls(posts=posts, claims=claims, embeddings=embeddings, model=model)

    def get_similar_examples(self, query_post: str, k: int = 3) -> list[tuple[str, str, float]]:
        """Find k most similar training examples for a query post.

        Returns list of (post, claim, similarity_score) tuples.
        """
        if self.embeddings is None or self.model is None:
            return []

        # E5 models need "query: " prefix for queries
        query_embedding = self.model.encode(
            f"query: {query_post[:1000]}",
            convert_to_numpy=True
        )

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_k_idx = np.argsort(similarities)[-k:][::-1]

        return [
            (self.posts[i], self.claims[i], float(similarities[i]))
            for i in top_k_idx
        ]

    def get_best_match(self, query_post: str, threshold: float = 0.85) -> tuple[str, float] | None:
        """Get the best matching claim if similarity exceeds threshold.

        Returns (claim, similarity) if match found above threshold, None otherwise.
        This enables retrieval-based fallback for high-similarity cases.
        """
        if self.embeddings is None or self.model is None:
            return None

        # E5 models need "query: " prefix for queries
        query_embedding = self.model.encode(
            f"query: {query_post[:1000]}",
            convert_to_numpy=True
        )

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get best match
        best_idx = np.argmax(similarities)
        best_sim = float(similarities[best_idx])

        if best_sim >= threshold:
            return (self.claims[best_idx], best_sim)
        return None


# =============================================================================
# Local GPU LLM
# =============================================================================

# Local models that fit in 24GB A10 VRAM (bfloat16)
LOCAL_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",  # Fits in 24GB with bfloat16
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

# Models that require quantization to fit in 24GB VRAM
QUANTIZED_MODELS = set()  # Mixtral-8x7B too large even with 4-bit

DEFAULT_LOCAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


@dataclass
class LocalLLM:
    """Local LLM running on GPU for fast inference."""

    model: object = None
    tokenizer: object = None
    device: str = "cuda"
    model_name: str = ""

    @classmethod
    def load(cls, model_name: str = DEFAULT_LOCAL_MODEL) -> "LocalLLM":
        """Load model onto GPU with optimizations."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"\n{'='*60}")
        print(f"Loading local model: {model_name}")
        print(f"{'='*60}")

        # Check GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Local mode requires GPU.")

        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check if model needs quantization (large MoE models)
        needs_quantization = model_name in QUANTIZED_MODELS

        # Check if flash_attn is available
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"
            print("Using SDPA attention (flash-attn not installed)")

        if needs_quantization:
            print("Using 4-bit quantization (model too large for full precision)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
            )
        else:
            # Use bfloat16 for A10 (good balance of speed and quality)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=attn_impl,
            )

        print(f"Model loaded successfully!")
        print(f"{'='*60}\n")

        return cls(model=model, tokenizer=tokenizer, device="cuda", model_name=model_name)

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 8,
    ) -> list[str]:
        """Generate responses for a batch of prompts efficiently."""
        import torch

        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only new tokens
            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                generated = self.tokenizer.decode(
                    output[input_len:],
                    skip_special_tokens=True
                ).strip()
                results.append(generated)

        return results

    def build_chat_prompt(self, system: str, user: str) -> str:
        """Build prompt using model's chat template."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


# =============================================================================
# Prompts
# =============================================================================

# System prompt for few-shot claim normalization
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

# Few-shot user prompt with examples
USER_PROMPT_TEMPLATE_FEWSHOT = """Here are some examples of claim normalization:

{examples}
Now normalize this post:

Post: {post}

Normalized claim:"""

# Zero-shot fallback (no examples)
USER_PROMPT_TEMPLATE_ZEROSHOT = """Post: {post}

Normalized claim:"""


def format_examples(examples: list[tuple[str, str, float]], max_chars: int = 500) -> str:
    """Format examples for the prompt."""
    formatted = []
    for i, (post, claim, _score) in enumerate(examples, 1):
        # Truncate long posts in examples
        post_truncated = post[:max_chars] + "..." if len(post) > max_chars else post
        formatted.append(f"Example {i}:\nPost: {post_truncated}\nNormalized claim: {claim}")
    return "\n\n".join(formatted)


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
    retriever: ExampleRetriever | None = None,
    num_examples: int = 3,
    timeout_seconds: float = 30.0,
    retrieval_threshold: float = 1.0,
) -> NormalizationResult:
    """Normalize a single post with rate limiting, concurrency control, and timeout."""
    async with semaphore:
        await rate_limiter.acquire()

        # Truncate very long posts
        post_truncated = post[:4000] if len(post) > 4000 else post

        start = time.perf_counter()

        # Check for high-similarity retrieval match first (skip API call if match found)
        if retriever is not None and retrieval_threshold < 1.0:
            match = retriever.get_best_match(post_truncated, threshold=retrieval_threshold)
            if match is not None:
                claim, sim = match
                latency_ms = (time.perf_counter() - start) * 1000
                meteor = compute_meteor(claim, gold_claim) if gold_claim else None
                return NormalizationResult(
                    idx=idx,
                    post=post,
                    predicted_claim=claim,
                    gold_claim=gold_claim,
                    meteor=meteor,
                    model=f"retrieval@{sim:.2f}",
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                )

        # Build prompt with few-shot examples if retriever is available
        if retriever is not None and num_examples > 0:
            examples = retriever.get_similar_examples(post_truncated, k=num_examples)
            examples_str = format_examples(examples)
            user_prompt = USER_PROMPT_TEMPLATE_FEWSHOT.format(examples=examples_str, post=post_truncated)
        else:
            user_prompt = USER_PROMPT_TEMPLATE_ZEROSHOT.format(post=post_truncated)

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
    retriever: ExampleRetriever | None = None,
    num_examples: int = 3,
    retrieval_threshold: float = 1.0,
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
        retriever: ExampleRetriever for few-shot prompting (None for zero-shot)
        num_examples: Number of few-shot examples to include
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
                model_name, semaphore, rate_limiter,
                retriever=retriever, num_examples=num_examples,
                timeout_seconds=timeout_seconds,
                retrieval_threshold=retrieval_threshold,
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
    retriever: ExampleRetriever | None = None,
    num_examples: int = 3,
    retrieval_threshold: float = 1.0,
) -> list[NormalizationResult]:
    """Run evaluation for a single model."""
    return asyncio.run(
        run_model_async(
            model_name, samples, parallel, rate_limit,
            checkpoint_path, timeout_seconds,
            retriever=retriever, num_examples=num_examples,
            retrieval_threshold=retrieval_threshold,
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
    retriever: ExampleRetriever | None = None,
    num_examples: int = 3,
    retrieval_threshold: float = 1.0,
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
                retriever=retriever,
                num_examples=num_examples,
                retrieval_threshold=retrieval_threshold,
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
# Local GPU Inference
# =============================================================================

def run_local_model(
    llm: LocalLLM,
    samples: list[dict],
    retriever: ExampleRetriever | None = None,
    num_examples: int = 3,
    batch_size: int = 8,
    checkpoint_path: Path | None = None,
    retrieval_threshold: float = 0.85,
) -> list[NormalizationResult]:
    """Run evaluation using local GPU model with batched inference.

    Uses retrieval-first approach: if a training example has similarity > threshold,
    use its claim directly instead of generating with the LLM.
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
        print(f"Resuming: {len(completed_idxs)} samples already completed")

    remaining = [s for s in samples if s["idx"] not in completed_idxs]

    if not remaining:
        print("All samples already completed")
        return existing_results

    print(f"Processing {len(remaining)} samples with batch_size={batch_size}")
    if retrieval_threshold < 1.0:
        print(f"Retrieval threshold: {retrieval_threshold} (direct copy if similarity >= threshold)")

    results = existing_results.copy()
    retrieval_hits = 0
    llm_generations = 0

    # Ensure checkpoint dir exists
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Process in batches
    pbar = tqdm(total=len(remaining), desc=f"Local: {llm.model_name.split('/')[-1]}")

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        start_time = time.perf_counter()

        # Separate samples into retrieval hits and LLM generation needed
        retrieval_results = []  # (sample, claim, similarity)
        llm_samples = []  # samples needing LLM generation

        for s in batch:
            post = s["post"][:4000]

            # Check for high-similarity retrieval match first
            if retriever is not None and retrieval_threshold < 1.0:
                match = retriever.get_best_match(post, threshold=retrieval_threshold)
                if match is not None:
                    claim, sim = match
                    retrieval_results.append((s, claim, sim))
                    retrieval_hits += 1
                    continue

            llm_samples.append(s)

        # Build prompts only for samples needing LLM generation
        prompts = []
        for s in llm_samples:
            post = s["post"][:4000]

            # Get few-shot examples if retriever available
            if retriever is not None and num_examples > 0:
                examples = retriever.get_similar_examples(post, k=num_examples)
                examples_str = format_examples(examples)
                user_prompt = USER_PROMPT_TEMPLATE_FEWSHOT.format(examples=examples_str, post=post)
            else:
                user_prompt = USER_PROMPT_TEMPLATE_ZEROSHOT.format(post=post)

            # Build chat prompt
            full_prompt = llm.build_chat_prompt(SYSTEM_PROMPT, user_prompt)
            prompts.append(full_prompt)

        # Generate batch (only if there are samples needing generation)
        if prompts:
            try:
                predictions = llm.generate_batch(prompts, batch_size=batch_size)
                llm_generations += len(predictions)
            except Exception as e:
                print(f"\nError in batch: {e}")
                predictions = [f"ERROR: {e}"] * len(llm_samples)
        else:
            predictions = []

        batch_time_ms = (time.perf_counter() - start_time) * 1000
        per_sample_ms = batch_time_ms / len(batch) if batch else 0

        # Create results from retrieval hits
        batch_results = []
        for s, claim, sim in retrieval_results:
            gold = s.get("gold_claim")
            meteor = compute_meteor(claim, gold) if gold else None

            result = NormalizationResult(
                idx=s["idx"],
                post=s["post"],
                predicted_claim=claim,
                gold_claim=gold,
                meteor=meteor,
                model=f"retrieval@{sim:.2f}",  # Mark as retrieval with similarity
                latency_ms=per_sample_ms,
                input_tokens=0,
                output_tokens=0,
            )
            batch_results.append(result)

        # Create results from LLM predictions
        for s, pred in zip(llm_samples, predictions):
            # Clean up prediction (remove quotes, extra whitespace)
            pred_clean = pred.strip().strip('"\'').split('\n')[0].strip()

            # Compute METEOR
            gold = s.get("gold_claim")
            meteor = compute_meteor(pred_clean, gold) if gold else None

            result = NormalizationResult(
                idx=s["idx"],
                post=s["post"],
                predicted_claim=pred_clean,
                gold_claim=gold,
                meteor=meteor,
                model=llm.model_name,
                latency_ms=per_sample_ms,
                input_tokens=0,
                output_tokens=0,
            )
            batch_results.append(result)

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

    # Print summary
    meteors = [r.meteor for r in results if r.meteor is not None]
    if meteors:
        import statistics
        avg_meteor = statistics.mean(meteors)
        print(f"\nResults: METEOR = {avg_meteor:.4f} (n={len(meteors)})")
        print(f"Retrieval hits: {retrieval_hits}, LLM generations: {llm_generations}")

    return results


# =============================================================================
# Results Aggregation
# =============================================================================

def compute_stats(results: list[NormalizationResult], config: ModelConfig | None = None) -> dict:
    """Compute aggregate statistics."""
    import statistics

    meteors = [r.meteor for r in results if r.meteor is not None]
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)
    total_latency = sum(r.latency_ms for r in results)

    # Cost is 0 for local models (no config)
    if config is not None:
        cost = (total_in / 1_000_000 * config.cost_per_1m_input +
                total_out / 1_000_000 * config.cost_per_1m_output)
    else:
        cost = 0.0

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
    # Few-shot options
    parser.add_argument("--num-examples", type=int, default=3,
                        help="Number of few-shot examples (default 3, use 0 for zero-shot)")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Use zero-shot prompting (no examples)")
    # Local GPU options
    parser.add_argument("--local", action="store_true",
                        help="Use local GPU model instead of API (faster, no cost)")
    parser.add_argument("--local-model", type=str, default=DEFAULT_LOCAL_MODEL,
                        help=f"Local model to use. Shortcuts: {list(LOCAL_MODELS.keys())}")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for local GPU inference (default 8)")
    parser.add_argument("--retrieval-threshold", type=float, default=0.85,
                        help="Similarity threshold for retrieval fallback (default 0.85). "
                             "If a training example has similarity >= threshold, use its claim directly. "
                             "Set to 1.0 to disable retrieval fallback.")

    args = parser.parse_args()

    # Override num_examples if zero-shot flag is set
    num_examples = 0 if args.zero_shot else args.num_examples

    # Resolve local model shortcut
    local_model = LOCAL_MODELS.get(args.local_model, args.local_model)

    # Load data
    df = load_data(args.split)
    if args.limit:
        df = df.head(args.limit)

    samples = df.to_dicts()
    print(f"\nLoaded {len(samples)} samples from {args.split} split")

    # Show mode
    if args.local:
        print(f"\n=== LOCAL GPU MODE ===")
        print(f"Model: {local_model}")
        print(f"Batch size: {args.batch_size}")
        print("=" * 40)
    else:
        # Debug: Show API keys being used
        import os
        print("\n=== API Keys ===")
        openai_key = os.getenv("OPENAI_API_KEY")
        together_key = os.getenv("TOGETHER_API_KEY")
        print(f"OPENAI_API_KEY:   {openai_key[:30]}..." if openai_key else "OPENAI_API_KEY:   NOT SET")
        print(f"TOGETHER_API_KEY: {together_key[:30]}..." if together_key else "TOGETHER_API_KEY: NOT SET")
        print("================\n")

    # Initialize few-shot retriever if using few-shot prompting
    retriever = None
    if num_examples > 0:
        print(f"\n=== Few-Shot Setup ({num_examples} examples) ===")
        # Load training data for few-shot examples
        train_df = load_data("train")
        # Also include dev data for more examples (if not evaluating on dev)
        if args.split != "dev":
            dev_df = load_data("dev")
            train_df = pl.concat([train_df, dev_df])
        retriever = ExampleRetriever.from_dataframe(train_df)
        print(f"Few-shot retriever ready with {len(retriever.posts)} examples")
        print("=" * 40 + "\n")
    else:
        print("\n=== Zero-Shot Mode (no examples) ===\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOCAL GPU MODE
    # =========================================================================
    if args.local:
        # Load local model
        llm = LocalLLM.load(local_model)

        # Set checkpoint path
        model_short = local_model.split("/")[-1]
        checkpoint = None if args.no_resume else RESULTS_DIR / f"local_{model_short}_{args.split}.jsonl"

        # Run local inference
        results = run_local_model(
            llm=llm,
            samples=samples,
            retriever=retriever,
            num_examples=num_examples,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint,
            retrieval_threshold=args.retrieval_threshold,
        )

        if results:
            stats = compute_stats(results, config=None)
            print_results(local_model, stats)
            save_model_results(model_short, stats, args.split, RESULTS_DIR)

    # =========================================================================
    # API MODE
    # =========================================================================
    elif not API_MODE_AVAILABLE:
        print("\nERROR: API mode requires the full project (src.checkworthiness.config)")
        print("Use --local for local GPU inference instead.")
        sys.exit(1)

    elif args.compare_models:
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
                retriever=retriever,
                num_examples=num_examples,
                retrieval_threshold=args.retrieval_threshold,
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
            retriever=retriever,
            num_examples=num_examples,
            retrieval_threshold=args.retrieval_threshold,
        )

        if results and args.model in MODELS:
            stats = compute_stats(results, MODELS[args.model])
            print_results(args.model, stats)
            save_model_results(args.model, stats, args.split, RESULTS_DIR)


if __name__ == "__main__":
    main()
