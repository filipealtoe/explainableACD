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

    # Use fine-tuned LoRA model
    python run_claim_normalization_ct25.py --local --local-model checkpoints/mistral-7b_20250114_123456/final --split test

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

CHANGELOG:
---------
Best known config: --retrieval-threshold 0.85 --claim-verify-threshold 0.65 -> METEOR 0.4835

2025-01-15: Added hybrid retrieval mode
  - NEW: --retrieval-mode {embedding,hybrid} argument
  - hybrid mode combines: embedding similarity + BM25 lexical + entity overlap
  - Weights: alpha=0.5 (embedding), beta=0.3 (BM25), gamma=0.2 (entity)
  - Entity extraction: numbers, dates, capitalized words/phrases
  - Usage: --retrieval-mode hybrid --retrieval-threshold 0.7

2025-01-14: Added OpenAI GPT-5.2 models
  - NEW: gpt-5.2 ($1.75-14.00/M) - reasoning model, 400K context
  - NEW: gpt-5.2-pro ($1.75-14.00/M) - xhigh reasoning variant

2025-01-14: Added OpenAI models
  - NEW: gpt-4o ($2.50-10.00/M) - OpenAI flagship
  - NEW: gpt-4o-mini ($0.15-0.60/M) - cheap OpenAI option

2025-01-14: Added new Together.ai models
  - NEW: deepseek-r1 (671B reasoning, $3-7/M) - SOTA reasoning
  - NEW: llama-4-scout (17B MoE, $0.18-0.59/M) - newest Llama, cheap
  - NEW: mistral-small-24b (24B, $0.80/M) - strong mid-tier
  - NEW: kimi-k2 (1T MoE, $1-3/M) - Moonshot reasoning model

2025-01-14: Cleaned up experimental code
  - REMOVED: All cluster detection and cluster-aware code (no improvement)
  - KEPT: DeepSeek V3/V3.1 via Together.ai
  - KEPT: qwen-2.5-14b model
  - KEPT: Two-stage retrieval verification (post similarity + claim verification)

Failed experiments (DO NOT RE-TRY without new approach):
  - Skipping retrieval for multi_claim posts -> LLM worse than retrieval
  - Keyword overlap verification -> Too strict, rejected good matches
  - ERROR markers in postprocess_claim -> Tanks METEOR to 0
  - Raising claim-verify-threshold above 0.65 -> Rejects too many good matches
  - "HANDLING DIFFICULT CASES" prompt section -> No measurable improvement
  - Cluster-aware few-shot selection -> No improvement over similarity-based
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
from dotenv import load_dotenv
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
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
# For standalone package: scripts/ -> parent is package root
# For full project: experiments/scripts/ -> parent.parent is project root
if SCRIPT_DIR.name == "scripts" and SCRIPT_DIR.parent.name == "experiments":
    REPO_ROOT = SCRIPT_DIR.parents[1]  # Full project structure
elif SCRIPT_DIR.name == "scripts":
    REPO_ROOT = SCRIPT_DIR.parent      # Standalone package
else:
    REPO_ROOT = SCRIPT_DIR

# Try to load from project structure, fall back to standalone mode
try:
    sys.path.insert(0, str(REPO_ROOT))
    load_dotenv(REPO_ROOT / ".env", override=True)
    from src.checkworthiness.config import MODELS, ModelConfig
    API_MODE_AVAILABLE = True
except ImportError:
    # Standalone mode - define minimal config for Together.ai API
    import os
    from dataclasses import dataclass as config_dataclass

    @config_dataclass
    class ModelConfig:
        """Minimal model config for standalone mode."""
        model_name: str
        api_base: str
        api_key_env: str
        cost_per_1m_input: float = 0.0
        cost_per_1m_output: float = 0.0

        def get_api_key(self) -> str | None:
            return os.getenv(self.api_key_env)

    # Together.ai models available in standalone mode
    MODELS = {
        "mixtral-8x7b": ModelConfig(
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=0.60,
            cost_per_1m_output=0.60,
        ),
        "mistral-7b-v0.3": ModelConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=0.20,
            cost_per_1m_output=0.20,
        ),
        "llama-3.3-70b": ModelConfig(
            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=0.88,
            cost_per_1m_output=0.88,
        ),
        "qwen-2.5-72b": ModelConfig(
            model_name="Qwen/Qwen2.5-72B-Instruct-Turbo",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=1.20,
            cost_per_1m_output=1.20,
        ),
        "qwen-2.5-7b": ModelConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=0.30,
            cost_per_1m_output=0.30,
        ),
        "qwen-2.5-14b": ModelConfig(
            model_name="Qwen/Qwen2.5-14B-Instruct-Turbo",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=0.80,
            cost_per_1m_output=0.80,
        ),
        "deepseek-v3": ModelConfig(
            model_name="deepseek-ai/DeepSeek-V3",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=0.90,
            cost_per_1m_output=0.90,
        ),
        "deepseek-v3.1": ModelConfig(
            model_name="deepseek-ai/DeepSeek-V3.1",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=0.90,
            cost_per_1m_output=0.90,
        ),
        # New models added Jan 2025
        "deepseek-r1": ModelConfig(
            model_name="deepseek-ai/DeepSeek-R1",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=3.00,
            cost_per_1m_output=7.00,
        ),
        "llama-4-scout": ModelConfig(
            model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=0.18,
            cost_per_1m_output=0.59,
        ),
        "mistral-small-24b": ModelConfig(
            model_name="mistralai/Mistral-Small-24B-Instruct-2501",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=0.80,
            cost_per_1m_output=0.80,
        ),
        "kimi-k2": ModelConfig(
            model_name="moonshotai/Kimi-K2-Instruct",
            api_base="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            cost_per_1m_input=1.00,
            cost_per_1m_output=3.00,
        ),
        # OpenAI models
        "gpt-4o": ModelConfig(
            model_name="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            cost_per_1m_input=2.50,
            cost_per_1m_output=10.00,
        ),
        "gpt-4o-mini": ModelConfig(
            model_name="gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            cost_per_1m_input=0.15,
            cost_per_1m_output=0.60,
        ),
        "gpt-5.2": ModelConfig(
            model_name="gpt-5.2",
            api_base="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            cost_per_1m_input=1.75,
            cost_per_1m_output=14.00,
        ),
        "gpt-5.2-pro": ModelConfig(
            model_name="gpt-5.2-pro",
            api_base="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            cost_per_1m_input=1.75,
            cost_per_1m_output=14.00,
        ),
    }
    API_MODE_AVAILABLE = True
    print("Note: Running in standalone mode with Together.ai API support")


# =============================================================================
# Configuration
# =============================================================================

# Data directory - check multiple possible locations
_possible_data_dirs = [
    REPO_ROOT / "data" / "raw" / "check_that_25",  # Full project structure
    REPO_ROOT / "data",                             # Standalone package structure
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
    # Tier 1: Cheapest (< $0.30/M) - start here
    "gpt-4o-mini",          # OpenAI, Oct 2023 cutoff - $0.15-0.60/M
    "llama-4-scout",        # 17B MoE, 2024 cutoff - $0.18-0.59/M (newest Llama)
    "mistral-7b-v0.3",      # 7B, 2023 cutoff - $0.20/M
    "qwen-2.5-7b",          # 7B, End 2023 cutoff - $0.30/M
    # Tier 2: Mid-range ($0.30-$1.00/M)
    "mixtral-8x7b",         # 8x7B MoE, 2023 cutoff - $0.60/M
    "qwen-2.5-14b",         # 14B, End 2023 cutoff - $0.80/M
    "mistral-small-24b",    # 24B, Jan 2025 release - $0.80/M
    "llama-3.3-70b",        # 70B, Dec 2023 cutoff - $0.88/M
    "llama-3.1-70b",        # 70B, Dec 2023 cutoff - $0.88/M
    "deepseek-v3",          # MoE 671B, Jul 2024 cutoff - $0.90/M
    "deepseek-v3.1",        # MoE 671B, improved - $0.90/M
    # Tier 3: Premium ($1.00+/M)
    "kimi-k2",              # MoE 1T, 2024 cutoff - $1.00-3.00/M (reasoning)
    "qwen-2.5-72b",         # 72B, End 2023 cutoff - best on CT24, $1.20/M
    "gpt-5.2",              # OpenAI, Aug 2025 cutoff - $1.75-14.00/M (reasoning)
    "gpt-4o",               # OpenAI, Oct 2023 cutoff - $2.50-10.00/M
    "deepseek-r1",          # 671B, reasoning model - $3.00-7.00/M (SOTA reasoning)
    "llama-3.1-405b",       # 405B, Dec 2023 cutoff - largest, $3.50/M
    "gpt-5.2-pro",          # OpenAI Pro, Aug 2025 cutoff - $1.75-14.00/M (xhigh reasoning)
]

# Embedding model for few-shot retrieval
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# =============================================================================
# Cluster-Aware Prompting
# =============================================================================

# Cluster types and their specific instructions (concise, single focus)
CLUSTER_INSTRUCTIONS = {
    "multi_claim": (
        "This post contains MULTIPLE claims. "
        "Select the single MOST VERIFIABLE claim that can be fact-checked."
    ),
    "sarcasm": (
        "This post may be SARCASTIC. "
        "Extract the underlying factual claim, not the literal sarcastic statement."
    ),
    "numeric": (
        "Preserve ALL numbers, percentages, and statistics exactly as written."
    ),
    "no_cluster": (
        "Reframe into a clear, verifiable factual claim."
    ),
    "political": (
        "Focus on the factual assertion. Preserve exact attributions (who said/did what)."
    ),
    "media_reference": (
        "Format as 'Photo/Video shows [what it depicts]'."
    ),
    "noisy_emoji": (
        "Ignore emojis and hashtags. Extract only the core factual claim."
    ),
}

# Suffix disabled - testing if cluster instructions alone help
PRESERVE_INFO_SUFFIX = ""

# Model-specific prompt styles
# Each model family has different strengths in prompt interpretation
# NOTE: All styles now use only PRIMARY cluster (clusters[0]) to avoid instruction overload
MODEL_PROMPT_STYLES = {
    # DeepSeek: Excels with structured, detailed prompts with clear sections
    "deepseek": {
        "style": "structured",
        "format_cluster": lambda clusters, instructions: (
            f"\n\n## SPECIAL INSTRUCTION\n{instructions.get(clusters[0], '')}"
            if clusters and clusters[0] in instructions else ""
        ),
        "system_prefix": "",
        "system_suffix": "",
    },
    # GPT-5.x: Natural language, concise, handles nuanced instructions well
    "gpt-5": {
        "style": "natural",
        "format_cluster": lambda clusters, instructions: (
            f"\n\nNote: {instructions.get(clusters[0], '')}" if clusters else ""
        ),
        "system_prefix": "",
        "system_suffix": "",
    },
    # Mistral-7B: Benefits from direct, simple instructions
    "mistral-7b": {
        "style": "direct",
        "format_cluster": lambda clusters, instructions: (
            f"\n\n[IMPORTANT] {instructions.get(clusters[0], 'Extract the main factual claim.')}"
            if clusters else ""
        ),
        "system_prefix": "[INST] ",
        "system_suffix": " [/INST]",
    },
    # Mixtral-8x7B: Can handle more complexity than 7B, structured works well
    "mixtral": {
        "style": "structured",
        "format_cluster": lambda clusters, instructions: (
            f"\n\n### Special Instruction ###\n{instructions.get(clusters[0], '')}"
            if clusters and clusters[0] in instructions else ""
        ),
        "system_prefix": "",
        "system_suffix": "",
    },
    # GPT-4o: Similar to GPT-5 but slightly more verbose is OK
    "gpt-4o": {
        "style": "natural",
        "format_cluster": lambda clusters, instructions: (
            f"\n\nContext: {instructions.get(clusters[0], '')}" if clusters else ""
        ),
        "system_prefix": "",
        "system_suffix": "",
    },
    # Default for unknown models
    "default": {
        "style": "direct",
        "format_cluster": lambda clusters, instructions: (
            f"\n\nNote: {instructions.get(clusters[0], '')}" if clusters else ""
        ),
        "system_prefix": "",
        "system_suffix": "",
    },
}


def get_model_style(model_name: str) -> dict:
    """Get the prompt style configuration for a model."""
    model_lower = model_name.lower()

    if "deepseek" in model_lower:
        return MODEL_PROMPT_STYLES["deepseek"]
    elif "gpt-5" in model_lower:
        return MODEL_PROMPT_STYLES["gpt-5"]
    elif "mistral-7b" in model_lower or "mistral/mistral-7b" in model_lower:
        return MODEL_PROMPT_STYLES["mistral-7b"]
    elif "mixtral" in model_lower:
        return MODEL_PROMPT_STYLES["mixtral"]
    elif "gpt-4o" in model_lower:
        return MODEL_PROMPT_STYLES["gpt-4o"]
    else:
        return MODEL_PROMPT_STYLES["default"]


def load_cluster_mapping(cluster_file: Path) -> dict[tuple[int, str], list[str]]:
    """Load cluster assignments from file.

    Returns: dict mapping (idx, split) -> list of clusters
    """
    if not cluster_file.exists():
        return {}

    df = pl.read_csv(cluster_file)
    mapping = {}

    for row in df.iter_rows(named=True):
        key = (row["idx"], row["split"])
        if "clusters" in row and row["clusters"]:
            clusters = row["clusters"].split(",")
        else:
            clusters = [row.get("cluster", "no_cluster")]
        mapping[key] = clusters

    return mapping


def build_cluster_aware_prompt(
    base_prompt: str,
    clusters: list[str],
    model_name: str,
) -> str:
    """Build a cluster-aware system prompt for a specific model.

    Args:
        base_prompt: The base system prompt (v1 or v2)
        clusters: List of detected cluster types for this post (only primary used)
        model_name: The model being used (for style selection)

    Returns:
        Modified system prompt with cluster-specific instructions + preserve info suffix
    """
    if not clusters:
        # Even without clusters, add the preserve info suffix
        return base_prompt + PRESERVE_INFO_SUFFIX

    style = get_model_style(model_name)

    # Build cluster-specific section (uses only primary cluster via format_cluster)
    cluster_section = style["format_cluster"](clusters, CLUSTER_INSTRUCTIONS)

    # Combine: base prompt + cluster instruction + preserve info suffix
    full_prompt = (
        style["system_prefix"] +
        base_prompt +
        cluster_section +
        PRESERVE_INFO_SUFFIX +
        style["system_suffix"]
    )

    return full_prompt


# =============================================================================
# Entity Extraction (for hybrid retrieval)
# =============================================================================

def extract_entities(text: str) -> set[str]:
    """Extract simple entities from text for overlap scoring.

    Extracts:
    - Numbers (including percentages, currencies)
    - Capitalized words/phrases (likely proper nouns)
    - Dates in common formats
    """
    entities = set()

    # Numbers with optional % or currency symbols
    numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?', text)
    entities.update(n.lower() for n in numbers)

    # Dates (various formats)
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', text)
    entities.update(dates)

    # Month names with optional day/year
    months = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?\b', text, re.IGNORECASE)
    entities.update(m.lower() for m in months)

    # Capitalized words (potential proper nouns) - exclude sentence starters
    # Split into sentences first
    sentences = re.split(r'[.!?]\s+', text)
    for sent in sentences:
        words = sent.split()
        # Skip first word of each sentence
        for word in words[1:]:
            # Check if capitalized and not all caps
            if word and word[0].isupper() and not word.isupper() and len(word) > 1:
                # Remove trailing punctuation
                clean = re.sub(r'[^\w\s]', '', word)
                if clean and len(clean) > 1:
                    entities.add(clean.lower())

    # Multi-word capitalized phrases (like "White House", "United States")
    phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    entities.update(p.lower() for p in phrases)

    return entities


def compute_entity_overlap(text1: str, text2: str) -> float:
    """Compute Jaccard similarity of entities between two texts."""
    entities1 = extract_entities(text1)
    entities2 = extract_entities(text2)

    if not entities1 and not entities2:
        return 0.5  # Neutral score if no entities found
    if not entities1 or not entities2:
        return 0.0

    intersection = len(entities1 & entities2)
    union = len(entities1 | entities2)

    return intersection / union if union > 0 else 0.0


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
    bm25: BM25Okapi | None = None
    tokenized_posts: list[list[str]] = field(default_factory=list)
    # Topic clustering fields
    cluster_labels: np.ndarray | None = None
    cluster_centroids: np.ndarray | None = None
    n_clusters: int = 0

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        embedding_model: str = EMBEDDING_MODEL,
        cache_dir: Path | None = None,
        n_clusters: int = 0,
    ) -> "ExampleRetriever":
        """Create retriever from training DataFrame with optional caching.

        Args:
            df: Training DataFrame with 'post' and 'gold_claim' columns
            embedding_model: Model for embeddings
            cache_dir: Directory for caching embeddings
            n_clusters: Number of topic clusters (0 = disabled)
        """
        import hashlib
        import torch

        # Filter to rows with gold claims
        df_valid = df.filter(pl.col("gold_claim").is_not_null())

        posts = df_valid["post"].to_list()
        claims = df_valid["gold_claim"].to_list()

        # Set up cache paths
        if cache_dir is None:
            cache_dir = SCRIPT_DIR / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache key based on model name and data hash
        data_hash = hashlib.md5(f"{len(posts)}_{posts[0][:50]}_{posts[-1][:50]}".encode()).hexdigest()[:8]
        model_short = embedding_model.split("/")[-1]
        cache_embeddings = cache_dir / f"embeddings_{model_short}_{data_hash}.npy"
        cache_metadata = cache_dir / f"metadata_{model_short}_{data_hash}.json"

        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {embedding_model} (device: {device})")
        model = SentenceTransformer(embedding_model, device=device)

        # Try to load cached embeddings
        if cache_embeddings.exists() and cache_metadata.exists():
            print(f"Loading cached embeddings from {cache_embeddings.name}")
            embeddings = np.load(cache_embeddings)
            with open(cache_metadata) as f:
                metadata = json.load(f)
            cached_posts = metadata["posts"]
            cached_claims = metadata["claims"]
            # Verify cache is valid - must use cached posts/claims to match embeddings!
            if len(cached_posts) == len(embeddings):
                print(f"Loaded {len(embeddings)} cached embeddings")
                # Build BM25 index
                print("Building BM25 index...")
                tokenized_posts = [word_tokenize(p.lower()[:1000]) for p in cached_posts]
                bm25 = BM25Okapi(tokenized_posts)

                # Build topic clusters if requested
                cluster_labels, cluster_centroids = None, None
                if n_clusters > 0:
                    cluster_labels, cluster_centroids = cls._build_clusters(
                        embeddings, n_clusters, cache_dir, data_hash
                    )

                return cls(
                    posts=cached_posts,
                    claims=cached_claims,
                    embeddings=embeddings,
                    model=model,
                    bm25=bm25,
                    tokenized_posts=tokenized_posts,
                    cluster_labels=cluster_labels,
                    cluster_centroids=cluster_centroids,
                    n_clusters=n_clusters if cluster_labels is not None else 0,
                )
            else:
                print("Cache mismatch, regenerating embeddings...")

        # Generate embeddings
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

        # Save to cache
        print(f"Saving embeddings to cache: {cache_embeddings.name}")
        np.save(cache_embeddings, embeddings)
        with open(cache_metadata, "w") as f:
            json.dump({"posts": posts, "claims": claims}, f)

        # Build BM25 index
        print("Building BM25 index...")
        tokenized_posts = [word_tokenize(p.lower()[:1000]) for p in posts]
        bm25 = BM25Okapi(tokenized_posts)

        # Build topic clusters if requested
        cluster_labels, cluster_centroids = None, None
        if n_clusters > 0:
            cluster_labels, cluster_centroids = cls._build_clusters(
                embeddings, n_clusters, cache_dir, data_hash
            )

        return cls(
            posts=posts,
            claims=claims,
            embeddings=embeddings,
            model=model,
            bm25=bm25,
            tokenized_posts=tokenized_posts,
            cluster_labels=cluster_labels,
            cluster_centroids=cluster_centroids,
            n_clusters=n_clusters if cluster_labels is not None else 0,
        )

    @staticmethod
    def _build_clusters(
        embeddings: np.ndarray,
        n_clusters: int,
        cache_dir: Path,
        data_hash: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build topic clusters using K-means on embeddings."""
        from sklearn.cluster import KMeans

        cache_file = cache_dir / f"clusters_{n_clusters}_{data_hash}.npz"

        if cache_file.exists():
            print(f"Loading cached clusters from {cache_file.name}")
            data = np.load(cache_file)
            return data["labels"], data["centroids"]

        print(f"Building {n_clusters} topic clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # Cache clusters
        np.savez(cache_file, labels=labels, centroids=centroids)
        print(f"Saved clusters to {cache_file.name}")

        # Print cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("Cluster distribution:")
        for c, cnt in zip(unique, counts):
            print(f"  Cluster {c}: {cnt} examples ({100*cnt/len(labels):.1f}%)")

        return labels, centroids

    def assign_cluster(self, query_post: str) -> int:
        """Assign a query post to the nearest topic cluster."""
        if self.cluster_centroids is None or self.model is None:
            return -1

        # Embed query
        query_embedding = self.model.encode(
            f"query: {query_post[:1000]}",
            convert_to_numpy=True
        )

        # Find nearest centroid
        distances = np.linalg.norm(self.cluster_centroids - query_embedding, axis=1)
        return int(np.argmin(distances))

    def get_similar_examples_from_cluster(
        self,
        query_post: str,
        k: int = 3,
        fallback_to_all: bool = True,
    ) -> list[tuple[str, str, float]]:
        """Find k most similar examples from the same topic cluster.

        Args:
            query_post: Query post text
            k: Number of examples to return
            fallback_to_all: If cluster has fewer than k examples, include from other clusters

        Returns:
            List of (post, claim, similarity_score) tuples
        """
        if self.cluster_labels is None:
            # No clustering, fall back to regular retrieval
            return self.get_similar_examples(query_post, k)

        # Assign query to cluster
        cluster_id = self.assign_cluster(query_post)

        # Get indices of examples in this cluster
        cluster_mask = self.cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            return self.get_similar_examples(query_post, k)

        # Embed query
        query_embedding = self.model.encode(
            f"query: {query_post[:1000]}",
            convert_to_numpy=True
        )

        # Compute similarities only for cluster examples
        cluster_embeddings = self.embeddings[cluster_indices]
        similarities = np.dot(cluster_embeddings, query_embedding) / (
            np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k from cluster
        n_from_cluster = min(k, len(cluster_indices))
        top_k_local = np.argsort(similarities)[-n_from_cluster:][::-1]
        top_k_global = cluster_indices[top_k_local]

        results = [
            (self.posts[i], self.claims[i], float(similarities[top_k_local[j]]))
            for j, i in enumerate(top_k_global)
        ]

        # If not enough examples in cluster, fill from other clusters
        if fallback_to_all and len(results) < k:
            remaining = k - len(results)
            all_examples = self.get_similar_examples(query_post, k + remaining)
            # Add examples not already in results
            existing_posts = {r[0] for r in results}
            for ex in all_examples:
                if ex[0] not in existing_posts and len(results) < k:
                    results.append(ex)

        return results

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

    def get_best_match(
        self,
        query_post: str,
        threshold: float = 0.85,
        claim_verification_threshold: float = 0.5,
    ) -> tuple[str, float] | None:
        """Get the best matching claim with two-stage verification.

        Stage 1: Find similar posts (embedding similarity)
        Stage 2: Verify claim-post embedding similarity

        Returns (claim, similarity) if match found and verified, None otherwise.
        """
        if self.embeddings is None or self.model is None:
            return None

        # E5 models need "query: " prefix for queries
        query_embedding = self.model.encode(
            f"query: {query_post[:1000]}",
            convert_to_numpy=True
        )

        # Stage 1: Find similar posts
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-5 candidates (not just the best one)
        top_k = 5
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Stage 2: Verify claim relevance for each candidate
        for idx in top_indices:
            post_sim = float(similarities[idx])

            # Skip if post similarity is below threshold
            if post_sim < threshold:
                break  # Sorted descending, so no more candidates above threshold

            candidate_claim = self.claims[idx]

            # Compute similarity between query post and candidate claim
            claim_embedding = self.model.encode(
                f"passage: {candidate_claim[:500]}",
                convert_to_numpy=True
            )
            claim_sim = float(
                np.dot(query_embedding, claim_embedding) /
                (np.linalg.norm(query_embedding) * np.linalg.norm(claim_embedding))
            )

            # Only return if claim is also relevant to the query post
            if claim_sim >= claim_verification_threshold:
                combined_sim = (post_sim + claim_sim) / 2
                return (candidate_claim, combined_sim)

        return None

    def get_best_match_hybrid(
        self,
        query_post: str,
        threshold: float = 0.85,
        claim_verification_threshold: float = 0.5,
        alpha: float = 0.5,  # Weight for embedding similarity
        beta: float = 0.3,   # Weight for BM25
        gamma: float = 0.2,  # Weight for entity overlap
    ) -> tuple[str, float] | None:
        """Get the best matching claim using hybrid retrieval.

        Combines three signals:
        1. Embedding similarity (semantic matching)
        2. BM25 (lexical matching)
        3. Entity overlap (numbers, dates, proper nouns)

        Args:
            query_post: Input post to match
            threshold: Minimum combined score threshold
            claim_verification_threshold: Threshold for claim-post similarity
            alpha: Weight for embedding similarity (default 0.5)
            beta: Weight for BM25 score (default 0.3)
            gamma: Weight for entity overlap (default 0.2)

        Returns (claim, combined_score) if match found, None otherwise.
        """
        if self.embeddings is None or self.model is None or self.bm25 is None:
            return None

        query_truncated = query_post[:1000]

        # 1. Embedding similarity
        query_embedding = self.model.encode(
            f"query: {query_truncated}",
            convert_to_numpy=True
        )
        emb_similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 2. BM25 scores
        query_tokens = word_tokenize(query_truncated.lower())
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        # Normalize BM25 scores to [0, 1]
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_normalized = bm25_scores / bm25_max

        # 3. Entity overlap (computed on-the-fly for top candidates)
        # First, get top-20 by embedding+BM25 to limit entity computation
        prelim_scores = alpha * emb_similarities + beta * bm25_normalized
        top_20_indices = np.argsort(prelim_scores)[-20:][::-1]

        # Compute entity overlap for top candidates
        entity_scores = np.zeros(len(self.posts))
        for idx in top_20_indices:
            entity_scores[idx] = compute_entity_overlap(query_truncated, self.posts[idx])

        # Combined score
        combined_scores = (
            alpha * emb_similarities +
            beta * bm25_normalized +
            gamma * entity_scores
        )

        # Get top-5 candidates
        top_k = 5
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]

        # Verify claim relevance for each candidate
        for idx in top_indices:
            combined_score = float(combined_scores[idx])

            # Skip if combined score is below threshold
            if combined_score < threshold:
                break

            candidate_claim = self.claims[idx]

            # Compute similarity between query post and candidate claim
            claim_embedding = self.model.encode(
                f"passage: {candidate_claim[:500]}",
                convert_to_numpy=True
            )
            claim_sim = float(
                np.dot(query_embedding, claim_embedding) /
                (np.linalg.norm(query_embedding) * np.linalg.norm(claim_embedding))
            )

            # Only return if claim is also relevant to the query post
            if claim_sim >= claim_verification_threshold:
                final_score = (combined_score + claim_sim) / 2
                return (candidate_claim, final_score)

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
        """Load model onto GPU with optimizations.

        Supports:
        - Standard HuggingFace models (e.g., mistralai/Mistral-7B-Instruct-v0.3)
        - Fine-tuned LoRA adapters (path to directory with adapter_config.json)
        """
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

        # Check if this is a LoRA adapter (has adapter_config.json)
        model_path = Path(model_name)
        is_lora_adapter = (model_path / "adapter_config.json").exists()

        if is_lora_adapter:
            return cls._load_lora_adapter(model_path, gpu_mem)

        # Standard model loading
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

    @classmethod
    def _load_lora_adapter(cls, adapter_path: Path, gpu_mem: float) -> "LocalLLM":
        """Load a fine-tuned LoRA adapter model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print(f"Detected LoRA adapter at: {adapter_path}")

        # Load training config to get base model name
        config_path = adapter_path / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                training_config = json.load(f)
            base_model_name = training_config.get("base_model")
            print(f"Base model (from training config): {base_model_name}")
        else:
            # Try to get from adapter_config.json
            with open(adapter_path / "adapter_config.json") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
            print(f"Base model (from adapter config): {base_model_name}")

        if not base_model_name:
            raise ValueError("Could not determine base model from adapter config")

        # Load tokenizer from adapter path (it should have been saved there)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check if flash_attn is available
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"
            print("Using SDPA attention (flash-attn not installed)")

        # Load base model with 4-bit quantization (same as training)
        print(f"Loading base model with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )

        # Load LoRA adapter
        print(f"Loading LoRA adapter weights...")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # Merge adapter into base model for faster inference (optional)
        # model = model.merge_and_unload()

        # Print adapter info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Adapter params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% trainable)")

        print(f"Fine-tuned model loaded successfully!")
        print(f"{'='*60}\n")

        # Use adapter path as model name for identification
        return cls(model=model, tokenizer=tokenizer, device="cuda", model_name=str(adapter_path))

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
SYSTEM_PROMPT = """You are a claim normalization specialist. Your task is to transform social media posts into clean, factual claims for fact-checking.

INSTRUCTIONS:
1. Extract the main factual assertion from the post
2. Rewrite it as a single, clear declarative sentence
3. Keep all specific details: names, numbers, dates, locations
4. Remove: emojis, hashtags, @mentions, URLs, "See More", repeated text
5. Use third person and neutral tone

CRITICAL RULES:
- Output ONLY the normalized claim - no explanations, no quotes, no prefixes
- Never start with "The claim is" or "Normalized claim:"
- Keep the claim concise (1 sentence, under 50 words)
- Preserve the original meaning exactly - do not add or infer information"""

# Improved prompt based on error analysis (80% MISSING_INFO, 36% WRONG_RELATION)
SYSTEM_PROMPT_V2 = """You are a claim normalization specialist. Your task is to transform social media posts into normalized claims for fact-checking.

TASK: Convert the post into a single, clear factual claim.

CRITICAL - DO NOT OMIT INFORMATION:
- Include ALL key entities: people, organizations, places mentioned
- Include ALL numbers, percentages, dates, and statistics exactly as written
- Include the CONTEXT: what happened, to whom, where, when
- If the post is about a PHOTO or VIDEO, your claim should be "Photo/Video shows [what it depicts]"

PRESERVE RELATIONSHIPS EXACTLY:
- Do NOT change: killed→died, supports→opposes, increase→decrease, accused→convicted
- Keep exact attribution: "Trump said X" NOT "The president said X"
- Keep the direction: "A attacked B" NOT "B was involved with A"

FORMAT RULES:
- Output ONLY the normalized claim - no explanations, no quotes, no prefixes
- Never start with "The claim is" or "Normalized claim:"
- One sentence, under 50 words
- Third person, neutral tone
- Remove: emojis, hashtags, @mentions, URLs, repeated text

EXAMPLES OF COMMON MISTAKES TO AVOID:
- BAD: "A person died" → GOOD: "John Smith, 45, was killed in Chicago shooting"
- BAD: "Political event occurred" → GOOD: "Photo shows Trump rally in Georgia on Jan 6"
- BAD: "Disaster happened" → GOOD: "Video shows flooding in Kenya's Kisumu county" """

# Map of prompt versions
PROMPT_VERSIONS = {
    "v1": SYSTEM_PROMPT,
    "v2": SYSTEM_PROMPT_V2,
}

def postprocess_claim(text: str) -> str:
    """Clean up LLM output to extract just the normalized claim."""
    import re

    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Remove common prefixes
    prefixes_to_remove = [
        r"^normalized\s*claim\s*:\s*",
        r"^the\s*claim\s*is\s*:\s*",
        r"^the\s*claim\s*is\s+that\s+",
        r"^the\s*normalized\s*claim\s*:\s*",
        r"^claim\s*:\s*",
        r"^output\s*:\s*",
        r"^answer\s*:\s*",
        r"^here\s*is\s*the\s*normalized\s*claim\s*:\s*",
    ]
    for pattern in prefixes_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Take only the first line/sentence if multiple lines
    text = text.split('\n')[0].strip()

    # Remove trailing incomplete sentences (if ends with common patterns)
    if text.endswith('...'):
        text = text[:-3].strip()

    # Remove surrounding quotes again (in case they were inside prefix)
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Remove any remaining leading/trailing punctuation except period
    text = text.strip('*_`')

    # Ensure ends with period if it's a complete sentence
    if text and text[-1] not in '.!?':
        text = text + '.'

    return text


# Few-shot user prompt with examples
USER_PROMPT_TEMPLATE_FEWSHOT = """Here are some examples of claim normalization:

{examples}
Now normalize this post:

Post: {post}

Normalized claim:"""

# Contrastive few-shot prompt with good AND bad examples
USER_PROMPT_TEMPLATE_CONTRASTIVE = """Learn from these examples of claim normalization:

=== GOOD EXAMPLES (follow these) ===
{good_examples}

=== BAD EXAMPLES (avoid these mistakes) ===
{bad_examples}

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


def format_contrastive_bad_examples(
    examples: list[dict],
    max_chars: int = 500,
) -> str:
    """Format bad examples showing the mistake and correct answer."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        post = ex["post"]
        post_truncated = post[:max_chars] + "..." if len(post) > max_chars else post
        bad_claim = ex["predicted_claim"]
        good_claim = ex["gold_claim"]
        error_type = ex.get("error_type", "ERROR")

        formatted.append(
            f"Bad Example {i} ({error_type}):\n"
            f"Post: {post_truncated}\n"
            f"❌ Wrong: {bad_claim}\n"
            f"✓ Correct: {good_claim}"
        )
    return "\n\n".join(formatted)


@dataclass
class ContrastiveExamples:
    """Holds good and bad examples loaded from a results file."""

    good_examples: list[dict] = field(default_factory=list)  # High METEOR
    bad_examples: list[dict] = field(default_factory=list)   # Low METEOR
    bad_embeddings: np.ndarray | None = None
    model: SentenceTransformer | None = None

    @classmethod
    def from_results_file(
        cls,
        results_file: Path,
        good_threshold: float = 0.7,
        bad_threshold: float = 0.3,
        embedding_model: str = EMBEDDING_MODEL,
        max_bad: int = 500,
    ) -> "ContrastiveExamples":
        """Load contrastive examples from a results JSONL file.

        Args:
            results_file: Path to .jsonl with predictions
            good_threshold: METEOR >= this is a "good" example
            bad_threshold: METEOR <= this is a "bad" example
            embedding_model: Model for embedding bad examples (for retrieval)
            max_bad: Maximum bad examples to keep (for memory)
        """
        import torch

        good = []
        bad = []

        print(f"Loading contrastive examples from {results_file.name}...")

        with open(results_file) as f:
            for line in f:
                data = json.loads(line)
                meteor = data.get("meteor", 0)

                # Skip errors/timeouts
                pred = data.get("predicted_claim", "")
                if pred.startswith(("ERROR:", "TIMEOUT:")):
                    continue

                if meteor >= good_threshold:
                    good.append(data)
                elif meteor <= bad_threshold:
                    bad.append(data)

        print(f"  Good examples (METEOR >= {good_threshold}): {len(good)}")
        print(f"  Bad examples (METEOR <= {bad_threshold}): {len(bad)}")

        # Limit bad examples for memory
        if len(bad) > max_bad:
            # Keep diverse bad examples by sampling
            import random
            random.seed(42)
            bad = random.sample(bad, max_bad)

        # Embed bad examples for similarity-based retrieval
        bad_embeddings = None
        model = None
        if bad:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Embedding {len(bad)} bad examples for retrieval...")
            model = SentenceTransformer(embedding_model, device=device)
            bad_posts = [f"passage: {ex['post'][:1000]}" for ex in bad]
            bad_embeddings = model.encode(
                bad_posts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=64 if device == "cuda" else 32,
            )

        return cls(
            good_examples=good,
            bad_examples=bad,
            bad_embeddings=bad_embeddings,
            model=model,
        )

    def get_similar_bad_examples(
        self,
        query_post: str,
        k: int = 1,
    ) -> list[dict]:
        """Get k bad examples most similar to the query post."""
        if not self.bad_examples or self.bad_embeddings is None or self.model is None:
            return []

        # Embed query
        query_embedding = self.model.encode(
            f"query: {query_post[:1000]}",
            convert_to_numpy=True,
        )

        # Find most similar bad examples
        similarities = np.dot(self.bad_embeddings, query_embedding) / (
            np.linalg.norm(self.bad_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_k_idx = np.argsort(similarities)[-k:][::-1]

        return [self.bad_examples[i] for i in top_k_idx]

    def get_random_bad_examples(self, k: int = 1) -> list[dict]:
        """Get k random bad examples."""
        import random
        if not self.bad_examples:
            return []
        return random.sample(self.bad_examples, min(k, len(self.bad_examples)))


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
# Ensemble / Self-Consistency
# =============================================================================

def select_centroid_candidate(candidates: list[str]) -> tuple[str, int]:
    """Select the candidate most similar to all others (self-consistency).

    Uses METEOR score as similarity metric between candidates.
    Returns the centroid candidate and its index.
    """
    if len(candidates) == 1:
        return candidates[0], 0

    n = len(candidates)
    avg_similarities = []

    for i, cand_i in enumerate(candidates):
        # Compute average METEOR similarity to all other candidates
        similarities = []
        for j, cand_j in enumerate(candidates):
            if i != j:
                sim = compute_meteor(cand_i, cand_j)
                similarities.append(sim)
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        avg_similarities.append(avg_sim)

    # Select candidate with highest average similarity (most central)
    best_idx = max(range(n), key=lambda i: avg_similarities[i])
    return candidates[best_idx], best_idx


# =============================================================================
# Data Loading
# =============================================================================

def load_data(split: str, filtered: bool = False, single_claim: bool = False) -> pl.DataFrame:
    """Load CT25 data for given split.

    Args:
        split: Data split ("train", "dev", "test")
        filtered: If True and split="train", use filtered training data
        single_claim: If True, use single-claim filtered datasets (excludes multi_claim)
    """
    if single_claim:
        file_map = {
            "train": "train-eng-single-claim.csv",
            "dev": "dev-eng-single-claim.csv",
            "test": "test-eng-single-claim.csv",
        }
    else:
        file_map = {
            "train": "train-eng-filtered.csv" if filtered else "train-eng.csv",
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


def load_pipeline_data(pipeline_dir: Path, top_k: int = 5) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load claims and their source tweets from streaming pipeline output.

    Args:
        pipeline_dir: Path to pipeline output directory containing claims.parquet and tweets.parquet
        top_k: Number of representative tweets per cluster (sorted by cluster_similarity desc)

    Returns:
        claims_df: Original claims DataFrame (for updating later)
        tweets_df: DataFrame with columns matching CT25 format:
            - idx: row index
            - claim_id: FK to claims
            - post: concatenated top-K tweet texts (\\n\\n separated)
            - gold_claim: None (no ground truth)

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If no claims found in claims.parquet
    """
    claims_path = pipeline_dir / "claims.parquet"
    tweets_path = pipeline_dir / "tweets.parquet"

    # Check required files exist
    missing = []
    if not claims_path.exists():
        missing.append("claims.parquet")
    if not tweets_path.exists():
        missing.append("tweets.parquet")
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {pipeline_dir}: {', '.join(missing)}"
        )

    # Load data
    claims_df = pl.read_parquet(claims_path)
    tweets_df = pl.read_parquet(tweets_path)

    if len(claims_df) == 0:
        raise ValueError(f"No claims found in {claims_path}")

    print(f"Loaded {len(claims_df)} claims, {len(tweets_df)} tweets from pipeline")

    # Build mapping: claim_id -> list of cluster_ids
    # ClaimInfo schema has cluster_ids as list[int]
    rows = []
    skipped = 0

    for claim in claims_df.iter_rows(named=True):
        claim_id = claim["claim_id"]
        cluster_ids = claim.get("cluster_ids", [])

        # Handle case where cluster_ids might be stored as string or be None
        if cluster_ids is None:
            cluster_ids = []
        if isinstance(cluster_ids, str):
            import json
            try:
                cluster_ids = json.loads(cluster_ids)
            except json.JSONDecodeError:
                cluster_ids = []

        if not cluster_ids:
            # Fallback: use trigger_cluster_id if available
            trigger = claim.get("trigger_cluster_id")
            if trigger is not None:
                cluster_ids = [trigger]

        if not cluster_ids:
            skipped += 1
            continue

        # Get tweets for these clusters, sorted by similarity
        cluster_tweets = tweets_df.filter(
            pl.col("cluster_id").is_in(cluster_ids)
        )

        if len(cluster_tweets) == 0:
            skipped += 1
            continue

        # Sort by cluster_similarity descending, take top K
        if "cluster_similarity" in cluster_tweets.columns:
            cluster_tweets = cluster_tweets.sort("cluster_similarity", descending=True)

        top_tweets = cluster_tweets.head(top_k)

        # Concatenate tweet texts
        texts = top_tweets["text"].to_list()
        combined_post = "\n\n".join(texts)

        rows.append({
            "claim_id": claim_id,
            "post": combined_post,
            "gold_claim": None,
        })

    if skipped > 0:
        print(f"⚠️ Skipped {skipped} claims with no matching tweets")

    if not rows:
        raise ValueError("No claims with matching tweets found")

    # Create DataFrame matching CT25 format
    result_df = pl.DataFrame(rows).with_row_index("idx")

    print(f"Prepared {len(result_df)} claims for normalization (top-{top_k} tweets each)")

    return claims_df, result_df


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
    claim_verify_threshold: float = 0.5,
    retrieval_mode: str = "embedding",
    system_prompt: str = SYSTEM_PROMPT,
    ensemble: int = 1,
    ensemble_temperature: float = 0.7,
    clusters: list[str] | None = None,
    cluster_aware: bool = False,
    contrastive: ContrastiveExamples | None = None,
    contrastive_k: int = 1,
    contrastive_mode: str = "similar",
) -> NormalizationResult:
    """Normalize a single post with rate limiting, concurrency control, and timeout.

    Args:
        ensemble: Number of candidates to generate (1 = no ensemble, >1 = self-consistency)
        ensemble_temperature: Temperature for ensemble generation (default 0.7)
        clusters: List of detected cluster types for this post (for cluster-aware prompting)
        cluster_aware: Whether to use cluster-aware prompting
        contrastive: ContrastiveExamples object with good/bad examples
        contrastive_k: Number of bad examples to include
        contrastive_mode: How to select bad examples ("similar" or "random")
    """
    async with semaphore:
        await rate_limiter.acquire()

        # Truncate very long posts
        post_truncated = post[:4000] if len(post) > 4000 else post

        start = time.perf_counter()

        # Check for high-similarity retrieval match first (skip API call if match found)
        if retriever is not None and retrieval_threshold < 1.0:
            if retrieval_mode == "hybrid":
                match = retriever.get_best_match_hybrid(
                    post_truncated,
                    threshold=retrieval_threshold,
                    claim_verification_threshold=claim_verify_threshold,
                )
            else:
                match = retriever.get_best_match(
                    post_truncated,
                    threshold=retrieval_threshold,
                    claim_verification_threshold=claim_verify_threshold,
                )
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
            # Use topic-based retrieval if clusters are enabled
            if retriever.n_clusters > 0:
                examples = retriever.get_similar_examples_from_cluster(post_truncated, k=num_examples)
            else:
                examples = retriever.get_similar_examples(post_truncated, k=num_examples)
            good_examples_str = format_examples(examples)

            # Use contrastive template if contrastive examples are available
            if contrastive is not None and contrastive.bad_examples:
                if contrastive_mode == "similar":
                    bad_examples = contrastive.get_similar_bad_examples(post_truncated, k=contrastive_k)
                else:
                    bad_examples = contrastive.get_random_bad_examples(k=contrastive_k)
                bad_examples_str = format_contrastive_bad_examples(bad_examples)
                user_prompt = USER_PROMPT_TEMPLATE_CONTRASTIVE.format(
                    good_examples=good_examples_str,
                    bad_examples=bad_examples_str,
                    post=post_truncated,
                )
            else:
                user_prompt = USER_PROMPT_TEMPLATE_FEWSHOT.format(examples=good_examples_str, post=post_truncated)
        else:
            user_prompt = USER_PROMPT_TEMPLATE_ZEROSHOT.format(post=post_truncated)

        # Build cluster-aware prompt if enabled
        if cluster_aware and clusters:
            effective_prompt = build_cluster_aware_prompt(system_prompt, clusters, model_name)
        else:
            effective_prompt = system_prompt

        # Retry logic for transient API errors
        max_retries = 3
        retry_delay = 2.0
        predicted = None
        input_tokens = 0
        output_tokens = 0

        # GPT-5.x uses max_completion_tokens instead of max_tokens
        is_gpt5 = config.model_name.startswith("gpt-5")
        token_param = {"max_completion_tokens": 512} if is_gpt5 else {"max_tokens": 512}

        # Determine temperature based on ensemble setting
        temperature = ensemble_temperature if ensemble > 1 else 0.0
        num_candidates = ensemble if ensemble > 1 else 1

        async def single_api_call() -> tuple[str | None, int, int]:
            """Make a single API call and return (prediction, input_tokens, output_tokens)."""
            for attempt in range(max_retries):
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=config.model_name,
                            messages=[
                                {"role": "system", "content": effective_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=temperature,
                            **token_param,
                        ),
                        timeout=timeout_seconds,
                    )
                    raw_output = response.choices[0].message.content
                    pred = postprocess_claim(raw_output)
                    in_tok = response.usage.prompt_tokens if response.usage else 0
                    out_tok = response.usage.completion_tokens if response.usage else 0
                    return pred, in_tok, out_tok

                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    return f"TIMEOUT: No response after {max_retries} attempts", 0, 0

                except Exception as e:
                    error_str = str(e)
                    is_transient = any(code in error_str for code in ["503", "429", "502", "504", "Connection", "connection"])
                    if is_transient and attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    return f"ERROR: {e}", 0, 0

            return None, 0, 0

        # Generate candidates (1 for normal, N for ensemble)
        if num_candidates == 1:
            predicted, input_tokens, output_tokens = await single_api_call()
        else:
            # Ensemble: generate N candidates in parallel, select centroid
            results = await asyncio.gather(*[single_api_call() for _ in range(num_candidates)])
            candidates = []
            total_input_tokens = 0
            total_output_tokens = 0

            for pred, in_tok, out_tok in results:
                if pred and not pred.startswith(("ERROR:", "TIMEOUT:")):
                    candidates.append(pred)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

            if candidates:
                predicted, _ = select_centroid_candidate(candidates)
            else:
                # All candidates failed
                predicted = results[0][0] if results else "ERROR: No candidates generated"

            input_tokens = total_input_tokens
            output_tokens = total_output_tokens

        latency_ms = (time.perf_counter() - start) * 1000

        # Compute METEOR if gold is available
        meteor = compute_meteor(predicted, gold_claim) if gold_claim else None

        return NormalizationResult(
            idx=idx,
            post=post,
            predicted_claim=predicted,
            gold_claim=gold_claim,
            meteor=meteor,
            model=model_name + (f"@ens{num_candidates}" if num_candidates > 1 else ""),
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
    claim_verify_threshold: float = 0.5,
    retrieval_mode: str = "embedding",
    system_prompt: str = SYSTEM_PROMPT,
    ensemble: int = 1,
    ensemble_temperature: float = 0.7,
    cluster_mapping: dict[tuple[int, str], list[str]] | None = None,
    cluster_aware: bool = False,
    split: str = "dev",
    contrastive: ContrastiveExamples | None = None,
    contrastive_k: int = 1,
    contrastive_mode: str = "similar",
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
        ensemble: Number of candidates for self-consistency (1 = disabled)
        ensemble_temperature: Temperature for ensemble generation
        cluster_mapping: Mapping from (idx, split) to list of clusters
        cluster_aware: Whether to use cluster-aware prompting
        split: Data split name (for cluster lookup)
        contrastive: ContrastiveExamples for bad example injection
        contrastive_k: Number of bad examples per prompt
        contrastive_mode: "similar" or "random" bad example selection
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
                claim_verify_threshold=claim_verify_threshold,
                retrieval_mode=retrieval_mode,
                system_prompt=system_prompt,
                ensemble=ensemble,
                ensemble_temperature=ensemble_temperature,
                clusters=cluster_mapping.get((s["idx"], split), []) if cluster_mapping else None,
                cluster_aware=cluster_aware,
                contrastive=contrastive,
                contrastive_k=contrastive_k,
                contrastive_mode=contrastive_mode,
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
    claim_verify_threshold: float = 0.5,
    retrieval_mode: str = "embedding",
    system_prompt: str = SYSTEM_PROMPT,
    ensemble: int = 1,
    ensemble_temperature: float = 0.7,
    cluster_mapping: dict[tuple[int, str], list[str]] | None = None,
    cluster_aware: bool = False,
    split: str = "dev",
    contrastive: ContrastiveExamples | None = None,
    contrastive_k: int = 1,
    contrastive_mode: str = "similar",
) -> list[NormalizationResult]:
    """Run evaluation for a single model."""
    return asyncio.run(
        run_model_async(
            model_name, samples, parallel, rate_limit,
            checkpoint_path, timeout_seconds,
            retriever=retriever, num_examples=num_examples,
            retrieval_threshold=retrieval_threshold,
            claim_verify_threshold=claim_verify_threshold,
            retrieval_mode=retrieval_mode,
            system_prompt=system_prompt,
            ensemble=ensemble,
            ensemble_temperature=ensemble_temperature,
            cluster_mapping=cluster_mapping,
            cluster_aware=cluster_aware,
            split=split,
            contrastive=contrastive,
            contrastive_k=contrastive_k,
            contrastive_mode=contrastive_mode,
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
    no_save: bool,
    split: str,
    retriever: ExampleRetriever | None = None,
    num_examples: int = 3,
    retrieval_threshold: float = 1.0,
    claim_verify_threshold: float = 0.5,
    retrieval_mode: str = "embedding",
    system_prompt: str = SYSTEM_PROMPT,
    ensemble: int = 1,
    ensemble_temperature: float = 0.7,
    cluster_mapping: dict[tuple[int, str], list[str]] | None = None,
    cluster_aware: bool = False,
    contrastive: ContrastiveExamples | None = None,
    contrastive_k: int = 1,
    contrastive_mode: str = "similar",
) -> dict[str, dict]:
    """Run all models in the same event loop to avoid cleanup issues."""
    all_stats = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        checkpoint = None if no_save else results_dir / f"{model}_{split}.jsonl"
        # If --no-resume, delete existing checkpoint to start fresh
        if no_resume and checkpoint and checkpoint.exists():
            checkpoint.unlink()
            print(f"Deleted existing checkpoint: {checkpoint.name}")

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
                claim_verify_threshold=claim_verify_threshold,
                retrieval_mode=retrieval_mode,
                system_prompt=system_prompt,
                ensemble=ensemble,
                ensemble_temperature=ensemble_temperature,
                cluster_mapping=cluster_mapping,
                cluster_aware=cluster_aware,
                split=split,
                contrastive=contrastive,
                contrastive_k=contrastive_k,
                contrastive_mode=contrastive_mode,
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
    claim_verify_threshold: float = 0.5,
    retrieval_mode: str = "embedding",
    system_prompt: str = SYSTEM_PROMPT,
    cluster_mapping: dict[tuple[int, str], list[str]] | None = None,
    cluster_aware: bool = False,
    split: str = "dev",
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
                if retrieval_mode == "hybrid":
                    match = retriever.get_best_match_hybrid(
                        post,
                        threshold=retrieval_threshold,
                        claim_verification_threshold=claim_verify_threshold,
                    )
                else:
                    match = retriever.get_best_match(
                        post,
                        threshold=retrieval_threshold,
                        claim_verification_threshold=claim_verify_threshold,
                    )
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
                # Use topic-based retrieval if clusters are enabled
                if retriever.n_clusters > 0:
                    examples = retriever.get_similar_examples_from_cluster(post, k=num_examples)
                else:
                    examples = retriever.get_similar_examples(post, k=num_examples)
                examples_str = format_examples(examples)
                user_prompt = USER_PROMPT_TEMPLATE_FEWSHOT.format(examples=examples_str, post=post)
            else:
                user_prompt = USER_PROMPT_TEMPLATE_ZEROSHOT.format(post=post)

            # Build cluster-aware prompt if enabled
            if cluster_aware and cluster_mapping:
                clusters = cluster_mapping.get((s["idx"], split), [])
                effective_prompt = build_cluster_aware_prompt(system_prompt, clusters, llm.model_name)
            else:
                effective_prompt = system_prompt

            # Build chat prompt
            full_prompt = llm.build_chat_prompt(effective_prompt, user_prompt)
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
            # Clean up prediction using post-processing
            pred_clean = postprocess_claim(pred)

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
    parser.add_argument("--model", type=str, default="mixtral-8x7b",
                        help=f"Model to use. Available: {list(MODELS.keys())}")
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
                        help="Start fresh, ignore existing checkpoint (still saves results)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save checkpoint file")
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
                        help=f"Local model to use. Shortcuts: {list(LOCAL_MODELS.keys())}. "
                             "Can also be a path to a fine-tuned LoRA adapter directory.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for local GPU inference (default 8)")
    parser.add_argument("--retrieval-threshold", type=float, default=0.85,
                        help="Similarity threshold for retrieval fallback (default 0.85). "
                             "If a training example has similarity >= threshold, use its claim directly. "
                             "Set to 1.0 to disable retrieval fallback.")
    parser.add_argument("--claim-verify-threshold", type=float, default=0.5,
                        help="Threshold for claim verification in retrieval (default 0.5). "
                             "Retrieved claim must be this similar to query post to be used.")
    parser.add_argument("--retrieval-mode", type=str, default="embedding",
                        choices=["embedding", "hybrid"],
                        help="Retrieval mode: 'embedding' (default) uses only embedding similarity, "
                             "'hybrid' combines embedding + BM25 + entity overlap.")
    parser.add_argument("--prompt-version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="Prompt version: 'v1' (original), 'v2' (improved based on error analysis). "
                             "v2 addresses MISSING_INFO and WRONG_RELATION errors.")
    parser.add_argument("--filtered-examples", action="store_true",
                        help="Use filtered training data for few-shot retrieval. "
                             "Removes verbatim copies, examples that drop numbers, etc.")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Number of candidates for self-consistency ensemble. "
                             "1 = disabled (default), 3-5 recommended. Cost scales linearly.")
    parser.add_argument("--ensemble-temperature", type=float, default=0.7,
                        help="Temperature for ensemble candidate generation (default 0.7)")
    parser.add_argument("--single-claim", action="store_true",
                        help="Use single-claim filtered datasets (excludes multi_claim samples). "
                             "Requires running filter_multi_claim.py first.")
    parser.add_argument("--cluster-aware", action="store_true",
                        help="Use cluster-aware prompting. Detects post characteristics (multi_claim, "
                             "sarcasm, numeric, etc.) and injects model-optimized instructions. "
                             "Requires clusters.csv from compute_claim_clusters.py.")
    parser.add_argument("--cluster-file", type=str, default=None,
                        help="Path to clusters.csv file (default: auto-detect in data dir)")
    parser.add_argument("--topic-clusters", type=int, default=0,
                        help="Number of topic clusters for few-shot retrieval (0 = disabled). "
                             "When enabled, retrieves examples from same topic cluster first. "
                             "Recommended: 10-30 clusters.")
    parser.add_argument("--contrastive-file", type=str, default=None,
                        help="Path to results .jsonl file for contrastive learning. "
                             "Extracts good (high METEOR) and bad (low METEOR) examples.")
    parser.add_argument("--contrastive-examples", type=int, default=1,
                        help="Number of bad examples to include in contrastive prompt (default: 1)")
    parser.add_argument("--contrastive-mode", type=str, default="similar",
                        choices=["similar", "random"],
                        help="How to select bad examples: 'similar' (default) finds bad examples "
                             "similar to query post, 'random' picks randomly.")
    parser.add_argument("--auto-contrastive", action="store_true",
                        help="Automatically run two-stage contrastive learning: "
                             "1) Run baseline if results don't exist, 2) Run contrastive using baseline results. "
                             "Looks for {model}_{split}.jsonl in results directory.")
    parser.add_argument("--claim", type=str, default=None,
                        help="Single claim/post to normalize (instead of running on a dataset split). "
                             "Prints the normalized claim and exits.")

    # Full pipeline mode (for streaming pipeline output)
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Load data from streaming pipeline output instead of CT25 benchmark. "
                             "Updates claims.parquet with normalized claim_text.")
    parser.add_argument("--pipeline-dir", type=str, default=None,
                        help="Path to pipeline output directory (required if --full-pipeline)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of representative tweets per cluster for normalization (default: 5)")

    args = parser.parse_args()

    # Handle --claim: single claim inference mode
    if args.claim:
        async def run_single_claim():
            """Run inference on a single claim and print result."""
            import os

            # Get model config
            if args.model not in MODELS:
                print(f"❌ Unknown model: {args.model}")
                print(f"Available models: {list(MODELS.keys())}")
                return

            config = MODELS[args.model]
            api_key = os.getenv(config.api_key_env)
            if not api_key:
                print(f"❌ Missing API key: {config.api_key_env}")
                return

            client = AsyncOpenAI(api_key=api_key, base_url=config.api_base)

            # Initialize retriever for few-shot examples
            retriever = None
            num_examples = 0 if args.zero_shot else args.num_examples
            if num_examples > 0:
                try:
                    train_df = load_data("train")
                    retriever = ExampleRetriever.from_dataframe(train_df, n_clusters=args.topic_clusters)
                    print(f"Loaded {len(retriever.posts)} training examples for few-shot")
                except Exception as e:
                    print(f"⚠️ Could not load training data for few-shot: {e}")
                    print("   Running in zero-shot mode")
                    num_examples = 0

            # Create minimal semaphore and rate limiter
            semaphore = asyncio.Semaphore(1)
            rate_limiter = RateLimiter(requests_per_minute=600)

            print(f"\n{'='*60}")
            print(f"Model: {args.model}")
            print(f"Input: {args.claim[:100]}{'...' if len(args.claim) > 100 else ''}")
            print(f"{'='*60}\n")

            result = await normalize_single(
                client=client,
                config=config,
                idx=0,
                post=args.claim,
                gold_claim=None,
                model_name=args.model,
                semaphore=semaphore,
                rate_limiter=rate_limiter,
                retriever=retriever,
                num_examples=num_examples,
                timeout_seconds=args.timeout,
                retrieval_threshold=args.retrieval_threshold,
                claim_verify_threshold=args.claim_verify_threshold,
                retrieval_mode=args.retrieval_mode,
                system_prompt=PROMPT_VERSIONS[args.prompt_version],
            )

            print(f"Normalized claim: {result.predicted_claim}")
            print(f"\nLatency: {result.latency_ms:.0f}ms")
            if result.input_tokens > 0:
                cost = (result.input_tokens * config.cost_per_1m_input +
                        result.output_tokens * config.cost_per_1m_output) / 1_000_000
                print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")
                print(f"Cost: ${cost:.6f}")

        asyncio.run(run_single_claim())
        sys.exit(0)

    # Handle --auto-contrastive: automatically run two-stage contrastive learning
    if args.auto_contrastive:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_results_path = RESULTS_DIR / f"{args.model}_{args.split}.jsonl"

        if not baseline_results_path.exists():
            print("\n" + "=" * 60)
            print("AUTO-CONTRASTIVE: Stage 1 - Running baseline (no contrastive)")
            print("=" * 60)

            # Build command for baseline run (without auto-contrastive to avoid recursion)
            import subprocess
            baseline_cmd = [
                sys.executable, __file__,
                "--model", args.model,
                "--split", args.split,
                "--parallel", str(args.parallel),
                "--rate-limit", str(args.rate_limit),
                "--timeout", str(args.timeout),
                "--num-examples", str(args.num_examples),
                "--retrieval-threshold", str(args.retrieval_threshold),
                "--claim-verify-threshold", str(args.claim_verify_threshold),
                "--retrieval-mode", args.retrieval_mode,
                "--prompt-version", args.prompt_version,
            ]
            if args.limit:
                baseline_cmd.extend(["--limit", str(args.limit)])
            if args.topic_clusters > 0:
                baseline_cmd.extend(["--topic-clusters", str(args.topic_clusters)])
            if args.zero_shot:
                baseline_cmd.append("--zero-shot")
            if args.filtered_examples:
                baseline_cmd.append("--filtered-examples")
            if args.single_claim:
                baseline_cmd.append("--single-claim")
            if args.local:
                baseline_cmd.extend(["--local", "--local-model", args.local_model, "--batch-size", str(args.batch_size)])
            if args.no_resume:
                baseline_cmd.append("--no-resume")

            print(f"Command: {' '.join(baseline_cmd)}\n")
            result = subprocess.run(baseline_cmd)

            if result.returncode != 0:
                print(f"\n❌ Baseline run failed with exit code {result.returncode}")
                sys.exit(result.returncode)

            if not baseline_results_path.exists():
                print(f"\n❌ Baseline results not found at {baseline_results_path}")
                print("   Check if the model name matches the output filename.")
                sys.exit(1)
        else:
            print(f"\n✓ Found existing baseline results: {baseline_results_path.name}")

        print("\n" + "=" * 60)
        print("AUTO-CONTRASTIVE: Stage 2 - Running with contrastive examples")
        print("=" * 60 + "\n")

        # Set contrastive file to baseline results and proceed
        args.contrastive_file = str(baseline_results_path)
        args.auto_contrastive = False  # Prevent re-triggering

    # Override num_examples if zero-shot flag is set
    num_examples = 0 if args.zero_shot else args.num_examples

    # Resolve local model shortcut
    local_model = LOCAL_MODELS.get(args.local_model, args.local_model)

    # Track pipeline mode for output handling
    pipeline_mode = False
    pipeline_path = None
    original_claims_df = None

    # Handle --full-pipeline: load from streaming pipeline output
    if args.full_pipeline:
        if not args.pipeline_dir:
            parser.error("--pipeline-dir is required when using --full-pipeline")
        pipeline_path = Path(args.pipeline_dir)
        if not pipeline_path.exists():
            parser.error(f"Pipeline directory not found: {pipeline_path}")

        print("\n" + "=" * 60)
        print("FULL PIPELINE MODE")
        print(f"Loading from: {pipeline_path}")
        print("=" * 60 + "\n")

        try:
            original_claims_df, df = load_pipeline_data(pipeline_path, top_k=args.top_k)
            pipeline_mode = True
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Error loading pipeline data: {e}")
            sys.exit(1)

        if args.limit:
            df = df.head(args.limit)

        samples = df.to_dicts()
        print(f"Processing {len(samples)} claims from pipeline output")

        # Auto-discover contrastive file in pipeline mode
        if args.contrastive_file is None:
            auto_contrastive_path = RESULTS_DIR / f"{args.model}_train.jsonl"
            if auto_contrastive_path.exists():
                args.contrastive_file = str(auto_contrastive_path)
                print(f"✓ Auto-discovered contrastive file: {auto_contrastive_path.name}")
            else:
                print(f"ℹ No contrastive file found at {auto_contrastive_path.name} - proceeding without contrastive learning")
    else:
        # Standard CT25 benchmark mode
        df = load_data(args.split, single_claim=args.single_claim)
        if args.single_claim:
            print(f"Using SINGLE-CLAIM filtered dataset")
        if args.limit:
            df = df.head(args.limit)

        samples = df.to_dicts()
        print(f"\nLoaded {len(samples)} samples from {args.split} split")

    # Load cluster mapping if cluster-aware prompting is enabled
    cluster_mapping = None
    if args.cluster_aware:
        cluster_file = Path(args.cluster_file) if args.cluster_file else DATA_DIR / "clusters.csv"
        if cluster_file.exists():
            print(f"\n=== Cluster-Aware Prompting ===")
            cluster_mapping = load_cluster_mapping(cluster_file)
            print(f"Loaded {len(cluster_mapping)} cluster assignments from {cluster_file.name}")
            # Show model style being used
            model_for_style = args.model if not args.local else local_model
            style = get_model_style(model_for_style)
            print(f"Model prompt style: {style['style']} (for {model_for_style})")
            print("=" * 40)
        else:
            print(f"\n⚠️ WARNING: --cluster-aware enabled but clusters.csv not found at {cluster_file}")
            print("   Run compute_claim_clusters.py first, or specify --cluster-file path")
            print("   Proceeding without cluster-aware prompting.\n")
            args.cluster_aware = False

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
    use_topic_clusters = args.topic_clusters > 0
    if num_examples > 0:
        print(f"\n=== Few-Shot Setup ({num_examples} examples) ===")
        # Load training data for few-shot examples
        train_df = load_data("train", filtered=args.filtered_examples)
        if args.filtered_examples:
            print(f"Using FILTERED training data ({len(train_df)} examples)")
        else:
            print(f"Using full training data ({len(train_df)} examples)")
        # Also include dev data for more examples (if not evaluating on dev)
        if args.split != "dev":
            dev_df = load_data("dev")
            train_df = pl.concat([train_df, dev_df])
        retriever = ExampleRetriever.from_dataframe(
            train_df,
            n_clusters=args.topic_clusters,
        )
        print(f"Few-shot retriever ready with {len(retriever.posts)} examples")
        if use_topic_clusters:
            print(f"Topic-based retrieval: {args.topic_clusters} clusters")
        print("=" * 40 + "\n")
    else:
        print("\n=== Zero-Shot Mode (no examples) ===\n")

    # Load contrastive examples if specified
    contrastive = None
    if args.contrastive_file:
        contrastive_path = Path(args.contrastive_file)
        if contrastive_path.exists():
            print(f"\n=== Contrastive Learning Setup ===")
            contrastive = ContrastiveExamples.from_results_file(contrastive_path)
            print(f"Contrastive mode: {args.contrastive_mode}")
            print(f"Bad examples per prompt: {args.contrastive_examples}")
            print("=" * 40 + "\n")
        else:
            print(f"\n⚠️ WARNING: Contrastive file not found: {contrastive_path}")
            print("   Proceeding without contrastive examples.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOCAL GPU MODE
    # =========================================================================
    if args.local:
        # Load local model
        llm = LocalLLM.load(local_model)

        # Set checkpoint path
        model_short = local_model.split("/")[-1]
        checkpoint = None if args.no_save else RESULTS_DIR / f"local_{model_short}_{args.split}.jsonl"
        # If --no-resume, delete existing checkpoint to start fresh
        if args.no_resume and checkpoint and checkpoint.exists():
            checkpoint.unlink()
            print(f"Deleted existing checkpoint: {checkpoint.name}")

        # Run local inference
        results = run_local_model(
            llm=llm,
            samples=samples,
            retriever=retriever,
            num_examples=num_examples,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint,
            retrieval_threshold=args.retrieval_threshold,
            claim_verify_threshold=args.claim_verify_threshold,
            retrieval_mode=args.retrieval_mode,
            system_prompt=PROMPT_VERSIONS[args.prompt_version],
            cluster_mapping=cluster_mapping,
            cluster_aware=args.cluster_aware,
            split=args.split,
        )

        if results:
            stats = compute_stats(results, config=None)
            print_results(local_model, stats)
            save_model_results(model_short, stats, args.split, RESULTS_DIR)

            # === PIPELINE MODE: Update claims.parquet with normalized claims ===
            if pipeline_mode and original_claims_df is not None and pipeline_path is not None:
                print("\n" + "=" * 60)
                print("UPDATING CLAIMS.PARQUET")
                print("=" * 60)

                claim_text_map = {}
                for result in results:
                    if result.predicted_claim:
                        sample_idx = result.idx
                        if sample_idx < len(samples):
                            claim_id = samples[sample_idx].get("claim_id")
                            if claim_id:
                                claim_text_map[claim_id] = result.predicted_claim

                if claim_text_map:
                    updated_claims_df = original_claims_df.with_columns(
                        pl.when(pl.col("claim_id").is_in(list(claim_text_map.keys())))
                        .then(
                            pl.col("claim_id").replace_strict(
                                claim_text_map,
                                default=pl.col("claim_text")
                            )
                        )
                        .otherwise(pl.col("claim_text"))
                        .alias("claim_text")
                    )
                    output_path = pipeline_path / "claims.parquet"
                    updated_claims_df.write_parquet(output_path)
                    print(f"✓ Updated {len(claim_text_map)} claims in {output_path}")
                    print(f"  (out of {len(original_claims_df)} total claims)")
                else:
                    print("⚠️ No claims were successfully normalized")

    # =========================================================================
    # API MODE
    # =========================================================================
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
                no_save=args.no_save,
                split=args.split,
                retriever=retriever,
                num_examples=num_examples,
                retrieval_threshold=args.retrieval_threshold,
                claim_verify_threshold=args.claim_verify_threshold,
                retrieval_mode=args.retrieval_mode,
                system_prompt=PROMPT_VERSIONS[args.prompt_version],
                ensemble=args.ensemble,
                ensemble_temperature=args.ensemble_temperature,
                cluster_mapping=cluster_mapping,
                cluster_aware=args.cluster_aware,
                contrastive=contrastive,
                contrastive_k=args.contrastive_examples,
                contrastive_mode=args.contrastive_mode,
            )
        )

        # Print and save comparison
        print_comparison_table(all_stats)
        if all_stats:
            save_comparison_summary(all_stats, args.split, RESULTS_DIR)

    else:
        # Use different filename for contrastive mode
        if contrastive is not None:
            checkpoint = None if args.no_save else RESULTS_DIR / f"contrastive_{args.model}_{args.split}.jsonl"
        else:
            checkpoint = None if args.no_save else RESULTS_DIR / f"{args.model}_{args.split}.jsonl"
        # If --no-resume, delete existing checkpoint to start fresh
        if args.no_resume and checkpoint and checkpoint.exists():
            checkpoint.unlink()
            print(f"Deleted existing checkpoint: {checkpoint.name}")

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
            claim_verify_threshold=args.claim_verify_threshold,
            retrieval_mode=args.retrieval_mode,
            system_prompt=PROMPT_VERSIONS[args.prompt_version],
            ensemble=args.ensemble,
            ensemble_temperature=args.ensemble_temperature,
            cluster_mapping=cluster_mapping,
            cluster_aware=args.cluster_aware,
            split=args.split,
            contrastive=contrastive,
            contrastive_k=args.contrastive_examples,
            contrastive_mode=args.contrastive_mode,
        )

        if results and args.model in MODELS:
            stats = compute_stats(results, MODELS[args.model])
            # Use different name for contrastive mode
            display_name = f"contrastive_{args.model}" if contrastive is not None else args.model
            print_results(display_name, stats)
            save_model_results(display_name, stats, args.split, RESULTS_DIR)

        # === PIPELINE MODE: Update claims.parquet with normalized claims ===
        if pipeline_mode and results and original_claims_df is not None and pipeline_path is not None:
            print("\n" + "=" * 60)
            print("UPDATING CLAIMS.PARQUET")
            print("=" * 60)

            # Build mapping: claim_id -> normalized claim text
            # Results have idx which maps to samples, and samples have claim_id
            claim_text_map = {}
            for result in results:
                if result.predicted_claim:
                    # Get claim_id from the corresponding sample
                    sample_idx = result.idx
                    if sample_idx < len(samples):
                        claim_id = samples[sample_idx].get("claim_id")
                        if claim_id:
                            claim_text_map[claim_id] = result.predicted_claim

            if claim_text_map:
                # Update claim_text column in original claims DataFrame
                updated_claims_df = original_claims_df.with_columns(
                    pl.when(pl.col("claim_id").is_in(list(claim_text_map.keys())))
                    .then(
                        pl.col("claim_id").replace_strict(
                            claim_text_map,
                            default=pl.col("claim_text")
                        )
                    )
                    .otherwise(pl.col("claim_text"))
                    .alias("claim_text")
                )

                # Write back to claims.parquet
                output_path = pipeline_path / "claims.parquet"
                updated_claims_df.write_parquet(output_path)

                print(f"✓ Updated {len(claim_text_map)} claims in {output_path}")
                print(f"  (out of {len(original_claims_df)} total claims)")
            else:
                print("⚠️ No claims were successfully normalized")


if __name__ == "__main__":
    main()
