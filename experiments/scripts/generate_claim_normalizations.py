#!/usr/bin/env python3
"""Generate normalized claims for CT24 train/dev/test splits.

Claim normalization transforms raw social media text into clean, verifiable claims
that are self-contained, faithful to the original, and focused on fact-checkability.

This script processes the CT24 datasets and adds a normalized_claim column.

FEATURES:
- Saves results in batches (default: every 100 samples) to avoid losing progress
- Supports resuming from last checkpoint if interrupted
- Merges all batches into final output file
- Rate limiting optimized for Together AI (600 req/min)

Usage:
    # Generate normalizations for all splits
    python experiments/scripts/generate_claim_normalizations.py

    # Generate for specific split
    python experiments/scripts/generate_claim_normalizations.py --split train

    # Use different model (default: mistral-small-24b)
    python experiments/scripts/generate_claim_normalizations.py --model qwen-2.5-72b

    # Custom batch size and rate limit
    python experiments/scripts/generate_claim_normalizations.py --batch-size 50 --rate-limit 0.5

    # Resume from checkpoint
    python experiments/scripts/generate_claim_normalizations.py --resume

    # Dry run (5 samples per split)
    python experiments/scripts/generate_claim_normalizations.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock

import polars as pl
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Optional: json_repair for robust parsing
try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkworthiness.config import MODELS

# Load environment variables
load_dotenv(override=True)

# =============================================================================
# Constants
# =============================================================================

# Default model - mistral-small-24b has been the workhorse for LLM features
DEFAULT_MODEL = "mistral-small-24b"

# Batch settings (smaller than confidence features since normalization is simpler)
DEFAULT_BATCH_SIZE = 100

# Parallelization settings optimized for Together AI
# Together AI limit: 600 requests/minute = 10 requests/second
# We use 0.11s between calls (slightly under limit for safety buffer)
DEFAULT_RATE_LIMIT = 0.11  # ~9 requests/second (540/min, 10% safety buffer)
DEFAULT_MAX_WORKERS = 10   # More workers to saturate the rate limit

# Dataset paths
RAW_DATA_DIR = Path("data/raw/CT24_checkworthy_english")
V4_FEATURES_DIR = Path("data/processed/CT24_llm_features_v4")
OUTPUT_DIR = Path("data/processed/CT24_normalized_claims")
CHECKPOINT_DIR = Path("data/processed/CT24_normalized_claims/checkpoints")

# Raw data files (contain text and labels)
RAW_FILES = {
    "train": "CT24_checkworthy_english_train.tsv",
    "dev": "CT24_checkworthy_english_dev.tsv",
    "test": "CT24_checkworthy_english_test_gold.tsv",
}

# V4 feature files (for joining back later)
V4_FILES = {
    "train": "train_llm_features.parquet",
    "dev": "dev_llm_features.parquet",
    "test": "test_llm_features.parquet",
}

# =============================================================================
# Prompt Configuration
# =============================================================================

# Prompt file path (YAML format, Mistral 24B optimized with XML tags)
PROMPT_FILE = Path("prompts/claim_normalization.yaml")


def load_prompts(prompt_file: Path) -> tuple[str, str, dict]:
    """Load prompts from YAML file.

    Args:
        prompt_file: Path to the YAML prompt file

    Returns:
        Tuple of (system_prompt, user_template, json_schema)
    """
    with open(prompt_file) as f:
        config = yaml.safe_load(f)

    normalization = config["normalization"]
    system_prompt = normalization["system"].strip()
    user_template = normalization["user"].strip()

    # Load JSON schema
    json_schema = config.get("json_schema", {
        "type": "object",
        "properties": {
            "has_claim": {"type": "boolean"},
            "normalized_claim": {"type": ["string", "null"]}
        },
        "required": ["has_claim", "normalized_claim"],
        "additionalProperties": False
    })

    return system_prompt, user_template, json_schema


# Load prompts at module level
SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, JSON_SCHEMA = load_prompts(PROMPT_FILE)


# =============================================================================
# JSON Parsing with Fallbacks
# =============================================================================

def parse_json_response(content: str) -> dict | None:
    """Parse JSON from LLM response with multiple fallback strategies.

    Strategy order:
    1. Direct JSON parse
    2. Extract JSON from markdown code blocks
    3. json_repair library (if available)
    4. Regex extraction of fields

    Returns:
        Parsed dict with 'has_claim' and 'normalized_claim', or None if all fail
    """
    content = content.strip()

    # Strategy 1: Direct parse
    try:
        result = json.loads(content)
        if "has_claim" in result:
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    code_block_patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"`({.*?})`",
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if "has_claim" in result:
                    return result
            except json.JSONDecodeError:
                continue

    # Strategy 3: json_repair library
    if HAS_JSON_REPAIR:
        try:
            repaired = repair_json(content)
            result = json.loads(repaired)
            if "has_claim" in result:
                return result
        except Exception:
            pass

    # Strategy 4: Regex extraction of individual fields
    try:
        # Extract has_claim
        has_claim_match = re.search(r'"has_claim"\s*:\s*(true|false)', content, re.IGNORECASE)
        if has_claim_match:
            has_claim = has_claim_match.group(1).lower() == "true"

            # Extract normalized_claim
            claim_match = re.search(r'"normalized_claim"\s*:\s*(".*?"|null)', content, re.DOTALL)
            if claim_match:
                claim_raw = claim_match.group(1)
                if claim_raw.lower() == "null":
                    normalized_claim = None
                else:
                    # Remove surrounding quotes and unescape
                    normalized_claim = json.loads(claim_raw)

                return {
                    "has_claim": has_claim,
                    "normalized_claim": normalized_claim
                }
    except Exception:
        pass

    # Strategy 5: Plain text fallback - if content looks like a claim
    content_lower = content.lower()
    if "[no claim]" in content_lower or "no claim" in content_lower:
        return {"has_claim": False, "normalized_claim": None}

    # If it's a single sentence without JSON markers, treat as the claim
    if not any(c in content for c in ["{", "}", "[", "]"]) and len(content) < 500:
        return {"has_claim": True, "normalized_claim": content}

    return None


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last_call = 0.0
        self.lock = Lock()

    def wait(self):
        """Wait if needed to respect rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NormalizationResult:
    """Result of claim normalization for a single sample."""

    sentence_id: str
    text: str  # Original text
    label: str  # Ground truth: "Yes" or "No"
    normalized_claim: str | None  # The normalized claim
    has_claim: bool  # Whether a verifiable claim was found
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str | None = None


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(split: str) -> pl.DataFrame:
    """Load a CT24 dataset split from raw TSV files.

    Args:
        split: One of "train", "dev", "test"

    Returns:
        DataFrame with columns: sentence_id, text, label
    """
    if split not in RAW_FILES:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(RAW_FILES.keys())}")

    file_path = RAW_DATA_DIR / RAW_FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pl.read_csv(file_path, separator="\t")

    # Standardize column names (raw data has: Sentence_id, Text, class_label)
    df = df.rename({
        "Sentence_id": "sentence_id",
        "Text": "text",
        "class_label": "label",
    })

    # Ensure sentence_id is string for consistent joining
    df = df.with_columns(pl.col("sentence_id").cast(pl.Utf8))

    return df.select(["sentence_id", "text", "label"])


# =============================================================================
# Checkpoint Management
# =============================================================================

def get_checkpoint_path(split: str, model_name: str, batch_num: int) -> Path:
    """Get path for a checkpoint file."""
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return CHECKPOINT_DIR / f"{split}_{safe_model}_norm_batch_{batch_num:04d}.parquet"


def get_checkpoint_pattern(split: str, model_name: str) -> str:
    """Get glob pattern for checkpoint files."""
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return f"{split}_{safe_model}_norm_batch_*.parquet"


def find_existing_checkpoints(split: str, model_name: str) -> list[Path]:
    """Find all existing checkpoint files for a split."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pattern = get_checkpoint_pattern(split, model_name)
    checkpoints = sorted(CHECKPOINT_DIR.glob(pattern))
    return checkpoints


def load_checkpoints(split: str, model_name: str) -> tuple[set[str], list[NormalizationResult]]:
    """Load existing checkpoints and return processed sample IDs and results.

    Returns:
        Tuple of (set of processed sentence_ids, list of all results from checkpoints)
    """
    checkpoints = find_existing_checkpoints(split, model_name)
    if not checkpoints:
        return set(), []

    processed_ids = set()
    all_results = []

    for cp_path in checkpoints:
        df = pl.read_parquet(cp_path)
        processed_ids.update(df["sentence_id"].to_list())

        # Convert back to NormalizationResult
        for row in df.to_dicts():
            all_results.append(NormalizationResult(**row))

    print(f"  Loaded {len(checkpoints)} checkpoints with {len(processed_ids)} samples")
    return processed_ids, all_results


def save_checkpoint(
    results: list[NormalizationResult],
    split: str,
    model_name: str,
    batch_num: int,
) -> Path:
    """Save a batch of results as a checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp_path = get_checkpoint_path(split, model_name, batch_num)

    data = [asdict(r) for r in results]
    df = pl.DataFrame(data, infer_schema_length=None)
    df.write_parquet(cp_path)

    return cp_path


def merge_checkpoints(split: str, model_name: str, output_dir: Path) -> Path:
    """Merge all checkpoints into a single final file."""
    checkpoints = find_existing_checkpoints(split, model_name)
    if not checkpoints:
        raise ValueError(f"No checkpoints found for {split}")

    # Load and concatenate all checkpoints
    dfs = [pl.read_parquet(cp) for cp in checkpoints]
    merged_df = pl.concat(dfs).sort("sentence_id")

    # Save final file
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"CT24_{split}_normalized_{safe_model}_{timestamp}.parquet"

    merged_df.write_parquet(output_path)

    print(f"  Merged {len(checkpoints)} checkpoints -> {output_path.name}")
    print(f"  Total samples: {len(merged_df)}")

    return output_path


def cleanup_checkpoints(split: str, model_name: str) -> int:
    """Delete checkpoint files after successful merge."""
    checkpoints = find_existing_checkpoints(split, model_name)
    for cp in checkpoints:
        cp.unlink()
    return len(checkpoints)


# =============================================================================
# Normalization
# =============================================================================

def normalize_claim(
    client: OpenAI,
    model_name: str,
    text: str,
    sentence_id: str,
    label: str,
    rate_limiter: RateLimiter | None = None,
    max_retries: int = 3,
    use_json_schema: bool = True,
) -> NormalizationResult:
    """Normalize a single claim using the LLM with JSON output.

    Args:
        client: OpenAI client (works with Together AI too)
        model_name: Model name for API
        text: Original text to normalize
        sentence_id: Sample identifier
        label: Ground truth label
        rate_limiter: Optional rate limiter
        max_retries: Number of retries on failure
        use_json_schema: Whether to use JSON schema enforcement (Together AI)

    Returns:
        NormalizationResult with normalized claim or error
    """
    if rate_limiter:
        rate_limiter.wait()

    user_prompt = USER_PROMPT_TEMPLATE.format(text=text)

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            # Build API params
            api_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
            }

            # Add JSON schema for structured output (Together AI supports this)
            if use_json_schema:
                api_params["response_format"] = {
                    "type": "json_object",
                    "schema": JSON_SCHEMA,
                }

            response = client.chat.completions.create(**api_params)

            latency_ms = (time.time() - start_time) * 1000

            # Extract response content
            content = response.choices[0].message.content
            if content is None:
                content = ""
            content = content.strip()

            # Parse JSON with fallbacks
            parsed = parse_json_response(content)

            if parsed is not None:
                has_claim = parsed.get("has_claim", False)
                normalized = parsed.get("normalized_claim")
            else:
                # All parsing failed - log the raw content for debugging
                has_claim = False
                normalized = None

            # Get token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            return NormalizationResult(
                sentence_id=sentence_id,
                text=text,
                label=label,
                normalized_claim=normalized,
                has_claim=has_claim,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                error=None if parsed else f"Parse failed: {content[:100]}",
            )

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue

            return NormalizationResult(
                sentence_id=sentence_id,
                text=text,
                label=label,
                normalized_claim=None,
                has_claim=False,
                model_name=model_name,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                error=str(e),
            )

    # Should never reach here, but satisfy type checker
    return NormalizationResult(
        sentence_id=sentence_id,
        text=text,
        label=label,
        normalized_claim=None,
        has_claim=False,
        model_name=model_name,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        error="Max retries exceeded",
    )


def normalize_claims_parallel(
    client: OpenAI,
    model_name: str,
    samples: list[dict],
    max_workers: int = DEFAULT_MAX_WORKERS,
    rate_limit: float = DEFAULT_RATE_LIMIT,
    desc: str = "Processing",
    use_json_schema: bool = True,
) -> list[NormalizationResult]:
    """Normalize claims for multiple samples in parallel.

    Args:
        client: OpenAI client
        model_name: Model name
        samples: List of sample dicts with sentence_id, text, label
        max_workers: Number of parallel workers
        rate_limit: Seconds between API calls per thread
        desc: Progress bar description
        use_json_schema: Whether to use JSON schema enforcement

    Returns:
        List of NormalizationResult
    """
    rate_limiter = RateLimiter(rate_limit)
    results: list[NormalizationResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                normalize_claim,
                client,
                model_name,
                sample["text"],
                sample["sentence_id"],
                sample["label"],
                rate_limiter,
                3,  # max_retries
                use_json_schema,
            ): sample
            for sample in samples
        }

        with tqdm(total=len(samples), desc=desc) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    # Sort by sentence_id to maintain order
    results.sort(key=lambda x: x.sentence_id)
    return results


# =============================================================================
# Reporting
# =============================================================================

def print_summary(results: list[NormalizationResult], split: str) -> dict:
    """Print and return summary statistics."""
    n_total = len(results)
    n_api_errors = sum(1 for r in results if r.error and not r.error.startswith("Parse failed"))
    n_parse_errors = sum(1 for r in results if r.error and r.error.startswith("Parse failed"))
    n_has_claim = sum(1 for r in results if r.has_claim)
    n_no_claim = sum(1 for r in results if not r.has_claim and r.error is None)

    # Token usage
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)

    # Latency stats (exclude API errors)
    latencies = [r.latency_ms for r in results if not (r.error and not r.error.startswith("Parse"))]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Check correlation with labels
    checkworthy_with_claim = sum(1 for r in results if r.has_claim and r.label == "Yes")
    checkworthy_total = sum(1 for r in results if r.label == "Yes")
    non_checkworthy_with_claim = sum(1 for r in results if r.has_claim and r.label == "No")
    non_checkworthy_total = sum(1 for r in results if r.label == "No")

    print(f"\n{'='*60}")
    print(f"Summary for {split.upper()} split ({n_total} samples)")
    print(f"{'='*60}")
    print(f"  API errors: {n_api_errors} ({100*n_api_errors/n_total:.1f}%)")
    print(f"  Parse failures: {n_parse_errors} ({100*n_parse_errors/n_total:.1f}%)")
    print(f"  Has verifiable claim: {n_has_claim} ({100*n_has_claim/n_total:.1f}%)")
    print(f"  No claim found: {n_no_claim} ({100*n_no_claim/n_total:.1f}%)")
    print()
    print("  Claim detection by label:")
    if checkworthy_total > 0:
        print(f"    - Checkworthy (Yes): {checkworthy_with_claim}/{checkworthy_total} "
              f"({100*checkworthy_with_claim/checkworthy_total:.1f}%) have claims")
    if non_checkworthy_total > 0:
        print(f"    - Non-checkworthy (No): {non_checkworthy_with_claim}/{non_checkworthy_total} "
              f"({100*non_checkworthy_with_claim/non_checkworthy_total:.1f}%) have claims")
    print()
    print(f"  Token usage: {total_input:,} input + {total_output:,} output = {total_input + total_output:,} total")
    print(f"  Avg latency: {avg_latency:.0f}ms")
    print(f"{'='*60}")

    # Show sample parse failures for debugging
    if n_parse_errors > 0:
        print("\n  Sample parse failures (first 3):")
        parse_fails = [r for r in results if r.error and r.error.startswith("Parse failed")][:3]
        for r in parse_fails:
            print(f"    - {r.sentence_id}: {r.error}")

    return {
        "split": split,
        "n_total": n_total,
        "n_api_errors": n_api_errors,
        "n_parse_errors": n_parse_errors,
        "n_has_claim": n_has_claim,
        "n_no_claim": n_no_claim,
        "total_tokens": total_input + total_output,
        "avg_latency_ms": avg_latency,
    }


# =============================================================================
# Main Processing
# =============================================================================

def process_split(
    split: str,
    model_name: str,
    n_samples: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rate_limit: float = DEFAULT_RATE_LIMIT,
    resume: bool = True,
    use_json_schema: bool = True,
) -> tuple[list[NormalizationResult], Path]:
    """Process a single dataset split with batch checkpointing.

    Args:
        split: Dataset split ("train", "dev", "test")
        model_name: Model to use
        n_samples: Limit number of samples (for testing)
        max_workers: Parallel workers
        batch_size: Save checkpoint every N samples
        rate_limit: Seconds between API calls
        resume: Whether to resume from existing checkpoints
        use_json_schema: Whether to use JSON schema enforcement

    Returns:
        Tuple of (results list, output path)
    """
    print(f"\n{'#'*60}")
    print(f"# Processing {split.upper()} split with {model_name}")
    print(f"{'#'*60}")

    # Load dataset
    df = load_dataset(split)
    print(f"Loaded {len(df)} samples from {split}")

    # Limit samples if requested
    if n_samples is not None and n_samples < len(df):
        df = df.head(n_samples)
        print(f"Limited to {n_samples} samples")

    # Check for existing checkpoints
    processed_ids: set[str] = set()
    all_results: list[NormalizationResult] = []

    if resume:
        processed_ids, all_results = load_checkpoints(split, model_name)
        if processed_ids:
            print(f"  Resuming: {len(processed_ids)} samples already processed")

    # Filter out already processed samples
    samples = [s for s in df.to_dicts() if s["sentence_id"] not in processed_ids]
    print(f"  Remaining: {len(samples)} samples to process")

    if not samples:
        print("  All samples already processed!")
        output_path = merge_checkpoints(split, model_name, OUTPUT_DIR)
        return all_results, output_path

    # Initialize client
    model_config = MODELS[model_name]
    client = OpenAI(
        api_key=model_config.get_api_key(),
        base_url=model_config.api_base,
    )

    # Process in batches
    n_batches = (len(samples) + batch_size - 1) // batch_size
    start_batch = len(find_existing_checkpoints(split, model_name))

    print(f"  Processing {len(samples)} samples in {n_batches} batches of {batch_size}")
    print(f"  Checkpoints will be saved to: {CHECKPOINT_DIR}")

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        batch_num = start_batch + batch_idx

        print(f"\n  Batch {batch_idx + 1}/{n_batches} ({len(batch_samples)} samples)")

        # Process batch
        batch_results = normalize_claims_parallel(
            client=client,
            model_name=model_config.model_name,
            samples=batch_samples,
            max_workers=max_workers,
            rate_limit=rate_limit,
            desc=f"    Batch {batch_idx + 1}",
            use_json_schema=use_json_schema,
        )

        # Save checkpoint
        cp_path = save_checkpoint(batch_results, split, model_name, batch_num)
        print(f"    Saved checkpoint: {cp_path.name}")

        # Track progress
        all_results.extend(batch_results)

        # Print batch stats
        n_errors = sum(1 for r in batch_results if r.error is not None)
        n_claims = sum(1 for r in batch_results if r.has_claim)
        print(f"    Claims found: {n_claims}/{len(batch_results)}, Errors: {n_errors}")

    # Merge all checkpoints into final file
    print(f"\n  Merging checkpoints...")
    output_path = merge_checkpoints(split, model_name, OUTPUT_DIR)

    # Print summary
    print_summary(all_results, split)

    # Cleanup checkpoints after successful merge
    n_deleted = cleanup_checkpoints(split, model_name)
    print(f"  Cleaned up {n_deleted} checkpoint files")

    return all_results, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate normalized claims for CT24 datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test", "all"],
        default="all",
        help="Dataset split to process (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit samples per split (for testing)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Parallel workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Save checkpoint every N samples (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help=f"Seconds between API calls (default: {DEFAULT_RATE_LIMIT})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 5 samples per split",
    )
    parser.add_argument(
        "--no-json-schema",
        action="store_true",
        help="Disable JSON schema enforcement (fallback to prompt-only JSON)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(PROMPT_FILE),
        help=f"Path to prompt YAML file (default: {PROMPT_FILE})",
    )

    args = parser.parse_args()

    # Load prompts from file (allows override via CLI)
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}")
        sys.exit(1)

    global SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, JSON_SCHEMA
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, JSON_SCHEMA = load_prompts(prompt_path)

    # Validate model
    if args.model not in MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {list(MODELS.keys())}")
        sys.exit(1)

    # Dry run limits to 5 samples
    n_samples = 5 if args.dry_run else args.n_samples

    # Determine splits to process (order: test -> dev -> train, smallest first)
    if args.split == "all":
        splits = ["test", "dev", "train"]
    else:
        splits = [args.split]

    # Determine batch size (use 5 for dry-run)
    batch_size = 5 if args.dry_run else args.batch_size
    resume = not args.no_resume
    use_json_schema = not args.no_json_schema

    # Estimate cost
    model_config = MODELS[args.model]
    cost_per_1k_input = model_config.cost_per_1m_input / 1000
    cost_per_1k_output = model_config.cost_per_1m_output / 1000

    print("\n" + "=" * 60)
    print("CLAIM NORMALIZATION")
    print("=" * 60)
    print(f"Model: {args.model} ({model_config.model_name})")
    print(f"Prompt file: {prompt_path}")
    print(f"Splits: {splits}")
    print(f"Samples per split: {n_samples or 'all'}")
    print(f"Batch size: {batch_size}")
    print(f"Resume from checkpoints: {resume}")
    print(f"Max workers: {args.max_workers}")
    print(f"Rate limit: {args.rate_limit}s between calls")
    print(f"JSON schema enforcement: {use_json_schema}")
    print(f"json_repair available: {HAS_JSON_REPAIR}")
    print(f"Cost: ${cost_per_1k_input:.4f}/1K input, ${cost_per_1k_output:.4f}/1K output")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print("=" * 60)

    # Process each split
    all_paths = []
    total_stats = {"tokens": 0, "samples": 0}

    for split in splits:
        results, path = process_split(
            split=split,
            model_name=args.model,
            n_samples=n_samples,
            max_workers=args.max_workers,
            batch_size=batch_size,
            rate_limit=args.rate_limit,
            resume=resume,
            use_json_schema=use_json_schema,
        )
        all_paths.append(path)
        total_stats["samples"] += len(results)
        total_stats["tokens"] += sum(r.input_tokens + r.output_tokens for r in results)

    # Estimate total cost
    total_cost = (total_stats["tokens"] / 1_000_000) * (
        model_config.cost_per_1m_input + model_config.cost_per_1m_output
    ) / 2  # Rough average

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE - Output files:")
    print("=" * 60)
    for path in all_paths:
        print(f"  {path}")
    print()
    print(f"Total samples processed: {total_stats['samples']:,}")
    print(f"Total tokens used: {total_stats['tokens']:,}")
    print(f"Estimated cost: ${total_cost:.2f}")
    print()
    print("Next steps:")
    print("  1. Review normalized claims for quality")
    print("  2. Add normalized_claim column to feature datasets")
    print("  3. Use as additional feature or for claim-level analysis")


if __name__ == "__main__":
    main()
