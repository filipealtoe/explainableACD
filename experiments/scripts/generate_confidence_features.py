#!/usr/bin/env python3
"""Generate confidence features for classifier training.

This script processes CT24 datasets (train/dev/test) and adds 6 confidence columns:
- checkability_self_conf: Self-reported confidence for checkability
- verifiability_self_conf: Self-reported confidence for verifiability
- harm_self_conf: Self-reported confidence for harm potential
- checkability_logprob_conf: Logprob-derived confidence for checkability
- verifiability_logprob_conf: Logprob-derived confidence for verifiability
- harm_logprob_conf: Logprob-derived confidence for harm potential

These features can then be used to train a classifier to predict checkworthiness.

FEATURES:
- Saves results in batches (default: every 500 samples) to avoid losing progress
- Supports resuming from last checkpoint if interrupted
- Merges all batches into final output file

Usage:
    # Generate features for all splits (with batch saving)
    python experiments/scripts/generate_confidence_features.py

    # Generate features for specific split
    python experiments/scripts/generate_confidence_features.py --split train

    # Use specific model
    python experiments/scripts/generate_confidence_features.py --model llama-3.1-70b

    # Custom batch size
    python experiments/scripts/generate_confidence_features.py --batch-size 1000

    # Resume from checkpoint
    python experiments/scripts/generate_confidence_features.py --resume

    # Dry run (1 sample per split)
    python experiments/scripts/generate_confidence_features.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkworthiness.config import MODELS, ModelProvider
from src.checkworthiness.prompting_baseline import PromptingBaseline

# Load environment variables
load_dotenv(override=True)

# =============================================================================
# Constants
# =============================================================================

# Default model - gpt-3.5-turbo has 100% extraction rate and safe cutoff
DEFAULT_MODEL = "gpt-3.5-turbo"

# Batch settings
DEFAULT_BATCH_SIZE = 500  # Save checkpoint every N samples

# Parallelization settings (optimized for Together AI: 600 req/min = 10 req/sec)
# Each sample makes 3 API calls (checkability, verifiability, harm_potential)
# So effective rate is ~3.3 samples/sec with 10 workers
DEFAULT_MAX_WORKERS = 10
DEFAULT_RATE_LIMIT = 0.11  # seconds between requests (~9 req/sec with safety buffer)

# Default dataset paths (can be overridden via CLI)
DEFAULT_DATA_DIR = Path("data/raw/CT24_checkworthy_english")
DEFAULT_OUTPUT_DIR = Path("data/processed/CT24_with_confidences")

# These will be set in main() based on CLI args
DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"

# Dataset files (default naming, can be overridden)
DEFAULT_DATASET_FILES = {
    "train": "CT24_checkworthy_english_train.tsv",
    "dev": "CT24_checkworthy_english_dev.tsv",
    "test": "CT24_checkworthy_english_test_gold.tsv",
}
DATASET_FILES = DEFAULT_DATASET_FILES.copy()

# Default prompt file (can be overridden via CLI)
DEFAULT_PROMPT_FILE = Path("prompts/checkworthiness_prompts_zeroshot_v3.yaml")
PROMPT_FILE = DEFAULT_PROMPT_FILE


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter for API requests.

    Together AI allows 600 requests/minute = 10 req/sec.
    Each sample makes 3 API calls, so with 10 workers we need
    to space out requests to stay under the limit.
    """

    def __init__(self, min_interval: float = DEFAULT_RATE_LIMIT):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.last_request_time = 0.0

    def wait(self):
        """Wait if needed to respect rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()


# Global rate limiter instance (set in main)
rate_limiter: RateLimiter | None = None


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ConfidenceFeatures:
    """Confidence features for a single sample."""

    sample_id: str
    text: str
    label: str  # Ground truth: "Yes" or "No"

    # Self-reported confidences (from model's JSON output)
    checkability_self_conf: float | None
    verifiability_self_conf: float | None
    harm_self_conf: float | None

    # Logprob-derived confidences
    checkability_logprob_conf: float | None
    verifiability_logprob_conf: float | None
    harm_logprob_conf: float | None

    # Average confidence (for reference)
    average_confidence: float | None

    # Model prediction (for comparison)
    model_prediction: str

    # Quality flags
    json_parse_failed: bool = False
    logprobs_missing: bool = False
    error: str | None = None


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(split: str) -> pl.DataFrame:
    """Load a CT24 dataset split.

    Args:
        split: One of "train", "dev", "test"

    Returns:
        DataFrame with columns: sample_id, text, label
    """
    if split not in DATASET_FILES:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(DATASET_FILES.keys())}")

    file_path = DATA_DIR / DATASET_FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pl.read_csv(file_path, separator="\t")

    # Standardize column names (handle different capitalizations)
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ("sentence", "text"):
            rename_map[col] = "text"
        elif col_lower in ("class_label", "label"):
            rename_map[col] = "label"
        elif col_lower in ("sentence_id", "sample_id", "id"):
            rename_map[col] = "sample_id"

    if rename_map:
        df = df.rename(rename_map)

    # Ensure sample_id is string
    df = df.with_columns(pl.col("sample_id").cast(pl.Utf8))

    return df.select(["sample_id", "text", "label"])


# =============================================================================
# Checkpoint Management
# =============================================================================

def get_checkpoint_path(split: str, model_name: str, batch_num: int) -> Path:
    """Get path for a checkpoint file."""
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return CHECKPOINT_DIR / f"{split}_{safe_model}_batch_{batch_num:04d}.parquet"


def get_checkpoint_pattern(split: str, model_name: str) -> str:
    """Get glob pattern for checkpoint files."""
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return f"{split}_{safe_model}_batch_*.parquet"


def find_existing_checkpoints(split: str, model_name: str) -> list[Path]:
    """Find all existing checkpoint files for a split."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pattern = get_checkpoint_pattern(split, model_name)
    checkpoints = sorted(CHECKPOINT_DIR.glob(pattern))
    return checkpoints


def load_checkpoints(split: str, model_name: str) -> tuple[set[str], list[ConfidenceFeatures]]:
    """Load existing checkpoints and return processed sample IDs and features.

    Returns:
        Tuple of (set of processed sample_ids, list of all features from checkpoints)
    """
    checkpoints = find_existing_checkpoints(split, model_name)
    if not checkpoints:
        return set(), []

    processed_ids = set()
    all_features = []

    for cp_path in checkpoints:
        df = pl.read_parquet(cp_path)
        processed_ids.update(df["sample_id"].to_list())

        # Convert back to ConfidenceFeatures
        for row in df.to_dicts():
            all_features.append(ConfidenceFeatures(**row))

    print(f"  Loaded {len(checkpoints)} checkpoints with {len(processed_ids)} samples")
    return processed_ids, all_features


def save_checkpoint(
    features: list[ConfidenceFeatures],
    split: str,
    model_name: str,
    batch_num: int,
) -> Path:
    """Save a batch of features as a checkpoint.

    Args:
        features: List of features to save
        split: Dataset split
        model_name: Model name
        batch_num: Batch number

    Returns:
        Path to saved checkpoint
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp_path = get_checkpoint_path(split, model_name, batch_num)

    data = [asdict(f) for f in features]
    df = pl.DataFrame(data, infer_schema_length=None)
    df.write_parquet(cp_path)

    return cp_path


def merge_checkpoints(split: str, model_name: str, output_dir: Path) -> Path:
    """Merge all checkpoints into a single final file.

    Args:
        split: Dataset split
        model_name: Model name
        output_dir: Output directory for final file

    Returns:
        Path to merged file
    """
    checkpoints = find_existing_checkpoints(split, model_name)
    if not checkpoints:
        raise ValueError(f"No checkpoints found for {split}")

    # Load and concatenate all checkpoints
    dfs = [pl.read_parquet(cp) for cp in checkpoints]
    merged_df = pl.concat(dfs).sort("sample_id")

    # Save final file
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"CT24_{split}_with_confidences_{safe_model}_{timestamp}.parquet"

    merged_df.write_parquet(output_path)

    print(f"  Merged {len(checkpoints)} checkpoints -> {output_path.name}")
    print(f"  Total samples: {len(merged_df)}")

    return output_path


def cleanup_checkpoints(split: str, model_name: str) -> int:
    """Delete checkpoint files after successful merge.

    Returns:
        Number of files deleted
    """
    checkpoints = find_existing_checkpoints(split, model_name)
    for cp in checkpoints:
        cp.unlink()
    return len(checkpoints)


# =============================================================================
# Feature Generation
# =============================================================================

def generate_features_for_sample(
    baseline: PromptingBaseline,
    sample: dict,
) -> ConfidenceFeatures:
    """Generate confidence features for a single sample.

    Args:
        baseline: The PromptingBaseline instance
        sample: Dict with sample_id, text, label

    Returns:
        ConfidenceFeatures with all confidence values
    """
    try:
        # Rate limit before API call
        if rate_limiter is not None:
            rate_limiter.wait()

        # Run the 3-module pipeline
        result, usage, all_logprobs, reasoning, reasoning_logprobs = baseline(sample["text"])

        # Determine quality flags
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

        return ConfidenceFeatures(
            sample_id=sample["sample_id"],
            text=sample["text"],
            label=sample["label"],
            # Self-reported confidences
            checkability_self_conf=result.checkability.self_confidence,
            verifiability_self_conf=result.verifiability.self_confidence,
            harm_self_conf=result.harm_potential.self_confidence,
            # Logprob-derived confidences
            checkability_logprob_conf=result.checkability.logprob_confidence,
            verifiability_logprob_conf=result.verifiability.logprob_confidence,
            harm_logprob_conf=result.harm_potential.logprob_confidence,
            # Aggregates
            average_confidence=result.average_confidence,
            model_prediction=result.prediction,
            # Quality
            json_parse_failed=json_parse_failed,
            logprobs_missing=logprobs_missing,
        )

    except Exception as e:
        return ConfidenceFeatures(
            sample_id=sample["sample_id"],
            text=sample["text"],
            label=sample["label"],
            checkability_self_conf=None,
            verifiability_self_conf=None,
            harm_self_conf=None,
            checkability_logprob_conf=None,
            verifiability_logprob_conf=None,
            harm_logprob_conf=None,
            average_confidence=None,
            model_prediction="No",
            json_parse_failed=True,
            logprobs_missing=True,
            error=str(e),
        )


def generate_features_parallel(
    baseline: PromptingBaseline,
    samples: list[dict],
    max_workers: int = DEFAULT_MAX_WORKERS,
    desc: str = "Processing",
) -> list[ConfidenceFeatures]:
    """Generate features for multiple samples in parallel.

    Args:
        baseline: The PromptingBaseline instance
        samples: List of sample dicts
        max_workers: Number of parallel workers
        desc: Progress bar description

    Returns:
        List of ConfidenceFeatures
    """
    results: list[ConfidenceFeatures] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_features_for_sample, baseline, sample): sample
            for sample in samples
        }

        with tqdm(total=len(samples), desc=desc) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    # Sort by sample_id to maintain order
    results.sort(key=lambda x: x.sample_id)
    return results


# =============================================================================
# Saving Results
# =============================================================================

def save_features(
    features: list[ConfidenceFeatures],
    split: str,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save generated features to parquet file.

    Args:
        features: List of ConfidenceFeatures
        split: Dataset split name
        model_name: Model used for generation
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to list of dicts
    data = [asdict(f) for f in features]

    # Create DataFrame
    df = pl.DataFrame(data, infer_schema_length=None)

    # Save as parquet
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"CT24_{split}_with_confidences_{safe_model_name}_{timestamp}.parquet"

    df.write_parquet(output_path)

    return output_path


def print_summary(features: list[ConfidenceFeatures], split: str) -> dict:
    """Print and return summary statistics.

    Args:
        features: List of ConfidenceFeatures
        split: Dataset split name

    Returns:
        Dict with summary statistics
    """
    n_total = len(features)
    n_errors = sum(1 for f in features if f.error is not None)
    n_json_failed = sum(1 for f in features if f.json_parse_failed)
    n_logprobs_missing = sum(1 for f in features if f.logprobs_missing)

    # Count non-null confidences
    n_self_check = sum(1 for f in features if f.checkability_self_conf is not None)
    n_self_verif = sum(1 for f in features if f.verifiability_self_conf is not None)
    n_self_harm = sum(1 for f in features if f.harm_self_conf is not None)
    n_logprob_check = sum(1 for f in features if f.checkability_logprob_conf is not None)
    n_logprob_verif = sum(1 for f in features if f.verifiability_logprob_conf is not None)
    n_logprob_harm = sum(1 for f in features if f.harm_logprob_conf is not None)

    # Model accuracy (for reference)
    n_correct = sum(1 for f in features if f.model_prediction == f.label)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Summary for {split.upper()} split ({n_total} samples)")
    print(f"{'='*60}")
    print(f"  Errors: {n_errors} ({100*n_errors/n_total:.1f}%)")
    print(f"  JSON parse failures: {n_json_failed} ({100*n_json_failed/n_total:.1f}%)")
    print(f"  Logprobs missing: {n_logprobs_missing} ({100*n_logprobs_missing/n_total:.1f}%)")
    print()
    print("  Self-reported confidence extraction:")
    print(f"    - checkability: {n_self_check}/{n_total} ({100*n_self_check/n_total:.1f}%)")
    print(f"    - verifiability: {n_self_verif}/{n_total} ({100*n_self_verif/n_total:.1f}%)")
    print(f"    - harm: {n_self_harm}/{n_total} ({100*n_self_harm/n_total:.1f}%)")
    print()
    print("  Logprob-derived confidence extraction:")
    print(f"    - checkability: {n_logprob_check}/{n_total} ({100*n_logprob_check/n_total:.1f}%)")
    print(f"    - verifiability: {n_logprob_verif}/{n_total} ({100*n_logprob_verif/n_total:.1f}%)")
    print(f"    - harm: {n_logprob_harm}/{n_total} ({100*n_logprob_harm/n_total:.1f}%)")
    print()
    print(f"  Model zero-shot accuracy: {accuracy:.3f}")
    print(f"{'='*60}")

    return {
        "split": split,
        "n_total": n_total,
        "n_errors": n_errors,
        "n_json_failed": n_json_failed,
        "n_logprobs_missing": n_logprobs_missing,
        "self_conf_extraction": {
            "checkability": n_self_check / n_total,
            "verifiability": n_self_verif / n_total,
            "harm": n_self_harm / n_total,
        },
        "logprob_conf_extraction": {
            "checkability": n_logprob_check / n_total,
            "verifiability": n_logprob_verif / n_total,
            "harm": n_logprob_harm / n_total,
        },
        "model_accuracy": accuracy,
    }


# =============================================================================
# Main
# =============================================================================

def process_split(
    split: str,
    model_name: str,
    n_samples: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = True,
) -> tuple[list[ConfidenceFeatures], Path]:
    """Process a single dataset split with batch checkpointing.

    Args:
        split: Dataset split ("train", "dev", "test")
        model_name: Model to use
        n_samples: Limit number of samples (for testing)
        max_workers: Parallel workers
        batch_size: Save checkpoint every N samples
        resume: Whether to resume from existing checkpoints

    Returns:
        Tuple of (features list, output path)
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
    all_features: list[ConfidenceFeatures] = []

    if resume:
        processed_ids, all_features = load_checkpoints(split, model_name)
        if processed_ids:
            print(f"  Resuming: {len(processed_ids)} samples already processed")

    # Filter out already processed samples
    samples = [s for s in df.to_dicts() if s["sample_id"] not in processed_ids]
    print(f"  Remaining: {len(samples)} samples to process")

    if not samples:
        print("  All samples already processed!")
        # Merge and return
        output_path = merge_checkpoints(split, model_name, OUTPUT_DIR)
        return all_features, output_path

    # Initialize baseline
    model_config = MODELS[model_name]
    baseline = PromptingBaseline(
        model_config=model_config,
        prompts_path=PROMPT_FILE,
        temperature=0.0,
        threshold=50.0,
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
        batch_features = generate_features_parallel(
            baseline=baseline,
            samples=batch_samples,
            max_workers=max_workers,
            desc=f"    Batch {batch_idx + 1}",
        )

        # Save checkpoint
        cp_path = save_checkpoint(batch_features, split, model_name, batch_num)
        print(f"    Saved checkpoint: {cp_path.name}")

        # Track progress
        all_features.extend(batch_features)

        # Print batch stats
        n_errors = sum(1 for f in batch_features if f.error is not None)
        n_success = len(batch_features) - n_errors
        print(f"    Success: {n_success}/{len(batch_features)}, Errors: {n_errors}")

    # Merge all checkpoints into final file
    print(f"\n  Merging checkpoints...")
    output_path = merge_checkpoints(split, model_name, OUTPUT_DIR)

    # Print summary
    summary = print_summary(all_features, split)

    # Cleanup checkpoints after successful merge
    n_deleted = cleanup_checkpoints(split, model_name)
    print(f"  Cleaned up {n_deleted} checkpoint files")

    return all_features, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate confidence features for CT24 datasets"
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
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help=f"Min seconds between requests (default: {DEFAULT_RATE_LIMIT}, ~9 req/sec for Together AI)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Save checkpoint every N samples (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 1 sample per split",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Input data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default=None,
        help="Custom file pattern, e.g. 'CT24_checkworthy_english_{split}_v5.tsv' (use {split} placeholder)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(DEFAULT_PROMPT_FILE),
        help=f"Path to prompt YAML file (default: {DEFAULT_PROMPT_FILE})",
    )

    args = parser.parse_args()

    # Set global paths from CLI args
    global DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, DATASET_FILES, PROMPT_FILE
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    PROMPT_FILE = Path(args.prompt_file)

    # Update dataset files if custom pattern provided
    if args.file_pattern:
        DATASET_FILES = {
            "train": args.file_pattern.format(split="train"),
            "dev": args.file_pattern.format(split="dev"),
            "test": args.file_pattern.format(split="test"),
        }

    # Validate model
    if args.model not in MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {list(MODELS.keys())}")
        sys.exit(1)

    # Initialize rate limiter
    global rate_limiter
    rate_limiter = RateLimiter(min_interval=args.rate_limit)

    # Dry run limits to 1 sample
    n_samples = 1 if args.dry_run else args.n_samples

    # Determine splits to process (order: test -> dev -> train, smallest first)
    if args.split == "all":
        splits = ["test", "dev", "train"]
    else:
        splits = [args.split]

    # Determine batch size (use 1 for dry-run)
    batch_size = 1 if args.dry_run else args.batch_size
    resume = not args.no_resume

    print("\n" + "=" * 60)
    print("CONFIDENCE FEATURE GENERATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Splits: {splits}")
    print(f"Samples per split: {n_samples or 'all'}")
    print(f"Batch size: {batch_size}")
    print(f"Resume from checkpoints: {resume}")
    print(f"Max workers: {args.max_workers}")
    print(f"Rate limit: {args.rate_limit}s between requests (~{1/args.rate_limit:.1f} req/sec)")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Prompt file: {PROMPT_FILE}")
    print("=" * 60)

    # Process each split
    all_paths = []
    for split in splits:
        features, path = process_split(
            split=split,
            model_name=args.model,
            n_samples=n_samples,
            max_workers=args.max_workers,
            batch_size=batch_size,
            resume=resume,
        )
        all_paths.append(path)

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE - Output files:")
    print("=" * 60)
    for path in all_paths:
        print(f"  {path}")
    print()
    print("Next step: Run classifier training with these features")
    print("  python experiments/scripts/train_confidence_classifier.py")


if __name__ == "__main__":
    main()
