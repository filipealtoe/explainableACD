"""
Post-hoc checkworthiness scoring for pipeline output claims.

This script loads claims from a completed pipeline run and scores them
using the ensemble checkworthiness model (DeBERTa + LLM + XGBoost).

Usage:
    python experiments/scripts/score_claims_checkworthiness.py \
        --input data/pipeline_output/streaming_full/2026-01-17_XX-XX \
        --output data/pipeline_output/streaming_full/2026-01-17_XX-XX/claims_scored.parquet

Modes:
    --mode full      Score using full ensemble (DeBERTa + LLM + XGBoost) [default]
    --mode deberta   Score using only DeBERTa (fast, no API calls)
    --mode llm       Score using only LLM features + XGBoost (no GPU needed)
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.checkworthiness.predictor import (
    CheckworthinessConfig,
    CheckworthinessPredictor,
    CheckworthinessOutput,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_claims(input_dir: Path) -> pl.DataFrame:
    """Load claims from pipeline output."""
    claims_path = input_dir / "claims.parquet"
    if not claims_path.exists():
        # Try JSON format
        json_path = input_dir / "claims.json"
        if json_path.exists():
            with open(json_path) as f:
                claims_data = json.load(f)
            return pl.DataFrame(list(claims_data.values()))
        raise FileNotFoundError(f"No claims found in {input_dir}")

    return pl.read_parquet(claims_path)


async def score_claims_full(
    claims_df: pl.DataFrame,
    config: CheckworthinessConfig,
    batch_size: int = 10,
) -> list[CheckworthinessOutput]:
    """Score claims using full ensemble."""
    predictor = CheckworthinessPredictor(config)
    predictor.load()

    claim_texts = claims_df["claim_text"].to_list()
    results = []

    logger.info(f"Scoring {len(claim_texts)} claims with full ensemble...")

    for i in range(0, len(claim_texts), batch_size):
        batch = claim_texts[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(claim_texts) + batch_size - 1) // batch_size}")

        for text in batch:
            try:
                result = await predictor.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to score claim: {e}")
                results.append(
                    CheckworthinessOutput(
                        claim_text=text,
                        deberta_prob=0.0,
                        llm_prob=0.0,
                        fused_prob=0.0,
                        prediction="No",
                        is_checkworthy=False,
                    )
                )

    return results


def score_claims_deberta_only(
    claims_df: pl.DataFrame,
    config: CheckworthinessConfig,
) -> list[dict]:
    """Score claims using only DeBERTa (fast, no API calls)."""
    predictor = CheckworthinessPredictor(config)
    predictor._deberta.load()

    claim_texts = claims_df["claim_text"].to_list()

    logger.info(f"Scoring {len(claim_texts)} claims with DeBERTa only...")

    probs, probs_by_seed = predictor._deberta.predict(claim_texts)

    results = []
    for i, (text, prob) in enumerate(zip(claim_texts, probs)):
        results.append({
            "claim_text": text,
            "deberta_prob": float(prob),
            "llm_prob": None,
            "fused_prob": float(prob),  # Use DeBERTa as final
            "is_checkworthy": prob >= config.threshold,
            "prediction": "Yes" if prob >= config.threshold else "No",
            "checkability_score": None,
            "verifiability_score": None,
            "harm_score": None,
        })

    return results


def merge_scores_to_claims(
    claims_df: pl.DataFrame,
    results: list[CheckworthinessOutput] | list[dict],
) -> pl.DataFrame:
    """Merge checkworthiness scores back into claims DataFrame."""
    # Convert results to dicts if needed
    if results and isinstance(results[0], CheckworthinessOutput):
        result_dicts = [r.to_dict() for r in results]
    else:
        result_dicts = results

    # Create scores DataFrame
    scores_df = pl.DataFrame(result_dicts)

    # Select only the checkworthiness columns
    score_cols = [
        "deberta_prob",
        "llm_prob",
        "fused_prob",
        "is_checkworthy",
        "checkability_score",
        "verifiability_score",
        "harm_score",
    ]

    # Filter to available columns
    available_cols = [c for c in score_cols if c in scores_df.columns]

    # Add row index for joining
    claims_with_idx = claims_df.with_row_index("_idx")
    scores_with_idx = scores_df.select(available_cols).with_row_index("_idx")

    # Join
    merged = claims_with_idx.join(scores_with_idx, on="_idx", how="left")

    # Rename fused_prob to checkworthiness_prob
    if "fused_prob" in merged.columns:
        merged = merged.rename({"fused_prob": "checkworthiness_prob"})

    # Add llm_xgboost_prob alias
    if "llm_prob" in merged.columns:
        merged = merged.rename({"llm_prob": "llm_xgboost_prob"})

    # Drop index
    merged = merged.drop("_idx")

    return merged


def print_summary(claims_df: pl.DataFrame) -> None:
    """Print summary statistics."""
    n_total = len(claims_df)

    if "is_checkworthy" in claims_df.columns:
        n_checkworthy = claims_df.filter(pl.col("is_checkworthy") == True).height
        n_not_checkworthy = claims_df.filter(pl.col("is_checkworthy") == False).height

        logger.info(f"\n{'='*60}")
        logger.info("CHECKWORTHINESS SCORING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total claims scored: {n_total}")
        logger.info(f"Checkworthy: {n_checkworthy} ({100*n_checkworthy/n_total:.1f}%)")
        logger.info(f"Not checkworthy: {n_not_checkworthy} ({100*n_not_checkworthy/n_total:.1f}%)")

    if "checkworthiness_prob" in claims_df.columns:
        probs = claims_df["checkworthiness_prob"].drop_nulls()
        if len(probs) > 0:
            logger.info(f"\nProbability distribution:")
            logger.info(f"  Mean: {probs.mean():.3f}")
            logger.info(f"  Std:  {probs.std():.3f}")
            logger.info(f"  Min:  {probs.min():.3f}")
            logger.info(f"  Max:  {probs.max():.3f}")

    # Show top checkworthy claims
    if "checkworthiness_prob" in claims_df.columns and "claim_text" in claims_df.columns:
        logger.info(f"\nTop 10 most checkworthy claims:")
        top_claims = (
            claims_df
            .filter(pl.col("checkworthiness_prob").is_not_null())
            .sort("checkworthiness_prob", descending=True)
            .head(10)
        )
        for i, row in enumerate(top_claims.iter_rows(named=True)):
            prob = row.get("checkworthiness_prob", 0)
            text = row.get("claim_text", "")[:80]
            logger.info(f"  {i+1}. [{prob:.3f}] {text}...")


def main():
    parser = argparse.ArgumentParser(description="Score claims for checkworthiness")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to pipeline output directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save scored claims (default: {input}/claims_scored.parquet)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "deberta", "llm"],
        default="deberta",
        help="Scoring mode: full (ensemble), deberta (fast), llm (no GPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device for DeBERTa: cpu, cuda, mps",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--deberta-weight",
        type=float,
        default=0.6,
        help="Weight for DeBERTa in late fusion",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for feature extraction",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=30,
        help="LLM rate limit (requests per minute)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output) if args.output else input_dir / "claims_scored.parquet"

    # Load claims
    logger.info(f"Loading claims from {input_dir}")
    claims_df = load_claims(input_dir)
    logger.info(f"Loaded {len(claims_df)} claims")

    # Configure predictor
    config = CheckworthinessConfig(
        device=args.device,
        threshold=args.threshold,
        deberta_weight=args.deberta_weight,
        llm_model=args.llm_model,
        llm_rate_limit_rpm=args.rate_limit,
        xgboost_path="model/checkworthiness_xgboost.pkl",
        train_xgboost_on_init=True,
    )

    # Score claims
    if args.mode == "full":
        logger.info("Using full ensemble (DeBERTa + LLM + XGBoost)")
        results = asyncio.run(score_claims_full(claims_df, config))
    elif args.mode == "deberta":
        logger.info("Using DeBERTa only (fast mode)")
        results = score_claims_deberta_only(claims_df, config)
    else:  # llm
        logger.info("LLM-only mode not yet implemented")
        logger.info("Use --mode full or --mode deberta")
        return

    # Merge scores
    scored_df = merge_scores_to_claims(claims_df, results)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.write_parquet(output_path)
    logger.info(f"Saved scored claims to {output_path}")

    # Print summary
    print_summary(scored_df)


if __name__ == "__main__":
    main()
