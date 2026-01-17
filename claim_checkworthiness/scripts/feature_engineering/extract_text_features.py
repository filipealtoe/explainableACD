#!/usr/bin/env python3
"""
Text Feature Extraction for Checkworthiness Classification

Extracts 35 text-based features from cleaned CT24 datasets.
All features are regex/string-based (no LLM calls, zero cost).

Feature groups:
- A: Quantification (numbers, precision, scale, ranges)
- B: Temporal (years, relative time, anchoring)
- C: Source/Evidence (attribution, agencies, evidence nouns)
- D: Comparative (comparisons, superlatives, rankings)
- E: Action/Change (increase/decrease, voting, negation)
- F: Opinion/Hedge (stance, desire, future, hedging)
- G: Rhetorical (fillers, fact assertions, questions)
- H: Metadata (word count, avg length, alpha ratio)
- I: Interactions (combined signals)

Usage:
    python experiments/scripts/extract_text_features.py
    python experiments/scripts/extract_text_features.py --split train
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl


# =============================================================================
# Paths
# =============================================================================

INPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_clean"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"


# =============================================================================
# Feature Extraction Functions
# =============================================================================

# --- Group A: Quantification Features ---

SPELLED_NUMBERS = r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b"


def has_number(text: str) -> bool:
    """Has any numeric content (digits or spelled-out)."""
    return bool(re.search(r"\d+", text) or re.search(SPELLED_NUMBERS, text.lower()))


def has_precise_number(text: str) -> bool:
    """Has precise numbers: decimals, percentages, or dollar amounts."""
    patterns = [
        r"\d+\.\d+",              # decimals
        r"\d+\s*%",               # percentages
        r"\bpercent\b",           # spelled percent
        r"\$\s*\d+",              # dollar sign
        r"\d+\s*dollars?\b",      # spelled dollars
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def has_large_scale(text: str) -> bool:
    """Has large scale numbers (million, billion, trillion)."""
    return bool(re.search(r"\b(million|billion|trillion)s?\b", text.lower()))


def count_numbers(text: str) -> int:
    """Count numeric tokens in text."""
    digit_matches = len(re.findall(r"\d+", text))
    spelled_matches = len(re.findall(SPELLED_NUMBERS, text.lower()))
    return digit_matches + spelled_matches


def has_range(text: str) -> bool:
    """Has range expressions (from X to Y, between X and Y)."""
    patterns = [
        r"\bfrom\s+\d+\s+to\s+\d+",
        r"\bbetween\s+\d+\s+and\s+\d+",
        r"\d+\s*[-–—]\s*\d+",     # 10-20, 10–20
        r"\d+\s+to\s+\d+",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def has_delta(text: str) -> bool:
    """Has change magnitude (by X, up/down X, twice, doubled)."""
    patterns = [
        r"\b(by|up|down)\s+\d+",
        r"\b(twice|half|double|triple|quadruple)d?\b",
        r"\b(doubled|tripled|halved)\b",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


# --- Group B: Temporal Features ---

def has_specific_year(text: str) -> bool:
    """Has specific year reference (1900s-2000s)."""
    return bool(re.search(r"\b(19|20)\d{2}\b", text))


def has_relative_time(text: str) -> bool:
    """Has relative time expressions."""
    patterns = [
        r"\b(last|this|next)\s+(year|month|week|decade)\b",
        r"\byesterday\b",
        r"\b(in the past|over the past|in the last)\s+\d+",
        r"\b\d+\s*(years?|months?|weeks?|days?)\s+ago\b",
        r"\brecently\b",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def has_temporal_anchor(text: str) -> bool:
    """Has temporal anchoring words (since, until, before, after)."""
    patterns = [
        r"\bsince\s+(19|20)\d{2}\b",
        r"\bsince\s+(the|last|january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b(until|before|after)\s+(19|20)\d{2}\b",
        r"\b(before|after)\s+the\s+\w+\b",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


# --- Group C: Source/Evidence Features ---

def has_source_attribution(text: str) -> bool:
    """Has source attribution phrases."""
    patterns = [
        r"\baccording to\b",
        r"\breport\s+(says?|shows?|found|indicates?)\b",
        r"\bstudy\s+(says?|shows?|found|indicates?)\b",
        r"\bdata\s+(says?|shows?|found|indicates?)\b",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def has_evidence_noun(text: str) -> bool:
    """Has evidence-related nouns."""
    return bool(re.search(
        r"\b(report|data|study|survey|statistics|records?|analysis|research|evidence|findings?)\b",
        text.lower()
    ))


def has_official_source(text: str) -> bool:
    """Has official agency/organization references."""
    agencies = [
        r"\b(CBO|FBI|CDC|BLS|CIA|DOJ|EPA|IRS|NSA|DEA|ATF|DHS|FEMA)\b",
        r"\b(Congress|Senate|House|Pentagon|White House)\b",
        r"\b(Federal Reserve|Treasury|State Department)\b",
        r"\b(United Nations|UN|NATO|WHO|IMF|World Bank)\b",
    ]
    return any(re.search(p, text) for p in agencies)  # Case sensitive for acronyms


def has_said_claimed(text: str) -> bool:
    """Has reporting verbs (said that, claimed that)."""
    return bool(re.search(
        r"\b(said|says|claimed|claims|stated|states|reported|reports)\s+that\b",
        text.lower()
    ))


# --- Group D: Comparative Features ---

def has_comparative(text: str) -> bool:
    """Has comparative expressions with 'than'."""
    return bool(re.search(
        r"\b(more|less|fewer|higher|lower|better|worse|greater|larger|smaller)\s+than\b",
        text.lower()
    ))


def has_superlative(text: str) -> bool:
    """Has superlative expressions."""
    return bool(re.search(
        r"\b(first|last|worst|best|highest|lowest|largest|smallest|biggest|greatest|most|least|record)\b",
        text.lower()
    ))


def has_ranking(text: str) -> bool:
    """Has ranking expressions."""
    return bool(re.search(
        r"\b(ranked|ranking|number one|number two|top\s+\d+|bottom\s+\d+|#\d+)\b",
        text.lower()
    ))


# --- Group E: Action/Change Features ---

def has_increase_decrease(text: str) -> bool:
    """Has change verbs."""
    return bool(re.search(
        r"\b(increased?|decreased?|grew|grown|fell|fallen|rose|risen|dropped|gained|lost|surged|plunged|spiked|declined?)\b",
        text.lower()
    ))


def has_voted(text: str) -> bool:
    """Has voting references."""
    return bool(re.search(r"\b(voted?|voting)\s+(for|against|to|on)\b", text.lower()))


def has_negation_claim(text: str) -> bool:
    """Has negation in claim (did not, never, failed to)."""
    patterns = [
        r"\b(did not|didn't|does not|doesn't|has not|hasn't|have not|haven't)\b",
        r"\b(never|no one|nobody|nothing|none of)\b",
        r"\b(failed to|refused to|denied)\b",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


# --- Group F: Opinion/Hedge Features ---

def has_first_person_stance(text: str) -> bool:
    """Has first-person opinion markers."""
    return bool(re.search(
        r"\b(i think|i believe|i feel|in my opinion|in my view|my view is|i\'m convinced)\b",
        text.lower()
    ))


def has_desire_intent(text: str) -> bool:
    """Has desire/intent expressions."""
    return bool(re.search(
        r"\b(i want|i hope|i wish|we should|we must|we need to|we have to|should be|ought to)\b",
        text.lower()
    ))


def has_future_modal(text: str) -> bool:
    """Has future modal verbs."""
    return bool(re.search(
        r"\b(will be|will have|going to|gonna|expect to|plan to|intend to)\b",
        text.lower()
    ))


def has_hedge(text: str) -> bool:
    """Has hedging/uncertainty words."""
    return bool(re.search(
        r"\b(probably|likely|possibly|perhaps|maybe|seems|appears|might|could be|may be|sort of|kind of)\b",
        text.lower()
    ))


def has_vague_quantifier(text: str) -> bool:
    """Has vague quantifier expressions."""
    return bool(re.search(
        r"\b(many people|some people|a lot of|lots of|everyone knows|people say|they say)\b",
        text.lower()
    ))


# --- Group G: Rhetorical Features ---

def has_rhetorical_filler(text: str) -> bool:
    """Has rhetorical filler phrases."""
    patterns = [
        r"\blet me\b",
        r"\blet\'s\b",
        r"\b(look|listen),",
        r"\bhere\'s the (thing|deal|point)\b",
        r"\byou know what\b",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def has_fact_assertion(text: str) -> bool:
    """Has fact assertion phrases."""
    return bool(re.search(
        r"\b(the fact is|the truth is|the reality is|fact of the matter|in fact)\b",
        text.lower()
    ))


def is_question(text: str) -> bool:
    """Text is a question."""
    return text.strip().endswith("?")


def has_transcript_artifact(text: str) -> bool:
    """Has transcript artifacts (stage directions, speaker tags)."""
    patterns = [
        r"\[.+\]",                          # [applause], [laughter]
        r"^[A-Z]+:",                          # MODERATOR:, BIDEN:
        r"\b(APPLAUSE|LAUGHTER|CROSSTALK)\b",
    ]
    return any(re.search(p, text) for p in patterns)


# --- Group H: Metadata Features ---

def compute_word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def compute_avg_word_length(text: str) -> float:
    """Average word length."""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def compute_alpha_ratio(text: str) -> float:
    """Ratio of alphabetic characters."""
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


# --- Group I: Interaction Features ---

def has_number_and_time(text: str) -> bool:
    """Interaction: number + temporal reference."""
    return has_number(text) and (has_specific_year(text) or has_relative_time(text))


def has_number_and_comparative(text: str) -> bool:
    """Interaction: number + comparative."""
    return has_number(text) and has_comparative(text)


def has_change_and_time(text: str) -> bool:
    """Interaction: increase/decrease + temporal."""
    return has_increase_decrease(text) and (has_specific_year(text) or has_relative_time(text))


def has_source_and_number(text: str) -> bool:
    """Interaction: source/evidence + number."""
    has_source = has_source_attribution(text) or has_evidence_noun(text) or has_official_source(text)
    return has_source and has_number(text)


# =============================================================================
# Main Feature Extraction
# =============================================================================

def extract_features(text: str) -> dict:
    """Extract all features from text."""
    return {
        # Group A: Quantification
        "has_number": has_number(text),
        "has_precise_number": has_precise_number(text),
        "has_large_scale": has_large_scale(text),
        "number_count": count_numbers(text),
        "has_range": has_range(text),
        "has_delta": has_delta(text),

        # Group B: Temporal
        "has_specific_year": has_specific_year(text),
        "has_relative_time": has_relative_time(text),
        "has_temporal_anchor": has_temporal_anchor(text),

        # Group C: Source/Evidence
        "has_source_attribution": has_source_attribution(text),
        "has_evidence_noun": has_evidence_noun(text),
        "has_official_source": has_official_source(text),
        "has_said_claimed": has_said_claimed(text),

        # Group D: Comparative
        "has_comparative": has_comparative(text),
        "has_superlative": has_superlative(text),
        "has_ranking": has_ranking(text),

        # Group E: Action/Change
        "has_increase_decrease": has_increase_decrease(text),
        "has_voted": has_voted(text),
        "has_negation_claim": has_negation_claim(text),

        # Group F: Opinion/Hedge
        "has_first_person_stance": has_first_person_stance(text),
        "has_desire_intent": has_desire_intent(text),
        "has_future_modal": has_future_modal(text),
        "has_hedge": has_hedge(text),
        "has_vague_quantifier": has_vague_quantifier(text),

        # Group G: Rhetorical
        "has_rhetorical_filler": has_rhetorical_filler(text),
        "has_fact_assertion": has_fact_assertion(text),
        "is_question": is_question(text),
        "has_transcript_artifact": has_transcript_artifact(text),

        # Group H: Metadata
        "word_count": compute_word_count(text),
        "avg_word_length": round(compute_avg_word_length(text), 2),
        "alpha_ratio": round(compute_alpha_ratio(text), 3),

        # Group I: Interactions
        "has_number_and_time": has_number_and_time(text),
        "has_number_and_comparative": has_number_and_comparative(text),
        "has_change_and_time": has_change_and_time(text),
        "has_source_and_number": has_source_and_number(text),
    }


def process_dataset(df: pl.DataFrame, verbose: bool = False) -> pl.DataFrame:
    """Extract features for all rows in dataset."""

    # Use cleaned_text for feature extraction
    text_col = "cleaned_text"

    if verbose:
        print(f"  Extracting features from {len(df)} rows...")

    # Extract features for each row
    features_list = []
    for i, text in enumerate(df[text_col].to_list()):
        features = extract_features(text or "")
        features_list.append(features)

        if verbose and (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{len(df)} rows...")

    # Create features DataFrame
    features_df = pl.DataFrame(features_list)

    # Combine with original data
    # Drop old features from cleaning script that we're replacing
    cols_to_drop = [
        "is_fragment", "is_compound", "is_long",
        "is_all_caps", "is_mostly_tags", "is_retweet",
        "has_numbers", "char_count", "sentence_count",
        "is_question", "word_count", "alpha_ratio",  # Also in new features
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(cols_to_drop)

    # Concatenate
    result = pl.concat([df, features_df], how="horizontal")

    return result


def print_summary(df: pl.DataFrame, split: str) -> None:
    """Print feature summary statistics."""
    print(f"\n{'='*60}")
    print(f"Feature Summary for {split}")
    print(f"{'='*60}")

    print(f"Total rows: {len(df)}")

    # Boolean features
    bool_cols = [c for c in df.columns if df[c].dtype == pl.Boolean]
    print(f"\nBoolean features ({len(bool_cols)}):")
    for col in sorted(bool_cols):
        true_count = df[col].sum()
        pct = 100 * true_count / len(df)
        print(f"  {col}: {true_count} ({pct:.1f}%)")

    # Numeric features
    numeric_cols = ["number_count", "word_count", "avg_word_length", "alpha_ratio"]
    print(f"\nNumeric features:")
    for col in numeric_cols:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Extract text features from CT24 datasets")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="all",
        help="Which split to process (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine splits to process
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    for split in splits:
        print(f"\n{'#'*60}")
        print(f"Processing {split} set")
        print(f"{'#'*60}")

        # Load cleaned data
        input_path = INPUT_DIR / f"CT24_{split}_clean.parquet"
        if not input_path.exists():
            print(f"  ERROR: Input file not found: {input_path}")
            continue

        df = pl.read_parquet(input_path)
        print(f"  Loaded {len(df)} rows from {input_path.name}")

        # Extract features
        df_features = process_dataset(df, verbose=args.verbose)

        # Print summary
        print_summary(df_features, split)

        # Save
        output_path = args.output_dir / f"CT24_{split}_features.parquet"
        df_features.write_parquet(output_path)
        print(f"\nSaved to: {output_path}")

        # Also save as TSV for inspection
        tsv_path = args.output_dir / f"CT24_{split}_features.tsv"
        df_features.write_csv(tsv_path, separator="\t")
        print(f"Saved TSV to: {tsv_path}")

    print("\n" + "="*60)
    print("Feature extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
