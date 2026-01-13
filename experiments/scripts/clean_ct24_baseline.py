#!/usr/bin/env python3
"""
CT24 Baseline Cleaning Script

Applies deterministic cleaning and flagging to train/dev/test sets.

Behavior by split:
- train/dev: Rows are REMOVED based on quality issues, text is cleaned
- test: Rows are NEVER removed. Both original_text and cleaned_text are kept.
        Flags indicate what WOULD have been removed (would_exclude).

This preserves test set integrity for SOTA comparison while allowing
cleaned text for our pipeline.

Usage:
    python experiments/scripts/clean_ct24_baseline.py
    python experiments/scripts/clean_ct24_baseline.py --split train
    python experiments/scripts/clean_ct24_baseline.py --verbose
"""

from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path

import polars as pl

# Note: langdetect was removed - unreliable for short texts (high false negative rate)

# Optional: ftfy for mojibake fixing
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False
    print("Warning: ftfy not installed. Using basic encoding fixes.")
    print("Install with: uv add ftfy")


# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_clean"

DATA_PATHS = {
    "train": DATA_DIR / "CT24_checkworthy_english_train.tsv",
    "dev": DATA_DIR / "CT24_checkworthy_english_dev.tsv",
    "test": DATA_DIR / "CT24_checkworthy_english_test_gold.tsv",
}


# =============================================================================
# Cleaning Functions
# =============================================================================

def is_corrupted(text: str | None) -> bool:
    """Check if text is corrupted (Excel errors, etc.)."""
    if not text or not isinstance(text, str):
        return True
    corrupted_patterns = ["#NAME?", "#REF!", "#VALUE!", "#DIV/0!", "#NULL!", "#N/A"]
    return any(p in text for p in corrupted_patterns)


def fix_mojibake(text: str) -> str:
    """Fix encoding issues (mojibake). Maps directly to ASCII equivalents."""
    if HAS_FTFY:
        # ftfy handles this properly, then normalize_to_ascii will strip non-ASCII
        return ftfy.fix_text(text)
    # Basic fallback: map mojibake patterns directly to ASCII equivalents
    # (since we normalize to ASCII right after anyway)
    replacements = {
        # Accented vowels -> base letter
        "\xc3\xa9": "e", "\xc3\xa8": "e", "\xc3\xa0": "a", "\xc3\xa2": "a",
        "\xc3\xb4": "o", "\xc3\xae": "i", "\xc3\xaf": "i", "\xc3\xa7": "c",
        # Smart quotes -> ASCII quotes
        "\xe2\x80\x9c": '"', "\xe2\x80\x9d": '"',  # left/right double quotes
        "\xe2\x80\x98": "'", "\xe2\x80\x99": "'",  # left/right single quotes
        # Dashes -> hyphen
        "\xe2\x80\x94": "-",  # em-dash
        "\xe2\x80\x93": "-",  # en-dash
        # Ellipsis -> three dots
        "\xe2\x80\xa6": "...",
        # Common garbage
        "\xc2": "", "\xc3": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_to_ascii(text: str) -> str:
    """Transliterate accented chars to ASCII, remove non-ASCII."""
    # Normalize unicode (decompose accents)
    text = unicodedata.normalize("NFKD", text)
    # Keep only ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    # Match http/https URLs
    text = re.sub(r"https?://\S+", "", text)
    # Match www URLs
    text = re.sub(r"www\.\S+", "", text)
    # Match t.co short URLs
    text = re.sub(r"t\.co/\S+", "", text)
    return text


def strip_wrapping_quotes(text: str) -> str:
    """Remove matching quotes that wrap the entire text."""
    text = text.strip()
    # Check for matching wrapping quotes (using unicode escapes for non-ASCII)
    quote_pairs = [
        ('"', '"'),           # ASCII double quotes
        ("'", "'"),           # ASCII single quotes
        ("\u201c", "\u201d"), # curly double quotes ""
        ("\u2018", "\u2019"), # curly single quotes ''
        ("\u00ab", "\u00bb"), # guillemets «»
    ]
    for open_q, close_q in quote_pairs:
        if text.startswith(open_q) and text.endswith(close_q) and len(text) > 2:
            text = text[1:-1].strip()
    return text


def escape_internal_quotes(text: str) -> str:
    """Convert internal double quotes to single quotes for prompt safety."""
    # Replace double quotes with single quotes
    text = text.replace('"', "'")
    text = text.replace("\u201c", "'")  # left double quote
    text = text.replace("\u201d", "'")  # right double quote
    return text


def remove_control_chars(text: str) -> str:
    """Remove control characters and normalize newlines."""
    # Replace newlines and tabs with spaces
    text = re.sub(r"[\n\r\t]", " ", text)
    # Remove other control characters (except space)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text


def normalize_punctuation(text: str) -> str:
    """Reduce excessive punctuation."""
    # Reduce multiple punctuation to max 3
    text = re.sub(r"([!?.]){4,}", r"\1\1\1", text)
    # Reduce multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text


def decode_html_entities(text: str) -> str:
    """Decode HTML entities like &amp; -> &."""
    return html.unescape(text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str | None) -> str:
    """Apply all cleaning transformations in order."""
    if text is None or not isinstance(text, str):
        return ""

    # Order matters!
    text = fix_mojibake(text)           # Fix encoding first
    text = decode_html_entities(text)   # Decode HTML
    text = normalize_to_ascii(text)     # Then ASCII normalize
    text = remove_urls(text)            # Remove URLs
    text = strip_wrapping_quotes(text)  # Strip outer quotes
    text = escape_internal_quotes(text) # Escape internal quotes
    text = remove_control_chars(text)   # Remove control chars
    text = normalize_punctuation(text)  # Reduce excessive punctuation
    text = normalize_whitespace(text)   # Final whitespace cleanup

    return text


# =============================================================================
# Flag Functions
# =============================================================================

def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences (by terminal punctuation)."""
    if not text:
        return 0
    # Count .!? that are followed by space or end of string
    return len(re.findall(r"[.!?]+(?:\s|$)", text))


def is_question(text: str) -> bool:
    """Check if text is a question."""
    if not text:
        return False
    return text.rstrip().endswith("?")


def has_numbers(text: str) -> bool:
    """Check if text contains numbers."""
    if not text:
        return False
    return bool(re.search(r"\d+", text))


def is_all_caps(text: str) -> bool:
    """Check if text is mostly uppercase (>80% of letters)."""
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 5:
        return False
    uppercase = sum(1 for c in letters if c.isupper())
    return uppercase / len(letters) > 0.8


def is_mostly_tags(text: str) -> bool:
    """Check if text is mostly @mentions and #hashtags (>50% of tokens)."""
    if not text:
        return False
    tokens = text.split()
    if len(tokens) < 2:
        return False
    tag_tokens = sum(1 for t in tokens if t.startswith("@") or t.startswith("#"))
    return tag_tokens / len(tokens) > 0.5


def is_retweet(text: str) -> bool:
    """Check if text starts with RT (retweet indicator)."""
    if not text:
        return False
    return text.strip().upper().startswith("RT ")


def has_verb_heuristic(text: str) -> bool:
    """Simple heuristic: check for common verb patterns."""
    if not text:
        return False
    # Common verb indicators (very basic, not NLP)
    verb_patterns = [
        r"\b(is|are|was|were|be|been|being)\b",
        r"\b(have|has|had|having)\b",
        r"\b(do|does|did|doing|done)\b",
        r"\b(will|would|shall|should|can|could|may|might|must)\b",
        r"\b(say|said|says|saying)\b",
        r"\b(get|got|gets|getting)\b",
        r"\b(make|made|makes|making)\b",
        r"\b(go|goes|went|going|gone)\b",
        r"\b(know|knows|knew|knowing|known)\b",
        r"\b(think|thinks|thought|thinking)\b",
        r"\b(take|takes|took|taking|taken)\b",
        r"\b(see|sees|saw|seeing|seen)\b",
        r"\b(come|comes|came|coming)\b",
        r"\b(want|wants|wanted|wanting)\b",
        r"\b(use|uses|used|using)\b",
        r"\b(find|finds|found|finding)\b",
        r"\b(give|gives|gave|giving|given)\b",
        r"\b(tell|tells|told|telling)\b",
        r"\b(work|works|worked|working)\b",
        r"\b(call|calls|called|calling)\b",
        r"\b(try|tries|tried|trying)\b",
        r"\b(need|needs|needed|needing)\b",
        r"\b(feel|feels|felt|feeling)\b",
        r"\b(become|becomes|became|becoming)\b",
        r"\b(leave|leaves|left|leaving)\b",
        r"\b(put|puts|putting)\b",
        r"\b(mean|means|meant|meaning)\b",
        r"\b(keep|keeps|kept|keeping)\b",
        r"\b(let|lets|letting)\b",
        r"\b(begin|begins|began|beginning|begun)\b",
        r"\b(show|shows|showed|showing|shown)\b",
        r"\b(hear|hears|heard|hearing)\b",
        r"\b(play|plays|played|playing)\b",
        r"\b(run|runs|ran|running)\b",
        r"\b(move|moves|moved|moving)\b",
        r"\b(live|lives|lived|living)\b",
        r"\b(believe|believes|believed|believing)\b",
        r"\b(hold|holds|held|holding)\b",
        r"\b(bring|brings|brought|bringing)\b",
        r"\b(happen|happens|happened|happening)\b",
        r"\b(write|writes|wrote|writing|written)\b",
        r"\b(provide|provides|provided|providing)\b",
        r"\b(sit|sits|sat|sitting)\b",
        r"\b(stand|stands|stood|standing)\b",
        r"\b(lose|loses|lost|losing)\b",
        r"\b(pay|pays|paid|paying)\b",
        r"\b(meet|meets|met|meeting)\b",
        r"\b(include|includes|included|including)\b",
        r"\b(continue|continues|continued|continuing)\b",
        r"\b(set|sets|setting)\b",
        r"\b(learn|learns|learned|learning)\b",
        r"\b(change|changes|changed|changing)\b",
        r"\b(lead|leads|led|leading)\b",
        r"\b(understand|understands|understood|understanding)\b",
        r"\b(watch|watches|watched|watching)\b",
        r"\b(follow|follows|followed|following)\b",
        r"\b(stop|stops|stopped|stopping)\b",
        r"\b(create|creates|created|creating)\b",
        r"\b(speak|speaks|spoke|speaking|spoken)\b",
        r"\b(read|reads|reading)\b",
        r"\b(spend|spends|spent|spending)\b",
        r"\b(grow|grows|grew|growing|grown)\b",
        r"\b(open|opens|opened|opening)\b",
        r"\b(walk|walks|walked|walking)\b",
        r"\b(win|wins|won|winning)\b",
        r"\b(offer|offers|offered|offering)\b",
        r"\b(remember|remembers|remembered|remembering)\b",
        r"\b(love|loves|loved|loving)\b",
        r"\b(consider|considers|considered|considering)\b",
        r"\b(appear|appears|appeared|appearing)\b",
        r"\b(buy|buys|bought|buying)\b",
        r"\b(wait|waits|waited|waiting)\b",
        r"\b(serve|serves|served|serving)\b",
        r"\b(die|dies|died|dying)\b",
        r"\b(send|sends|sent|sending)\b",
        r"\b(expect|expects|expected|expecting)\b",
        r"\b(build|builds|built|building)\b",
        r"\b(stay|stays|stayed|staying)\b",
        r"\b(fall|falls|fell|falling|fallen)\b",
        r"\b(cut|cuts|cutting)\b",
        r"\b(reach|reaches|reached|reaching)\b",
        r"\b(kill|kills|killed|killing)\b",
        r"\b(remain|remains|remained|remaining)\b",
        r"\b(suggest|suggests|suggested|suggesting)\b",
        r"\b(raise|raises|raised|raising)\b",
        r"\b(pass|passes|passed|passing)\b",
        r"\b(sell|sells|sold|selling)\b",
        r"\b(require|requires|required|requiring)\b",
        r"\b(report|reports|reported|reporting)\b",
        r"\b(decide|decides|decided|deciding)\b",
        r"\b(pull|pulls|pulled|pulling)\b",
        r"\b(voted|votes|voting|vote)\b",
        r"\b(increased|increases|increasing|increase)\b",
        r"\b(decreased|decreases|decreasing|decrease)\b",
        r"\b(announced|announces|announcing|announce)\b",
        r"\b(claimed|claims|claiming|claim)\b",
        r"\b(stated|states|stating|state)\b",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in verb_patterns)


def compute_alpha_ratio(text: str) -> float:
    """Compute ratio of alphabetic characters to total characters."""
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text) if len(text) > 0 else 0.0


def compute_flags(cleaned_text: str, original_text: str) -> dict:
    """Compute all flag columns for a text."""
    word_count = count_words(cleaned_text)
    sentence_count = count_sentences(cleaned_text)

    return {
        # Quality flags (computed on original where it matters)
        "is_fragment": word_count <= 8 and not has_verb_heuristic(cleaned_text),
        "is_long": len(cleaned_text) > 500 if cleaned_text else False,
        "is_compound": sentence_count > 1,
        "is_all_caps": is_all_caps(original_text),
        "is_mostly_tags": is_mostly_tags(original_text),
        "is_retweet": is_retweet(original_text),

        # Predictive flags (from our lift analysis)
        "is_question": is_question(cleaned_text),
        "has_numbers": has_numbers(cleaned_text),

        # Metadata
        "word_count": word_count,
        "char_count": len(cleaned_text) if cleaned_text else 0,
        "sentence_count": sentence_count,
        "alpha_ratio": round(compute_alpha_ratio(cleaned_text), 3),
    }


# =============================================================================
# Main Processing
# =============================================================================

def load_dataset(split: str) -> pl.DataFrame:
    """Load a CT24 dataset split."""
    path = DATA_PATHS[split]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pl.read_csv(path, separator="\t", infer_schema_length=None)
    print(f"Loaded {split}: {len(df)} rows")
    return df


def process_dataset(df: pl.DataFrame, split: str, verbose: bool = False) -> pl.DataFrame:
    """
    Apply cleaning and flagging to a dataset.

    Behavior:
    - train/dev: Remove corrupted, empty, duplicate rows. Clean text.
    - test: Keep ALL rows. Add both original_text and cleaned_text.
            Add would_exclude flag for rows that would be removed in train/dev.
    """
    is_test = split == "test"
    original_count = len(df)

    # Get column names (may vary by split)
    # Find text column (case-insensitive)
    text_col = None
    for col in df.columns:
        if col.lower() in ("sentence_text", "text"):
            text_col = col
            break
    if text_col is None:
        raise ValueError(f"No text column found. Available: {df.columns}")

    # Find ID column (case-insensitive)
    id_col = None
    for col in df.columns:
        if col.lower() in ("sentence_id", "id"):
            id_col = col
            break
    if id_col is None:
        raise ValueError(f"No ID column found. Available: {df.columns}")

    # -------------------------------------------------------------------------
    # Step 1: Create original_text and cleaned_text columns
    # -------------------------------------------------------------------------
    df = df.with_columns([
        pl.col(text_col).alias("original_text"),
        pl.col(text_col).map_elements(clean_text, return_dtype=pl.Utf8).alias("cleaned_text"),
    ])

    # -------------------------------------------------------------------------
    # Step 2: Identify issues (for flagging and/or removal)
    # -------------------------------------------------------------------------

    # 2a: Corrupted entries
    df = df.with_columns([
        pl.col("original_text").map_elements(is_corrupted, return_dtype=pl.Boolean).alias("_is_corrupted")
    ])

    # 2b: Empty after cleaning
    df = df.with_columns([
        ((pl.col("cleaned_text").is_null()) |
         (pl.col("cleaned_text").str.len_chars() == 0) |
         (pl.col("cleaned_text").str.strip_chars() == "")).alias("_is_empty")
    ])

    # 2c: Duplicate IDs
    id_counts = df.group_by(id_col).len()
    dup_ids_set = set(id_counts.filter(pl.col("len") > 1)[id_col].to_list())
    if dup_ids_set:
        print(f"  Found {len(dup_ids_set)} duplicate IDs")
    df = df.with_columns([
        pl.col(id_col).map_elements(lambda x: x in dup_ids_set, return_dtype=pl.Boolean).alias("_is_dup_id")
    ])

    # 2d: Duplicate text (mark all but first as duplicate)
    df = df.with_columns([
        pl.cum_count("cleaned_text").over("cleaned_text").alias("_text_occurrence")
    ])
    df = df.with_columns([
        (pl.col("_text_occurrence") > 1).alias("_is_dup_text")
    ])

    # -------------------------------------------------------------------------
    # Step 3: Compute would_exclude flag (for all sets)
    # -------------------------------------------------------------------------
    df = df.with_columns([
        (pl.col("_is_corrupted") | pl.col("_is_empty") | pl.col("_is_dup_text")).alias("would_exclude")
    ])

    # Build exclusion reasons string
    def build_exclusion_reasons(row: dict) -> str:
        reasons = []
        if row["_is_corrupted"]:
            reasons.append("corrupted")
        if row["_is_empty"]:
            reasons.append("empty")
        if row["_is_dup_id"]:
            reasons.append("dup_id")
        if row["_is_dup_text"]:
            reasons.append("dup_text")
        return "|".join(reasons)

    exclusion_reasons = [build_exclusion_reasons(row) for row in df.iter_rows(named=True)]
    df = df.with_columns([pl.Series("exclusion_reasons", exclusion_reasons)])

    # -------------------------------------------------------------------------
    # Step 4: For train/dev, actually remove bad rows
    # -------------------------------------------------------------------------
    if not is_test:
        pre_filter = len(df)

        # Remove corrupted
        df = df.filter(~pl.col("_is_corrupted"))
        if verbose:
            removed = pre_filter - len(df)
            if removed > 0:
                print(f"  Removed {removed} corrupted rows")
            pre_filter = len(df)

        # Remove empty
        df = df.filter(~pl.col("_is_empty"))
        if verbose:
            removed = pre_filter - len(df)
            if removed > 0:
                print(f"  Removed {removed} empty rows")
            pre_filter = len(df)

        # Remove duplicate text (keep first)
        df = df.filter(~pl.col("_is_dup_text"))
        if verbose:
            removed = pre_filter - len(df)
            if removed > 0:
                print(f"  Removed {removed} duplicate text rows")

    # -------------------------------------------------------------------------
    # Step 5: Detect label inconsistencies
    # -------------------------------------------------------------------------
    if "class_label" in df.columns:
        # Find texts that have conflicting labels
        label_check = (
            df.group_by("cleaned_text")
            .agg([pl.col("class_label").n_unique().alias("n_labels")])
            .filter(pl.col("n_labels") > 1)
        )
        conflict_texts = set(label_check["cleaned_text"].to_list())

        if conflict_texts:
            print(f"  WARNING: {len(conflict_texts)} texts have conflicting labels!")

        df = df.with_columns([
            pl.col("cleaned_text").map_elements(
                lambda x: x in conflict_texts, return_dtype=pl.Boolean
            ).alias("has_label_conflict")
        ])
    else:
        df = df.with_columns([pl.lit(False).alias("has_label_conflict")])

    # -------------------------------------------------------------------------
    # Step 6: Compute all flags
    # -------------------------------------------------------------------------
    print("  Computing flags...")
    flags_list = []
    for row in df.iter_rows(named=True):
        flags = compute_flags(row["cleaned_text"] or "", row["original_text"] or "")
        flags_list.append(flags)

    flags_df = pl.DataFrame(flags_list)
    df = pl.concat([df, flags_df], how="horizontal")

    # -------------------------------------------------------------------------
    # Step 7: Clean up temporary columns
    # -------------------------------------------------------------------------
    temp_cols = ["_is_corrupted", "_is_empty", "_is_dup_id", "_is_dup_text", "_text_occurrence"]
    df = df.drop([c for c in temp_cols if c in df.columns])

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------
    if is_test:
        would_exclude_count = df["would_exclude"].sum()
        print(f"  Final: {len(df)} rows (0 removed, {would_exclude_count} flagged as would_exclude)")
    else:
        print(f"  Final: {len(df)} rows (removed {original_count - len(df)} total)")

    return df


def print_summary(df: pl.DataFrame, split: str) -> None:
    """Print summary statistics for cleaned dataset."""
    print(f"\n{'='*60}")
    print(f"Summary for {split}")
    print(f"{'='*60}")

    print(f"Total rows: {len(df)}")

    # Would exclude (test only)
    if "would_exclude" in df.columns and split == "test":
        would_exclude = df["would_exclude"].sum()
        print(f"Would exclude in train/dev: {would_exclude} ({100*would_exclude/len(df):.1f}%)")

    # Flag counts
    flag_cols = sorted([c for c in df.columns if c.startswith("is_") or c.startswith("has_")])
    print("\nFlag distributions:")
    for col in flag_cols:
        if col in df.columns and df[col].dtype == pl.Boolean:
            true_count = df[col].sum()
            pct = 100 * true_count / len(df)
            print(f"  {col}: {true_count} ({pct:.1f}%)")

    # Label distribution
    if "class_label" in df.columns:
        print("\nLabel distribution:")
        label_counts = df.group_by("class_label").len().sort("class_label")
        for row in label_counts.iter_rows(named=True):
            pct = 100 * row["len"] / len(df)
            print(f"  {row['class_label']}: {row['len']} ({pct:.1f}%)")

    # Word count stats
    if "word_count" in df.columns:
        print("\nWord count stats:")
        print(f"  Mean: {df['word_count'].mean():.1f}")
        print(f"  Median: {df['word_count'].median():.1f}")
        print(f"  Min: {df['word_count'].min()}")
        print(f"  Max: {df['word_count'].max()}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean CT24 datasets (train/dev rows removed, test rows preserved)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="all",
        help="Which split to clean (default: all)",
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

        # Load
        df = load_dataset(split)

        # Process (clean + flag)
        df_clean = process_dataset(df, split=split, verbose=args.verbose)

        # Print summary
        print_summary(df_clean, split)

        # Save as parquet (primary format)
        output_path = args.output_dir / f"CT24_{split}_clean.parquet"
        df_clean.write_parquet(output_path)
        print(f"\nSaved to: {output_path}")

        # Also save as TSV for inspection
        tsv_path = args.output_dir / f"CT24_{split}_clean.tsv"
        df_clean.write_csv(tsv_path, separator="\t")
        print(f"Saved TSV to: {tsv_path}")

    print("\n" + "="*60)
    print("Cleaning complete!")
    print("="*60)


if __name__ == "__main__":
    main()
