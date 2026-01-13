#!/usr/bin/env python3
"""
Analyze temporal distribution of train/dev/test sets.

Extracts year from Sentence_id format (e.g., "2016_1_45" → 2016)
and shows distribution to understand domain shift.

Usage:
    python experiments/scripts/analyze_temporal_distribution.py
"""

from pathlib import Path
import polars as pl
import re
from collections import Counter

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "CT24_checkworthy_english"


def extract_year(sentence_id: str) -> int | None:
    """Extract year from Sentence_id format."""
    # Try patterns like "2016_1_45" or "1996_3_12"
    match = re.match(r"(\d{4})_", str(sentence_id))
    if match:
        year = int(match.group(1))
        if 1980 <= year <= 2025:
            return year
    return None


def analyze_split(split: str):
    """Analyze temporal distribution of a split."""
    path = DATA_DIR / f"CT24_checkworthy_english_{split}.tsv"
    if not path.exists():
        print(f"  {split}: NOT FOUND")
        return None

    df = pl.read_csv(path, separator="\t")

    # Extract years
    years = [extract_year(sid) for sid in df["Sentence_id"].to_list()]
    valid_years = [y for y in years if y is not None]

    if not valid_years:
        print(f"  {split}: Could not extract years from Sentence_id")
        return None

    year_counts = Counter(valid_years)
    total = len(valid_years)

    # Get label distribution by year
    labels = df["class_label"].to_list()
    year_label_counts = {}
    for year, label in zip(years, labels):
        if year is not None:
            if year not in year_label_counts:
                year_label_counts[year] = {"Yes": 0, "No": 0}
            year_label_counts[year][label] += 1

    return {
        "total": len(df),
        "years_extracted": len(valid_years),
        "year_counts": year_counts,
        "year_range": (min(valid_years), max(valid_years)),
        "year_label_counts": year_label_counts,
    }


def main():
    print("=" * 70)
    print("TEMPORAL DISTRIBUTION ANALYSIS")
    print("=" * 70)

    results = {}
    for split in ["train", "dev", "dev-test", "test"]:
        print(f"\n📊 {split.upper()}")
        result = analyze_split(split)
        if result:
            results[split] = result
            print(f"   Total samples: {result['total']}")
            print(f"   Year range: {result['year_range'][0]} - {result['year_range'][1]}")

            # Show top years
            top_years = sorted(result["year_counts"].items(), key=lambda x: -x[1])[:10]
            print(f"   Top years:")
            for year, count in top_years:
                pct = 100 * count / result["years_extracted"]
                yes_count = result["year_label_counts"][year]["Yes"]
                no_count = result["year_label_counts"][year]["No"]
                yes_pct = 100 * yes_count / (yes_count + no_count)
                print(f"      {year}: {count:4d} ({pct:5.1f}%) | Yes: {yes_pct:.0f}%")

    # Cross-split comparison
    print(f"\n{'='*70}")
    print("CROSS-SPLIT COMPARISON")
    print("=" * 70)

    if "train" in results and "test" in results:
        train_years = set(results["train"]["year_counts"].keys())
        test_years = set(results["test"]["year_counts"].keys())

        overlap = train_years & test_years
        train_only = train_years - test_years
        test_only = test_years - train_years

        print(f"\n   Train years: {sorted(train_years)}")
        print(f"   Test years: {sorted(test_years)}")
        print(f"\n   Overlapping years: {sorted(overlap) if overlap else 'NONE!'}")
        print(f"   Train-only years: {sorted(train_only)}")
        print(f"   Test-only years (DOMAIN SHIFT!): {sorted(test_only)}")

        if test_only:
            print(f"\n   ⚠️  WARNING: Test set has years not in training!")
            test_only_samples = sum(results["test"]["year_counts"][y] for y in test_only)
            test_total = results["test"]["years_extracted"]
            print(f"   {test_only_samples}/{test_total} test samples ({100*test_only_samples/test_total:.0f}%) are from unseen years")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("=" * 70)

    if "test" in results:
        test_years = results["test"]["year_counts"]
        recent_years = [y for y in test_years if y >= 2017]
        if recent_years:
            recent_count = sum(test_years[y] for y in recent_years)
            print(f"\n   🎯 {recent_count} test samples are from 2017+")
            print(f"   Prioritize generating synthetic data for these years:")
            for y in sorted(recent_years, reverse=True):
                print(f"      - {y}: {test_years[y]} samples")


if __name__ == "__main__":
    main()
