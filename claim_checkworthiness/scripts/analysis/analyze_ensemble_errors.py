#!/usr/bin/env python3
"""
Error Analysis for Ensemble Predictions.

Analyzes the errors made by the ensemble model to understand:
1. What types of samples are misclassified (FP vs FN)
2. Linguistic patterns in errors (length, questions, numbers, etc.)
3. Model agreement patterns (all wrong vs some right)
4. Confidence distribution of errors
5. Actionable insights for improvement

Usage:
    python analyze_ensemble_errors.py \
        --ensemble-dir ~/ensemble_results \
        --data-dir ~/data \
        --seeds 42 123 456 \
        --temperature 0.5 \
        --threshold 0.55
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import confusion_matrix, f1_score


# =============================================================================
# Text Feature Extraction
# =============================================================================

@dataclass
class TextFeatures:
    """Linguistic features extracted from text."""
    length: int  # character count
    word_count: int
    has_question: bool
    has_numbers: bool
    has_percentage: bool
    has_dollar: bool
    has_quotes: bool
    has_hashtag: bool
    has_mention: bool
    has_url: bool
    is_all_caps: bool
    has_exclamation: bool
    starts_with_i: bool  # "I think", "I believe" patterns
    has_opinion_words: bool
    has_claim_words: bool
    avg_word_length: float


def extract_features(text: str) -> TextFeatures:
    """Extract linguistic features from text."""
    words = text.split()
    word_count = len(words)

    # Opinion indicators
    opinion_patterns = [
        r'\bi think\b', r'\bi believe\b', r'\bi feel\b', r'\bin my opinion\b',
        r'\bseems like\b', r'\bprobably\b', r'\bmaybe\b', r'\bmight be\b'
    ]
    has_opinion = any(re.search(p, text.lower()) for p in opinion_patterns)

    # Claim indicators (factual language)
    claim_patterns = [
        r'\baccording to\b', r'\bstudies show\b', r'\bresearch\b', r'\bdata\b',
        r'\bpercent\b', r'\b\d+%', r'\bmillion\b', r'\bbillion\b', r'\bfact\b',
        r'\bproven\b', r'\bconfirmed\b', r'\breported\b'
    ]
    has_claim = any(re.search(p, text.lower()) for p in claim_patterns)

    return TextFeatures(
        length=len(text),
        word_count=word_count,
        has_question='?' in text,
        has_numbers=bool(re.search(r'\d', text)),
        has_percentage=bool(re.search(r'\d+%|\bpercent\b', text.lower())),
        has_dollar=bool(re.search(r'\$\d|dollars?', text.lower())),
        has_quotes='"' in text or "'" in text or '"' in text or '"' in text,
        has_hashtag='#' in text,
        has_mention='@' in text,
        has_url=bool(re.search(r'https?://|www\.', text.lower())),
        is_all_caps=text.isupper() and len(text) > 10,
        has_exclamation='!' in text,
        starts_with_i=text.lower().startswith('i '),
        has_opinion_words=has_opinion,
        has_claim_words=has_claim,
        avg_word_length=sum(len(w) for w in words) / max(word_count, 1),
    )


# =============================================================================
# Path Utilities (from ensemble script)
# =============================================================================

def find_results_dir(model_dir: Path) -> Path:
    """Find the actual directory containing results.json and probs."""
    if (model_dir / "results.json").exists():
        return model_dir

    if model_dir.exists():
        subdirs = [d for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        for subdir in subdirs:
            if (subdir / "results.json").exists():
                return subdir

    return model_dir


# =============================================================================
# Data Loading
# =============================================================================

def load_test_data(data_dir: Path) -> pl.DataFrame:
    """Load test data with texts and labels."""
    clean_dir = data_dir / "processed" / "CT24_clean"

    # Try different paths
    for name in ["CT24_test_clean.parquet", "CT24_test.parquet"]:
        path = clean_dir / name
        if path.exists():
            return pl.read_parquet(path)

    # Try TSV
    for name in ["CT24_test_clean.tsv", "CT24_test.tsv"]:
        path = clean_dir / name
        if path.exists():
            return pl.read_csv(path, separator="\t")

    # Fall back to raw
    raw_path = data_dir / "raw" / "CT24_checkworthy_english" / "CT24_checkworthy_english_test_gold.tsv"
    if raw_path.exists():
        return pl.read_csv(raw_path, separator="\t")

    raise FileNotFoundError(f"Test data not found in {data_dir}")


def load_model_probs(ensemble_dir: Path, seeds: list[int]) -> list[np.ndarray]:
    """Load probability arrays for each seed."""
    probs_list = []

    for seed in seeds:
        seed_dir = ensemble_dir / f"seed_{seed}"
        results_dir = find_results_dir(seed_dir)
        prob_file = results_dir / "test_probs.npy"

        if prob_file.exists():
            probs_list.append(np.load(prob_file))
        else:
            raise FileNotFoundError(f"Probabilities not found: {prob_file}")

    return probs_list


# =============================================================================
# Temperature Scaling
# =============================================================================

def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
    epsilon = 1e-8
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))


def ensemble_with_temperature(probs_list: list[np.ndarray], temperature: float) -> np.ndarray:
    """Apply temperature scaling and average."""
    scaled = [apply_temperature(p, temperature) for p in probs_list]
    return np.mean(scaled, axis=0)


# =============================================================================
# Error Analysis
# =============================================================================

@dataclass
class ErrorSample:
    """A single misclassified sample."""
    idx: int
    text: str
    true_label: int
    pred_label: int
    ensemble_prob: float
    individual_probs: list[float]
    features: TextFeatures
    error_type: str  # "FP" or "FN"
    model_agreement: int  # how many models predicted correctly
    confidence: float  # distance from 0.5


def analyze_errors(
    texts: list[str],
    labels: list[int],
    probs_list: list[np.ndarray],
    ensemble_probs: np.ndarray,
    threshold: float,
) -> tuple[list[ErrorSample], list[ErrorSample], list[int]]:
    """
    Analyze all errors made by the ensemble.

    Returns:
        - false_positives: List of FP samples
        - false_negatives: List of FN samples
        - correct_indices: Indices of correctly classified samples
    """
    predictions = (ensemble_probs >= threshold).astype(int)

    false_positives = []
    false_negatives = []
    correct_indices = []

    for i, (text, label, pred, ens_prob) in enumerate(zip(texts, labels, predictions, ensemble_probs)):
        ind_probs = [float(p[i]) for p in probs_list]
        features = extract_features(text)

        # How many individual models got it right?
        ind_preds = [(p >= threshold) for p in ind_probs]
        model_agreement = sum(1 for p in ind_preds if p == label)

        if pred == label:
            correct_indices.append(i)
            continue

        error_type = "FP" if pred == 1 and label == 0 else "FN"
        confidence = abs(ens_prob - 0.5)

        error = ErrorSample(
            idx=i,
            text=text,
            true_label=label,
            pred_label=pred,
            ensemble_prob=ens_prob,
            individual_probs=ind_probs,
            features=features,
            error_type=error_type,
            model_agreement=model_agreement,
            confidence=confidence,
        )

        if error_type == "FP":
            false_positives.append(error)
        else:
            false_negatives.append(error)

    return false_positives, false_negatives, correct_indices


def compute_feature_stats(errors: list[ErrorSample], all_samples_features: list[TextFeatures]) -> dict:
    """Compare feature distributions between errors and all samples."""
    if not errors:
        return {}

    error_features = [e.features for e in errors]

    stats = {}
    feature_names = [
        'has_question', 'has_numbers', 'has_percentage', 'has_dollar',
        'has_quotes', 'has_hashtag', 'has_mention', 'has_url',
        'has_exclamation', 'starts_with_i', 'has_opinion_words', 'has_claim_words'
    ]

    for feat in feature_names:
        error_rate = sum(1 for f in error_features if getattr(f, feat)) / len(error_features)
        overall_rate = sum(1 for f in all_samples_features if getattr(f, feat)) / len(all_samples_features)

        # Lift: how much more likely is this feature in errors?
        lift = error_rate / max(overall_rate, 0.001)

        stats[feat] = {
            'error_rate': error_rate,
            'overall_rate': overall_rate,
            'lift': lift,
        }

    # Numeric features
    for feat in ['length', 'word_count', 'avg_word_length']:
        error_vals = [getattr(f, feat) for f in error_features]
        overall_vals = [getattr(f, feat) for f in all_samples_features]

        stats[feat] = {
            'error_mean': np.mean(error_vals),
            'overall_mean': np.mean(overall_vals),
            'error_std': np.std(error_vals),
        }

    return stats


def print_error_examples(errors: list[ErrorSample], title: str, n: int = 10):
    """Print example errors with details."""
    print(f"\n{'='*70}")
    print(f"{title} ({len(errors)} total)")
    print('='*70)

    # Sort by confidence (most confident errors first - these are worst)
    sorted_errors = sorted(errors, key=lambda e: -e.confidence)

    for i, err in enumerate(sorted_errors[:n]):
        agreement_str = f"{err.model_agreement}/{len(err.individual_probs)} models correct"
        conf_str = f"conf={err.confidence:.3f}"
        prob_str = f"p={err.ensemble_prob:.3f}"

        print(f"\n[{i+1}] {err.error_type} | {prob_str} | {conf_str} | {agreement_str}")
        print(f"    Text: {err.text[:150]}{'...' if len(err.text) > 150 else ''}")
        print(f"    Individual probs: {[f'{p:.3f}' for p in err.individual_probs]}")

        # Highlight relevant features
        feats = []
        if err.features.has_question:
            feats.append("QUESTION")
        if err.features.has_numbers:
            feats.append("HAS_NUMS")
        if err.features.has_opinion_words:
            feats.append("OPINION")
        if err.features.has_claim_words:
            feats.append("CLAIM_WORDS")
        if err.features.starts_with_i:
            feats.append("STARTS_I")
        if feats:
            print(f"    Features: {', '.join(feats)}")


def print_feature_analysis(fp_stats: dict, fn_stats: dict):
    """Print feature analysis comparing FP and FN errors."""
    print(f"\n{'='*70}")
    print("FEATURE ANALYSIS")
    print('='*70)

    print("\n📊 Boolean Features (lift > 1.5 = overrepresented in errors):")
    print(f"{'Feature':<20} {'FP Lift':<10} {'FN Lift':<10} {'Interpretation'}")
    print("-" * 70)

    bool_features = [
        'has_question', 'has_numbers', 'has_percentage', 'has_opinion_words',
        'has_claim_words', 'starts_with_i', 'has_exclamation', 'has_quotes'
    ]

    for feat in bool_features:
        fp_lift = fp_stats.get(feat, {}).get('lift', 0)
        fn_lift = fn_stats.get(feat, {}).get('lift', 0)

        # Interpretation
        interp = ""
        if fp_lift > 1.5:
            interp += "↑ in FP "
        if fn_lift > 1.5:
            interp += "↑ in FN "
        if fp_lift < 0.5:
            interp += "↓ in FP "
        if fn_lift < 0.5:
            interp += "↓ in FN "

        print(f"{feat:<20} {fp_lift:<10.2f} {fn_lift:<10.2f} {interp}")


def print_model_agreement_analysis(errors: list[ErrorSample], n_models: int):
    """Analyze how often models agree on errors."""
    print(f"\n{'='*70}")
    print("MODEL AGREEMENT ANALYSIS")
    print('='*70)

    agreement_counts = Counter(e.model_agreement for e in errors)

    print(f"\nHow many models got the error samples RIGHT?")
    print(f"(0 = all wrong, {n_models} = all right but ensemble wrong due to threshold)")
    print("-" * 40)

    for agreement in range(n_models + 1):
        count = agreement_counts.get(agreement, 0)
        pct = count / len(errors) * 100 if errors else 0
        bar = "█" * int(pct / 2)
        print(f"  {agreement} models correct: {count:>4} ({pct:>5.1f}%) {bar}")

    # Categorize errors
    hard_errors = [e for e in errors if e.model_agreement == 0]
    fixable_errors = [e for e in errors if e.model_agreement > 0]

    print(f"\n📍 HARD errors (all models wrong): {len(hard_errors)} ({len(hard_errors)/len(errors)*100:.1f}%)")
    print(f"   → These likely need better data or are inherently ambiguous")
    print(f"📍 FIXABLE errors (some models right): {len(fixable_errors)} ({len(fixable_errors)/len(errors)*100:.1f}%)")
    print(f"   → Better ensembling or model selection could help")

    return hard_errors, fixable_errors


def print_confidence_analysis(errors: list[ErrorSample]):
    """Analyze confidence distribution of errors."""
    print(f"\n{'='*70}")
    print("CONFIDENCE ANALYSIS")
    print('='*70)

    confidences = [e.confidence for e in errors]

    # Bucket by confidence
    high_conf = [e for e in errors if e.confidence > 0.3]  # p > 0.8 or p < 0.2
    med_conf = [e for e in errors if 0.1 < e.confidence <= 0.3]
    low_conf = [e for e in errors if e.confidence <= 0.1]  # p near 0.5

    print(f"\nError confidence distribution:")
    print(f"  High confidence errors (p > 0.8 or p < 0.2): {len(high_conf):>4} - WORST, model was sure but wrong")
    print(f"  Medium confidence errors:                    {len(med_conf):>4}")
    print(f"  Low confidence errors (p near 0.5):          {len(low_conf):>4} - OK, model was uncertain")

    if confidences:
        print(f"\n  Mean confidence of errors: {np.mean(confidences):.3f}")
        print(f"  Median confidence of errors: {np.median(confidences):.3f}")


def generate_recommendations(
    fp_errors: list[ErrorSample],
    fn_errors: list[ErrorSample],
    fp_stats: dict,
    fn_stats: dict,
    hard_errors: list[ErrorSample],
) -> list[str]:
    """Generate actionable recommendations based on error analysis."""
    recommendations = []

    # Check question patterns
    if fp_stats.get('has_question', {}).get('lift', 0) > 2:
        recommendations.append(
            "🔴 Questions are overrepresented in FP errors - model thinks questions are checkworthy. "
            "Consider: (1) Adding more question examples to training with 'No' labels, "
            "(2) Adding a 'is_question' feature to the model."
        )

    # Check opinion patterns
    if fn_stats.get('has_claim_words', {}).get('lift', 0) > 1.5:
        recommendations.append(
            "🔴 Claim-like language is overrepresented in FN errors - model misses checkworthy claims. "
            "Consider: Data augmentation with more factual claims, especially with statistics."
        )

    if fp_stats.get('has_opinion_words', {}).get('lift', 0) < 0.5:
        recommendations.append(
            "🟢 Model correctly identifies opinion words as non-checkworthy (low FP lift)."
        )

    # Check hard vs fixable ratio
    hard_pct = len(hard_errors) / (len(fp_errors) + len(fn_errors)) * 100 if (fp_errors or fn_errors) else 0
    if hard_pct > 60:
        recommendations.append(
            f"⚠️ {hard_pct:.0f}% of errors are HARD (all models wrong). "
            "This suggests the errors are due to: (1) label noise, (2) inherent ambiguity, "
            "or (3) distribution shift. Data augmentation may have limited impact."
        )
    else:
        recommendations.append(
            f"🟢 {100-hard_pct:.0f}% of errors are FIXABLE (some models got it right). "
            "Better model selection or architecture diversity could help."
        )

    # Check FP vs FN balance
    if len(fp_errors) > len(fn_errors) * 2:
        recommendations.append(
            "🔴 Many more FP than FN - model is too aggressive at predicting checkworthy. "
            "Consider: Raising threshold, or adding negative examples to training."
        )
    elif len(fn_errors) > len(fp_errors) * 2:
        recommendations.append(
            "🔴 Many more FN than FP - model misses checkworthy claims. "
            "Consider: Lowering threshold, or adding more positive examples."
        )

    return recommendations


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze ensemble prediction errors")
    parser.add_argument("--ensemble-dir", type=Path, required=True,
                        help="Directory containing ensemble results")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Data directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Seeds used for ensemble")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for ensemble (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Classification threshold (default: 0.55)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file for detailed results (JSON)")
    args = parser.parse_args()

    print("=" * 70)
    print("ENSEMBLE ERROR ANALYSIS")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Seeds: {args.seeds}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Threshold: {args.threshold}")

    # Load data
    print("\n📂 Loading data...")
    test_df = load_test_data(args.data_dir)
    texts = test_df["Text"].to_list()
    labels = [1 if l == "Yes" else 0 for l in test_df["class_label"].to_list()]
    print(f"   Loaded {len(texts)} test samples")
    print(f"   Class distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")

    # Load model probabilities
    print("\n📂 Loading model probabilities...")
    probs_list = load_model_probs(args.ensemble_dir, args.seeds)
    print(f"   Loaded probabilities for {len(probs_list)} models")

    # Create ensemble
    print("\n🔧 Creating ensemble...")
    ensemble_probs = ensemble_with_temperature(probs_list, args.temperature)
    predictions = (ensemble_probs >= args.threshold).astype(int)

    # Overall metrics
    f1 = f1_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    print(f"\n📊 Overall Metrics:")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Confusion Matrix:")
    print(f"              Pred No  Pred Yes")
    print(f"   Actual No    {cm[0,0]:>4}     {cm[0,1]:>4}")
    print(f"   Actual Yes   {cm[1,0]:>4}     {cm[1,1]:>4}")

    # Analyze errors
    print("\n🔍 Analyzing errors...")
    fp_errors, fn_errors, correct_indices = analyze_errors(
        texts, labels, probs_list, ensemble_probs, args.threshold
    )

    total_errors = len(fp_errors) + len(fn_errors)
    print(f"\n   Total errors: {total_errors} ({total_errors/len(texts)*100:.1f}%)")
    print(f"   False Positives: {len(fp_errors)} (predicted Yes, actual No)")
    print(f"   False Negatives: {len(fn_errors)} (predicted No, actual Yes)")

    # Extract features for all samples (for comparison)
    all_features = [extract_features(t) for t in texts]

    # Feature analysis
    fp_stats = compute_feature_stats(fp_errors, all_features)
    fn_stats = compute_feature_stats(fn_errors, all_features)

    print_feature_analysis(fp_stats, fn_stats)

    # Model agreement analysis
    all_errors = fp_errors + fn_errors
    hard_errors, fixable_errors = print_model_agreement_analysis(all_errors, len(args.seeds))

    # Confidence analysis
    print_confidence_analysis(all_errors)

    # Print example errors
    print_error_examples(fp_errors, "FALSE POSITIVES (predicted Yes, actual No)", n=10)
    print_error_examples(fn_errors, "FALSE NEGATIVES (predicted No, actual Yes)", n=10)

    # Hard errors (all models wrong)
    hard_fp = [e for e in hard_errors if e.error_type == "FP"]
    hard_fn = [e for e in hard_errors if e.error_type == "FN"]
    if hard_fp:
        print_error_examples(hard_fp, "HARD FALSE POSITIVES (all models wrong)", n=5)
    if hard_fn:
        print_error_examples(hard_fn, "HARD FALSE NEGATIVES (all models wrong)", n=5)

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print('='*70)

    recommendations = generate_recommendations(
        fp_errors, fn_errors, fp_stats, fn_stats, hard_errors
    )
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    # Save detailed results
    if args.output:
        # Helper to convert numpy types to Python native types
        def to_native(obj):
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output_data = {
            "settings": {
                "seeds": args.seeds,
                "temperature": args.temperature,
                "threshold": args.threshold,
            },
            "metrics": {
                "f1": float(f1),
                "total_errors": total_errors,
                "false_positives": len(fp_errors),
                "false_negatives": len(fn_errors),
                "hard_errors": len(hard_errors),
                "fixable_errors": len(fixable_errors),
            },
            "fp_stats": to_native(fp_stats),
            "fn_stats": to_native(fn_stats),
            "fp_examples": [
                {"idx": e.idx, "text": e.text, "prob": float(e.ensemble_prob), "agreement": e.model_agreement}
                for e in sorted(fp_errors, key=lambda x: -x.confidence)[:50]
            ],
            "fn_examples": [
                {"idx": e.idx, "text": e.text, "prob": float(e.ensemble_prob), "agreement": e.model_agreement}
                for e in sorted(fn_errors, key=lambda x: -x.confidence)[:50]
            ],
            "recommendations": recommendations,
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Detailed results saved to: {args.output}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"\n   📊 F1 Score: {f1:.4f}")
    print(f"   ❌ Total Errors: {total_errors} ({total_errors/len(texts)*100:.1f}%)")
    print(f"   🔴 Hard Errors (all models wrong): {len(hard_errors)} ({len(hard_errors)/total_errors*100:.1f}% of errors)")
    print(f"   🟡 Fixable Errors (some models right): {len(fixable_errors)} ({len(fixable_errors)/total_errors*100:.1f}% of errors)")

    print("\n✅ Error analysis complete!")


if __name__ == "__main__":
    main()
