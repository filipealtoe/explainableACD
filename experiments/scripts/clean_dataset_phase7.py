"""
Phase 7: Text Enrichment & Filtering

Semantic enrichment and quality flagging.

Input: data/processed/tweets_v6_text_processed.parquet
Output: data/processed/tweets_v7_enriched.parquet

Operations:
1. Language Detection: Use FastText to flag non-English tweets (is_english).
2. Word Segmentation: Split joined words/hashtags (e.g., DataScience -> Data Science).
3. Acronym Expansion: Convert tech acronyms (AI -> Artificial Intelligence).
4. Bot Detection: Calculate bot_score and is_bot_like based on content features.
"""

import sys
import time
import urllib.request
from pathlib import Path

import fasttext
import mlflow
import polars as pl
import wordsegment

# Path setup
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# ==============================================================================
# Configuration & Dictionaries
# ==============================================================================

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_PATH = repo_root / "experiments" / ".cache" / "lid.176.bin"

TECH_ACRONYMS = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "iot": "internet of things",
    "vr": "virtual reality",
    "ar": "augmented reality",
    "api": "application programming interface",
    "ui": "user interface",
    "ux": "user experience",
    "saas": "software as a service",
    "llm": "large language model",
    "genai": "generative ai",
}

SLANG_DICT = {
    "idk": "i do not know",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "tbh": "to be honest",
    "thx": "thanks",
    "thnx": "thanks",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "r": "are",
    "ur": "your",
    "bc": "because",
    "b/c": "because",
    "w/": "with",
    "w/o": "without",
    "dm": "direct message",
}

# Combine dictionaries
EXPANSION_DICT = {**TECH_ACRONYMS, **SLANG_DICT}


# ==============================================================================
# Helper Functions
# ==============================================================================


def download_fasttext_model():
    """Download FastText language identification model if not present."""
    if not FASTTEXT_MODEL_PATH.exists():
        print(f"Downloading FastText model to {FASTTEXT_MODEL_PATH}...")
        FASTTEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)
        print("Download complete.")
    return str(FASTTEXT_MODEL_PATH)


def detect_language(model, text: str) -> bool:
    """Return True if English, False otherwise."""
    if not isinstance(text, str) or len(text) < 3:
        return False
    # FastText expects single line
    clean_text = text.replace("\n", " ")
    try:
        # Predict top 1 label
        labels, scores = model.predict(clean_text)
        if not labels:
            return False
        label = labels[0]
        return label == "__label__en"
    except Exception:
        return False


def segment_and_expand(text: str) -> str:
    """
    1. Segment joined words (CamelCase or joinedlower).
    2. Expand acronyms/slang.
    """
    if not isinstance(text, str) or not text:
        return ""

    tokens = text.split()
    processed_tokens = []

    for token in tokens:
        lower_token = token.lower()

        # 1. Direct Expansion (if it matches dictionary exactly)
        if lower_token in EXPANSION_DICT:
            processed_tokens.append(EXPANSION_DICT[lower_token])
            continue

        # 2. Segmentation check
        # Only attempt segmentation if:
        # - Contains uppercase (CamelCase)
        # - OR is very long (> 20 chars)
        # - AND contains only letters (no numbers/symbols)
        needs_segmentation = (
            (any(c.isupper() for c in token) and not token.isupper()) or len(token) > 20
        ) and token.isalpha()

        if needs_segmentation:
            # "DeepLearning" -> ["deep", "learning"]
            # Limit segmentation length to prevent hanging on garbage
            if len(token) > 50:
                segmented = [token]  # Too long, likely garbage
            else:
                segmented = wordsegment.segment(token)

            # Expand each part if needed
            expanded_segments = []
            for seg in segmented:
                if seg.lower() in EXPANSION_DICT:
                    expanded_segments.append(EXPANSION_DICT[seg.lower()])
                else:
                    expanded_segments.append(seg)
            processed_tokens.append(" ".join(expanded_segments))
        else:
            processed_tokens.append(token)

    return " ".join(processed_tokens)


def main():
    print("=" * 60)
    print("PHASE 7: TEXT ENRICHMENT & FILTERING")
    print("=" * 60)

    # 0. Setup
    model_path = download_fasttext_model()
    # Suppress FastText warning
    fasttext.FastText.eprint = lambda x: None
    ft_model = fasttext.load_model(model_path)
    wordsegment.load()

    # =========================================================================
    # Step 1: Load Phase 6 Output
    # =========================================================================
    print("\n[1] Loading Phase 6 output...")
    input_path = repo_root / "data" / "processed" / "tweets_v6_text_processed.parquet"
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    df = pl.read_parquet(input_path)
    print(f"    Loaded {len(df):,} rows")

    # =========================================================================
    # Step 2: Language Detection
    # =========================================================================
    print("\n[2] Detecting language...")

    # Process in chunks or plain list to avoid memory/pickling issues with FastText object
    texts = df["tweet"].to_list()
    # Use list comprehension (single threaded but safe)
    start_time = time.time()
    is_english_list = [detect_language(ft_model, t) for t in texts]
    print(f"    Detection took {time.time() - start_time:.2f}s")

    df = df.with_columns(pl.Series(name="is_english", values=is_english_list))

    non_en_count = len(df) - sum(is_english_list)
    print(f"    Flagged {non_en_count:,} non-English tweets ({100 * non_en_count / len(df):.1f}%)")

    # =========================================================================
    # Step 3: Segmentation & Expansion
    # =========================================================================
    print("\n[3] Segmenting words and expanding acronyms...")
    print("    (This may take 1-2 minutes...)")

    # We will use Polars map_elements.
    # To speed up, we can filter to only rows that likely need it?
    # No, we need it for everyone (acronyms are everywhere).

    start_time = time.time()
    # Using python map_elements is slow but correct.
    # For 800k rows, with optimized function, might take 2-3 mins.
    df = df.with_columns(
        pl.col("tweet").map_elements(segment_and_expand, return_dtype=pl.String).alias("tweet_enriched")
    )
    print(f"    Enrichment took {time.time() - start_time:.2f}s")

    # =========================================================================
    # Step 4: Bot Likeness Score
    # =========================================================================
    print("\n[4] Calculating Bot Score...")

    # Ensure boolean columns are treated as 0/1
    score_expr = (
        pl.col("has_cashtags").cast(pl.Int8) * 1
        + pl.col("has_urls").cast(pl.Int8) * 1
        + pl.col("has_many_urls").cast(pl.Int8) * 2
        + pl.col("excessive_hashtags").cast(pl.Int8) * 2
        + pl.col("is_manual_retweet").cast(pl.Int8) * 2
        + pl.col("has_all_caps").cast(pl.Int8) * 1
        + (pl.col("mention_count") > 3).cast(pl.Int8) * 1
        + pl.col("is_reply_tweet").cast(pl.Int8) * -1
    )

    df = df.with_columns(score_expr.clip(0, 20).alias("bot_score"))

    # Define boolean flag (e.g., score >= 4 is suspicious)
    df = df.with_columns((pl.col("bot_score") >= 4).alias("is_bot_like"))

    bot_count = df.filter(pl.col("is_bot_like")).shape[0]
    print(f"    Identified {bot_count:,} potential bot-like tweets ({100 * bot_count / len(df):.1f}%)")
    print(f"    Avg Bot Score: {df['bot_score'].mean():.2f}")

    # =========================================================================
    # Step 5: Save Output
    # =========================================================================
    print("\n[5] Saving enriched dataset...")
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "tweets_v7_enriched.parquet"
    df.write_parquet(output_path)
    print(f"    Saved to: {output_path}")

    # =========================================================================
    # Step 6: Log to MLflow
    # =========================================================================
    print("\n[6] Logging to MLflow...")
    mlflow.set_experiment("dataset_processing")

    with mlflow.start_run(run_name="tweets_v7_enriched"):
        mlflow.log_params(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "columns": len(df.columns),
                "operations": "lang_detect,word_segment,acronym_expand,bot_score",
            }
        )

        mlflow.log_metrics(
            {
                "pct_non_english": 100 * non_en_count / len(df),
                "pct_bot_like": 100 * bot_count / len(df),
                "avg_bot_score": df["bot_score"].mean(),
                "final_row_count": len(df),
            }
        )

        mlflow.log_artifact(str(output_path))
        print(f"    MLflow Run ID: {mlflow.active_run().info.run_id}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Location: {output_path}")
    print("-" * 60)

    print("Examples of enrichment:")
    sample = df.filter(pl.col("tweet") != pl.col("tweet_enriched")).head(5)
    for row in sample.iter_rows(named=True):
        print(f"ORIG:   {row['tweet']}")
        print(f"ENRICH: {row['tweet_enriched']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
