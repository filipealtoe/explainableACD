#!/usr/bin/env python3
"""
Generate LLM Features for Benchmark Datasets (ClaimBuster, CT23)

Runs the same 3 prompts (checkability, verifiability, harm_potential) on benchmark
datasets to produce features matching CT24 format (89 columns).

Usage:
    # Generate features for ClaimBuster groundtruth
    python experiments/scripts/generate_benchmark_llm_features.py --benchmark CB_groundtruth

    # Generate features for CT23
    python experiments/scripts/generate_benchmark_llm_features.py --benchmark CT23

    # Run both
    python experiments/scripts/generate_benchmark_llm_features.py --benchmark all

    # Dry run (5 samples)
    python experiments/scripts/generate_benchmark_llm_features.py --benchmark CB_groundtruth --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl
import yaml
from dotenv import load_dotenv
from json_repair import repair_json
from openai import OpenAI
from tqdm import tqdm

# Load environment
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "processed" / "benchmark_llm_features"
PROMPTS_PATH = Path(__file__).parent.parent.parent / "prompts" / "checkworthiness_prompts_zeroshot_v3.yaml"

# Together AI config
TOGETHER_API_BASE = "https://api.together.xyz/v1"
MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"

# Settings
CHECKPOINT_EVERY = 50
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_WORKERS = 30
REQUESTS_PER_MINUTE = 550
REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE

MODULES = ["checkability", "verifiability", "harm_potential"]

HEDGE_PATTERNS = [
    r"\buncertain\b", r"\bunclear\b", r"\bambiguous\b",
    r"\bdifficult to\b", r"\bhard to\b", r"\bnot sure\b",
    r"\bdepends\b", r"\bcould be\b", r"\bmight be\b",
]
HEDGE_REGEX = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    def __init__(self, requests_per_minute: float):
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.last_request = 0.0

    def acquire(self):
        with self.lock:
            now = time.time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.time()


class ThreadSafeWriter:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.lock = threading.Lock()

    def write(self, data: dict):
        with self.lock:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(data) + "\n")


# =============================================================================
# Data Loading
# =============================================================================

def load_benchmark(benchmark: str) -> pl.DataFrame:
    """Load benchmark dataset."""
    if benchmark == "CB_groundtruth":
        df = pl.read_csv(DATA_DIR / "raw" / "claim_buster" / "groundtruth.csv")
        # Standardize columns
        df = df.select([
            pl.col("Sentence_id").cast(pl.Utf8).alias("sample_id"),
            pl.col("Text").alias("text"),
            pl.when(pl.col("Verdict") == 1).then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("label"),
        ])
    elif benchmark == "CT23":
        df = pl.read_csv(
            DATA_DIR / "raw" / "check_that_23" / "CT23_1B_checkworthy_english_test_gold.tsv",
            separator="\t"
        )
        df = df.select([
            pl.col("ClasSentence_id").cast(pl.Utf8).alias("sample_id"),
            pl.col("Text").alias("text"),
            pl.col("class_label").alias("label"),
        ])
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return df


# =============================================================================
# API and Feature Extraction
# =============================================================================

def load_prompts() -> dict[str, dict]:
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def build_messages(prompt_config: dict, claim: str) -> list[dict]:
    messages = [
        {"role": "system", "content": prompt_config["system"].strip()},
        {"role": "user", "content": prompt_config["user"].format(claim=claim).strip()},
    ]
    if "assistant" in prompt_config:
        messages.append({"role": "assistant", "content": prompt_config["assistant"]})
    return messages


def call_api(
    client: OpenAI,
    messages: list[dict],
    rate_limiter: RateLimiter,
    max_tokens: int = 1024,
) -> tuple[dict | None, list | None, str | None, float]:
    """Call API with rate limiting and retries."""
    start = time.time()

    for attempt in range(MAX_RETRIES):
        try:
            rate_limiter.acquire()

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
                timeout=REQUEST_TIMEOUT,
                response_format={"type": "json_object"},
            )

            latency = (time.time() - start) * 1000

            choice = response.choices[0]
            response_dict = {
                "content": choice.message.content,
                "finish_reason": choice.finish_reason,
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
            }

            logprobs_list = None
            if choice.logprobs and choice.logprobs.content:
                logprobs_list = [
                    {
                        "token": lp.token,
                        "logprob": lp.logprob,
                        "top_logprobs": [
                            {"token": tlp.token, "logprob": tlp.logprob}
                            for tlp in (lp.top_logprobs or [])
                        ]
                    }
                    for lp in choice.logprobs.content
                ]

            return response_dict, logprobs_list, None, latency

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None, None, str(e), (time.time() - start) * 1000

    return None, None, "Max retries exceeded", (time.time() - start) * 1000


def parse_json_response(content: str, module: str) -> tuple[dict, bool]:
    """Parse JSON from response content."""
    if not content:
        return {"confidence": 50, "reasoning": ""}, True

    had_issues = False
    result = {}

    # Try json_repair
    try:
        result = json.loads(repair_json(content))
    except:
        had_issues = True
        # Regex fallback for confidence
        match = re.search(r'"confidence"\s*:\s*(\d+)', content)
        if match:
            result["confidence"] = int(match.group(1))
        else:
            result["confidence"] = 50

        # Regex for reasoning
        match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
        result["reasoning"] = match.group(1) if match else ""

    # Ensure confidence exists
    if "confidence" not in result:
        had_issues = True
        result["confidence"] = 50

    return result, had_issues


def extract_logprob_features(logprobs: list | None) -> dict:
    """Extract probability features from logprobs."""
    defaults = {
        "p_true": None, "p_false": None, "p_uncertain": None,
        "logit_p_true": None, "logit_p_false": None, "logit_p_uncertain": None,
        "entropy": None, "entropy_norm": None,
        "margin_p": None, "margin_logit": None, "top1_top2_gap": None,
        "p_uncertain_dominant": None, "is_argmax_match": None,
    }

    if not logprobs:
        return defaults

    # Find first token with "true"/"false" variants
    for lp in logprobs:
        top_lps = lp.get("top_logprobs", [])
        if not top_lps:
            continue

        # Look for true/false/uncertain tokens
        probs = {}
        for tlp in top_lps:
            token = tlp["token"].lower().strip()
            logprob = tlp["logprob"]
            if "true" in token or "yes" in token:
                probs["true"] = max(probs.get("true", -100), logprob)
            elif "false" in token or "no" in token:
                probs["false"] = max(probs.get("false", -100), logprob)
            elif "uncertain" in token or "maybe" in token:
                probs["uncertain"] = max(probs.get("uncertain", -100), logprob)

        if len(probs) >= 2:
            # Convert logprobs to probs
            logprob_vals = list(probs.values())
            max_lp = max(logprob_vals)
            exp_vals = [math.exp(lp - max_lp) for lp in logprob_vals]
            total = sum(exp_vals)

            p_true = math.exp(probs.get("true", -100) - max_lp) / total if "true" in probs else 0.01
            p_false = math.exp(probs.get("false", -100) - max_lp) / total if "false" in probs else 0.01
            p_uncertain = math.exp(probs.get("uncertain", -100) - max_lp) / total if "uncertain" in probs else 0.01

            # Normalize
            total_p = p_true + p_false + p_uncertain
            p_true /= total_p
            p_false /= total_p
            p_uncertain /= total_p

            # Entropy
            probs_arr = np.array([p_true, p_false, p_uncertain])
            probs_arr = np.clip(probs_arr, 1e-10, 1)
            entropy = -np.sum(probs_arr * np.log2(probs_arr))
            entropy_norm = entropy / np.log2(3)

            # Margins
            sorted_p = sorted([p_true, p_false, p_uncertain], reverse=True)
            margin_p = sorted_p[0] - sorted_p[1]

            return {
                "p_true": p_true,
                "p_false": p_false,
                "p_uncertain": p_uncertain,
                "logit_p_true": probs.get("true"),
                "logit_p_false": probs.get("false"),
                "logit_p_uncertain": probs.get("uncertain"),
                "entropy": entropy,
                "entropy_norm": entropy_norm,
                "margin_p": margin_p,
                "margin_logit": None,
                "top1_top2_gap": margin_p,
                "p_uncertain_dominant": p_uncertain > max(p_true, p_false),
                "is_argmax_match": None,
            }

    return defaults


def process_sample(
    sample_id: str,
    text: str,
    client: OpenAI,
    prompts: dict,
    rate_limiter: RateLimiter,
    writer: ThreadSafeWriter,
) -> dict:
    """Process a single sample through all modules."""
    features = {"sentence_id": sample_id}
    module_results = {}

    for module in MODULES:
        prompt_config = prompts[module]
        messages = build_messages(prompt_config, text)

        response, logprobs, error, latency = call_api(client, messages, rate_limiter)

        if error:
            module_results[module] = {"error": error}
            continue

        parsed, had_issues = parse_json_response(response["content"], module)
        logprob_feats = extract_logprob_features(logprobs)

        # Store raw response
        writer.write({
            "sentence_id": sample_id,
            "module": module,
            "response": response,
            "logprobs": logprobs,
            "parsed": parsed,
        })

        module_results[module] = {
            "parsed": parsed,
            "logprob_feats": logprob_feats,
            "completion_tokens": response.get("completion_tokens", 0),
            "had_issues": had_issues,
        }

    # Build feature dict
    for module, prefix in [("checkability", "check"), ("verifiability", "verif"), ("harm_potential", "harm")]:
        result = module_results.get(module, {})
        if "error" in result:
            features[f"{prefix}_score"] = None
            features[f"{prefix}_prediction"] = None
            continue

        parsed = result.get("parsed", {})
        lp_feats = result.get("logprob_feats", {})

        confidence = parsed.get("confidence", 50)
        features[f"{prefix}_score"] = confidence
        features[f"{prefix}_prediction"] = confidence >= 50

        reasoning = parsed.get("reasoning", "")
        features[f"{prefix}_reasoning_length"] = len(reasoning)
        features[f"{prefix}_reasoning_hedged"] = bool(HEDGE_REGEX.search(reasoning))

        for key, val in lp_feats.items():
            features[f"{prefix}_{key}"] = val

        features[f"{prefix}_completion_tokens"] = result.get("completion_tokens")
        features[f"{prefix}_parse_issue"] = result.get("had_issues", False)
        features[f"{prefix}_pred_derived"] = False

        # Score-probability residual
        if lp_feats.get("p_true") is not None:
            features[f"{prefix}_score_p_residual"] = confidence - (lp_feats["p_true"] * 100)
            features[f"{prefix}_pred_score_mismatch"] = (confidence >= 50) != (lp_feats["p_true"] >= 0.5)
        else:
            features[f"{prefix}_score_p_residual"] = None
            features[f"{prefix}_pred_score_mismatch"] = None

    # Harm sub-scores
    harm_parsed = module_results.get("harm_potential", {}).get("parsed", {})
    for sub in ["social_fragmentation", "spurs_action", "believability", "exploitativeness"]:
        features[f"harm_{sub}"] = harm_parsed.get(sub)
    features["harm_subscore_missing"] = any(
        features.get(f"harm_{s}") is None
        for s in ["social_fragmentation", "spurs_action", "believability", "exploitativeness"]
    )

    # Cross-module features
    scores = [features.get("check_score"), features.get("verif_score"), features.get("harm_score")]
    valid_scores = [s for s in scores if s is not None]

    if len(valid_scores) >= 2:
        features["score_variance"] = np.var(valid_scores)
        features["score_max_diff"] = max(valid_scores) - min(valid_scores)
    else:
        features["score_variance"] = None
        features["score_max_diff"] = None

    cs, vs, hs = features.get("check_score"), features.get("verif_score"), features.get("harm_score")
    features["check_minus_verif"] = cs - vs if cs is not None and vs is not None else None
    features["check_minus_harm"] = cs - hs if cs is not None and hs is not None else None
    features["verif_minus_harm"] = vs - hs if vs is not None and hs is not None else None

    cp = features.get("check_prediction")
    vp = features.get("verif_prediction")
    hp = features.get("harm_prediction")

    if all(p is not None for p in [cp, vp, hp]):
        features["yes_vote_count"] = sum([cp, vp, hp])
        features["unanimous_yes"] = all([cp, vp, hp])
        features["unanimous_no"] = not any([cp, vp, hp])
        features["check_verif_agree"] = cp == vp
        features["check_harm_agree"] = cp == hp
        features["verif_harm_agree"] = vp == hp
        features["pairwise_agreement_rate"] = sum([cp == vp, cp == hp, vp == hp]) / 3
        features["check_yes_verif_yes"] = cp and vp
        features["harm_high_verif_low"] = (hs is not None and vs is not None and hs >= 60 and vs < 40)

        p_yes = sum([cp, vp, hp]) / 3
        p_yes = max(0.01, min(0.99, p_yes))
        features["consensus_entropy"] = -p_yes * np.log2(p_yes) - (1-p_yes) * np.log2(1-p_yes)
    else:
        for key in ["yes_vote_count", "unanimous_yes", "unanimous_no", "check_verif_agree",
                    "check_harm_agree", "verif_harm_agree", "pairwise_agreement_rate",
                    "check_yes_verif_yes", "harm_high_verif_low", "consensus_entropy"]:
            features[key] = None

    # Quality flags
    features["row_has_parse_issues"] = any(
        features.get(f"{p}_parse_issue", False) for p in ["check", "verif", "harm"]
    )
    features["row_has_uncertain_pred"] = any(
        features.get(f"{p}_p_uncertain_dominant", False) for p in ["check", "verif", "harm"]
    )

    return features


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate LLM features for benchmarks")
    parser.add_argument("--benchmark", choices=["CB_groundtruth", "CT23", "all"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="Process only 5 samples")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    benchmarks = ["CB_groundtruth", "CT23"] if args.benchmark == "all" else [args.benchmark]

    # API client
    client = OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url=TOGETHER_API_BASE,
    )

    # Load prompts
    prompts = load_prompts()

    # Rate limiter
    rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)

    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Processing: {benchmark}")
        print(f"{'='*60}")

        # Load data
        df = load_benchmark(benchmark)
        if args.dry_run:
            df = df.head(5)

        print(f"Samples: {len(df)}")

        # Output files
        output_file = OUTPUT_DIR / f"{benchmark}_llm_features.parquet"
        jsonl_file = OUTPUT_DIR / f"{benchmark}_raw_responses.jsonl"
        checkpoint_file = OUTPUT_DIR / f"{benchmark}_checkpoint.json"

        # Resume logic
        processed_ids = set()
        if args.resume and checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
                processed_ids = set(checkpoint.get("processed_ids", []))
            print(f"Resuming from checkpoint: {len(processed_ids)} already processed")

        # Filter to unprocessed
        df = df.filter(~pl.col("sample_id").is_in(list(processed_ids)))
        print(f"To process: {len(df)}")

        if len(df) == 0:
            print("All samples already processed!")
            continue

        # Process
        writer = ThreadSafeWriter(jsonl_file)
        all_features = []

        samples = list(zip(df["sample_id"].to_list(), df["text"].to_list()))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_sample, sid, text, client, prompts, rate_limiter, writer
                ): sid
                for sid, text in samples
            }

            pbar = tqdm(total=len(futures), desc=benchmark)

            for future in as_completed(futures):
                sid = futures[future]
                try:
                    features = future.result()
                    all_features.append(features)
                    processed_ids.add(sid)
                except Exception as e:
                    print(f"\nError processing {sid}: {e}")

                pbar.update(1)

                # Checkpoint
                if len(all_features) % CHECKPOINT_EVERY == 0:
                    with open(checkpoint_file, "w") as f:
                        json.dump({"processed_ids": list(processed_ids)}, f)

            pbar.close()

        # Save final results
        if all_features:
            result_df = pl.DataFrame(all_features, infer_schema_length=None)

            # If resuming, merge with existing
            if args.resume and output_file.exists():
                existing_df = pl.read_parquet(output_file)
                result_df = pl.concat([existing_df, result_df])

            result_df.write_parquet(output_file)
            print(f"\nSaved {len(result_df)} samples to {output_file}")

            # Final checkpoint
            with open(checkpoint_file, "w") as f:
                json.dump({"processed_ids": list(processed_ids), "complete": True}, f)

        print(f"Done with {benchmark}!")

    print("\n" + "="*60)
    print("All benchmarks complete!")
    print("="*60)


if __name__ == "__main__":
    main()
