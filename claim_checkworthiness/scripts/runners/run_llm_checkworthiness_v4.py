#!/usr/bin/env python3
"""
LLM Checkworthiness Feature Extraction - V4 Binary Prompts

Same as run_llm_checkworthiness.py but uses v4 prompts (Yes/No only, no uncertain).
Outputs to separate directory to allow parallel comparison with v3.

Key differences from v3:
- Uses checkworthiness_prompts_zeroshot_v4.yaml
- Binary Yes/No output (no uncertain)
- Logprobs extract P(Yes) and P(No) only
- No p_uncertain features (2-class instead of 3-class)

Usage:
    python experiments/scripts/run_llm_checkworthiness_v4.py --split dev
    python experiments/scripts/run_llm_checkworthiness_v4.py --split all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import polars as pl
import yaml
from dotenv import load_dotenv
from json_repair import repair_json
from openai import OpenAI

# Load .env from repo root
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_features"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_llm_features_v4"
PROMPTS_PATH = Path(__file__).parent.parent.parent / "prompts" / "checkworthiness_prompts_zeroshot_v4.yaml"

# Together AI config for mistral-small-24b
TOGETHER_API_BASE = "https://api.together.xyz/v1"
MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"

# Settings
CHECKPOINT_EVERY = 100
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_WORKERS = 60
REQUESTS_PER_MINUTE = 580

MODULES = ["checkability", "verifiability", "harm_potential"]

HEDGE_PATTERNS = [
    r"\buncertain\b", r"\bunclear\b", r"\bambiguous\b",
    r"\bdifficult to\b", r"\bhard to\b", r"\bnot sure\b",
    r"\bdepends\b", r"\bcould be\b", r"\bmight be\b",
]
HEDGE_REGEX = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


# =============================================================================
# Rate Limiter & Writer
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


class RequestMetrics:
    """Track real-time request metrics."""
    def __init__(self, window_seconds: float = 60.0):
        self.window = window_seconds
        self.lock = threading.Lock()
        self.request_times: list[float] = []
        self.latencies: list[float] = []
        self.start_time = time.time()

    def record_request(self, latency_ms: float):
        now = time.time()
        with self.lock:
            self.request_times.append(now)
            self.latencies.append(latency_ms)
            # Prune old entries outside the window
            cutoff = now - self.window
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.pop(0)
                self.latencies.pop(0)

    def get_stats(self) -> dict:
        now = time.time()
        with self.lock:
            # Requests in last 60 seconds
            cutoff = now - self.window
            recent_count = sum(1 for t in self.request_times if t >= cutoff)
            recent_latencies = [l for t, l in zip(self.request_times, self.latencies) if t >= cutoff]

            rpm = recent_count * (60.0 / self.window)
            avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
            total_requests = len(self.request_times)
            elapsed = now - self.start_time

            return {
                "rpm": rpm,
                "avg_latency_ms": avg_latency,
                "total_in_window": recent_count,
                "elapsed_seconds": elapsed,
            }


class ThreadSafeWriter:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.lock = threading.Lock()

    def write(self, data: dict):
        with self.lock:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(data) + "\n")


# =============================================================================
# Prompts
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


# =============================================================================
# API Calls
# =============================================================================

def call_api(
    client: OpenAI,
    messages: list[dict],
    max_tokens: int = 1024,
) -> tuple[dict | None, dict | None, str | None, float]:
    start = time.time()

    for attempt in range(MAX_RETRIES):
        try:
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

            logprobs_dict = None
            if choice.logprobs and choice.logprobs.content:
                logprobs_dict = [
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

            return response_dict, logprobs_dict, None, latency

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None, None, str(e), (time.time() - start) * 1000

    return None, None, "Max retries exceeded", (time.time() - start) * 1000


def parse_json_response(content: str, module: str) -> tuple[dict, bool]:
    """Parse JSON response. Returns (result_dict, had_issues)."""
    if not content:
        return {"confidence": 50, "reasoning": ""}, True

    had_issues = False
    result = {}

    # Try json_repair first
    try:
        repaired = repair_json(content, return_objects=True)
        if isinstance(repaired, dict):
            result.update(repaired)
    except Exception:
        pass

    # Try standard JSON
    if "confidence" not in result:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                result.update(parsed)
        except json.JSONDecodeError:
            pass

    # Regex fallback for confidence
    if "confidence" not in result:
        conf_match = re.search(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)', content, re.I)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))
            had_issues = True

    # Extract prediction field (Yes/No for v4)
    pred_field_map = {
        "checkability": "is_checkable",
        "verifiability": "is_verifiable",
        "harm_potential": "is_harmful",
    }
    pred_field = pred_field_map.get(module, "")

    if pred_field and pred_field not in result:
        # Look for Yes/No (v4 format)
        pred_match = re.search(rf'"{pred_field}"\s*:\s*"?(Yes|No)"?', content, re.I)
        if pred_match:
            result[pred_field] = pred_match.group(1).capitalize()

    # Extract harm sub-scores
    if module == "harm_potential":
        for sub in ["social_fragmentation", "spurs_action", "believability", "exploitativeness"]:
            if sub not in result:
                sub_match = re.search(rf'"{sub}"[^}}]*"confidence"\s*:\s*(\d+)', content, re.I)
                if sub_match:
                    result[sub] = {"confidence": float(sub_match.group(1))}

    # Default confidence
    if "confidence" not in result:
        result["confidence"] = 50
        had_issues = True

    return result, had_issues


def extract_label_logprobs_binary(logprobs: list[dict] | None) -> tuple[float, float]:
    """
    Extract probabilities for Yes/No from logprobs (v4 binary format).

    Returns: (p_yes, p_no)
    """
    if not logprobs:
        return 0.5, 0.5

    p_yes = 0.0
    p_no = 0.0

    for token_info in logprobs:
        token = token_info.get("token", "").lower().strip()

        # Check if this is a Yes/No token
        if token in ["yes", "no", '"yes', '"no']:
            top_lps = token_info.get("top_logprobs", [])

            for tlp in top_lps:
                t = tlp.get("token", "").lower().strip().replace('"', '')
                lp = tlp.get("logprob", -100)
                prob = math.exp(lp)

                if t == "yes":
                    p_yes = max(p_yes, prob)
                elif t == "no":
                    p_no = max(p_no, prob)

            # Include actual token
            lp = token_info.get("logprob", -100)
            prob = math.exp(lp)
            t = token.replace('"', '')
            if t == "yes":
                p_yes = max(p_yes, prob)
            elif t == "no":
                p_no = max(p_no, prob)

    # Normalize
    total = p_yes + p_no
    if total > 0:
        p_yes /= total
        p_no /= total
    else:
        p_yes, p_no = 0.5, 0.5

    return p_yes, p_no


# =============================================================================
# Feature Extraction (Binary version)
# =============================================================================

def safe_logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def compute_entropy_binary(probs: list[float]) -> float:
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def extract_module_features(
    response_json: dict | None,
    logprobs: list[dict] | None,
    completion_tokens: int,
    module: str,
    parse_had_issues: bool = False,
) -> dict:
    """Extract features for a single module (v4 binary version)."""
    prefix_map = {
        "checkability": "check",
        "verifiability": "verif",
        "harm_potential": "harm",
    }
    prefix = prefix_map[module]

    features = {}

    if response_json is None:
        response_json = {"confidence": 50}
        parse_had_issues = True

    # Score
    score = response_json.get("confidence", 50)
    features[f"{prefix}_score"] = float(score) if score is not None else 50.0

    # Prediction (Yes/No for v4)
    pred_field = {
        "checkability": "is_checkable",
        "verifiability": "is_verifiable",
        "harm_potential": "is_harmful",
    }[module]

    pred_val = response_json.get(pred_field, "")
    if isinstance(pred_val, str):
        pred_val = pred_val.lower()
        if pred_val == "yes":
            features[f"{prefix}_prediction"] = True
        elif pred_val == "no":
            features[f"{prefix}_prediction"] = False
        else:
            # Derive from score
            features[f"{prefix}_prediction"] = features[f"{prefix}_score"] > 50
    else:
        features[f"{prefix}_prediction"] = features[f"{prefix}_score"] > 50

    # Reasoning
    reasoning = response_json.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning) if reasoning else ""
    features[f"{prefix}_reasoning_length"] = len(reasoning.split()) if reasoning else 0
    features[f"{prefix}_reasoning_hedged"] = bool(HEDGE_REGEX.search(reasoning)) if reasoning else False

    # Harm sub-scores
    if module == "harm_potential":
        main_score = features[f"{prefix}_score"]
        for sub in ["social_fragmentation", "spurs_action", "believability", "exploitativeness"]:
            sub_data = response_json.get(sub, {})
            if isinstance(sub_data, dict) and sub_data.get("confidence") is not None:
                features[f"harm_{sub}"] = float(sub_data["confidence"])
            else:
                features[f"harm_{sub}"] = main_score

    # Binary logprob features (v4: no p_uncertain)
    p_yes, p_no = extract_label_logprobs_binary(logprobs)

    features[f"{prefix}_p_yes"] = p_yes
    features[f"{prefix}_p_no"] = p_no
    features[f"{prefix}_logit_p_yes"] = safe_logit(p_yes)
    features[f"{prefix}_logit_p_no"] = safe_logit(p_no)

    # Binary entropy (2-class)
    entropy = compute_entropy_binary([p_yes, p_no])
    features[f"{prefix}_entropy"] = entropy
    features[f"{prefix}_entropy_norm"] = entropy / math.log(2)  # Normalize by max binary entropy

    # Margin
    features[f"{prefix}_margin_p"] = p_yes - p_no
    features[f"{prefix}_margin_logit"] = safe_logit(p_yes) - safe_logit(p_no)

    # Consistency
    features[f"{prefix}_score_p_residual"] = features[f"{prefix}_score"] - (p_yes * 100)

    # Metadata
    features[f"{prefix}_completion_tokens"] = completion_tokens
    features[f"{prefix}_parse_issue"] = parse_had_issues

    return features


def compute_cross_module_features(features: dict) -> dict:
    """Compute cross-module derived features."""
    cross = {}

    check_score = features.get("check_score", 50.0)
    verif_score = features.get("verif_score", 50.0)
    harm_score = features.get("harm_score", 50.0)

    scores = [check_score, verif_score, harm_score]
    mean_score = sum(scores) / 3
    cross["score_variance"] = sum((s - mean_score) ** 2 for s in scores) / 3
    cross["score_max_diff"] = max(scores) - min(scores)
    cross["check_minus_verif"] = check_score - verif_score
    cross["check_minus_harm"] = check_score - harm_score
    cross["verif_minus_harm"] = verif_score - harm_score

    check_pred = features.get("check_prediction", check_score > 50)
    verif_pred = features.get("verif_prediction", verif_score > 50)
    harm_pred = features.get("harm_prediction", harm_score > 50)

    preds = [check_pred, verif_pred, harm_pred]

    cross["yes_vote_count"] = sum(1 for p in preds if p == True)
    cross["unanimous_yes"] = all(p == True for p in preds)
    cross["unanimous_no"] = all(p == False for p in preds)
    cross["check_verif_agree"] = check_pred == verif_pred
    cross["check_harm_agree"] = check_pred == harm_pred
    cross["verif_harm_agree"] = verif_pred == harm_pred

    return cross


# =============================================================================
# Main Processing
# =============================================================================

def process_sample(
    client: OpenAI,
    prompts: dict,
    sentence_id: str,
    claim: str,
    writer: ThreadSafeWriter,
    rate_limiter: RateLimiter,
    metrics: RequestMetrics,
    sample_idx: int,
    total_samples: int,
) -> dict:
    """Process a single sample through all 3 modules."""
    all_features = {"sentence_id": sentence_id}

    for module in MODULES:
        rate_limiter.acquire()

        prompt_config = prompts[module]
        messages = build_messages(prompt_config, claim)

        response_dict, logprobs_dict, error, latency = call_api(
            client, messages, prompt_config.get("max_tokens", 1024)
        )

        # Record metrics
        metrics.record_request(latency)

        response_json = None
        parse_had_issues = True
        if response_dict and response_dict.get("content"):
            response_json, parse_had_issues = parse_json_response(response_dict["content"], module)

        # Save raw response
        raw_response = {
            "sentence_id": sentence_id,
            "module": module,
            "prompt_messages": messages,
            "response_raw": response_dict,
            "response_json": response_json,
            "logprobs": logprobs_dict,
            "error": error,
            "timestamp": time.time(),
            "latency_ms": latency,
        }
        writer.write(raw_response)

        # Extract features
        module_features = extract_module_features(
            response_json,
            logprobs_dict,
            response_dict.get("completion_tokens", 0) if response_dict else 0,
            module,
            parse_had_issues,
        )
        all_features.update(module_features)

    # Cross-module features
    cross_features = compute_cross_module_features(all_features)
    all_features.update(cross_features)

    # Print progress with metrics and ETA every 10 samples
    if sample_idx % 10 == 0 or sample_idx == total_samples:
        stats = metrics.get_stats()
        # Calculate ETA: remaining samples * 3 modules / requests per second
        remaining_samples = total_samples - sample_idx
        remaining_requests = remaining_samples * 3  # 3 modules per sample
        if stats['rpm'] > 0:
            remaining_minutes = remaining_requests / stats['rpm']
            if remaining_minutes >= 60:
                eta_str = f"{remaining_minutes/60:.1f}h"
            elif remaining_minutes >= 1:
                eta_str = f"{remaining_minutes:.1f}m"
            else:
                eta_str = f"{remaining_minutes*60:.0f}s"
        else:
            eta_str = "..."
        print(f"  [{sample_idx}/{total_samples}] RPM: {stats['rpm']:.0f} | Latency: {stats['avg_latency_ms']:.0f}ms | ETA: {eta_str}")
    else:
        print(f"  [{sample_idx}/{total_samples}] {sentence_id[:15]}... check={all_features.get('check_score'):.0f}")

    return all_features


def get_processed_ids(responses_file: Path) -> set[str]:
    if not responses_file.exists():
        return set()

    counts = {}
    with open(responses_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                sid = data.get("sentence_id")
                if sid:
                    counts[sid] = counts.get(sid, 0) + 1
            except json.JSONDecodeError:
                continue

    return {sid for sid, count in counts.items() if count >= 3}


def run_inference(split: str, resume: bool = False) -> None:
    print(f"\n{'='*70}")
    print(f"V4 Binary Prompts - Processing: {split}")
    print(f"{'='*70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    responses_file = OUTPUT_DIR / f"{split}_responses.jsonl"
    features_file = OUTPUT_DIR / f"{split}_llm_features.parquet"

    data_file = DATA_DIR / f"CT24_{split}_features.parquet"
    if not data_file.exists():
        print(f"ERROR: {data_file} not found")
        return

    df = pl.read_parquet(data_file)
    print(f"Loaded {len(df)} samples")

    processed_ids = set()
    if resume and responses_file.exists():
        processed_ids = get_processed_ids(responses_file)
        print(f"Resuming: {len(processed_ids)} already processed")
    elif not resume and responses_file.exists():
        responses_file.unlink()

    prompts = load_prompts()
    print(f"Using prompts: {PROMPTS_PATH.name}")

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set")
        return

    client = OpenAI(api_key=api_key, base_url=TOGETHER_API_BASE)
    rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)
    writer = ThreadSafeWriter(responses_file)
    metrics = RequestMetrics(window_seconds=60.0)

    samples_to_process = [
        (str(row["Sentence_id"]), row["cleaned_text"])
        for row in df.iter_rows(named=True)
        if str(row["Sentence_id"]) not in processed_ids
    ]

    print(f"To process: {len(samples_to_process)}")
    print(f"Using {MAX_WORKERS} workers")

    all_features = []
    features_lock = threading.Lock()

    def process_wrapper(args):
        idx, (sentence_id, claim) = args
        try:
            return process_sample(
                client, prompts, sentence_id, claim,
                writer, rate_limiter, metrics, idx + 1, len(samples_to_process)
            )
        except Exception as e:
            print(f"  [{idx + 1}] {sentence_id[:15]}... ERROR: {e}")
            return None

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_wrapper, (idx, sample)): idx
            for idx, sample in enumerate(samples_to_process)
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                with features_lock:
                    all_features.append(result)

    elapsed = time.time() - start_time
    final_stats = metrics.get_stats()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Samples processed: {len(all_features)}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Total API calls: {len(all_features) * 3} (3 modules × {len(all_features)} samples)")
    print(f"  Effective RPM: {(len(all_features) * 3) / (elapsed / 60):.0f}")
    print(f"  Final window RPM: {final_stats['rpm']:.0f}")
    print(f"  Avg latency: {final_stats['avg_latency_ms']:.0f}ms")
    print(f"  Rate limit: {REQUESTS_PER_MINUTE} RPM")

    if final_stats['rpm'] < REQUESTS_PER_MINUTE * 0.8:
        print(f"\n  ⚠️  Bottleneck: Server latency ({final_stats['avg_latency_ms']:.0f}ms avg)")
        print(f"      Max theoretical RPM with this latency: {60000 / final_stats['avg_latency_ms'] * MAX_WORKERS:.0f}")
    else:
        print(f"\n  ✓ Running near rate limit - rate limiter is the bottleneck")

    if all_features:
        features_df = pl.DataFrame(all_features)
        features_df.write_parquet(features_file)
        print(f"\nSaved to {features_file}")


def main():
    parser = argparse.ArgumentParser(description="V4 Binary LLM Feature Extraction")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    for split in splits:
        run_inference(split, resume=args.resume)


if __name__ == "__main__":
    main()
