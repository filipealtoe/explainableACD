#!/usr/bin/env python3
"""
LLM Checkworthiness Feature Extraction

Runs 3 prompts (checkability, verifiability, harm_potential) on CT24 dataset
using mistral-small-24b via Together AI. Extracts 79 features per sample.

Features:
- Saves raw responses to JSONL for later reprocessing
- Periodic checkpoints (configurable)
- Resume from last checkpoint
- Timeout handling for stuck requests
- Extracts all features to parquet

Usage:
    # Run inference on dev set
    python experiments/scripts/run_llm_checkworthiness.py --split dev

    # Run on all splits
    python experiments/scripts/run_llm_checkworthiness.py --split all

    # Resume interrupted run
    python experiments/scripts/run_llm_checkworthiness.py --split train --resume

    # Only extract features from existing responses
    python experiments/scripts/run_llm_checkworthiness.py --split dev --extract-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
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
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "CT24_llm_features"
PROMPTS_PATH = Path(__file__).parent.parent.parent / "prompts" / "checkworthiness_prompts_zeroshot_v3.yaml"

# Together AI config for mistral-small-24b
TOGETHER_API_BASE = "https://api.together.xyz/v1"
MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501"

# Checkpoint settings
CHECKPOINT_EVERY = 100  # Save every N samples
REQUEST_TIMEOUT = 60  # Seconds before timing out a request
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds between retries

# Parallelization settings
MAX_WORKERS = 60  # Concurrent threads (need many due to ~6s API latency)
REQUESTS_PER_MINUTE = 580  # Stay just under 600 limit (retry handles occasional 429s)
REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE  # ~0.10 seconds between requests

# Modules to run
MODULES = ["checkability", "verifiability", "harm_potential"]

# Hedge words for reasoning_hedged feature
HEDGE_PATTERNS = [
    r"\buncertain\b", r"\bunclear\b", r"\bambiguous\b",
    r"\bdifficult to\b", r"\bhard to\b", r"\bnot sure\b",
    r"\bdepends\b", r"\bcould be\b", r"\bmight be\b",
    r"\bnot enough\b", r"\bmissing\b", r"\bwithout context\b",
]
HEDGE_REGEX = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, requests_per_minute: float):
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.last_request = 0.0

    def acquire(self):
        """Wait until we can make the next request."""
        with self.lock:
            now = time.time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.time()


# Thread-safe file writer
class ThreadSafeWriter:
    """Thread-safe JSONL writer."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.lock = threading.Lock()

    def write(self, data: dict):
        """Append a JSON line to the file."""
        with self.lock:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(data) + "\n")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RawResponse:
    """Raw API response storage."""
    sentence_id: str
    module: str
    prompt_messages: list[dict]
    response_raw: dict | None
    response_json: dict | None
    logprobs: dict | None
    completion_tokens: int
    prompt_tokens: int
    finish_reason: str | None
    error: str | None
    timestamp: float
    latency_ms: float


@dataclass
class SampleFeatures:
    """All 79 features for one sample."""
    sentence_id: str

    # Checkability (20 features)
    check_score: float | None
    check_prediction: bool | None
    check_p_true: float | None
    check_p_false: float | None
    check_p_uncertain: float | None
    check_logit_p_true: float | None
    check_logit_p_false: float | None
    check_logit_p_uncertain: float | None
    check_entropy: float | None
    check_entropy_norm: float | None
    check_margin_p: float | None
    check_margin_logit: float | None
    check_top1_top2_gap: float | None
    check_p_uncertain_dominant: bool | None
    check_is_argmax_match: bool | None
    check_score_p_residual: float | None
    check_pred_score_mismatch: bool | None
    check_completion_tokens: int | None
    check_reasoning_length: int | None
    check_reasoning_hedged: bool | None

    # Verifiability (20 features)
    verif_score: float | None
    verif_prediction: bool | None
    verif_p_true: float | None
    verif_p_false: float | None
    verif_p_uncertain: float | None
    verif_logit_p_true: float | None
    verif_logit_p_false: float | None
    verif_logit_p_uncertain: float | None
    verif_entropy: float | None
    verif_entropy_norm: float | None
    verif_margin_p: float | None
    verif_margin_logit: float | None
    verif_top1_top2_gap: float | None
    verif_p_uncertain_dominant: bool | None
    verif_is_argmax_match: bool | None
    verif_score_p_residual: float | None
    verif_pred_score_mismatch: bool | None
    verif_completion_tokens: int | None
    verif_reasoning_length: int | None
    verif_reasoning_hedged: bool | None

    # Harm (20 features)
    harm_score: float | None
    harm_prediction: bool | None
    harm_p_true: float | None
    harm_p_false: float | None
    harm_p_uncertain: float | None
    harm_logit_p_true: float | None
    harm_logit_p_false: float | None
    harm_logit_p_uncertain: float | None
    harm_entropy: float | None
    harm_entropy_norm: float | None
    harm_margin_p: float | None
    harm_margin_logit: float | None
    harm_top1_top2_gap: float | None
    harm_p_uncertain_dominant: bool | None
    harm_is_argmax_match: bool | None
    harm_score_p_residual: float | None
    harm_pred_score_mismatch: bool | None
    harm_completion_tokens: int | None
    harm_reasoning_length: int | None
    harm_reasoning_hedged: bool | None

    # Harm sub-scores (4 features)
    harm_social_fragmentation: float | None
    harm_spurs_action: float | None
    harm_believability: float | None
    harm_exploitativeness: float | None

    # Cross-module derived (15 features)
    score_variance: float | None
    score_max_diff: float | None
    yes_vote_count: int | None
    unanimous_yes: bool | None
    unanimous_no: bool | None
    check_verif_agree: bool | None
    check_harm_agree: bool | None
    verif_harm_agree: bool | None
    pairwise_agreement_rate: float | None
    check_minus_verif: float | None
    check_minus_harm: float | None
    verif_minus_harm: float | None
    harm_high_verif_low: bool | None
    check_yes_verif_yes: bool | None
    consensus_entropy: float | None


# =============================================================================
# Prompt Loading
# =============================================================================

def load_prompts() -> dict[str, dict]:
    """Load prompts from YAML file."""
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def build_messages(prompt_config: dict, claim: str) -> list[dict]:
    """Build message list for API call."""
    messages = [
        {"role": "system", "content": prompt_config["system"].strip()},
        {"role": "user", "content": prompt_config["user"].format(claim=claim).strip()},
    ]
    # Add assistant prefill if present
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
    """
    Call Together AI API with timeout and retry logic.

    Returns: (response_dict, logprobs_dict, error_string, latency_ms)
    """
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
                response_format={"type": "json_object"},  # Force JSON output
            )

            latency = (time.time() - start) * 1000

            # Extract response data
            choice = response.choices[0]
            response_dict = {
                "content": choice.message.content,
                "finish_reason": choice.finish_reason,
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
            }

            # Extract logprobs if available
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
            error_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                print(f"      Retry {attempt + 1}/{MAX_RETRIES} after error: {error_msg[:50]}...")
                time.sleep(RETRY_DELAY)
            else:
                latency = (time.time() - start) * 1000
                return None, None, error_msg, latency

    return None, None, "Max retries exceeded", (time.time() - start) * 1000


def parse_json_response(content: str, module: str) -> tuple[dict, bool]:
    """Parse JSON from response content with multiple fallback strategies.

    Returns: (result_dict, had_issues)
        - result_dict: dict with at least 'confidence' key (defaults to 50 if not found)
        - had_issues: True if confidence was missing or we had to use regex fallbacks
    Never returns None - always returns a usable dict.
    """
    if not content:
        return {"confidence": 50, "reasoning": ""}, True

    had_issues = False

    result = {}

    # Strategy 1: Try json_repair library first (handles most malformed JSON)
    try:
        repaired = repair_json(content, return_objects=True)
        if isinstance(repaired, dict):
            result.update(repaired)
    except Exception:
        pass

    # Strategy 2: Try standard JSON parse on the content
    if "confidence" not in result:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                result.update(parsed)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract fields with regex (handles free-form text)
    # If we reach here without confidence, we're using fallbacks
    used_regex = False

    # Extract confidence
    if "confidence" not in result:
        conf_patterns = [
            r'"confidence"\s*:\s*(\d+(?:\.\d+)?)',
            r'confidence["\s:]+(\d+(?:\.\d+)?)',
        ]
        for pattern in conf_patterns:
            conf_match = re.search(pattern, content, re.I)
            if conf_match:
                result["confidence"] = float(conf_match.group(1))
                used_regex = True
                break

    # Extract reasoning
    if "reasoning" not in result:
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', content)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1)
        else:
            # Take text before confidence field as reasoning
            conf_pos = re.search(r'"?confidence"?\s*:', content, re.I)
            if conf_pos:
                reasoning_text = content[:conf_pos.start()].strip()
                reasoning_text = re.sub(r'^[{\s"]*reasoning["\s:]*', '', reasoning_text, flags=re.I)
                result["reasoning"] = reasoning_text.strip(' "{}')
            else:
                result["reasoning"] = content[:200]

    # Extract prediction field
    pred_field_map = {
        "checkability": "is_checkable",
        "verifiability": "is_verifiable",
        "harm_potential": "is_harmful",
    }
    pred_field = pred_field_map.get(module, "")

    if pred_field and pred_field not in result:
        pred_patterns = [
            rf'"{pred_field}"\s*:\s*"?(true|false|uncertain)"?',
            rf'{pred_field}["\s:]+["\s]*(true|false|uncertain)',
        ]
        for pattern in pred_patterns:
            pred_match = re.search(pattern, content, re.I)
            if pred_match:
                result[pred_field] = pred_match.group(1).lower()
                break

    # Extract harm sub-scores
    if module == "harm_potential":
        for sub in ["social_fragmentation", "spurs_action", "believability", "exploitativeness"]:
            if sub not in result:
                sub_patterns = [
                    rf'"{sub}"\s*:\s*\{{\s*"confidence"\s*:\s*(\d+)',
                    rf'"{sub}"[^}}]*"confidence"\s*:\s*(\d+)',
                    rf'{sub}["\s:]+(\d+)',
                ]
                for pattern in sub_patterns:
                    sub_match = re.search(pattern, content, re.I)
                    if sub_match:
                        result[sub] = {"confidence": float(sub_match.group(1))}
                        break

    # Default confidence to 50 if still missing
    if "confidence" not in result:
        result["confidence"] = 50
        had_issues = True

    # Mark as having issues if we used regex fallback
    if used_regex:
        had_issues = True

    return result, had_issues


def extract_label_logprobs(logprobs: list[dict] | None) -> tuple[float, float, float]:
    """
    Extract probabilities for true/false/uncertain from logprobs.

    Returns: (p_true, p_false, p_uncertain)
    """
    if not logprobs:
        return 0.33, 0.33, 0.34  # Default uniform

    # Look for the token that contains the prediction
    # It should be one of: true, false, uncertain (or variations)
    p_true = 0.0
    p_false = 0.0
    p_uncertain = 0.0

    for token_info in logprobs:
        token = token_info.get("token", "").lower().strip()

        # Check if this token is a label token
        if token in ["true", "false", "uncertain", '"true', '"false', '"uncertain']:
            # Get all top logprobs at this position
            top_lps = token_info.get("top_logprobs", [])

            for tlp in top_lps:
                t = tlp.get("token", "").lower().strip().replace('"', '')
                lp = tlp.get("logprob", -100)
                prob = math.exp(lp)

                if t == "true":
                    p_true = max(p_true, prob)
                elif t == "false":
                    p_false = max(p_false, prob)
                elif t == "uncertain":
                    p_uncertain = max(p_uncertain, prob)

            # Also include the actual token
            lp = token_info.get("logprob", -100)
            prob = math.exp(lp)
            t = token.replace('"', '')
            if t == "true":
                p_true = max(p_true, prob)
            elif t == "false":
                p_false = max(p_false, prob)
            elif t == "uncertain":
                p_uncertain = max(p_uncertain, prob)

    # Normalize
    total = p_true + p_false + p_uncertain
    if total > 0:
        p_true /= total
        p_false /= total
        p_uncertain /= total
    else:
        # Fallback to uniform
        p_true, p_false, p_uncertain = 0.33, 0.33, 0.34

    return p_true, p_false, p_uncertain


# =============================================================================
# Feature Extraction
# =============================================================================

def safe_logit(p: float, eps: float = 1e-6) -> float:
    """Compute logit with clipping to avoid inf."""
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def compute_entropy(probs: list[float]) -> float:
    """Compute entropy of probability distribution."""
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
    """Extract features for a single module. Never returns None values.

    Also tracks issue flags:
    - {prefix}_parse_issue: JSON parsing had problems
    - {prefix}_pred_derived: prediction was missing/uncertain, derived from score
    - {prefix}_subscore_missing: (harm only) one or more sub-scores were missing
    """
    prefix_map = {
        "checkability": "check",
        "verifiability": "verif",
        "harm_potential": "harm",
    }
    prefix = prefix_map[module]

    features = {}

    # Track issues for this module
    pred_derived = False
    subscore_missing = False

    # Ensure response_json is a dict
    if response_json is None:
        response_json = {"confidence": 50}
        parse_had_issues = True

    # Base features from JSON
    score = response_json.get("confidence", 50)
    features[f"{prefix}_score"] = float(score) if score is not None else 50.0

    # Prediction field name varies by module
    pred_field = {
        "checkability": "is_checkable",
        "verifiability": "is_verifiable",
        "harm_potential": "is_harmful",
    }[module]
    pred_val = response_json.get(pred_field, "")
    if isinstance(pred_val, bool):
        features[f"{prefix}_prediction"] = pred_val
    elif isinstance(pred_val, str):
        pred_val = pred_val.lower()
        if pred_val == "true":
            features[f"{prefix}_prediction"] = True
        elif pred_val == "false":
            features[f"{prefix}_prediction"] = False
        else:
            # "uncertain" or missing -> derive from score
            features[f"{prefix}_prediction"] = features[f"{prefix}_score"] > 50
            pred_derived = True
    else:
        # Derive from score
        features[f"{prefix}_prediction"] = features[f"{prefix}_score"] > 50
        pred_derived = True

    # Reasoning features
    reasoning = response_json.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning) if reasoning else ""
    features[f"{prefix}_reasoning_length"] = len(reasoning.split()) if reasoning else 0
    features[f"{prefix}_reasoning_hedged"] = bool(HEDGE_REGEX.search(reasoning)) if reasoning else False

    # Harm sub-scores - default to main harm_score if missing
    if module == "harm_potential":
        main_score = features[f"{prefix}_score"]
        for sub in ["social_fragmentation", "spurs_action", "believability", "exploitativeness"]:
            sub_data = response_json.get(sub, {})
            if isinstance(sub_data, dict) and sub_data.get("confidence") is not None:
                features[f"harm_{sub}"] = float(sub_data["confidence"])
            else:
                # Default to main harm score
                features[f"harm_{sub}"] = main_score
                subscore_missing = True

    # Logprob features
    p_true, p_false, p_uncertain = extract_label_logprobs(logprobs)

    features[f"{prefix}_p_true"] = p_true
    features[f"{prefix}_p_false"] = p_false
    features[f"{prefix}_p_uncertain"] = p_uncertain
    features[f"{prefix}_logit_p_true"] = safe_logit(p_true)
    features[f"{prefix}_logit_p_false"] = safe_logit(p_false)
    features[f"{prefix}_logit_p_uncertain"] = safe_logit(p_uncertain)

    # Entropy
    entropy = compute_entropy([p_true, p_false, p_uncertain])
    features[f"{prefix}_entropy"] = entropy
    features[f"{prefix}_entropy_norm"] = entropy / math.log(3)  # Normalize by max entropy

    # Margin features
    features[f"{prefix}_margin_p"] = p_true - p_false
    features[f"{prefix}_margin_logit"] = safe_logit(p_true) - safe_logit(p_false)

    probs_sorted = sorted([p_true, p_false, p_uncertain], reverse=True)
    features[f"{prefix}_top1_top2_gap"] = probs_sorted[0] - probs_sorted[1]
    features[f"{prefix}_p_uncertain_dominant"] = p_uncertain > max(p_true, p_false)

    # Consistency features (pred and score are guaranteed non-None)
    argmax_prob = max(p_true, p_false, p_uncertain)
    argmax_label = "true" if p_true == argmax_prob else ("false" if p_false == argmax_prob else "uncertain")
    pred = features[f"{prefix}_prediction"]  # Guaranteed boolean from above
    features[f"{prefix}_is_argmax_match"] = (pred and argmax_label == "true") or (not pred and argmax_label == "false")

    score = features[f"{prefix}_score"]  # Guaranteed float from above
    features[f"{prefix}_score_p_residual"] = score - (p_true * 100)
    features[f"{prefix}_pred_score_mismatch"] = (pred == True and score <= 50) or (pred == False and score > 50)

    # Metadata
    features[f"{prefix}_completion_tokens"] = completion_tokens

    # Issue flags for this module
    features[f"{prefix}_parse_issue"] = parse_had_issues
    features[f"{prefix}_pred_derived"] = pred_derived
    if module == "harm_potential":
        features[f"{prefix}_subscore_missing"] = subscore_missing

    return features


def compute_cross_module_features(features: dict) -> dict:
    """Compute cross-module derived features. Never returns None values."""
    cross = {}

    # Get scores with defaults
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
    cross["harm_high_verif_low"] = harm_score > 70 and verif_score < 30

    # Get predictions with defaults (derive from score if missing)
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

    agreements = sum([
        check_pred == verif_pred,
        check_pred == harm_pred,
        verif_pred == harm_pred,
    ])
    cross["pairwise_agreement_rate"] = agreements / 3
    cross["check_yes_verif_yes"] = check_pred == True and verif_pred == True

    # Consensus entropy
    yes_count = cross["yes_vote_count"]
    no_count = 3 - yes_count
    p_yes = yes_count / 3
    p_no = no_count / 3
    cross["consensus_entropy"] = compute_entropy([p_yes, p_no]) if yes_count not in [0, 3] else 0.0

    # Summary flag: True if ANY module had actual JSON parse failures
    has_parse_issues = any([
        features.get("check_parse_issue", False),
        features.get("verif_parse_issue", False),
        features.get("harm_parse_issue", False),
        features.get("harm_subscore_missing", False),
    ])
    cross["row_has_parse_issues"] = has_parse_issues

    # Summary flag: True if ANY module said "uncertain" (valid behavior, not an error)
    has_uncertain = any([
        features.get("check_pred_derived", False),
        features.get("verif_pred_derived", False),
        features.get("harm_pred_derived", False),
    ])
    cross["row_has_uncertain_pred"] = has_uncertain

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
    sample_idx: int,
    total_samples: int,
) -> dict:
    """Process a single sample through all 3 modules."""
    all_features = {"sentence_id": sentence_id}
    errors = []

    for module in MODULES:
        # Rate limit before each API call
        rate_limiter.acquire()

        prompt_config = prompts[module]
        messages = build_messages(prompt_config, claim)

        # Call API
        response_dict, logprobs_dict, error, latency = call_api(
            client, messages, prompt_config.get("max_tokens", 1024)
        )

        # Parse JSON response
        response_json = None
        parse_had_issues = True  # Default to True if no content
        if response_dict and response_dict.get("content"):
            response_json, parse_had_issues = parse_json_response(response_dict["content"], module)

        # Save raw response (thread-safe)
        raw_response = RawResponse(
            sentence_id=sentence_id,
            module=module,
            prompt_messages=messages,
            response_raw=response_dict,
            response_json=response_json,
            logprobs=logprobs_dict,
            completion_tokens=response_dict.get("completion_tokens", 0) if response_dict else 0,
            prompt_tokens=response_dict.get("prompt_tokens", 0) if response_dict else 0,
            finish_reason=response_dict.get("finish_reason") if response_dict else None,
            error=error,
            timestamp=time.time(),
            latency_ms=latency,
        )

        writer.write(asdict(raw_response))

        # Extract features
        module_features = extract_module_features(
            response_json,
            logprobs_dict,
            raw_response.completion_tokens,
            module,
            parse_had_issues,
        )
        all_features.update(module_features)

        if error:
            errors.append(f"{module}: {error[:30]}")

    # Compute cross-module features
    cross_features = compute_cross_module_features(all_features)
    all_features.update(cross_features)

    # Print progress (brief)
    check_score = all_features.get("check_score")
    if errors:
        print(f"  [{sample_idx}/{total_samples}] {sentence_id[:15]}... ERRORS: {', '.join(errors)}")
    else:
        print(f"  [{sample_idx}/{total_samples}] {sentence_id[:15]}... check={check_score}")

    return all_features


def get_processed_ids(responses_file: Path) -> set[str]:
    """Get set of sentence IDs that have been fully processed."""
    if not responses_file.exists():
        return set()

    # Count responses per sentence_id
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

    # Only return IDs with all 3 modules completed
    return {sid for sid, count in counts.items() if count >= 3}


def run_inference(
    split: str,
    resume: bool = False,
    checkpoint_every: int = 100,
) -> None:
    """Run inference on a split."""
    print(f"\n{'='*70}")
    print(f"Processing split: {split}")
    print(f"{'='*70}")

    # Setup paths
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    responses_file = OUTPUT_DIR / f"{split}_responses.jsonl"
    features_file = OUTPUT_DIR / f"{split}_llm_features.parquet"
    checkpoint_file = OUTPUT_DIR / f"{split}_checkpoint.json"

    # Load data
    data_file = DATA_DIR / f"CT24_{split}_features.parquet"
    if not data_file.exists():
        print(f"ERROR: {data_file} not found")
        return

    df = pl.read_parquet(data_file)
    print(f"Loaded {len(df)} samples")

    # Get already processed IDs if resuming
    processed_ids = set()
    if resume and responses_file.exists():
        processed_ids = get_processed_ids(responses_file)
        print(f"Resuming: {len(processed_ids)} samples already processed")
    elif not resume and responses_file.exists():
        # Clear old file
        responses_file.unlink()
        print("Cleared old responses file")

    # Load prompts
    prompts = load_prompts()

    # Setup client
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set")
        return

    client = OpenAI(api_key=api_key, base_url=TOGETHER_API_BASE)

    # Setup rate limiter and writer
    rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)
    writer = ThreadSafeWriter(responses_file)

    # Filter to samples that need processing
    samples_to_process = []
    for row in df.iter_rows(named=True):
        sentence_id = str(row["Sentence_id"])
        if sentence_id not in processed_ids:
            samples_to_process.append((sentence_id, row["cleaned_text"]))

    total = len(df)
    to_process = len(samples_to_process)
    skipped = total - to_process

    print(f"To process: {to_process}, Already done: {skipped}")
    print(f"Using {MAX_WORKERS} workers, {REQUESTS_PER_MINUTE} req/min limit")
    print(f"Estimated time: ~{(to_process * 3) / (REQUESTS_PER_MINUTE / 60):.1f} seconds")

    # Process samples in parallel
    all_features = []
    features_lock = threading.Lock()
    processed_count = [0]  # Use list for mutable counter in closure

    def process_wrapper(args):
        idx, (sentence_id, claim) = args
        try:
            features = process_sample(
                client, prompts, sentence_id, claim,
                writer, rate_limiter, idx + 1, to_process
            )
            return features
        except Exception as e:
            print(f"  [{idx + 1}/{to_process}] {sentence_id[:15]}... FATAL: {e}")
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
                    processed_count[0] += 1

                    # Checkpoint
                    if processed_count[0] % checkpoint_every == 0:
                        print(f"\n  === CHECKPOINT: {processed_count[0]} processed, saving... ===")
                        temp_df = pl.DataFrame(all_features)
                        temp_df.write_parquet(features_file)
                        with open(checkpoint_file, "w") as f:
                            json.dump({"processed": processed_count[0], "skipped": skipped}, f)

    elapsed = time.time() - start_time
    print(f"\nProcessed {processed_count[0]} samples in {elapsed:.1f}s ({processed_count[0] * 3 / elapsed:.1f} req/s)")

    # Final save
    print(f"\n{'='*70}")
    print(f"Completed: {processed_count[0]} processed, {skipped} skipped")

    if all_features:
        # If resuming, load old features and combine
        if resume and features_file.exists():
            old_df = pl.read_parquet(features_file)
            new_df = pl.DataFrame(all_features)
            combined = pl.concat([old_df, new_df])
            combined.write_parquet(features_file)
            print(f"Saved {len(combined)} total features to {features_file}")
        else:
            features_df = pl.DataFrame(all_features)
            features_df.write_parquet(features_file)
            print(f"Saved {len(features_df)} features to {features_file}")

    print(f"Raw responses: {responses_file}")
    print(f"{'='*70}")


def extract_features_only(split: str) -> None:
    """Extract features from existing responses file."""
    print(f"\n{'='*70}")
    print(f"Extracting features from: {split}")
    print(f"{'='*70}")

    responses_file = OUTPUT_DIR / f"{split}_responses.jsonl"
    features_file = OUTPUT_DIR / f"{split}_llm_features.parquet"

    if not responses_file.exists():
        print(f"ERROR: {responses_file} not found")
        return

    # Load all responses grouped by sentence_id
    responses_by_sample = {}
    with open(responses_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                sid = data.get("sentence_id")
                module = data.get("module")
                if sid and module:
                    if sid not in responses_by_sample:
                        responses_by_sample[sid] = {}
                    responses_by_sample[sid][module] = data
            except json.JSONDecodeError:
                continue

    print(f"Loaded responses for {len(responses_by_sample)} samples")

    # Extract features
    all_features = []
    for sentence_id, module_responses in responses_by_sample.items():
        features = {"sentence_id": sentence_id}

        for module in MODULES:
            if module not in module_responses:
                continue

            data = module_responses[module]

            # Re-parse from raw content using updated parser
            response_json = None
            parse_had_issues = True  # Default to True if no content
            raw_response = data.get("response_raw")
            if raw_response and raw_response.get("content"):
                response_json, parse_had_issues = parse_json_response(raw_response["content"], module)

            module_features = extract_module_features(
                response_json,
                data.get("logprobs"),
                data.get("completion_tokens", 0),
                module,
                parse_had_issues,
            )
            features.update(module_features)

        # Cross-module features
        cross_features = compute_cross_module_features(features)
        features.update(cross_features)

        all_features.append(features)

    # Save
    if all_features:
        features_df = pl.DataFrame(all_features)
        features_df.write_parquet(features_file)
        print(f"Saved {len(features_df)} features to {features_file}")

    print(f"{'='*70}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Checkworthiness Feature Extraction")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        required=True,
        help="Data split to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract features from existing responses",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help=f"Checkpoint frequency (default: {CHECKPOINT_EVERY})",
    )
    args = parser.parse_args()

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    for split in splits:
        if args.extract_only:
            extract_features_only(split)
        else:
            run_inference(split, resume=args.resume, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
