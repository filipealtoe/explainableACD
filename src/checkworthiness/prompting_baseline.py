"""Prompting Baseline - Direct API calls with hand-crafted prompts.

This module provides two implementations:
1. PromptingBaseline: Direct OpenAI API calls with YAML prompts
2. BAMLPromptingBaseline: BAML-based structured outputs with Collector for logprobs

Both implementations support logprob-based confidence calculation.
"""

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import openai
import yaml
from openai import OpenAI

# Retry configuration for rate limits
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0  # Exponential backoff: 1s, 2s, 4s

# BAML imports
try:
    import baml_py
    from baml_py import Collector

    BAML_AVAILABLE = True
except ImportError:
    BAML_AVAILABLE = False
    Collector = None  # type: ignore

from .config import ModelConfig, TokenUsage
from .schemas import (
    CheckabilityOutput,
    CheckworthinessResult,
    HarmPotentialOutput,
    HarmSubScores,
    LogprobConfidence,
    LogprobData,
    VerifiabilityOutput,
)

LOGPROBS_DEBUG = os.getenv("CHECKWORTHINESS_LOGPROBS_DEBUG", "").lower() in {"1", "true", "yes", "on"}
JSON_DEBUG = os.getenv("CHECKWORTHINESS_JSON_DEBUG", "").lower() in {"1", "true", "yes", "on"}


@dataclass
class EnvironmentInfo:
    """Information about the execution environment for reproducibility."""

    sdk_name: str
    sdk_version: str
    provider: str
    model_name: str
    api_base: str | None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "sdk_name": self.sdk_name,
            "sdk_version": self.sdk_version,
            "provider": self.provider,
            "model_name": self.model_name,
            "api_base": self.api_base or "default",
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"SDK: {self.sdk_name} v{self.sdk_version} | "
            f"Provider: {self.provider} | "
            f"Model: {self.model_name} | "
            f"API Base: {self.api_base or 'default'}"
        )


def load_prompts(prompts_path: Path | None = None) -> dict:
    """Load prompts from YAML file."""
    if prompts_path is None:
        # Default to v3 prompts (improved conciseness, edge cases, stronger constraints)
        prompts_path = Path(__file__).parent.parent.parent / "prompts" / "checkworthiness_prompts_zeroshot_v3.yaml"

    with open(prompts_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_json_response(response: str, prefill: str = "") -> tuple[dict, bool]:
    """Parse JSON response, handling assistant prefill.

    Args:
        response: The model's response text
        prefill: The assistant prefill that was used (to reconstruct full JSON)

    Returns:
        Tuple of (parsed dict, parse_failed flag).
        If parse_failed is True, the dict contains fallback values and should be
        treated with caution (e.g., excluded from metrics or flagged in results).
    """
    # Strategy: Try multiple parsing approaches in order of likelihood
    # 1. Try response as-is (model may return complete JSON)
    # 2. Try with prefill prepended (model continued from prefill)
    # 3. Try extracting JSON object from response
    # 4. Try to extract individual fields with regex (partial recovery)

    # Attempt 1: Parse response directly (model returned complete JSON)
    try:
        return json.loads(response), False
    except json.JSONDecodeError:
        pass

    # Attempt 2: Prepend prefill and parse (model continued from prefill)
    if prefill:
        try:
            return json.loads(prefill + response), False
        except json.JSONDecodeError:
            pass

    # Attempt 3: Extract JSON object from response text
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group()), False
        except json.JSONDecodeError:
            pass

    # Attempt 4: Try to extract individual fields with regex (partial recovery)
    # This helps when JSON is malformed but contains the data we need
    full_text = prefill + response if prefill else response

    # Try to extract confidence value (handles: "confidence":75, "confidence": 75, "confidence":"75")
    confidence_match = re.search(r'"confidence"\s*:\s*"?(\d+(?:\.\d+)?)"?', full_text)
    extracted_confidence = float(confidence_match.group(1)) if confidence_match else None

    # Try to extract reasoning (everything between "reasoning":" and the next field or closing brace)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', full_text)
    extracted_reasoning = reasoning_match.group(1) if reasoning_match else response

    if JSON_DEBUG:
        if extracted_confidence is None:
            print(f"[parse_json] WARN: Failed to parse JSON AND confidence regex extraction failed")
            print(f"[parse_json] Response preview: {full_text[:300]}...")
        else:
            print(f"[parse_json] INFO: JSON parse failed but extracted confidence={extracted_confidence} via regex")

    # Fallback: return extracted values with parse_failed flag
    # NOTE: confidence=None signals we couldn't extract it at all
    return {
        "reasoning": extracted_reasoning,
        "confidence": extracted_confidence,
        "_parse_failed": True,
    }, True


# Ternary token variants for true/false/uncertain classification
TRUE_TOKEN_VARIANTS = (" True", "True", "TRUE", " true", "true", '"true"', '"True"')
FALSE_TOKEN_VARIANTS = (" False", "False", "FALSE", " false", "false", '"false"', '"False"')
UNCERTAIN_TOKEN_VARIANTS = (" Uncertain", "Uncertain", "UNCERTAIN", " uncertain", "uncertain", '"uncertain"', '"Uncertain"')


def _get_attr(obj: object, name: str, default: object | None = None) -> object | None:
    """Fetch an attribute from dict-like or object-like responses."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_top_logprobs(top_logprobs: object | None) -> list[dict]:
    """Normalize top_logprobs entries into a list of dicts."""
    normalized: list[dict] = []
    if not top_logprobs:
        return normalized
    for item in top_logprobs:
        token = _get_attr(item, "token", "")
        logprob = _get_attr(item, "logprob", None)
        normalized.append({"token": token, "logprob": logprob})
    return normalized


def _extract_completion_logprobs(response: object) -> list[dict]:
    """Extract token-level logprobs from a chat completion response.

    Handles both OpenAI format and Together AI format:
    - OpenAI: logprobs.content = [{"token": ..., "logprob": ..., "top_logprobs": [...]}]
    - Together AI: logprobs.tokens, logprobs.token_logprobs, logprobs.top_logprobs (lists)
    """
    try:
        choice = _get_attr(response, "choices", [None])[0]
    except Exception:
        return []
    logprobs = _get_attr(choice, "logprobs", None)
    if not logprobs:
        return []

    # Try OpenAI format first (logprobs.content)
    content = _get_attr(logprobs, "content", None)
    if content:
        token_logprobs: list[dict] = []
        for item in content:
            token_logprobs.append(
                {
                    "token": _get_attr(item, "token", ""),
                    "logprob": _get_attr(item, "logprob", None),
                    "top_logprobs": _normalize_top_logprobs(_get_attr(item, "top_logprobs", None)),
                }
            )
        return token_logprobs

    # Try Together AI format (logprobs.tokens, logprobs.token_logprobs, logprobs.top_logprobs)
    tokens = _get_attr(logprobs, "tokens", None)
    token_probs = _get_attr(logprobs, "token_logprobs", None)
    top_probs_list = _get_attr(logprobs, "top_logprobs", None)

    if tokens and token_probs:
        token_logprobs = []
        for i, token in enumerate(tokens):
            # Together AI's top_logprobs is a list of dicts like [{"True": -0.006}, {".": -0.28}]
            # Convert to our normalized format: [{"token": "True", "logprob": -0.006}, ...]
            top_probs_raw = top_probs_list[i] if top_probs_list and i < len(top_probs_list) else {}
            top_probs_normalized = []
            if isinstance(top_probs_raw, dict):
                for tok, lp in top_probs_raw.items():
                    top_probs_normalized.append({"token": tok, "logprob": lp})

            token_logprobs.append(
                {
                    "token": token,
                    "logprob": token_probs[i] if i < len(token_probs) else None,
                    "top_logprobs": top_probs_normalized,
                }
            )
        return token_logprobs

    return []


def _find_ternary_match(text: str, field_name: str) -> tuple[str, int] | None:
    """Locate a ternary value (true/false/uncertain) for a field in the completion text."""
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"?((true|false|uncertain))"?'
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), match.start(1)


def _find_token_index_for_offset(tokens: list[str], offset: int) -> int | None:
    """Find which token span contains the given character offset."""
    position = 0
    for index, token in enumerate(tokens):
        next_position = position + len(token)
        if position <= offset < next_position:
            return index
        position = next_position
    return None


def _sum_variant_probs(
    top_logprobs: list[dict],
    variants: tuple[str, ...],
    normalized_target: str,
) -> tuple[float, int]:
    """Sum probabilities for variant tokens and count missing variants."""
    total = 0.0
    missing = 0
    matched_tokens: set[str] = set()
    for variant in variants:
        prob = 0.0
        for item in top_logprobs:
            if item.get("token") == variant:
                logprob = item.get("logprob")
                if logprob is not None:
                    prob = math.exp(logprob)
                break
        if prob == 0.0:
            missing += 1
        else:
            matched_tokens.add(variant)
        total += prob

    for item in top_logprobs:
        token = item.get("token")
        if not token or token in matched_tokens:
            continue
        normalized = token.lstrip().lower()
        if normalized == normalized_target:
            logprob = item.get("logprob")
            if logprob is not None:
                total += math.exp(logprob)

    return total, missing


def _compute_ternary_logprob_stats(
    completion_text: str,
    token_logprobs: list[dict],
    field_name: str,
) -> dict | None:
    """Compute ternary logprob stats (true/false/uncertain) for a specific field."""
    if not token_logprobs:
        return {"field_name": field_name, "error": "logprobs_missing"}

    match = _find_ternary_match(completion_text, field_name)
    if not match:
        return {"field_name": field_name, "error": "field_not_found"}

    emitted_value, value_offset = match
    token_texts = [item.get("token", "") for item in token_logprobs]
    token_index = _find_token_index_for_offset(token_texts, value_offset)
    if token_index is None:
        return {
            "field_name": field_name,
            "emitted_value": emitted_value,
            "token_index": None,
            "error": "token_not_found",
        }

    top_logprobs = token_logprobs[token_index].get("top_logprobs", [])
    top_tokens: list[tuple[str, float]] = []
    for item in top_logprobs:
        token = item.get("token")
        logprob = item.get("logprob")
        if token is None or logprob is None:
            continue
        top_tokens.append((token, math.exp(logprob)))

    sum_true, missing_true = _sum_variant_probs(top_logprobs, TRUE_TOKEN_VARIANTS, "true")
    sum_false, missing_false = _sum_variant_probs(top_logprobs, FALSE_TOKEN_VARIANTS, "false")
    sum_uncertain, missing_uncertain = _sum_variant_probs(top_logprobs, UNCERTAIN_TOKEN_VARIANTS, "uncertain")

    denom = sum_true + sum_false + sum_uncertain
    p_true = sum_true / denom if denom > 0 else 0.33
    p_false = sum_false / denom if denom > 0 else 0.33
    p_uncertain = sum_uncertain / denom if denom > 0 else 0.33

    # Compute Shannon entropy of the ternary distribution
    # H = -sum(p * log2(p)) for p > 0
    # Range: [0, log2(3)] ≈ [0, 1.585] where 0 = certain, 1.585 = uniform
    entropy = 0.0
    for p in [p_true, p_false, p_uncertain]:
        if p > 1e-10:
            entropy -= p * math.log2(p)

    return {
        "field_name": field_name,
        "emitted_value": emitted_value,
        "token_index": token_index,
        "top_tokens": top_tokens,
        "missing_true": missing_true,
        "missing_false": missing_false,
        "missing_uncertain": missing_uncertain,
        "p_true": p_true,
        "p_false": p_false,
        "p_uncertain": p_uncertain,
        "entropy": entropy,
    }


def _print_ternary_logprob_stats(
    field_name: str,
    completion_text: str,
    token_logprobs: list[dict],
    self_confidence: float,
) -> dict | None:
    """Print logprob diagnostics for a ternary field (true/false/uncertain)."""
    stats = _compute_ternary_logprob_stats(completion_text, token_logprobs, field_name)
    if stats is None:
        if LOGPROBS_DEBUG:
            print(f"[logprobs] {field_name}: logprobs unavailable")
        return None
    error = stats.get("error")
    if error == "logprobs_missing":
        if LOGPROBS_DEBUG:
            print(f"[logprobs] {field_name}: logprobs unavailable")
        return stats
    if error == "field_not_found":
        if LOGPROBS_DEBUG:
            print(f"[logprobs] {field_name}: ternary field not found")
        return stats
    if error == "token_not_found":
        if LOGPROBS_DEBUG:
            print(f"[logprobs] {field_name}: ternary token not found")
        return stats

    true_pct = stats["p_true"] * 100.0
    false_pct = stats["p_false"] * 100.0
    uncertain_pct = stats["p_uncertain"] * 100.0
    missing_true = stats["missing_true"]
    missing_false = stats["missing_false"]
    missing_uncertain = stats["missing_uncertain"]

    if LOGPROBS_DEBUG:
        print(
            f"[logprobs] {field_name}: self_conf={self_confidence:.1f} "
            f"p_true={true_pct:.1f} p_uncertain={uncertain_pct:.1f} emitted={stats['emitted_value']}"
        )
        print(
            f"[logprobs] {field_name}: p_true={true_pct:.1f} p_false={false_pct:.1f} p_uncertain={uncertain_pct:.1f} "
            f"missing_true={missing_true}/{len(TRUE_TOKEN_VARIANTS)} "
            f"missing_false={missing_false}/{len(FALSE_TOKEN_VARIANTS)} "
            f"missing_uncertain={missing_uncertain}/{len(UNCERTAIN_TOKEN_VARIANTS)}"
        )
        for token, prob in stats["top_tokens"]:
            print(f"[logprobs] {field_name}: token={repr(token)} prob={prob:.6f}")

    return stats


class PromptingBaseline:
    """Direct API calls with hand-crafted prompts (no DSPy)."""

    def __init__(
        self,
        model_config: ModelConfig,
        threshold: float = 50.0,
        temperature: float = 0.0,
        prompts_path: str | None = None,
    ):
        self.model_config = model_config
        self.threshold = threshold
        self.temperature = temperature
        self.prompts = load_prompts(Path(prompts_path) if prompts_path else None)

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=model_config.get_api_key(),
            base_url=model_config.api_base,
        )

        # Track token usage
        self.last_usage = TokenUsage()

        # Store environment info for reproducibility reporting
        self.environment_info = EnvironmentInfo(
            sdk_name="openai",
            sdk_version=openai.__version__,
            provider=model_config.provider.value,
            model_name=model_config.model_name,
            api_base=model_config.api_base,
        )

    def get_environment_info(self) -> EnvironmentInfo:
        """Get execution environment information for reproducibility."""
        return self.environment_info

    def print_environment_info(self) -> None:
        """Print environment info to stdout."""
        print("\n" + "=" * 70)
        print("EXECUTION ENVIRONMENT")
        print("=" * 70)
        print(f"  SDK: openai v{openai.__version__}")
        print(f"  Provider: {self.model_config.provider.value}")
        print(f"  Model: {self.model_config.model_name}")
        print(f"  API Base: {self.model_config.api_base or 'default (api.openai.com)'}")
        print(f"  Max Tokens: {self.model_config.max_tokens}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Threshold: {self.threshold}")
        print("=" * 70 + "\n")

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prefill: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, TokenUsage, list[dict], str | None, list[dict] | None]:
        """Make a single API call and return response + token usage + logprobs + reasoning.

        Args:
            system_prompt: The system message
            user_prompt: The user message
            assistant_prefill: Optional assistant message to prefill response format

        Returns:
            Tuple of (response content, token usage, token logprobs,
                      reasoning_content (for reasoning models), reasoning_logprobs)
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Add assistant prefill if provided (for JSON format guidance)
        # Skip for:
        # - thinking/reasoning models (they need to generate their own chain-of-thought first)
        # - models that don't support prefill (e.g., Mistral instruction models)
        prefill_supported = (
            not self.model_config.is_thinking_model
            and getattr(self.model_config, "supports_prefill", True)
        )
        if assistant_prefill and prefill_supported:
            messages.append({"role": "assistant", "content": assistant_prefill})

        call_max_tokens = max_tokens or self.model_config.max_tokens
        api_params = {
            "model": self.model_config.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        # Use correct parameter name based on model (GPT-5+ uses max_completion_tokens)
        if getattr(self.model_config, "uses_max_completion_tokens", False):
            api_params["max_completion_tokens"] = call_max_tokens
        else:
            api_params["max_tokens"] = call_max_tokens
        if self.model_config.supports_logprobs and self.model_config.logprobs:
            api_params["logprobs"] = True
            api_params["top_logprobs"] = self.model_config.top_logprobs

        # Retry with exponential backoff for rate limits and transient errors
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**api_params)
                break  # Success - exit retry loop
            except openai.RateLimitError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    backoff = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
                    time.sleep(backoff)
                    continue
                raise  # Final attempt failed
            except openai.APITimeoutError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    backoff = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
                    time.sleep(backoff)
                    continue
                raise
            except openai.APIConnectionError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    backoff = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
                    time.sleep(backoff)
                    continue
                raise
        else:
            # All retries exhausted
            raise last_error or RuntimeError("All API retries exhausted")

        message = response.choices[0].message
        content = message.content or ""

        # Extract reasoning tokens from API response
        # Different providers expose this differently:
        # - DeepSeek/OpenAI o1: response.usage.completion_tokens_details.reasoning_tokens
        # - Some APIs: response.usage.reasoning_tokens (direct attribute)
        reasoning_tokens = 0
        if response.usage:
            # Try completion_tokens_details first (DeepSeek Reasoner, OpenAI o1/o3)
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                details = response.usage.completion_tokens_details
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
            # Fallback: direct attribute
            elif hasattr(response.usage, "reasoning_tokens"):
                reasoning_tokens = response.usage.reasoning_tokens or 0

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            reasoning_tokens=reasoning_tokens,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        token_logprobs = _extract_completion_logprobs(response)

        # Extract reasoning content for reasoning models (e.g., DeepSeek Reasoner)
        # This is the internal chain-of-thought, separate from the final answer
        reasoning_content: str | None = None
        reasoning_logprobs: list[dict] | None = None

        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning_content = message.reasoning_content
            # Extract logprobs for reasoning content if available
            choice = response.choices[0]
            if choice.logprobs and hasattr(choice.logprobs, "reasoning_content"):
                reasoning_logprobs_raw = choice.logprobs.reasoning_content
                if reasoning_logprobs_raw:
                    reasoning_logprobs = []
                    for item in reasoning_logprobs_raw:
                        reasoning_logprobs.append({
                            "token": _get_attr(item, "token", ""),
                            "logprob": _get_attr(item, "logprob", None),
                            "top_logprobs": _normalize_top_logprobs(_get_attr(item, "top_logprobs", None)),
                        })

        return content, usage, token_logprobs, reasoning_content, reasoning_logprobs

    def assess_checkability(
        self, claim: str
    ) -> tuple[CheckabilityOutput, TokenUsage, list[dict], str | None, list[dict] | None]:
        """Assess checkability of a claim.

        Returns:
            Tuple of (output, usage, token_logprobs, reasoning_content, reasoning_logprobs)
        """
        system_prompt = self.prompts["checkability"]["system"]
        user_prompt = self.prompts["checkability"]["user"].format(claim=claim)
        assistant_prefill = self.prompts["checkability"].get("assistant")
        max_tokens = int(self.prompts["checkability"].get("max_tokens", self.model_config.max_tokens))

        response, usage, token_logprobs, reasoning_content, reasoning_logprobs = self._call_api(
            system_prompt,
            user_prompt,
            assistant_prefill,
            max_tokens=max_tokens,
        )

        # Parse JSON response (only prepend prefill if it was actually sent)
        prefill_supported = (
            not self.model_config.is_thinking_model
            and getattr(self.model_config, "supports_prefill", True)
        )
        prefill_used = assistant_prefill if (assistant_prefill and prefill_supported) else ""
        parsed, json_parse_failed = parse_json_response(response, prefill_used)

        # Get self-reported confidence (None if parse failed)
        raw_confidence = parsed.get("confidence")
        self_confidence = float(raw_confidence) if raw_confidence is not None else None
        reasoning = str(parsed.get("reasoning", response))

        stats = _print_ternary_logprob_stats(
            "is_checkable",
            response,
            token_logprobs,
            self_confidence=self_confidence if self_confidence is not None else 50.0,
        )

        # Determine final confidence with proper fallback chain:
        # 1. Use logprobs-derived confidence if available (most reliable)
        # 2. Fall back to model's self-reported confidence from JSON
        # 3. Only use 50.0 if both are missing (and log this as an issue)
        logprobs_missing = not (stats and "p_true" in stats)
        logprob_confidence: float | None = None
        if not logprobs_missing:
            logprob_confidence = stats["p_true"] * 100.0
            true_confidence = logprob_confidence
        elif self_confidence is not None:
            true_confidence = self_confidence
            if LOGPROBS_DEBUG:
                print("[logprobs] is_checkable: using self-reported confidence (logprobs missing)")
        else:
            true_confidence = 50.0
            if LOGPROBS_DEBUG:
                print("[logprobs] is_checkable: ERROR - both logprobs and JSON confidence missing, using 50.0")

        # Extract ternary probabilities and entropy from stats
        p_true = stats.get("p_true") if stats else None
        p_false = stats.get("p_false") if stats else None
        p_uncertain = stats.get("p_uncertain") if stats else None
        entropy = stats.get("entropy") if stats else None

        return (
            CheckabilityOutput(
                reasoning=reasoning,
                confidence=true_confidence,
                self_confidence=self_confidence,
                logprob_confidence=logprob_confidence,
                p_true=p_true,
                p_false=p_false,
                p_uncertain=p_uncertain,
                entropy=entropy,
                json_parse_failed=json_parse_failed,
                logprobs_missing=logprobs_missing,
            ),
            usage,
            token_logprobs,
            reasoning_content,
            reasoning_logprobs,
        )

    def assess_verifiability(
        self, claim: str
    ) -> tuple[VerifiabilityOutput, TokenUsage, list[dict], str | None, list[dict] | None]:
        """Assess verifiability of a claim.

        Returns:
            Tuple of (output, usage, token_logprobs, reasoning_content, reasoning_logprobs)
        """
        system_prompt = self.prompts["verifiability"]["system"]
        user_prompt = self.prompts["verifiability"]["user"].format(claim=claim)
        assistant_prefill = self.prompts["verifiability"].get("assistant")
        max_tokens = int(self.prompts["verifiability"].get("max_tokens", self.model_config.max_tokens))

        response, usage, token_logprobs, reasoning_content, reasoning_logprobs = self._call_api(
            system_prompt,
            user_prompt,
            assistant_prefill,
            max_tokens=max_tokens,
        )

        # Parse JSON response (only prepend prefill if it was actually sent)
        prefill_supported = (
            not self.model_config.is_thinking_model
            and getattr(self.model_config, "supports_prefill", True)
        )
        prefill_used = assistant_prefill if (assistant_prefill and prefill_supported) else ""
        parsed, json_parse_failed = parse_json_response(response, prefill_used)

        # Get self-reported confidence (None if parse failed)
        raw_confidence = parsed.get("confidence")
        self_confidence = float(raw_confidence) if raw_confidence is not None else None
        reasoning = str(parsed.get("reasoning", response))

        stats = _print_ternary_logprob_stats(
            "is_verifiable",
            response,
            token_logprobs,
            self_confidence=self_confidence if self_confidence is not None else 50.0,
        )

        # Determine final confidence with proper fallback chain
        logprobs_missing = not (stats and "p_true" in stats)
        logprob_confidence: float | None = None
        if not logprobs_missing:
            logprob_confidence = stats["p_true"] * 100.0
            true_confidence = logprob_confidence
        elif self_confidence is not None:
            true_confidence = self_confidence
            if LOGPROBS_DEBUG:
                print("[logprobs] is_verifiable: using self-reported confidence (logprobs missing)")
        else:
            true_confidence = 50.0
            if LOGPROBS_DEBUG:
                print("[logprobs] is_verifiable: ERROR - both logprobs and JSON confidence missing, using 50.0")

        # Extract ternary probabilities and entropy from stats
        p_true = stats.get("p_true") if stats else None
        p_false = stats.get("p_false") if stats else None
        p_uncertain = stats.get("p_uncertain") if stats else None
        entropy = stats.get("entropy") if stats else None

        return (
            VerifiabilityOutput(
                reasoning=reasoning,
                confidence=true_confidence,
                self_confidence=self_confidence,
                logprob_confidence=logprob_confidence,
                p_true=p_true,
                p_false=p_false,
                p_uncertain=p_uncertain,
                entropy=entropy,
                json_parse_failed=json_parse_failed,
                logprobs_missing=logprobs_missing,
            ),
            usage,
            token_logprobs,
            reasoning_content,
            reasoning_logprobs,
        )

    def assess_harm_potential(
        self, claim: str
    ) -> tuple[HarmPotentialOutput, TokenUsage, list[dict], str | None, list[dict] | None]:
        """Assess harm potential of a claim.

        Returns:
            Tuple of (output, usage, token_logprobs, reasoning_content, reasoning_logprobs)
        """
        system_prompt = self.prompts["harm_potential"]["system"]
        user_prompt = self.prompts["harm_potential"]["user"].format(claim=claim)
        assistant_prefill = self.prompts["harm_potential"].get("assistant")
        max_tokens = int(self.prompts["harm_potential"].get("max_tokens", self.model_config.max_tokens))

        response, usage, token_logprobs, reasoning_content, reasoning_logprobs = self._call_api(
            system_prompt,
            user_prompt,
            assistant_prefill,
            max_tokens=max_tokens,
        )

        # Parse JSON response (only prepend prefill if it was actually sent)
        prefill_supported = (
            not self.model_config.is_thinking_model
            and getattr(self.model_config, "supports_prefill", True)
        )
        prefill_used = assistant_prefill if (assistant_prefill and prefill_supported) else ""
        parsed, json_parse_failed = parse_json_response(response, prefill_used)

        # Get self-reported confidence (None if parse failed)
        raw_confidence = parsed.get("confidence")
        self_confidence = float(raw_confidence) if raw_confidence is not None else None
        overall_just = str(parsed.get("reasoning", response))

        # Extract sub-scores from nested JSON
        sf = parsed.get("social_fragmentation", {})
        sa = parsed.get("spurs_action", {})
        bv = parsed.get("believability", {})
        ex = parsed.get("exploitativeness", {})

        stats = _print_ternary_logprob_stats(
            "is_harmful",
            response,
            token_logprobs,
            self_confidence=self_confidence if self_confidence is not None else 50.0,
        )
        if isinstance(sf, dict):
            _print_ternary_logprob_stats(
                "is_social_fragmentation",
                response,
                token_logprobs,
                self_confidence=float(sf.get("confidence", 50.0)),
            )
        if isinstance(sa, dict):
            _print_ternary_logprob_stats(
                "is_spurs_action",
                response,
                token_logprobs,
                self_confidence=float(sa.get("confidence", 50.0)),
            )
        if isinstance(bv, dict):
            _print_ternary_logprob_stats(
                "is_believability",
                response,
                token_logprobs,
                self_confidence=float(bv.get("confidence", 50.0)),
            )
        if isinstance(ex, dict):
            _print_ternary_logprob_stats(
                "is_exploitativeness",
                response,
                token_logprobs,
                self_confidence=float(ex.get("confidence", 50.0)),
            )

        # Determine final confidence with proper fallback chain
        logprobs_missing = not (stats and "p_true" in stats)
        logprob_confidence: float | None = None
        if not logprobs_missing:
            logprob_confidence = stats["p_true"] * 100.0
            true_confidence = logprob_confidence
        elif self_confidence is not None:
            true_confidence = self_confidence
            if LOGPROBS_DEBUG:
                print("[logprobs] is_harmful: using self-reported confidence (logprobs missing)")
        else:
            true_confidence = 50.0
            if LOGPROBS_DEBUG:
                print("[logprobs] is_harmful: ERROR - both logprobs and JSON confidence missing, using 50.0")

        sub_scores_obj = HarmSubScores(
            social_fragmentation=float(sf.get("confidence", 50.0)) if isinstance(sf, dict) else 50.0,
            spurs_action=float(sa.get("confidence", 50.0)) if isinstance(sa, dict) else 50.0,
            believability=float(bv.get("confidence", 50.0)) if isinstance(bv, dict) else 50.0,
            exploitativeness=float(ex.get("confidence", 50.0)) if isinstance(ex, dict) else 50.0,
        )

        # Extract ternary probabilities and entropy from stats
        p_true = stats.get("p_true") if stats else None
        p_false = stats.get("p_false") if stats else None
        p_uncertain = stats.get("p_uncertain") if stats else None
        entropy = stats.get("entropy") if stats else None

        return (
            HarmPotentialOutput(
                reasoning=overall_just,
                confidence=true_confidence,
                self_confidence=self_confidence,
                logprob_confidence=logprob_confidence,
                p_true=p_true,
                p_false=p_false,
                p_uncertain=p_uncertain,
                entropy=entropy,
                sub_scores=sub_scores_obj,
                json_parse_failed=json_parse_failed,
                logprobs_missing=logprobs_missing,
            ),
            usage,
            token_logprobs,
            reasoning_content,
            reasoning_logprobs,
        )

    def __call__(
        self, claim: str
    ) -> tuple[CheckworthinessResult, TokenUsage, dict[str, list[dict]], str | None, list[dict] | None]:
        """Run full pipeline on a claim (parallel execution).

        Returns:
            Tuple of:
                - CheckworthinessResult: Combined assessment result
                - TokenUsage: Aggregated token usage across all calls
                - dict[str, list[dict]]: Logprobs per module (checkability, verifiability, harm)
                - str | None: Internal reasoning content (for reasoning models like DeepSeek Reasoner)
                - list[dict] | None: Logprobs for internal reasoning (for reasoning models)

        Note:
            For reasoning models (e.g., DeepSeek Reasoner), the internal reasoning is returned
            from ALL three assessments. We aggregate them into a single string if present.
        """
        total_usage = TokenUsage()

        # Run all three assessments in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_check = executor.submit(self.assess_checkability, claim)
            future_verif = executor.submit(self.assess_verifiability, claim)
            future_harm = executor.submit(self.assess_harm_potential, claim)

            check_out, check_usage, check_logprobs, check_reasoning, check_reasoning_lp = future_check.result()
            verif_out, verif_usage, verif_logprobs, verif_reasoning, verif_reasoning_lp = future_verif.result()
            harm_out, harm_usage, harm_logprobs, harm_reasoning, harm_reasoning_lp = future_harm.result()

        # Aggregate token usage
        total_usage.prompt_tokens = check_usage.prompt_tokens + verif_usage.prompt_tokens + harm_usage.prompt_tokens
        total_usage.completion_tokens = (
            check_usage.completion_tokens + verif_usage.completion_tokens + harm_usage.completion_tokens
        )
        total_usage.total_tokens = check_usage.total_tokens + verif_usage.total_tokens + harm_usage.total_tokens

        # Aggregate logprobs per module
        all_logprobs = {
            "checkability": check_logprobs,
            "verifiability": verif_logprobs,
            "harm_potential": harm_logprobs,
        }

        # Aggregate reasoning content from all modules (if present)
        # For reasoning models like DeepSeek Reasoner, all three calls may return reasoning
        reasoning_parts = []
        if check_reasoning:
            reasoning_parts.append(f"[Checkability]\n{check_reasoning}")
        if verif_reasoning:
            reasoning_parts.append(f"[Verifiability]\n{verif_reasoning}")
        if harm_reasoning:
            reasoning_parts.append(f"[Harm Potential]\n{harm_reasoning}")

        combined_reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None

        # Aggregate reasoning logprobs (combine all into single list)
        combined_reasoning_logprobs: list[dict] | None = None
        if check_reasoning_lp or verif_reasoning_lp or harm_reasoning_lp:
            combined_reasoning_logprobs = []
            for lp in [check_reasoning_lp, verif_reasoning_lp, harm_reasoning_lp]:
                if lp:
                    combined_reasoning_logprobs.extend(lp)

        # Combine results
        result = CheckworthinessResult.from_modules(
            claim_text=claim,
            checkability=check_out,
            verifiability=verif_out,
            harm_potential=harm_out,
            threshold=self.threshold,
        )

        self.last_usage = total_usage
        return result, total_usage, all_logprobs, combined_reasoning, combined_reasoning_logprobs


# =============================================================================
# BAML-based Implementation
# =============================================================================


class BAMLPromptingBaseline:
    """BAML-based prompting baseline with type-safe outputs and logprob extraction.

    Uses BAML for structured output parsing and Collector for raw response access.
    Prompts are defined in baml_src/checkworthiness.baml files.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        threshold: float = 50.0,
    ):
        if not BAML_AVAILABLE:
            raise RuntimeError(
                "BAML is not installed. Install with: uv add baml-py"
            )

        self.model_config = model_config
        self.threshold = threshold

        # Import BAML client (generated code in src/baml_client/baml_client/)
        try:
            import sys
            from pathlib import Path

            # Add baml_client to path if not already importable
            baml_client_path = Path(__file__).parent.parent / "baml_client"
            if str(baml_client_path) not in sys.path:
                sys.path.insert(0, str(baml_client_path))

            # Import sync client (not async) for synchronous usage
            from baml_client.sync_client import b
            from baml_client import types as baml_types

            self._b = b
            self._baml_types = baml_types
        except ImportError as e:
            raise RuntimeError(
                f"BAML client not generated. Run: baml-cli generate\n{e}"
            )

        # Track token usage
        self.last_usage = TokenUsage()

        # Get BAML version
        try:
            from importlib.metadata import version as pkg_version
            baml_version = pkg_version("baml-py")
        except Exception:
            baml_version = "unknown"

        # Store environment info for reproducibility reporting
        self.environment_info = EnvironmentInfo(
            sdk_name="baml",
            sdk_version=baml_version,
            provider=model_config.provider.value,
            model_name=model_config.model_name,
            api_base=model_config.api_base,
        )

    def get_environment_info(self) -> EnvironmentInfo:
        """Get execution environment information for reproducibility."""
        return self.environment_info

    def print_environment_info(self) -> None:
        """Print environment info to stdout."""
        print("\n" + "=" * 70)
        print("EXECUTION ENVIRONMENT (BAML)")
        print("=" * 70)
        print(f"  SDK: baml v{self.environment_info.sdk_version}")
        print(f"  Provider: {self.model_config.provider.value}")
        print(f"  Model: {self.model_config.model_name}")
        print(f"  API Base: {self.model_config.api_base or 'default (api.openai.com)'}")
        print(f"  Threshold: {self.threshold}")
        print("=" * 70 + "\n")

    def _extract_logprobs_from_collector(
        self, collector: Collector  # type: ignore[valid-type]
    ) -> tuple[TokenUsage, LogprobData | None]:
        """Extract token usage and logprobs from BAML Collector.

        Args:
            collector: BAML Collector with captured request/response

        Returns:
            Tuple of (TokenUsage, LogprobData or None if unavailable)
        """
        try:
            last = collector.last
            if not last:
                return TokenUsage(), None

            # Extract token usage including reasoning tokens
            reasoning_tokens = 0
            if last.usage:
                # BAML might expose reasoning tokens via completion_tokens_details
                if hasattr(last.usage, "completion_tokens_details") and last.usage.completion_tokens_details:
                    details = last.usage.completion_tokens_details
                    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                elif hasattr(last.usage, "reasoning_tokens"):
                    reasoning_tokens = last.usage.reasoning_tokens or 0

            usage = TokenUsage(
                prompt_tokens=last.usage.input_tokens if last.usage else 0,
                completion_tokens=last.usage.output_tokens if last.usage else 0,
                reasoning_tokens=reasoning_tokens,
                total_tokens=(last.usage.input_tokens + last.usage.output_tokens) if last.usage else 0,
            )

            # Extract logprobs from raw HTTP response
            if not last.calls:
                return usage, None

            last_call = last.calls[-1]
            if not last_call.http_response:
                return usage, None

            raw_body = last_call.http_response.body.json()
            if not raw_body or "choices" not in raw_body:
                return usage, None

            choices = raw_body.get("choices", [])
            if not choices:
                return usage, None

            logprobs_data = choices[0].get("logprobs", {})
            if not logprobs_data:
                return usage, None

            content = logprobs_data.get("content", [])
            logprob_data = LogprobData.from_api_response(content)

            return usage, logprob_data

        except Exception as e:
            print(f"[BAML] Warning: Failed to extract logprobs: {e}")
            return TokenUsage(), None

    def _calculate_confidence(
        self,
        baml_result: object,
        logprob_data: LogprobData | None,
        boolean_field: str,
    ) -> LogprobConfidence:
        """Calculate logprob-based confidence for a BAML result.

        Args:
            baml_result: The parsed BAML output
            logprob_data: Extracted logprob data (or None)
            boolean_field: Name of the boolean field (e.g., "is_checkable")

        Returns:
            LogprobConfidence with calculated values
        """
        # Get self-reported confidence from BAML result
        self_reported = getattr(baml_result, "confidence", 50.0)

        if not logprob_data:
            return LogprobConfidence.unavailable(self_reported)

        # Find logprobs for the ternary field
        top_logprobs = logprob_data.find_ternary_logprobs(boolean_field)

        if not top_logprobs:
            if LOGPROBS_DEBUG:
                print(f"[BAML] {boolean_field}: ternary token not found in logprobs")
            return LogprobConfidence.unavailable(self_reported)

        return LogprobConfidence.from_logprobs(
            top_logprobs=top_logprobs,
            self_reported=self_reported,
        )

    def assess_checkability(self, claim: str) -> tuple[CheckabilityOutput, TokenUsage, LogprobConfidence]:
        """Assess checkability of a claim using BAML."""
        collector = Collector(name="checkability")

        # Call BAML function with Collector
        result = self._b.AssessCheckability(claim, baml_options={"collector": collector})

        # Extract usage and logprobs from Collector
        usage, logprob_data = self._extract_logprobs_from_collector(collector)

        # Calculate confidence
        confidence = self._calculate_confidence(result, logprob_data, "is_checkable")

        # Print logprob stats
        if LOGPROBS_DEBUG:
            print(
                f"[BAML] is_checkable: self_conf={confidence.self_reported_confidence:.1f} "
                f"bool_conf={confidence.boolean_confidence:.1f}"
            )

        # Convert BAML type to our schema (use boolean confidence)
        output = CheckabilityOutput(
            reasoning=result.reasoning,
            confidence=confidence.boolean_confidence,
        )

        return output, usage, confidence

    def assess_verifiability(self, claim: str) -> tuple[VerifiabilityOutput, TokenUsage, LogprobConfidence]:
        """Assess verifiability of a claim using BAML."""
        collector = Collector(name="verifiability")

        # Call BAML function with Collector
        result = self._b.AssessVerifiability(claim, baml_options={"collector": collector})

        # Extract usage and logprobs from Collector
        usage, logprob_data = self._extract_logprobs_from_collector(collector)

        # Calculate confidence
        confidence = self._calculate_confidence(result, logprob_data, "is_verifiable")

        # Print logprob stats
        if LOGPROBS_DEBUG:
            print(
                f"[BAML] is_verifiable: self_conf={confidence.self_reported_confidence:.1f} "
                f"bool_conf={confidence.boolean_confidence:.1f}"
            )

        # Convert BAML type to our schema
        output = VerifiabilityOutput(
            reasoning=result.reasoning,
            confidence=confidence.boolean_confidence,
        )

        return output, usage, confidence

    def assess_harm_potential(
        self, claim: str
    ) -> tuple[HarmPotentialOutput, TokenUsage, LogprobConfidence, dict[str, LogprobConfidence]]:
        """Assess harm potential of a claim using BAML."""
        collector = Collector(name="harm_potential")

        # Call BAML function with Collector
        result = self._b.AssessHarmPotential(claim, baml_options={"collector": collector})

        # Extract usage and logprobs from Collector
        usage, logprob_data = self._extract_logprobs_from_collector(collector)

        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(result, logprob_data, "is_harmful")

        # Calculate sub-score confidences
        sub_confidences: dict[str, LogprobConfidence] = {}
        for field_name, sub_result in [
            ("social_fragmentation", result.social_fragmentation),
            ("spurs_action", result.spurs_action),
            ("believability", result.believability),
            ("exploitativeness", result.exploitativeness),
        ]:
            sub_conf = self._calculate_confidence(sub_result, logprob_data, f"is_harmful")
            sub_confidences[field_name] = sub_conf

        # Print logprob stats
        if LOGPROBS_DEBUG:
            print(
                f"[BAML] is_harmful: self_conf={overall_confidence.self_reported_confidence:.1f} "
                f"bool_conf={overall_confidence.boolean_confidence:.1f}"
            )

        # Convert BAML types to our schema
        sub_scores = HarmSubScores(
            social_fragmentation=sub_confidences["social_fragmentation"].boolean_confidence,
            spurs_action=sub_confidences["spurs_action"].boolean_confidence,
            believability=sub_confidences["believability"].boolean_confidence,
            exploitativeness=sub_confidences["exploitativeness"].boolean_confidence,
        )

        output = HarmPotentialOutput(
            reasoning=result.reasoning,
            confidence=overall_confidence.boolean_confidence,
            sub_scores=sub_scores,
        )

        return output, usage, overall_confidence, sub_confidences

    def __call__(self, claim: str) -> tuple[CheckworthinessResult, TokenUsage]:
        """Run full pipeline on a claim (parallel execution)."""
        total_usage = TokenUsage()

        # Run all three assessments in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_check = executor.submit(self.assess_checkability, claim)
            future_verif = executor.submit(self.assess_verifiability, claim)
            future_harm = executor.submit(self.assess_harm_potential, claim)

            check_out, check_usage, _ = future_check.result()
            verif_out, verif_usage, _ = future_verif.result()
            harm_out, harm_usage, _, _ = future_harm.result()

        # Aggregate token usage
        total_usage.prompt_tokens = check_usage.prompt_tokens + verif_usage.prompt_tokens + harm_usage.prompt_tokens
        total_usage.completion_tokens = (
            check_usage.completion_tokens + verif_usage.completion_tokens + harm_usage.completion_tokens
        )
        total_usage.total_tokens = check_usage.total_tokens + verif_usage.total_tokens + harm_usage.total_tokens

        # Combine results
        result = CheckworthinessResult.from_modules(
            claim_text=claim,
            checkability=check_out,
            verifiability=verif_out,
            harm_potential=harm_out,
            threshold=self.threshold,
        )

        self.last_usage = total_usage
        return result, total_usage
