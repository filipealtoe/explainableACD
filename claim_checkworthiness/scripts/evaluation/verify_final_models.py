#!/usr/bin/env python3
"""Verify final 5 models: logprobs support + reasoning content extraction.

Tests:
1. Does the model return logprobs?
2. Does the model return reasoning/CoT separately from the final answer?

Usage:
    python experiments/scripts/verify_final_models.py
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# =============================================================================
# Final 5 Models Configuration
# =============================================================================

FINAL_MODELS = {
    "gpt-4.1": {
        "model_name": "gpt-4.1",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY",
        "is_reasoning": False,
        "provider": "OpenAI",
    },
    "gpt-4.1-mini": {
        "model_name": "gpt-4.1-mini",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY",
        "is_reasoning": False,
        "provider": "OpenAI",
    },
    "deepseek-chat": {
        "model_name": "deepseek-chat",
        "api_base": "https://api.deepseek.com/beta",
        "api_key_env": "DEEPSEEK_API_KEY",
        "is_reasoning": False,
        "provider": "DeepSeek",
    },
    "deepseek-reasoner": {
        "model_name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com/beta",
        "api_key_env": "DEEPSEEK_API_KEY",
        "is_reasoning": True,
        "provider": "DeepSeek",
    },
    "grok-4.1-fast-reasoning": {
        "model_name": "grok-4-1-fast-reasoning",
        "api_base": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "is_reasoning": True,
        "provider": "xAI",
    },
}

# Test prompt that should elicit reasoning
TEST_PROMPT = """Is the following claim checkworthy? Answer Yes or No.

Claim: "The vaccine causes autism in 1 out of 100 children."

Think step by step, then give your final answer."""


def test_model(model_key: str, config: dict) -> dict:
    """Test a model for logprobs and reasoning content structure."""
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        return {
            "success": False,
            "error": f"API key {config['api_key_env']} not set",
        }

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=config["api_base"],
        )

        # Request with logprobs
        response = client.chat.completions.create(
            model=config["model_name"],
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=500,  # Enough for reasoning + answer
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )

        choice = response.choices[0]
        message = choice.message

        # Extract content
        content = message.content or ""

        # Check for separate reasoning content (DeepSeek R1 style)
        # DeepSeek reasoner returns reasoning in a separate field
        reasoning_content = None
        has_separate_reasoning = False

        # Check if message has reasoning_content attribute (DeepSeek R1)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning_content = message.reasoning_content
            has_separate_reasoning = True

        # Check raw response for any reasoning fields
        raw_message = response.choices[0].message.model_dump()

        # Check logprobs
        has_logprobs = choice.logprobs is not None
        logprobs_info = None

        if has_logprobs and choice.logprobs.content:
            first_token = choice.logprobs.content[0]
            logprobs_info = {
                "first_token": first_token.token,
                "logprob": first_token.logprob,
                "num_tokens_with_logprobs": len(choice.logprobs.content),
            }

            # Check if logprobs has reasoning_content (DeepSeek R1)
            if hasattr(choice.logprobs, "reasoning_content") and choice.logprobs.reasoning_content:
                logprobs_info["has_reasoning_logprobs"] = True
                logprobs_info["num_reasoning_tokens"] = len(choice.logprobs.reasoning_content)
            else:
                logprobs_info["has_reasoning_logprobs"] = False

        return {
            "success": True,
            "has_logprobs": has_logprobs,
            "logprobs_info": logprobs_info,
            "has_separate_reasoning": has_separate_reasoning,
            "content_length": len(content),
            "reasoning_length": len(reasoning_content) if reasoning_content else 0,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "reasoning_preview": (reasoning_content[:200] + "...") if reasoning_content and len(reasoning_content) > 200 else reasoning_content,
            "raw_message_keys": list(raw_message.keys()),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:300],
        }


def main():
    print("\n" + "=" * 80)
    print("FINAL 5 MODELS VERIFICATION")
    print("=" * 80)
    print(f"\nTest prompt: \"{TEST_PROMPT[:60]}...\"")
    print("\nTesting logprobs + reasoning content extraction...\n")

    results = {}

    for model_key, config in FINAL_MODELS.items():
        print(f"Testing {model_key}...", end=" ", flush=True)
        result = test_model(model_key, config)
        results[model_key] = result

        if not result["success"]:
            print(f"❌ FAILED: {result['error'][:60]}")
        else:
            logprobs_status = "✅" if result["has_logprobs"] else "❌"
            reasoning_status = "✅ SEPARATE" if result["has_separate_reasoning"] else "❌ INLINE"
            print(f"{logprobs_status} logprobs | {reasoning_status} reasoning")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\n{'Model':<28} {'Provider':<10} {'Logprobs':<10} {'Reasoning':<15} {'Content Len':<12} {'Reasoning Len'}")
    print("-" * 100)

    for model_key, config in FINAL_MODELS.items():
        result = results[model_key]
        provider = config.get("provider", "Unknown")

        if not result["success"]:
            print(f"{model_key:<28} {provider:<10} {'ERROR':<10} {'-':<15} {'-':<12} -")
            continue

        logprobs = "✅ YES" if result["has_logprobs"] else "❌ NO"
        reasoning = "✅ SEPARATE" if result["has_separate_reasoning"] else "INLINE"
        content_len = str(result["content_length"])
        reasoning_len = str(result["reasoning_length"]) if result["reasoning_length"] > 0 else "-"

        print(f"{model_key:<28} {provider:<10} {logprobs:<10} {reasoning:<15} {content_len:<12} {reasoning_len}")

    print("-" * 100)

    # Detailed output for models with separate reasoning
    print("\n" + "=" * 80)
    print("REASONING EXTRACTION DETAILS")
    print("=" * 80)

    for model_key, result in results.items():
        if not result["success"]:
            continue

        print(f"\n### {model_key} ###")
        print(f"Raw message keys: {result['raw_message_keys']}")

        if result["has_separate_reasoning"]:
            print(f"\n📝 REASONING (separate field):")
            print(f"   {result['reasoning_preview']}")
            print(f"\n💬 FINAL ANSWER:")
            print(f"   {result['content_preview']}")
        else:
            print(f"\n💬 CONTENT (reasoning inline):")
            print(f"   {result['content_preview']}")

        if result["logprobs_info"]:
            info = result["logprobs_info"]
            print(f"\n📊 LOGPROBS:")
            print(f"   First token: '{info['first_token']}' (logprob: {info['logprob']:.4f})")
            print(f"   Tokens with logprobs: {info['num_tokens_with_logprobs']}")
            if info.get("has_reasoning_logprobs"):
                print(f"   Reasoning tokens with logprobs: {info['num_reasoning_tokens']}")

    # Final recommendation
    print("\n" + "=" * 80)
    print("EXTRACTION RECOMMENDATION")
    print("=" * 80)

    for model_key, result in results.items():
        if not result["success"]:
            print(f"\n❌ {model_key}: FAILED - check API key/endpoint")
            continue

        if result["has_separate_reasoning"]:
            print(f"\n✅ {model_key}:")
            print(f"   → Use `message.reasoning_content` for CoT")
            print(f"   → Use `message.content` for final answer")
            print(f"   → Use `logprobs.content` for final answer logprobs")
            if result.get("logprobs_info", {}).get("has_reasoning_logprobs"):
                print(f"   → Use `logprobs.reasoning_content` for reasoning logprobs")
        else:
            print(f"\n⚠️  {model_key}:")
            print(f"   → Reasoning is INLINE with content")
            print(f"   → Need to parse CoT from `message.content`")
            print(f"   → Logprobs cover entire response")

    return results


if __name__ == "__main__":
    main()
