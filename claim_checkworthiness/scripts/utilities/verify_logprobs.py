#!/usr/bin/env python3
"""Verify logprobs support for all candidate models.

This script tests whether each model actually returns logprobs,
regardless of what their documentation claims.

Usage:
    python experiments/scripts/verify_logprobs.py
"""

import os
import sys
from pathlib import Path

from openai import OpenAI

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# =============================================================================
# Model Configurations to Test
# =============================================================================

MODELS_TO_TEST = {
    # =========================================================================
    # OpenAI Models - Temperature Experiment Candidates
    # =========================================================================
    "gpt-4o": {
        "model_name": "gpt-4o",
        "api_base": None,  # Default OpenAI
        "api_key_env": "OPENAI_API_KEY",
        "is_reasoning": False,
        "provider": "OpenAI",
        "experiment_role": "Baseline comparison",
    },
    "gpt-4.1": {
        "model_name": "gpt-4-1",
        "api_base": None,  # Default OpenAI
        "api_key_env": "OPENAI_API_KEY",
        "is_reasoning": False,
        "provider": "OpenAI",
        "experiment_role": "Standard LLM (large)",
    },
    "gpt-4.1-mini": {
        "model_name": "gpt-4.1-mini",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY",
        "is_reasoning": False,
        "provider": "OpenAI",
        "experiment_role": "Standard LLM (small)",
    },
    "o3": {
        "model_name": "o3",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY",
        "is_reasoning": True,
        "provider": "OpenAI",
        "experiment_role": "Reasoning model (no logprobs expected)",
    },
    "o4-mini": {
        "model_name": "o4-mini",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY",
        "is_reasoning": True,
        "provider": "OpenAI",
        "experiment_role": "Reasoning model (no logprobs expected)",
    },

    # =========================================================================
    # DeepSeek Models - Temperature Experiment Candidates
    # =========================================================================
    "deepseek-chat": {
        "model_name": "deepseek-chat",
        "api_base": "https://api.deepseek.com/beta",
        "api_key_env": "DEEPSEEK_API_KEY",
        "is_reasoning": False,
        "provider": "DeepSeek",
        "experiment_role": "Open-weight baseline (non-thinking)",
    },
    "deepseek-reasoner": {
        "model_name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com/beta",
        "api_key_env": "DEEPSEEK_API_KEY",
        "is_reasoning": True,
        "provider": "DeepSeek",
        "experiment_role": "Same model, thinking mode ON",
    },

    # =========================================================================
    # xAI Grok Models - Temperature Experiment Candidates
    # =========================================================================
    "grok-3-beta": {
        "model_name": "grok-3-beta",
        "api_base": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "is_reasoning": False,
        "provider": "xAI",
        "experiment_role": "Reasoning architecture comparison",
    },
    "grok-4.1-fast-reasoning": {
        "model_name": "grok-4-1-fast-reasoning",
        "api_base": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "is_reasoning": True,
        "provider": "xAI",
        "experiment_role": "Reasoning architecture (main candidate)",
    },

    # =========================================================================
    # Other Models for Comparison
    # =========================================================================
    "gemini-2.5-flash": {
        "model_name": "gemini-2.5-flash",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "is_reasoning": False,
        "provider": "Google",
        "experiment_role": "Comparison (no logprobs expected)",
    },
    
    "kimi-k2-thinking": {
        "model_name": "kimi-k2-thinking",
        "api_base": "https://api.moonshot.ai/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "is_reasoning": True,
        "provider": "Moonshot",
        "experiment_role": "Comparison (complex logprobs)",
    },
    
    "mistral-large-3": {
        "model_name": "mistral-large-2412",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "is_reasoning": False,
        "provider": "Mistral",
        "experiment_role": "Comparison (no logprobs expected)",
    },
}

# Simple test prompt
TEST_PROMPT = "Is the sky blue? Answer only 'Yes' or 'No'."


def test_logprobs(model_key: str, config: dict) -> dict:
    """Test if a model returns logprobs.

    Returns:
        dict with keys: success, has_logprobs, logprobs_sample, error
    """
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        return {
            "success": False,
            "has_logprobs": None,
            "logprobs_sample": None,
            "error": f"API key {config['api_key_env']} not set",
        }

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=config["api_base"],
        )

        # Make request with logprobs enabled
        response = client.chat.completions.create(
            model=config["model_name"],
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )

        choice = response.choices[0]
        content = choice.message.content

        # Check if logprobs were returned
        has_logprobs = choice.logprobs is not None
        logprobs_sample = None

        if has_logprobs and choice.logprobs.content:
            # Get first token's logprobs as sample
            first_token = choice.logprobs.content[0]
            logprobs_sample = {
                "token": first_token.token,
                "logprob": first_token.logprob,
                "top_logprobs": [
                    {"token": lp.token, "logprob": lp.logprob}
                    for lp in (first_token.top_logprobs or [])[:3]
                ],
            }

        return {
            "success": True,
            "has_logprobs": has_logprobs,
            "logprobs_sample": logprobs_sample,
            "response": content,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "has_logprobs": None,
            "logprobs_sample": None,
            "error": str(e)[:200],
        }


def main():
    print("\n" + "=" * 80)
    print("LOGPROBS VERIFICATION TEST")
    print("=" * 80)
    print(f"\nTest prompt: \"{TEST_PROMPT}\"")
    print("\nTesting each model to verify logprobs support...\n")

    results = {}

    for model_key, config in MODELS_TO_TEST.items():
        print(f"Testing {model_key}...", end=" ", flush=True)
        result = test_logprobs(model_key, config)
        results[model_key] = result

        if not result["success"]:
            print(f"❌ FAILED: {result['error'][:60]}")
        elif result["has_logprobs"]:
            print(f"✅ LOGPROBS WORK")
        else:
            print(f"⚠️  NO LOGPROBS (response OK, but logprobs=None)")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\n{'Model':<28} {'Provider':<10} {'Type':<12} {'Logprobs':<12} {'Status'}")
    print("-" * 100)

    for model_key, config in MODELS_TO_TEST.items():
        result = results[model_key]
        provider = config.get("provider", "Unknown")
        model_type = "🧠 Reasoning" if config.get("is_reasoning") else "Standard"

        if not result["success"]:
            logprobs_status = "⚪ N/A"
            status = f"Error: {result['error'][:35]}..."
        elif result["has_logprobs"]:
            logprobs_status = "✅ YES"
            status = "✓ Ready to use"
        else:
            logprobs_status = "❌ NO"
            status = "✗ Cannot use (no logprobs)"

        print(f"{model_key:<28} {provider:<10} {model_type:<12} {logprobs_status:<12} {status}")

    print("-" * 80)
    print("\n🧠 = Reasoning model (expected to not have logprobs)")

    # Show logprobs sample for models that support it
    print("\n" + "=" * 80)
    print("LOGPROBS SAMPLES (for models that support it)")
    print("=" * 80)

    for model_key, result in results.items():
        if result.get("has_logprobs") and result.get("logprobs_sample"):
            sample = result["logprobs_sample"]
            print(f"\n{model_key}:")
            print(f"  First token: '{sample['token']}' (logprob: {sample['logprob']:.4f})")
            if sample.get("top_logprobs"):
                print(f"  Top alternatives:")
                for alt in sample["top_logprobs"]:
                    prob = 100 * (2.718281828 ** alt["logprob"])  # exp(logprob) * 100
                    print(f"    '{alt['token']}': {prob:.1f}%")

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    working_models = [k for k, r in results.items() if r.get("has_logprobs")]
    non_working = [k for k, r in results.items() if r["success"] and not r.get("has_logprobs")]

    if working_models:
        print(f"\n✅ Models with logprobs support: {', '.join(working_models)}")
    if non_working:
        print(f"\n❌ Models WITHOUT logprobs: {', '.join(non_working)}")

    return results


if __name__ == "__main__":
    main()