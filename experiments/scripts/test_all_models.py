#!/usr/bin/env python3
"""Test script to verify logprobs extraction works across all configured models."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from checkworthiness.config import ModelConfig, ModelProvider
from checkworthiness.prompting_baseline import PromptingBaseline

# Test claim
TEST_CLAIM = "The COVID-19 vaccine causes autism."

# Model configurations to test
MODELS = [
    # OpenAI models
    {
        "name": "OpenAI GPT-4o",
        "config": {
            "model_name": "gpt-4o",
            "provider": ModelProvider.OPENAI,
            "api_key_env": "OPENAI_API_KEY",
            "logprobs": True,
            "top_logprobs": 5,
        },
    },
    {
        "name": "OpenAI GPT-4.1-mini",
        "config": {
            "model_name": "gpt-4.1-mini",
            "provider": ModelProvider.OPENAI,
            "api_key_env": "OPENAI_API_KEY",
            "logprobs": True,
            "top_logprobs": 5,
        },
    },
    # DeepSeek
    {
        "name": "DeepSeek V3",
        "config": {
            "model_name": "deepseek-chat",
            "provider": ModelProvider.DEEPSEEK,
            "api_key_env": "DEEPSEEK_API_KEY",
            "api_base": "https://api.deepseek.com/v1",
            "logprobs": True,
            "top_logprobs": 5,
        },
    },
    # xAI Grok
    {
        "name": "xAI Grok",
        "config": {
            "model_name": "grok-3-latest",
            "provider": ModelProvider.XAI,
            "api_key_env": "XAI_API_KEY",
            "api_base": "https://api.x.ai/v1",
            "logprobs": True,
            "top_logprobs": 5,
        },
    },
    # Moonshot Kimi K2
    {
        "name": "Moonshot Kimi K2",
        "config": {
            "model_name": "kimi-k2-0711-preview",
            "provider": ModelProvider.MOONSHOT,
            "api_key_env": "MOONSHOT_API_KEY",
            "api_base": "https://api.moonshot.cn/v1",
            "logprobs": True,
            "top_logprobs": 5,
        },
    },
]


def test_model(model_info: dict) -> dict:
    """Test a single model and return results."""
    name = model_info["name"]
    config_params = model_info["config"]

    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print("=" * 60)

    # Check if API key is available
    api_key_env = config_params.get("api_key_env", "")
    if api_key_env and not os.environ.get(api_key_env):
        print(f"⚠️  SKIPPED: {api_key_env} not set")
        return {"name": name, "status": "skipped", "reason": f"{api_key_env} not set"}

    try:
        # Create config
        config = ModelConfig(**config_params)

        # Create baseline
        baseline = PromptingBaseline(config)

        # Test just checkability (faster)
        result, usage = baseline.assess_checkability(TEST_CLAIM)

        print(f"\n✅ SUCCESS")
        print(f"   Confidence: {result.confidence:.1f}")
        print(f"   Reasoning: {result.reasoning[:80]}...")
        print(f"   Tokens: {usage.total_tokens}")

        return {
            "name": name,
            "status": "success",
            "confidence": result.confidence,
            "tokens": usage.total_tokens,
        }

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return {"name": name, "status": "failed", "error": str(e)}


def main():
    print("=" * 60)
    print("TESTING LOGPROBS EXTRACTION ACROSS ALL MODELS")
    print("=" * 60)
    print(f"Test claim: {TEST_CLAIM}")

    results = []
    for model_info in MODELS:
        result = test_model(model_info)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    print(f"✅ Success: {success_count}")
    print(f"⚠️  Skipped: {skipped_count}")
    print(f"❌ Failed: {failed_count}")

    print("\nDetails:")
    for r in results:
        status_icon = {"success": "✅", "skipped": "⚠️", "failed": "❌"}[r["status"]]
        if r["status"] == "success":
            print(f"  {status_icon} {r['name']}: confidence={r['confidence']:.1f}, tokens={r['tokens']}")
        elif r["status"] == "skipped":
            print(f"  {status_icon} {r['name']}: {r['reason']}")
        else:
            print(f"  {status_icon} {r['name']}: {r['error'][:50]}...")


if __name__ == "__main__":
    main()
