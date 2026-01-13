#!/usr/bin/env python3
"""
Generate synthetic training data using OpenAI-compatible APIs.

Supports: OpenAI, DeepSeek, Together AI
Prompts are loaded from prompts/synthetic_generation_prompts.yaml

Based on error analysis:
- 18 FP (45%) → Generate hard negatives (label=No that look checkworthy)
- 22 FN (55%) → Generate hard positives (label=Yes that look non-checkworthy)

Usage:
    # Preview prompts
    python experiments/scripts/generate_synthetic_data.py --dry-run

    # DeepSeek (recommended - smart + cheap)
    python experiments/scripts/generate_synthetic_data.py --provider deepseek --total_samples 500

    # OpenAI
    python experiments/scripts/generate_synthetic_data.py --provider openai --model gpt-4.1 --total_samples 500

    # Together AI
    python experiments/scripts/generate_synthetic_data.py --provider together --model meta-llama/Llama-3.1-70B-Instruct --total_samples 500
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import polars as pl
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Force load .env from repo root, overriding any existing env vars
REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(REPO_ROOT / ".env", override=True)

# Paths
PROMPTS_PATH = REPO_ROOT / "prompts" / "synthetic_generation_prompts.yaml"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "synthetic_data"

# Provider configurations
PROVIDERS = {
    "openai": {
        "base_url": None,  # Use default
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4.1-mini",
        "pricing": {  # per 1M tokens (input, output)
            "gpt-4.1-mini": (0.40, 1.60),
            "gpt-4.1": (2.00, 8.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
        },
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "pricing": {
            "deepseek-chat": (0.14, 0.28),  # V3.2, very cheap
            "deepseek-reasoner": (0.55, 2.19),
        },
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.1-70B-Instruct",
        "pricing": {
            "meta-llama/Llama-3.1-70B-Instruct": (0.88, 0.88),
            "meta-llama/Llama-3.1-8B-Instruct": (0.18, 0.18),
            "Qwen/Qwen2.5-72B-Instruct": (1.20, 1.20),
        },
    },
}


def load_prompts() -> dict:
    """Load prompts and categories from YAML file."""
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def build_generation_prompt(
    prompts: dict,
    category_name: str,
    category_info: dict,
    label: str,
    batch_size: int,
) -> str:
    """
    Build generation prompt using YAML template.

    Follows GPT-4.1 best practices:
    - XML-style structure for clear sections
    - Explicit instructions (model follows literally)
    - Sandwich method: constraints at start AND end
    """
    # Get label description (explicit about what makes checkworthy/not)
    label_description = prompts["label_descriptions"][label]

    # Format examples as bullet list
    examples_formatted = "\n".join(f"  - {ex}" for ex in category_info["examples"])

    # Fill in template
    prompt = prompts["generation_template"].format(
        batch_size=batch_size,
        label_description=label_description,
        category_name=category_name,
        category_description=category_info["description"],
        examples_formatted=examples_formatted,
    )

    return prompt


def generate_batch(
    client: OpenAI,
    prompts: dict,
    category: str,
    category_info: dict,
    label: str,
    batch_size: int,
    model: str,
) -> list[dict]:
    """Generate a batch of synthetic claims using GPT-4.1-mini optimized prompts."""
    user_prompt = build_generation_prompt(prompts, category, category_info, label, batch_size)
    system_prompt = prompts["system_prompt"]
    gen_params = prompts["generation_params"]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=gen_params["temperature"],
            max_tokens=gen_params["max_tokens"],
            presence_penalty=gen_params.get("presence_penalty", 0),
            frequency_penalty=gen_params.get("frequency_penalty", 0),
        )

        # Parse response into individual claims
        raw_text = response.choices[0].message.content.strip()

        # Clean up: remove numbering, bullets, empty lines
        claims = []
        for line in raw_text.split("\n"):
            line = line.strip()
            # Skip empty lines and meta-commentary
            if not line or len(line) < 10:
                continue
            # Remove common prefixes (1. 2. - * etc.)
            cleaned = line.lstrip("0123456789.-)*• ").strip()
            if cleaned and len(cleaned) >= 10:
                claims.append(cleaned)

        # Create records
        records = []
        for claim in claims[:batch_size]:  # Limit to requested batch size
            records.append(
                {
                    "Text": claim,
                    "class_label": label,
                    "category": category,
                    "source": "synthetic",
                    "model": model,
                }
            )

        return records

    except Exception as e:
        print(f"Error generating batch for {category}/{label}: {e}")
        return []


def estimate_cost(provider: str, model: str, total_samples: int) -> tuple[float, int, int]:
    """Estimate API cost based on provider, model and sample count."""
    # Estimates based on ~300 input tokens per prompt, ~30 output tokens per claim
    est_input_tokens = total_samples * 300  # Longer prompts with XML structure
    est_output_tokens = total_samples * 30

    # Get pricing from provider config
    pricing = PROVIDERS.get(provider, {}).get("pricing", {})
    input_price, output_price = pricing.get(model, (0.50, 2.00))  # Conservative default

    cost = (est_input_tokens * input_price + est_output_tokens * output_price) / 1_000_000

    return cost, est_input_tokens, est_output_tokens


def create_client(provider: str) -> tuple[OpenAI, str]:
    """Create OpenAI-compatible client for the given provider."""
    config = PROVIDERS[provider]
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        raise ValueError(f"{config['api_key_env']} not found in .env file")

    # Print masked key for debugging
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"\n🔑 Using {provider} API key: {masked_key}")

    if config["base_url"]:
        client = OpenAI(api_key=api_key, base_url=config["base_url"])
    else:
        client = OpenAI(api_key=api_key)

    return client, config["default_model"]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic checkworthiness training data")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["openai", "deepseek", "together"],
                        help="API provider (default: deepseek)")
    parser.add_argument("--model", type=str, default=None, help="Model to use (default: provider's default)")
    parser.add_argument("--total_samples", type=int, default=500, help="Total synthetic samples to generate (default: 500)")
    parser.add_argument("--batch_size", type=int, default=20, help="Claims per API call")
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts without API calls")
    parser.add_argument("--fp-ratio", type=float, default=0.45, help="Ratio of No samples (default: 0.45 from error analysis)")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    # Use provider's default model if not specified
    if args.model is None:
        args.model = PROVIDERS[args.provider]["default_model"]

    print("=" * 70)
    print("ADVERSARIAL SYNTHETIC DATA GENERATION")
    print("=" * 70)

    # Load prompts from YAML
    print(f"\n📄 Loading prompts from: {PROMPTS_PATH}")
    prompts = load_prompts()

    fp_categories = prompts["fp_categories"]
    fn_categories = prompts["fn_categories"]

    print(f"   FP categories: {len(fp_categories)}")
    print(f"   FN categories: {len(fn_categories)}")

    print(f"\n🤖 Provider: {args.provider}")
    print(f"🤖 Model: {args.model}")
    print(f"📊 Total samples: {args.total_samples}")
    print(f"📈 FP ratio (No): {args.fp_ratio:.0%}, FN ratio (Yes): {1-args.fp_ratio:.0%}")

    # Calculate distribution
    n_no = int(args.total_samples * args.fp_ratio)
    n_yes = args.total_samples - n_no
    print(f"\n📦 Distribution:")
    print(f"   label=No (hard negatives):  {n_no}")
    print(f"   label=Yes (hard positives): {n_yes}")

    # Calculate per-category counts
    print(f"\n📋 Per-category breakdown:")
    print(f"\n   --- HARD NEGATIVES (label=No) ---")
    category_counts = {}
    for cat, info in fp_categories.items():
        count = int(n_no * info["weight"])
        category_counts[(cat, "No")] = count
        print(f"      {cat}: {count}")

    print(f"\n   --- HARD POSITIVES (label=Yes) ---")
    for cat, info in fn_categories.items():
        count = int(n_yes * info["weight"])
        category_counts[(cat, "Yes")] = count
        print(f"      {cat}: {count}")

    # Cost estimate
    cost, est_input, est_output = estimate_cost(args.provider, args.model, args.total_samples)
    print(f"\n💰 Estimated cost: ${cost:.2f}")
    print(f"   (Input: ~{est_input/1000:.0f}K tokens, Output: ~{est_output/1000:.0f}K tokens)")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("🔍 DRY RUN - Previewing prompts (no API calls)")
        print("=" * 70)

        print("\n--- SYSTEM PROMPT ---")
        print(prompts["system_prompt"])

        print("\n--- EXAMPLE USER PROMPT (vague_rhetoric, No) ---")
        example_prompt = build_generation_prompt(
            prompts,
            "vague_rhetoric",
            fp_categories["vague_rhetoric"],
            "No",
            5,
        )
        print(example_prompt)

        print("\n--- EXAMPLE USER PROMPT (short_punchy_claims, Yes) ---")
        example_prompt = build_generation_prompt(
            prompts,
            "short_punchy_claims",
            fn_categories["short_punchy_claims"],
            "Yes",
            5,
        )
        print(example_prompt)

        return

    # Initialize client for the selected provider
    client, _ = create_client(args.provider)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate data
    all_records = []
    total_target = sum(category_counts.values())

    print(f"\n🚀 Starting generation...")
    with tqdm(total=total_target, desc="Generating") as pbar:
        for (category, label), target_count in category_counts.items():
            # Get category info from appropriate dict
            category_info = fp_categories.get(category) or fn_categories.get(category)
            generated = 0

            while generated < target_count:
                batch_size = min(args.batch_size, target_count - generated)
                records = generate_batch(
                    client, prompts, category, category_info, label, batch_size, args.model
                )

                if records:
                    all_records.extend(records)
                    generated += len(records)
                    pbar.update(len(records))
                else:
                    # If batch failed, wait longer and retry
                    time.sleep(2.0)
                    continue

                # Rate limiting
                time.sleep(args.rate_limit)

    # Create DataFrame
    df = pl.DataFrame(all_records)
    print(f"\n✅ Generated {len(df)} samples")

    # Stats
    print(f"\n📊 Label distribution:")
    label_counts = df.group_by("class_label").len().sort("class_label")
    for row in label_counts.iter_rows(named=True):
        print(f"   {row['class_label']}: {row['len']}")

    print(f"\n📊 Category distribution:")
    cat_counts = df.group_by("category").len().sort("len", descending=True)
    for row in cat_counts.iter_rows(named=True):
        print(f"   {row['category']}: {row['len']}")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]  # Handle Together AI model names like "meta-llama/..."
    output_path = OUTPUT_DIR / f"synthetic_{args.provider}_{model_short}_{len(df)}_{timestamp}.parquet"
    df.write_parquet(output_path)
    print(f"\n💾 Saved to: {output_path}")

    # Also save as CSV for inspection
    csv_path = output_path.with_suffix(".csv")
    df.write_csv(csv_path)
    print(f"💾 CSV copy: {csv_path}")

    # Save generation metadata
    meta = {
        "provider": args.provider,
        "model": args.model,
        "total_samples": len(df),
        "fp_ratio": args.fp_ratio,
        "batch_size": args.batch_size,
        "timestamp": timestamp,
        "prompts_file": str(PROMPTS_PATH),
        "label_distribution": {row["class_label"]: row["len"] for row in label_counts.iter_rows(named=True)},
        "category_distribution": {row["category"]: row["len"] for row in cat_counts.iter_rows(named=True)},
    }
    meta_path = OUTPUT_DIR / f"generation_metadata_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Metadata: {meta_path}")


if __name__ == "__main__":
    main()
