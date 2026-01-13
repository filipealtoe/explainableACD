#!/usr/bin/env python3
"""List all Together AI chat models with pricing."""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    print("ERROR: TOGETHER_API_KEY not found in .env")
    exit(1)

response = requests.get(
    "https://api.together.xyz/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code != 200:
    print(f"ERROR: {response.status_code} - {response.text}")
    exit(1)

data = response.json()

# Filter to chat models
chat_models = [m for m in data if m.get('type') == 'chat']

# Sort by input cost (proxy for model size)
chat_models.sort(key=lambda x: x.get('pricing', {}).get('input', 0))

print(f"Total chat models: {len(chat_models)}\n")
print(f"{'Model ID':<70} {'Ctx':>7} {'$/M In':>8} {'$/M Out':>8}")
print("=" * 100)

for m in chat_models:
    p = m.get('pricing', {})
    ctx = m.get('context_length', 0)
    ctx_s = f"{ctx//1000}k" if ctx >= 1000 else str(ctx)
    input_cost = p.get('input', 0)
    output_cost = p.get('output', 0)
    print(f"{m['id']:<70} {ctx_s:>7} ${input_cost:>7.3f} ${output_cost:>7.3f}")
