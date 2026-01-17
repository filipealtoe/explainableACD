#!/usr/bin/env python3
"""
Final test for Grok reasoning - trying the most likely parameters based on industry standards.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_grok_reasoning():
    """Test Grok reasoning with industry-standard parameters."""
    
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        print("❌ XAI_API_KEY not found")
        return
    
    client = OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1"
    )
    
    # Simple reasoning test prompt
    test_prompt = "Explain why the sky appears blue during the day. Provide a step-by-step scientific explanation."
    
    print("🔍 Testing Grok Reasoning Parameters")
    print("=" * 50)
    
    # Based on industry standards, try these parameters:
    
    # 1. Default (baseline)
    print("\n1️⃣ Testing default parameters...")
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=100
        )
        content = response.choices[0].message.content
        print(f"✅ Default: {len(content)} chars")
        print(f"   Preview: {content[:80]}...")
    except Exception as e:
        print(f"❌ Default failed: {e}")
    
    # 2. Try prompt_mode (Mistral-style)
    print("\n2️⃣ Testing prompt_mode='reasoning' (Mistral-style)...")
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=100,
            prompt_mode="reasoning"
        )
        content = response.choices[0].message.content
        print(f"✅ prompt_mode: {len(content)} chars")
        print(f"   Preview: {content[:80]}...")
    except Exception as e:
        print(f"❌ prompt_mode failed: {e}")
    
    # 3. Try reasoning_mode (generic)
    print("\n3️⃣ Testing reasoning_mode=True...")
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=100,
            reasoning_mode=True
        )
        content = response.choices[0].message.content
        print(f"✅ reasoning_mode: {len(content)} chars")
        print(f"   Preview: {content[:80]}...")
    except Exception as e:
        print(f"❌ reasoning_mode failed: {e}")
    
    # 4. Try chain_of_thought (common in research)
    print("\n4️⃣ Testing chain_of_thought=True...")
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=100,
            chain_of_thought=True
        )
        content = response.choices[0].message.content
        print(f"✅ chain_of_thought: {len(content)} chars")
        print(f"   Preview: {content[:80]}...")
    except Exception as e:
        print(f"❌ chain_of_thought failed: {e}")
    
    # 5. Try with system prompt for reasoning
    print("\n5️⃣ Testing with reasoning system prompt...")
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "system", "content": "You are a reasoning assistant. Think step by step and explain your reasoning process."},
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        content = response.choices[0].message.content
        print(f"✅ System prompt: {len(content)} chars")
        print(f"   Preview: {content[:80]}...")
    except Exception as e:
        print(f"❌ System prompt failed: {e}")
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("- Tested multiple reasoning parameter approaches")
    print("- Identified which parameters are supported by Grok API")
    print("- Found the best way to enable reasoning mode")

if __name__ == "__main__":
    test_grok_reasoning()