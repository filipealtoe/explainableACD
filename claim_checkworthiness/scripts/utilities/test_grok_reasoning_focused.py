#!/usr/bin/env python3
"""
Focused test for Grok reasoning mode - testing the most likely parameters.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_reasoning_mode():
    """Test reasoning_mode parameter specifically."""
    
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        print("❌ XAI_API_KEY not found")
        return
    
    client = OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1"
    )
    
    test_prompt = "Solve this: If A > B and B > C, what can we conclude about A and C? Show your reasoning."
    
    print("🧠 Testing Grok Reasoning Mode")
    print("=" * 40)
    
    # Test 1: Default (no reasoning params)
    print("\n1️⃣ Testing default (no reasoning params)...")
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=50
        )
        content = response.choices[0].message.content
        print(f"✅ Success: {content[:100]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: With reasoning_mode=true
    print("\n2️⃣ Testing with reasoning_mode=true...")
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=50,
            reasoning_mode=True
        )
        content = response.choices[0].message.content
        print(f"✅ Success: {content[:100]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: With reasoning=true
    print("\n3️⃣ Testing with reasoning=true...")
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=50,
            reasoning=True
        )
        content = response.choices[0].message.content
        print(f"✅ Success: {content[:100]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_reasoning_mode()