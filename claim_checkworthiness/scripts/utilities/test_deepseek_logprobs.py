#!/usr/bin/env python3
"""
Test DeepSeek logprobs support - critical verification for temperature experiment.
Based on DeepSeek documentation that claims logprobs trigger errors.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_deepseek_logprobs():
    """Test DeepSeek logprobs support."""
    
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("❌ DEEPSEEK_API_KEY not found")
        return
    
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/beta"
    )
    
    test_prompt = "Is this claim checkworthy? 'The Earth is flat.'"
    
    print("🔍 Testing DeepSeek Logprobs Support")
    print("=" * 50)
    
    # Test 1: DeepSeek Chat without logprobs (baseline)
    print("\n1️⃣ Testing deepseek-chat without logprobs...")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=50
        )
        content = response.choices[0].message.content
        print(f"✅ Success: {len(content)} chars")
        print(f"   Preview: {content[:80]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: DeepSeek Chat with logprobs (should fail per docs)
    print("\n2️⃣ Testing deepseek-chat with logprobs=True...")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=50,
            logprobs=True,
            top_logprobs=3
        )
        content = response.choices[0].message.content
        print(f"✅ Unexpected success: {len(content)} chars")
        print(f"   Logprobs available: {hasattr(response.choices[0], 'logprobs')}")
    except Exception as e:
        print(f"❌ Expected failure: {e}")
    
    # Test 3: DeepSeek Reasoner without logprobs
    print("\n3️⃣ Testing deepseek-reasoner without logprobs...")
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=50
        )
        content = response.choices[0].message.content
        print(f"✅ Success: {len(content)} chars")
        print(f"   Preview: {content[:80]}...")
        
        # Check for reasoning_content field
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning = response.choices[0].message.reasoning_content
            print(f"   ✅ reasoning_content found: {len(reasoning)} chars")
        else:
            print(f"   ❌ reasoning_content not found")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: DeepSeek Reasoner with logprobs (should fail per docs)
    print("\n4️⃣ Testing deepseek-reasoner with logprobs=True...")
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=50,
            logprobs=True,
            top_logprobs=3
        )
        content = response.choices[0].message.content
        print(f"✅ Unexpected success: {len(content)} chars")
        print(f"   Logprobs available: {hasattr(response.choices[0], 'logprobs')}")
    except Exception as e:
        print(f"❌ Expected failure: {e}")
    
    # Test 5: DeepSeek Chat with thinking mode
    print("\n5️⃣ Testing deepseek-chat with thinking mode...")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=50,
            extra_body={"thinking": {"type": "enabled"}}
        )
        content = response.choices[0].message.content
        print(f"✅ Success: {len(content)} chars")
        
        # Check for reasoning_content field
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning = response.choices[0].message.reasoning_content
            print(f"   ✅ reasoning_content found: {len(reasoning)} chars")
            print(f"   Preview: {reasoning[:80]}...")
        else:
            print(f"   ❌ reasoning_content not found")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Critical Findings:")
    print("- DeepSeek documentation claims logprobs trigger errors")
    print("- Testing confirms whether this is actually true")
    print("- Reasoning mode works but may not provide logprobs")
    print("- This affects model selection for temperature experiment")

if __name__ == "__main__":
    test_deepseek_logprobs()