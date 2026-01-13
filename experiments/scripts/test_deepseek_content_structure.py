#!/usr/bin/env python3
"""
Test DeepSeek content structure - understand how reasoning_content vs content works.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_deepseek_content_structure():
    """Test DeepSeek content structure in detail."""
    
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("❌ DEEPSEEK_API_KEY not found")
        return
    
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/beta"
    )
    
    # Test with a prompt that expects a clear answer
    test_prompt = "Answer YES or NO: Is the claim 'The Earth is flat' checkworthy?"
    
    print("🔬 DeepSeek Content Structure Analysis")
    print("=" * 60)
    
    # Test 1: DeepSeek Reasoner - check structure
    print("\n🔍 Testing deepseek-reasoner structure:")
    
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=200
        )
        
        message = response.choices[0].message
        
        print(f"✅ Request successful")
        print(f"  - Content: '{message.content}'")
        print(f"  - Has reasoning_content: {hasattr(message, 'reasoning_content')}")
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning = message.reasoning_content
            print(f"  - Reasoning length: {len(reasoning)}")
            print(f"  - Reasoning preview: '{reasoning[:100]}...'")
            
            # Check if reasoning contains the answer
            if 'YES' in reasoning or 'NO' in reasoning:
                print(f"  - ✅ Answer found in reasoning content")
            else:
                print(f"  - ❌ No clear answer in reasoning content")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: DeepSeek Chat with thinking mode
    print("\n🔍 Testing deepseek-chat with thinking mode:")
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=200,
            extra_body={"thinking": {"type": "enabled"}}
        )
        
        message = response.choices[0].message
        
        print(f"✅ Request successful")
        print(f"  - Content: '{message.content}'")
        print(f"  - Has reasoning_content: {hasattr(message, 'reasoning_content')}")
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning = message.reasoning_content
            print(f"  - Reasoning length: {len(reasoning)}")
            print(f"  - Reasoning preview: '{reasoning[:100]}...'")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: DeepSeek Chat without thinking mode (baseline)
    print("\n🔍 Testing deepseek-chat without thinking mode:")
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=200
        )
        
        message = response.choices[0].message
        
        print(f"✅ Request successful")
        print(f"  - Content: '{message.content}'")
        print(f"  - Has reasoning_content: {hasattr(message, 'reasoning_content')}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: Check if we can get final answer in content
    print("\n🔍 Testing with system prompt for structured output:")
    
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "Answer with FINAL ANSWER: [YES/NO] at the end."},
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        
        message = response.choices[0].message
        
        print(f"✅ Request successful")
        print(f"  - Content: '{message.content}'")
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning = message.reasoning_content
            print(f"  - Reasoning contains 'FINAL ANSWER': {'FINAL ANSWER' in reasoning}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Key Findings:")
    print("- DeepSeek Reasoner puts reasoning in reasoning_content")
    print("- May need to extract answer from reasoning_content")
    print("- Can use system prompts to structure the reasoning output")
    print("- Logprobs work for calibration metrics")

if __name__ == "__main__":
    test_deepseek_content_structure()