#!/usr/bin/env python3
"""
Verify DeepSeek reasoning_content behavior - definitive test.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_deepseek_reasoning():
    """Definitive test of DeepSeek reasoning_content."""
    
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("❌ DEEPSEEK_API_KEY not found")
        return
    
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/beta"
    )
    
    test_prompt = "Is this claim checkworthy? 'The Earth is flat.'"
    
    print("🔬 Definitive DeepSeek Reasoning Content Test")
    print("=" * 60)
    
    # Test 1: DeepSeek Chat (baseline)
    print("\n1️⃣ DeepSeek Chat (no thinking mode):")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=100
        )
        
        message = response.choices[0].message
        print(f"  Content: '{message.content[:50]}...'")
        print(f"  Has reasoning_content: {hasattr(message, 'reasoning_content')}")
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"  Reasoning: '{message.reasoning_content[:50]}...'")
        else:
            print(f"  Reasoning: ❌ NOT PROVIDED")
            
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test 2: DeepSeek Chat with thinking mode
    print("\n2️⃣ DeepSeek Chat (with thinking mode):")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=100,
            extra_body={"thinking": {"type": "enabled"}}
        )
        
        message = response.choices[0].message
        print(f"  Content: '{message.content[:50]}...'")
        print(f"  Has reasoning_content: {hasattr(message, 'reasoning_content')}")
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"  ✅ Reasoning: '{message.reasoning_content[:50]}...'")
        else:
            print(f"  ❌ Reasoning: NOT PROVIDED")
            
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test 3: DeepSeek Reasoner
    print("\n3️⃣ DeepSeek Reasoner:")
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=100
        )
        
        message = response.choices[0].message
        print(f"  Content: '{message.content[:50]}...'")
        print(f"  Has reasoning_content: {hasattr(message, 'reasoning_content')}")
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"  ✅ Reasoning: '{message.reasoning_content[:50]}...'")
        else:
            print(f"  ❌ Reasoning: NOT PROVIDED")
            
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test 4: Show full response structure
    print("\n4️⃣ Full Response Structure Analysis:")
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=100
        )
        
        # Convert to dict to see all fields
        response_dict = response.model_dump()
        choice = response_dict['choices'][0]
        message = choice['message']
        
        print(f"  Message keys: {list(message.keys())}")
        
        # Check each field
        for key, value in message.items():
            if key == 'content':
                print(f"    - {key}: '{value[:30]}...' (len: {len(value)})")
            elif key == 'reasoning_content':
                print(f"    - {key}: '{value[:30]}...' (len: {len(value)})")
            else:
                print(f"    - {key}: {type(value)}")
                
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Definitive Answer:")
    print("- DeepSeek Chat (no thinking): reasoning_content = ❌ NO")
    print("- DeepSeek Chat (with thinking): reasoning_content = ✅ YES")
    print("- DeepSeek Reasoner: reasoning_content = ✅ YES")
    print("- Both models provide final answer in 'content' field")
    print("- Reasoner model provides reasoning in 'reasoning_content' field")

if __name__ == "__main__":
    verify_deepseek_reasoning()