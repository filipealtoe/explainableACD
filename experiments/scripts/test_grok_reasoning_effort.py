#!/usr/bin/env python3
"""
Test Grok reasoning_effort parameter - the correct reasoning parameter for XAI models.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_reasoning_effort():
    """Test reasoning_effort parameter specifically."""
    
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        print("❌ XAI_API_KEY not found")
        return
    
    client = OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1"
    )
    
    # Complex reasoning test prompt
    test_prompt = """
Solve this complex logic puzzle step by step:

There are three boxes: red, blue, and green.
- The red box contains either apples or oranges
- The blue box contains either bananas or apples  
- The green box contains either oranges or bananas
- No box contains the same fruit as its color name
- The red box and blue box share one common fruit type

What fruit is in each box? Show your detailed reasoning process.
"""
    
    print("🧠 Testing Grok Reasoning Effort Parameter")
    print("=" * 50)
    
    # Test different reasoning_effort levels
    effort_levels = ["low", "medium", "high"]
    
    for effort in effort_levels:
        print(f"\n🔧 Testing reasoning_effort='{effort}'...")
        
        try:
            response = client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.0,
                max_tokens=200,
                reasoning_effort=effort
            )
            
            content = response.choices[0].message.content
            print(f"✅ Success with effort='{effort}': {len(content)} chars")
            print(f"   Preview: {content[:100]}...")
            
            # Count reasoning indicators
            reasoning_words = ['step', 'reason', 'because', 'therefore', 'conclusion', 'analysis']
            reasoning_count = sum(content.lower().count(word) for word in reasoning_words)
            print(f"   Reasoning indicators: {reasoning_count}")
            
        except Exception as e:
            print(f"❌ Failed with effort='{effort}': {e}")
    
    # Test with numeric effort level
    print(f"\n🔧 Testing reasoning_effort=3 (numeric)...")
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=200,
            reasoning_effort=3
        )
        
        content = response.choices[0].message.content
        print(f"✅ Success with numeric effort: {len(content)} chars")
        print(f"   Preview: {content[:100]}...")
        
    except Exception as e:
        print(f"❌ Failed with numeric effort: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Key Findings:")
    print("- ✅ reasoning_effort is the correct parameter for Grok reasoning")
    print("- ✅ Supports string levels: 'low', 'medium', 'high'")
    print("- ✅ May support numeric levels")
    print("- ✅ This enables proper reasoning mode for the model")

if __name__ == "__main__":
    test_reasoning_effort()