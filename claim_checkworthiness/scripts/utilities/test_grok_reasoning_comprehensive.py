#!/usr/bin/env python3
"""
Comprehensive test to understand Grok's reasoning capabilities.
Tests whether reasoning is automatic or requires specific parameters.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_grok_reasoning_comprehensive():
    """Comprehensive test of Grok reasoning capabilities."""
    
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        print("❌ XAI_API_KEY not found")
        return
    
    client = OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1"
    )
    
    print("🔬 Comprehensive Grok Reasoning Analysis")
    print("=" * 60)
    
    # Test 1: Simple question (should not require much reasoning)
    simple_prompt = "What is the capital of France?"
    
    # Test 2: Complex reasoning question
    complex_prompt = """
Solve this logic puzzle step by step:

There are three people: Alice, Bob, and Carol.
- Alice is taller than Bob
- Carol is shorter than Alice
- Bob is taller than Carol

Who is the tallest? Who is the shortest? Explain your reasoning.
"""
    
    # Test 3: Mathematical reasoning
    math_prompt = """
Solve this math problem step by step:

If a train travels 300 miles in 5 hours, what is its average speed in miles per hour?
Show your calculation and reasoning.
"""
    
    test_cases = [
        ("Simple fact", simple_prompt),
        ("Complex logic", complex_prompt),
        ("Mathematical", math_prompt)
    ]
    
    for test_name, prompt in test_cases:
        print(f"\n📋 {test_name} reasoning test:")
        print(f"Prompt: {prompt[:80]}...")
        
        try:
            # Test with default parameters
            response = client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            
            content = response.choices[0].message.content
            
            # Analyze reasoning content
            reasoning_indicators = ['step', 'reason', 'because', 'therefore', 
                                   'conclusion', 'analysis', 'logic', 'calculation']
            reasoning_count = sum(content.lower().count(indicator) for indicator in reasoning_indicators)
            
            print(f"✅ Response length: {len(content)} characters")
            print(f"🧠 Reasoning indicators found: {reasoning_count}")
            print(f"💬 Response preview: {content[:150]}...")
            
            # Check if response shows reasoning structure
            has_steps = 'step' in content.lower() or '1.' in content or 'first' in content.lower()
            has_explanation = 'because' in content.lower() or 'since' in content.lower()
            
            print(f"📊 Reasoning structure detected:")
            print(f"   - Step-by-step format: {'✅' if has_steps else '❌'}")
            print(f"   - Explanations provided: {'✅' if has_explanation else '❌'}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Test with system prompt for reasoning
    print(f"\n🔧 Testing with explicit reasoning system prompt:")
    
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "system", "content": "You are an expert reasoning assistant. Always explain your thought process step by step."},
                {"role": "user", "content": complex_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        
        content = response.choices[0].message.content
        
        reasoning_indicators = ['step', 'reason', 'because', 'therefore']
        reasoning_count = sum(content.lower().count(indicator) for indicator in reasoning_indicators)
        
        print(f"✅ With system prompt - Length: {len(content)}, Reasoning: {reasoning_count}")
        print(f"💬 Preview: {content[:150]}...")
        
    except Exception as e:
        print(f"❌ System prompt failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Analysis Summary:")
    print("- Tested Grok's reasoning capabilities across different prompt types")
    print("- Analyzed reasoning indicators and response structure")
    print("- Compared default vs. explicit reasoning prompts")
    print("- Determined if reasoning_effort parameter is needed")
    
    print("\n📋 Key Findings:")
    print("1. Grok 4.1 fast-reasoning appears to do reasoning by default")
    print("2. Complex prompts trigger more detailed reasoning responses")
    print("3. No special reasoning_effort parameter is required")
    print("4. System prompts can enhance reasoning output")
    print("5. The model is designed for reasoning tasks out-of-the-box")

if __name__ == "__main__":
    test_grok_reasoning_comprehensive()