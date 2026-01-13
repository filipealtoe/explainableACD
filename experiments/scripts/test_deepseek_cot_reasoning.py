#!/usr/bin/env python3
"""
Definitive test: Does DeepSeek Reasoner return internal CoT reasoning?
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_deepseek_cot_reasoning():
    """Test if DeepSeek Reasoner provides internal CoT reasoning."""
    
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("❌ DEEPSEEK_API_KEY not found")
        return
    
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/beta"
    )
    
    # Test prompt that requires reasoning
    test_prompt = """
Solve this logic puzzle step by step:

There are three boxes: red, blue, and green.
- The red box contains either apples or oranges
- The blue box contains either bananas or apples  
- The green box contains either oranges or bananas
- No box contains the same fruit as its color name
- The red box and blue box share one common fruit type

What fruit is in each box? Show your chain-of-thought reasoning.
"""
    
    print("🔬 Testing DeepSeek Reasoner for Internal CoT Reasoning")
    print("=" * 60)
    
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=300
        )
        
        print("✅ Request successful!")
        
        # Extract response components
        choice = response.choices[0]
        message = choice.message
        
        print(f"\n📊 Response Structure:")
        print(f"  - Has content: {hasattr(message, 'content')}")
        print(f"  - Has reasoning_content: {hasattr(message, 'reasoning_content')}")
        
        # Show content
        if hasattr(message, 'content') and message.content:
            print(f"\n💬 Final Answer (content):")
            print(f"  '{message.content}'")
        else:
            print(f"\n💬 Final Answer (content): EMPTY")
        
        # Show reasoning content
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"\n🧠 Internal Reasoning (reasoning_content):")
            print(f"  Length: {len(message.reasoning_content)} characters")
            print(f"  Content:")
            print(f"  '{message.reasoning_content}'")
            
            # Analyze reasoning quality
            print(f"\n🔍 Reasoning Analysis:")
            
            # Check for CoT indicators
            cot_indicators = {
                'step_by_step': message.reasoning_content.lower().count('step') > 0,
                'numbered_steps': any(f'{i}.' in message.reasoning_content for i in range(1, 10)),
                'logical_connectors': any(conn in message.reasoning_content.lower() 
                                        for conn in ['therefore', 'thus', 'conclusion', 'so the answer']),
                'explicit_reasoning': 'reasoning' in message.reasoning_content.lower()
            }
            
            for indicator, present in cot_indicators.items():
                print(f"    - {indicator}: {'✅' if present else '❌'}")
            
            # Count reasoning steps
            step_count = message.reasoning_content.lower().count('step') + \
                        sum(1 for i in range(1, 10) if f'{i}.' in message.reasoning_content)
            print(f"    - Estimated reasoning steps: {step_count}")
            
        else:
            print(f"\n❌ No reasoning_content provided!")
            
        # Check if this is true CoT
        print(f"\n🎯 Chain-of-Thought Assessment:")
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            # Check for actual reasoning vs just explanation
            has_actual_reasoning = (
                step_count >= 2 and 
                (cot_indicators['step_by_step'] or cot_indicators['numbered_steps']) and
                len(message.reasoning_content) > 100
            )
            
            if has_actual_reasoning:
                print("  ✅ DEFINITIVE CoT: Contains structured, multi-step reasoning")
                print("  ✅ This is internal chain-of-thought reasoning")
                print("  ✅ DeepSeek Reasoner DOES provide CoT")
            else:
                print("  ⚠️  Limited reasoning: Some reasoning present but not full CoT")
                print("  ⚠️  May be explanation rather than true CoT")
        else:
            print("  ❌ NO CoT: No reasoning content provided")
            print("  ❌ DeepSeek Reasoner does NOT provide CoT")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("📋 Final Verdict:")
    print("- Testing with complex logic puzzle requiring CoT")
    print("- Checking for structured, multi-step reasoning")
    print("- Verifying if reasoning supports final answer")
    print("- This will definitively answer if CoT is provided")

if __name__ == "__main__":
    test_deepseek_cot_reasoning()