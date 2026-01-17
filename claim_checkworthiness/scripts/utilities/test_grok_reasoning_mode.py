#!/usr/bin/env python3
"""
Test Grok 4.1 fast reasoning with proper reasoning mode parameters.
This script tests various reasoning-related parameters to ensure we're actually
using the reasoning capabilities of the model.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_grok_reasoning_parameters():
    """Test various reasoning parameters for Grok 4.1 fast reasoning."""
    
    print("🧠 Testing Grok 4.1 Fast Reasoning Mode Parameters")
    print("=" * 60)
    
    # Get API key
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        print("❌ XAI_API_KEY not found in environment variables")
        return
    
    # Initialize client
    client = OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1"
    )
    
    # Test prompt that should trigger reasoning
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
    
    print("📋 Test Setup:")
    print("  - Model: grok-4-1-fast-reasoning")
    print("  - Prompt: Complex logic puzzle requiring step-by-step reasoning")
    print()
    
    # Test different reasoning parameter combinations
    parameter_tests = [
        {
            "name": "Default (no reasoning params)",
            "params": {}
        },
        {
            "name": "With reasoning_mode=true",
            "params": {"reasoning_mode": True}
        },
        {
            "name": "With reasoning_mode='auto'",
            "params": {"reasoning_mode": "auto"}
        },
        {
            "name": "With reasoning_mode='full'",
            "params": {"reasoning_mode": "full"}
        },
        {
            "name": "With reasoning=true",
            "params": {"reasoning": True}
        },
        {
            "name": "With use_reasoning=true",
            "params": {"use_reasoning": True}
        },
        {
            "name": "With reasoning_steps=true",
            "params": {"reasoning_steps": True}
        },
        {
            "name": "With chain_of_thought=true",
            "params": {"chain_of_thought": True}
        }
    ]
    
    results = []
    
    for test_config in parameter_tests:
        print(f"🔧 Testing: {test_config['name']}")
        
        try:
            # Build request parameters
            request_params = {
                "model": "grok-4-1-fast-reasoning",
                "messages": [{"role": "user", "content": test_prompt}],
                "temperature": 0.0,
                "max_tokens": 150
            }
            
            # Add test-specific parameters
            request_params.update(test_config['params'])
            
            # Make API request
            response = client.chat.completions.create(**request_params)
            
            # Convert to dict for analysis
            response_dict = response.model_dump()
            choice = response_dict['choices'][0]
            message = choice['message']
            content = message['content']
            
            # Analyze response
            result = {
                "test_name": test_config['name'],
                "success": True,
                "content_length": len(content),
                "response_keys": list(response_dict.keys()),
                "choice_keys": list(choice.keys()),
                "message_keys": list(message.keys()),
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
            
            # Check for reasoning-related fields
            reasoning_fields = ['reasoning', 'reasoning_content', 'reasoning_steps', 
                              'thought_process', 'chain_of_thought', 'reasoning_trace']
            
            found_fields = []
            for field in reasoning_fields:
                if field in message:
                    found_fields.append(field)
                if field in choice:
                    found_fields.append(f"{field}_in_choice")
                if field in response_dict:
                    found_fields.append(f"{field}_in_response")
            
            result['reasoning_fields'] = found_fields
            
            # Count reasoning indicators in content
            reasoning_indicators = ['step', 'reason', 'because', 'therefore', 
                                   'conclusion', 'analysis', 'logic']
            reasoning_count = sum(content.lower().count(indicator) 
                                for indicator in reasoning_indicators)
            result['reasoning_indicators'] = reasoning_count
            
            print(f"  ✅ Success - Content length: {len(content)}, Reasoning indicators: {reasoning_count}")
            if found_fields:
                print(f"  🎯 Found reasoning fields: {found_fields}")
            
            results.append(result)
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Failed: {error_msg}")
            
            results.append({
                "test_name": test_config['name'],
                "success": False,
                "error": error_msg
            })
        
        print()
    
    # Summary analysis
    print("📊 Summary Analysis:")
    print("=" * 60)
    
    successful_tests = [r for r in results if r.get('success')]
    failed_tests = [r for r in results if not r.get('success')]
    
    print(f"Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"Failed tests: {len(failed_tests)}/{len(results)}")
    
    if failed_tests:
        print("\n❌ Failed parameter tests:")
        for test in failed_tests:
            print(f"  - {test['test_name']}: {test['error']}")
    
    if successful_tests:
        print("\n✅ Successful parameter tests:")
        for test in successful_tests:
            print(f"  - {test['test_name']}")
            print(f"    Content length: {test['content_length']}")
            print(f"    Reasoning indicators: {test['reasoning_indicators']}")
            if test['reasoning_fields']:
                print(f"    Reasoning fields: {test['reasoning_fields']}")
            print()
    
    # Find the best reasoning response
    if successful_tests:
        best_reasoning = max(successful_tests, key=lambda x: x['reasoning_indicators'])
        print(f"🎯 Best reasoning response: {best_reasoning['test_name']}")
        print(f"   Reasoning indicators: {best_reasoning['reasoning_indicators']}")
        print(f"   Content preview: {best_reasoning['content_preview'][:100]}...")
    
    return results

if __name__ == "__main__":
    results = test_grok_reasoning_parameters()
    
    print("\n" + "=" * 60)
    print("🎉 Grok Reasoning Mode Test Completed!")
    print("\nKey Findings:")
    print("- Tested multiple reasoning parameter combinations")
    print("- Identified which parameters work vs. which fail")
    print("- Found the best configuration for reasoning-heavy tasks")