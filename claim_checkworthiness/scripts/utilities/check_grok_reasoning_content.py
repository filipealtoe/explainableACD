#!/usr/bin/env python3
"""
Check if Grok 4.1 fast reasoning returns reasoning_content or encrypted reasoning.
Based on the documentation that mentions reasoning_content fields.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_grok_reasoning_fields():
    """Check all possible reasoning fields in Grok response."""
    
    print("🔍 Checking Grok 4.1 Fast Reasoning for Reasoning Content")
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
    
    # Test with a reasoning-heavy prompt
    reasoning_prompt = """
Solve this complex problem step by step:

Problem: A train leaves Station A at 60 mph. Another train leaves Station B at 40 mph, 30 minutes later, traveling towards Station A. The distance between stations is 200 miles. Where and when will they meet?

Show your reasoning process explicitly.
"""
    
    print(f"📋 Reasoning-heavy prompt:")
    print(f"  Type: Math/logic problem requiring step-by-step reasoning")
    print()
    
    try:
        # Make API request
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "user", "content": reasoning_prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        
        print("✅ API Request Successful!")
        print()
        
        # Convert to dict and examine all fields
        response_dict = response.model_dump()
        
        print("🔍 Examining Response for Reasoning Fields:")
        print("-" * 60)
        
        # Check main response structure
        print("Main Response Keys:")
        for key in response_dict.keys():
            print(f"  ✓ {key}")
        print()
        
        # Check choice structure
        choice = response_dict['choices'][0]
        print("Choice Keys:")
        for key in choice.keys():
            print(f"  ✓ {key}")
        print()
        
        # Check message structure
        message = choice['message']
        print("Message Keys:")
        for key in message.keys():
            print(f"  ✓ {key}")
        print()
        
        # Specifically check for reasoning fields mentioned in docs
        reasoning_fields = [
            'reasoning_content',
            'encrypted_content',
            'reasoning_trace',
            'thought_process',
            'internal_reasoning'
        ]
        
        print("🔎 Checking for Documented Reasoning Fields:")
        found_reasoning_fields = []
        
        # Check in message
        for field in reasoning_fields:
            if field in message:
                print(f"  ✅ Found {field} in message!")
                found_reasoning_fields.append(field)
                
                # Show content preview
                content = message[field]
                if isinstance(content, str):
                    print(f"     Content preview: {content[:100]}...")
                else:
                    print(f"     Content type: {type(content)}")
            else:
                print(f"  ❌ {field} not in message")
        
        # Check in choice
        for field in reasoning_fields:
            if field in choice:
                print(f"  ✅ Found {field} in choice!")
                found_reasoning_fields.append(field)
                
                content = choice[field]
                if isinstance(content, str):
                    print(f"     Content preview: {content[:100]}...")
                else:
                    print(f"     Content type: {type(content)}")
        
        # Check in response root
        for field in reasoning_fields:
            if field in response_dict:
                print(f"  ✅ Found {field} in response root!")
                found_reasoning_fields.append(field)
                
                content = response_dict[field]
                if isinstance(content, str):
                    print(f"     Content preview: {content[:100]}...")
                else:
                    print(f"     Content type: {type(content)}")
        
        print()
        
        # Show actual response content
        content = message['content']
        print("💬 Actual Response Content:")
        print("-" * 60)
        print(content)
        print("-" * 60)
        print()
        
        # Check for any additional fields that might contain reasoning
        print("🔎 Checking for Additional Fields:")
        
        # Check all fields in message
        for key, value in message.items():
            if key not in ['role', 'content']:
                print(f"  🔹 Additional message field: {key} (type: {type(value)})")
                if isinstance(value, str) and len(value) > 0:
                    print(f"     Preview: {value[:50]}...")
        
        # Check all fields in choice
        for key, value in choice.items():
            if key not in ['index', 'message', 'finish_reason', 'logprobs']:
                print(f"  🔹 Additional choice field: {key} (type: {type(value)})")
        
        print()
        
        if found_reasoning_fields:
            print("🎉 REASONING FIELDS FOUND:")
            for field in found_reasoning_fields:
                print(f"  ✅ {field}")
        else:
            print("❌ NO REASONING FIELDS FOUND")
            print("   According to docs, grok-4-fast-reasoning doesn't return reasoning_content")
            print("   The reasoning is embedded in the main content, not separate fields")
        
        print()
        print("📋 Documentation Summary:")
        print("  - grok-3-mini: Returns reasoning_content field")
        print("  - grok-3, grok-4, grok-4-fast-reasoning: Do NOT return reasoning_content")
        print("  - grok-4: May return encrypted_content if use_encrypted_content=true")
        print("  - Our finding: Reasoning is in main content, not separate fields")
        
    except Exception as e:
        print(f"❌ API Request Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_grok_reasoning_fields()