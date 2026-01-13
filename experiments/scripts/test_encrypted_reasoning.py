#!/usr/bin/env python3
"""
Test Grok 4.1 fast reasoning with use_encrypted_content parameter.
This tests the documentation claim that grok-4 models may return encrypted_content.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_encrypted_reasoning():
    """Test use_encrypted_content parameter with Grok 4.1 fast reasoning."""
    
    print("🔐 Testing Grok 4.1 Fast Reasoning with use_encrypted_content")
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

What fruit is in each box? Show your reasoning.
"""
    
    print("📋 Test Setup:")
    print("  - Model: grok-4-1-fast-reasoning")
    print("  - Parameter: use_encrypted_content=true")
    print("  - Prompt: Complex logic puzzle requiring reasoning")
    print()
    
    try:
        # Test WITH use_encrypted_content=true
        print("🔒 Testing WITH use_encrypted_content=true...")
        
        response_with_encrypted = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.0,
            max_tokens=100,
            use_encrypted_content=True  # This is the key parameter
        )
        
        print("✅ Request with use_encrypted_content=true successful!")
        
        # Convert to dict
        response_dict = response_with_encrypted.model_dump()
        choice = response_dict['choices'][0]
        message = choice['message']
        
        print()
        print("🔍 Checking for encrypted_content field:")
        
        # Check all possible locations for encrypted_content
        encrypted_found = False
        
        # Check in message
        if 'encrypted_content' in message:
            print("  ✅ FOUND encrypted_content in message!")
            encrypted_found = True
            encrypted_content = message['encrypted_content']
            print(f"     Type: {type(encrypted_content)}")
            print(f"     Length: {len(encrypted_content) if isinstance(encrypted_content, str) else 'N/A'}")
            if isinstance(encrypted_content, str):
                print(f"     Preview: {encrypted_content[:50]}...")
        else:
            print("  ❌ encrypted_content not in message")
        
        # Check in choice
        if 'encrypted_content' in choice:
            print("  ✅ FOUND encrypted_content in choice!")
            encrypted_found = True
            encrypted_content = choice['encrypted_content']
            print(f"     Type: {type(encrypted_content)}")
        else:
            print("  ❌ encrypted_content not in choice")
        
        # Check in response root
        if 'encrypted_content' in response_dict:
            print("  ✅ FOUND encrypted_content in response root!")
            encrypted_found = True
            encrypted_content = response_dict['encrypted_content']
            print(f"     Type: {type(encrypted_content)}")
        else:
            print("  ❌ encrypted_content not in response root")
        
        print()
        
        # Show actual response content
        content = message['content']
        print("💬 Response Content (with use_encrypted_content=true):")
        print("-" * 60)
        print(content[:500] + "..." if len(content) > 500 else content)
        print("-" * 60)
        print()
        
        # Now test WITHOUT use_encrypted_content for comparison
        print("🔄 Testing WITHOUT use_encrypted_content (default)...")
        
        response_without_encrypted = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.0,
            max_tokens=100
            # No use_encrypted_content parameter
        )
        
        print("✅ Request without use_encrypted_content successful!")
        
        response_dict_no_enc = response_without_encrypted.model_dump()
        choice_no_enc = response_dict_no_enc['choices'][0]
        message_no_enc = choice_no_enc['message']
        
        content_no_enc = message_no_enc['content']
        print("💬 Response Content (without use_encrypted_content):")
        print("-" * 60)
        print(content_no_enc[:500] + "..." if len(content_no_enc) > 500 else content_no_enc)
        print("-" * 60)
        print()
        
        # Compare responses
        print("🔎 Comparison Analysis:")
        print(f"  With encrypted_content: {len(content)} characters")
        print(f"  Without encrypted_content: {len(content_no_enc)} characters")
        print(f"  Content identical: {content == content_no_enc}")
        print()
        
        if encrypted_found:
            print("🎉 DOCUMENTATION CONFIRMED:")
            print("  ✅ use_encrypted_content parameter works")
            print("  ✅ Grok 4.1 fast reasoning DOES support encrypted reasoning")
            print("  ✅ Documentation is accurate")
        else:
            print("🤔 FINDING:")
            print("  ❌ use_encrypted_content parameter accepted but no encrypted_content returned")
            print("  ❌ Either:")
            print("     1. Documentation is incorrect for this model")
            print("     2. Encrypted content only available for grok-4 (not grok-4-fast-reasoning)")
            print("     3. Requires special access or additional parameters")
            print("     4. Only available in certain regions/data centers")
        
        # Check if there are any differences in response structure
        print()
        print("🔍 Full Structure Comparison:")
        
        # Get all keys from both responses
        all_keys_with = set(response_dict.keys())
        all_keys_without = set(response_dict_no_enc.keys())
        
        if all_keys_with != all_keys_without:
            print("  🔹 Response keys differ!")
            print(f"     With encrypted: {all_keys_with - all_keys_without}")
            print(f"     Without encrypted: {all_keys_without - all_keys_with}")
        else:
            print("  ✅ Response keys identical")
        
        # Check message keys
        msg_keys_with = set(message.keys())
        msg_keys_without = set(message_no_enc.keys())
        
        if msg_keys_with != msg_keys_without:
            print("  🔹 Message keys differ!")
            print(f"     With encrypted: {msg_keys_with - msg_keys_without}")
            print(f"     Without encrypted: {msg_keys_without - msg_keys_with}")
        else:
            print("  ✅ Message keys identical")
        
    except Exception as e:
        print(f"❌ API Request Failed: {e}")
        
        # Check if it's a parameter error
        if "use_encrypted_content" in str(e):
            print("🔴 PARAMETER ERROR: use_encrypted_content not supported")
            print("   This would confirm documentation is wrong for grok-4-fast-reasoning")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_encrypted_reasoning()