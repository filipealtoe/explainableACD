#!/usr/bin/env python3
"""
Test script to query Grok 4.1 fast reasoning model and verify its response format.
Focuses on logprobs support and response structure.
"""

import os
import json
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_grok_api():
    """Test Grok 4.1 fast reasoning API with logprobs request."""
    
    # Get API key
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        print("❌ XAI_API_KEY not found in environment variables")
        return
    
    # API endpoint
    api_url = "https://api.x.ai/v1/chat/completions"
    
    # Test prompt
    test_prompt = "Is this claim checkworthy? 'The Earth is flat.'"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {xai_api_key}",
        "Content-Type": "application/json"
    }
    
    # Payload with logprobs request
    payload = {
        "model": "grok-4-1-fast-reasoning",
        "messages": [
            {"role": "user", "content": test_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "logprobs": True,  # Request logprobs
        "top_logprobs": 5  # Get top 5 logprobs per token
    }
    
    print("🔍 Testing Grok 4.1 fast reasoning API...")
    print(f"📋 Prompt: {test_prompt}")
    print(f"🔧 Requesting logprobs: {payload['logprobs']}")
    print(f"🔧 Top logprobs: {payload['top_logprobs']}")
    
    try:
        # Make the API request
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n✅ API Request Successful!")
        print(f"📊 Response structure:")
        
        # Check if logprobs are present
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            
            print(f"  - Model: {result.get('model', 'N/A')}")
            print(f"  - Finish reason: {choice.get('finish_reason', 'N/A')}")
            print(f"  - Response length: {len(choice.get('message', {}).get('content', ''))} chars")
            
            # Check for logprobs
            if "logprobs" in choice:
                logprobs = choice["logprobs"]
                print(f"  - Logprobs present: ✅ YES")
                print(f"  - Logprobs structure: {type(logprobs)}")
                
                if isinstance(logprobs, dict):
                    print(f"  - Logprobs keys: {list(logprobs.keys())}")
                    
                    if "content" in logprobs:
                        content_logprobs = logprobs["content"]
                        print(f"  - Content logprobs length: {len(content_logprobs)}")
                        
                        if content_logprobs:
                            print(f"  - First token logprobs: {content_logprobs[0]}")
                            
                            # Show structure of first token
                            first_token = content_logprobs[0]
                            if isinstance(first_token, dict):
                                print(f"    - Token: {first_token.get('token', 'N/A')}")
                                print(f"    - Logprob: {first_token.get('logprob', 'N/A')}")
                                print(f"    - Top logprobs: {len(first_token.get('top_logprobs', []))}")
                                
                                if first_token.get('top_logprobs'):
                                    print(f"    - Top logprobs sample: {first_token['top_logprobs'][0]}")
            else:
                print(f"  - Logprobs present: ❌ NO")
        
        # Print full response for debugging
        print(f"\n📄 Full response (truncated):")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:1000] + "...")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None

def test_grok_without_logprobs():
    """Test Grok API without logprobs to compare response structure."""
    
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        return
    
    api_url = "https://api.x.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {xai_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "grok-4-1-fast-reasoning",
        "messages": [
            {"role": "user", "content": "Is this claim checkworthy? 'The Earth is flat.'"}
        ],
        "temperature": 0.0,
        "max_tokens": 50
        # No logprobs parameter
    }
    
    print("\n🔍 Testing Grok API without logprobs...")
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("✅ Request without logprobs successful")
        print(f"📊 Response has logprobs: {'logprobs' in result.get('choices', [{}])[0] if result.get('choices') else False}")
        
        return result
        
    except Exception as e:
        print(f"❌ Request without logprobs failed: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting Grok 4.1 Fast Reasoning API Test")
    print("=" * 50)
    
    # Test with logprobs
    result_with_logprobs = test_grok_api()
    
    # Test without logprobs for comparison
    result_without_logprobs = test_grok_without_logprobs()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("- Tested Grok 4.1 fast reasoning API")
    print("- Verified logprobs support")
    print("- Compared response structures")
    
    if result_with_logprobs:
        has_logprobs = "logprobs" in result_with_logprobs.get("choices", [{}])[0]
        print(f"- Logprobs supported: {'✅ YES' if has_logprobs else '❌ NO'}")
    else:
        print("- Could not determine logprobs support")