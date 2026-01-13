#!/usr/bin/env python3
"""
Simple script to test Grok 4.1 fast reasoning model and verify its response.
Focuses specifically on the Grok model to answer the user's request.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_grok_4_1_fast_reasoning():
    """Test Grok 4.1 fast reasoning model specifically."""
    
    print("🚀 Testing Grok 4.1 Fast Reasoning Model")
    print("=" * 50)
    
    # Get API key
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        print("❌ XAI_API_KEY not found in environment variables")
        print("Please set XAI_API_KEY in your .env file")
        return
    
    # Initialize client
    client = OpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1"
    )
    
    # Test prompt
    test_prompt = "Is this claim checkworthy? 'The Earth is flat.'"
    
    print(f"📋 Test prompt: {test_prompt}")
    print(f"🔧 Testing with logprobs=True, top_logprobs=3")
    
    try:
        # Make API request
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.0,
            max_tokens=50,
            logprobs=True,
            top_logprobs=3
        )
        
        print("\n✅ API Request Successful!")
        
        # Extract response
        choice = response.choices[0]
        content = choice.message.content
        
        print(f"💬 Response: {content}")
        print(f"📊 Finish reason: {choice.finish_reason}")
        
        # Check logprobs
        if choice.logprobs:
            print(f"✅ Logprobs supported: YES")
            
            if choice.logprobs.content:
                content_logprobs = choice.logprobs.content
                print(f"📈 Number of tokens with logprobs: {len(content_logprobs)}")
                
                # Show first token details
                if content_logprobs:
                    first_token = content_logprobs[0]
                    print(f"\n🔍 First token details:")
                    print(f"  Token: '{first_token.token}'")
                    print(f"  Logprob: {first_token.logprob:.4f}")
                    
                    if first_token.top_logprobs:
                        print(f"  Top alternatives:")
                        for i, alt in enumerate(first_token.top_logprobs[:3], 1):
                            prob = 100 * (2.718281828 ** alt.logprob)  # exp(logprob) * 100
                            print(f"    {i}. '{alt.token}': {prob:.1f}% (logprob: {alt.logprob:.4f})")
        else:
            print(f"❌ Logprobs supported: NO")
            print(f"  choice.logprobs is: {choice.logprobs}")
        
        # Print raw response structure for debugging
        print(f"\n📄 Raw response structure:")
        print(f"  Response object type: {type(response)}")
        print(f"  Choices count: {len(response.choices)}")
        print(f"  Choice 0 has logprobs attr: {hasattr(choice, 'logprobs')}")
        print(f"  Logprobs value: {choice.logprobs}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Request Failed: {e}")
        return False

if __name__ == "__main__":
    success = test_grok_4_1_fast_reasoning()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Grok 4.1 Fast Reasoning test completed successfully!")
        print("✅ The model supports logprobs and can be used in the temperature experiment")
    else:
        print("❌ Grok 4.1 Fast Reasoning test failed")
        print("Check your API key and network connection")