#!/usr/bin/env python3
"""
Simple script to show Grok 4.1 fast reasoning logprobs in a clear format.
Focuses ONLY on the Grok model and displays the logprobs structure.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_grok_logprobs():
    """Test Grok 4.1 fast reasoning and show logprobs clearly."""
    
    print("🔍 Testing Grok 4.1 Fast Reasoning Logprobs")
    print("=" * 50)
    
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
    
    # Test prompt
    test_prompt = "Is this claim checkworthy? 'The Earth is flat.'"
    
    print(f"📋 Test prompt: {test_prompt}")
    print(f"🔧 Parameters: logprobs=True, top_logprobs=3, temperature=0.0")
    print()
    
    try:
        # Make API request
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.0,
            max_tokens=20,  # Short response for clear demonstration
            logprobs=True,
            top_logprobs=3
        )
        
        print("✅ API Request Successful!")
        print()
        
        # Extract response
        choice = response.choices[0]
        content = choice.message.content
        
        print("💬 Response Content:")
        print(f"  {content}")
        print()
        
        # Check logprobs
        if choice.logprobs and choice.logprobs.content:
            print("✅ LOGPROBS SUPPORTED: YES")
            print()
            
            content_logprobs = choice.logprobs.content
            print(f"📊 Total tokens with logprobs: {len(content_logprobs)}")
            print()
            
            # Show first 5 tokens in detail
            print("🔍 Token-by-Token Logprobs (first 5 tokens):")
            print("-" * 60)
            
            for i, token_logprob in enumerate(content_logprobs[:5], 1):
                print(f"Token {i}:")
                print(f"  Token: '{token_logprob.token}'")
                print(f"  Logprob: {token_logprob.logprob:.6f}")
                
                if token_logprob.top_logprobs:
                    print(f"  Top alternatives:")
                    for j, alt in enumerate(token_logprob.top_logprobs[:3], 1):
                        prob_percent = 100 * (2.718281828 ** alt.logprob)
                        print(f"    {j}. '{alt.token}': {prob_percent:.2f}% (logprob: {alt.logprob:.4f})")
                print()
            
            # Show if there are more tokens
            if len(content_logprobs) > 5:
                print(f"⏭️  ... and {len(content_logprobs) - 5} more tokens")
            
            print("-" * 60)
            print()
            print("🎉 CONCLUSION: Grok 4.1 fast reasoning supports logprobs!")
            print("✅ Ready for temperature experiment calibration metrics")
            
        else:
            print("❌ LOGPROBS SUPPORTED: NO")
            print(f"  choice.logprobs: {choice.logprobs}")
            
    except Exception as e:
        print(f"❌ API Request Failed: {e}")

if __name__ == "__main__":
    test_grok_logprobs()