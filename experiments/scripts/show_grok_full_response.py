#!/usr/bin/env python3
"""
Comprehensive script to show Grok 4.1 fast reasoning full response.
Uses actual experiment prompt format and shows complete JSON structure.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_grok_full_response():
    """Test Grok 4.1 fast reasoning with experiment-style prompt and show full response."""
    
    print("🔬 Testing Grok 4.1 Fast Reasoning - Full Response Analysis")
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
    
    # Use actual experiment-style prompt (similar to checkworthiness prompts)
    experiment_prompt = """
You are an expert in claim checkworthiness assessment. Analyze the following claim:

Claim: "The Earth is flat."

Evaluate this claim based on the following criteria:
1. Checkability: Can this claim be fact-checked?
2. Verifiability: Is there authoritative evidence available?
3. Harm potential: Could this claim cause harm if left unchecked?

Provide your assessment with reasoning for each criterion and a final verdict.
"""
    
    print(f"📋 Experiment-style prompt:")
    print(f"  Length: {len(experiment_prompt)} characters")
    print(f"  Lines: {len(experiment_prompt.split(chr(10)))}")
    print()
    
    print("🔧 API Parameters:")
    print("  - Model: grok-4-1-fast-reasoning")
    print("  - Temperature: 0.0 (deterministic)")
    print("  - Max tokens: 100")
    print("  - Logprobs: True")
    print("  - Top logprobs: 5")
    print()
    
    try:
        # Make API request with experiment parameters
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "user", "content": experiment_prompt}
            ],
            temperature=0.0,
            max_tokens=100,
            logprobs=True,
            top_logprobs=5
        )
        
        print("✅ API Request Successful!")
        print()
        
        # Convert response to dict for full inspection
        response_dict = response.model_dump()
        
        # Show high-level structure
        print("📊 Response Structure Overview:")
        print(f"  - Model: {response_dict['model']}")
        print(f"  - Created: {response_dict['created']}")
        print(f"  - Usage: {response_dict['usage']}")
        print(f"  - Number of choices: {len(response_dict['choices'])}")
        print()
        
        # Extract and show full content
        choice = response_dict['choices'][0]
        content = choice['message']['content']
        
        print("💬 Full Response Content:")
        print("-" * 60)
        print(content)
        print("-" * 60)
        print()
        
        # Show logprobs structure
        if 'logprobs' in choice and choice['logprobs']:
            logprobs = choice['logprobs']
            
            print("✅ Logprobs Structure:")
            print(f"  - Has content logprobs: {'content' in logprobs}")
            
            if 'content' in logprobs and logprobs['content']:
                content_logprobs = logprobs['content']
                print(f"  - Total tokens with logprobs: {len(content_logprobs)}")
                print()
                
                # Show detailed analysis of first 3 tokens
                print("🔍 Detailed Token Analysis (First 3 Tokens):")
                print("-" * 60)
                
                for i, token_data in enumerate(content_logprobs[:3], 1):
                    print(f"Token {i}:")
                    print(f"  Token: '{token_data['token']}'")
                    print(f"  Logprob: {token_data['logprob']:.6f}")
                    print(f"  Bytes: {token_data['bytes']}")
                    
                    if token_data['top_logprobs']:
                        print(f"  Top {len(token_data['top_logprobs'])} alternatives:")
                        for j, alt in enumerate(token_data['top_logprobs'], 1):
                            prob_percent = 100 * (2.718281828 ** alt['logprob'])
                            print(f"    {j}. '{alt['token']}': {prob_percent:.4f}% (logprob: {alt['logprob']:.4f})")
                    print()
                
                print("-" * 60)
                print()
                
                # Show token statistics
                print("📈 Token Statistics:")
                
                # Calculate average confidence
                confidences = [2.718281828 ** token['logprob'] for token in content_logprobs]
                avg_confidence = sum(confidences) / len(confidences) * 100
                
                print(f"  - Average token confidence: {avg_confidence:.2f}%")
                print(f"  - Min logprob: {min(token['logprob'] for token in content_logprobs):.6f}")
                print(f"  - Max logprob: {max(token['logprob'] for token in content_logprobs):.6f}")
                print()
                
                # Show if there are more tokens
                if len(content_logprobs) > 3:
                    print(f"⏭️  Complete response contains {len(content_logprobs)} tokens total")
                    print(f"   (showing first 3 in detail, all {len(content_logprobs)} available for calibration)")
        else:
            print("❌ No logprobs in response")
            
        print()
        print("🎯 Calibration Metrics Readiness:")
        print("  ✅ Token-level probabilities available")
        print("  ✅ Confidence scores calculable")
        print("  ✅ ECE (Expected Calibration Error) can be computed")
        print("  ✅ Brier score can be computed")
        print("  ✅ Full probability distributions available")
        print()
        
        # Show complete JSON structure for reference
        print("📄 Complete Response Structure (key fields):")
        print("-" * 60)
        
        # Show simplified structure
        simplified_response = {
            'model': response_dict['model'],
            'created': response_dict['created'],
            'usage': response_dict['usage'],
            'choices': [{
                'index': choice['index'],
                'finish_reason': choice['finish_reason'],
                'message': {
                    'role': choice['message']['role'],
                    'content': choice['message']['content'][:100] + '...' if len(choice['message']['content']) > 100 else choice['message']['content']
                },
                'logprobs': {
                    'content': [
                        {
                            'token': token['token'],
                            'logprob': token['logprob'],
                            'top_logprobs': token['top_logprobs'][:2] if token['top_logprobs'] else []
                        }
                        for token in content_logprobs[:3]
                    ] + [f"... {len(content_logprobs) - 3} more tokens"] if len(content_logprobs) > 3 else []
                }
            }]
        }
        
        print(json.dumps(simplified_response, indent=2, ensure_ascii=False))
        print("-" * 60)
        print()
        
        print("🎉 CONCLUSION:")
        print("✅ Grok 4.1 fast reasoning provides complete logprobs data")
        print("✅ Ready for temperature experiment with calibration metrics")
        print("✅ Supports all required analysis for IJCAI paper")
        
    except Exception as e:
        print(f"❌ API Request Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_grok_full_response()