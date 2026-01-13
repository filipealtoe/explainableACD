# Logprobs Verification & Reasoning Analysis Scripts

This directory contains scripts to verify logprobs support and reasoning capabilities for the temperature experiment in the IJCAI paper.

## Scripts

### 1. `verify_logprobs.py`

**Purpose**: Comprehensive verification of logprobs support across all candidate models.

**Usage**:
```bash
python experiments/scripts/verify_logprobs.py
```

**Features**:
- Tests 13 models across 6 providers
- Validates actual API responses (not just documentation claims)
- Provides detailed summary with experiment role information
- Generates markdown table for documentation
- Gives specific recommendation for temperature experiment

**Models Tested**:
- **OpenAI**: gpt-4o, gpt-4.1, gpt-4.1-mini, o3, o4-mini
- **DeepSeek**: deepseek-chat, deepseek-reasoner  
- **xAI**: grok-3-beta, grok-4.1-fast-reasoning
- **Google**: gemini-2.5-flash
- **Moonshot**: kimi-k2-thinking
- **Mistral**: mistral-large-3

### 2. `test_grok_simple.py`

**Purpose**: Simple test specifically for Grok 4.1 fast reasoning model.

**Usage**:
```bash
python experiments/scripts/test_grok_simple.py
```

**Features**:
- Focuses only on Grok 4.1 fast reasoning
- Shows detailed logprobs structure
- Displays token-level probabilities
- Quick validation for API connectivity

### 3. `test_grok_reasoning_final.py`

**Purpose**: Comprehensive test of Grok reasoning parameters.

**Usage**:
```bash
python experiments/scripts/test_grok_reasoning_final.py
```

**Features**:
- Tests multiple reasoning parameter approaches
- Identifies which parameters are supported by Grok API
- Compares default vs. explicit reasoning configurations
- Provides clear summary of findings

### 4. `test_grok_reasoning_effort.py`

**Purpose**: Test the `reasoning_effort` parameter specifically.

**Usage**:
```bash
python experiments/scripts/test_grok_reasoning_effort.py
```

**Features**:
- Tests different reasoning effort levels (low, medium, high)
- Confirms parameter support status
- Shows API error messages for documentation

### 5. `test_grok_reasoning_comprehensive.py`

**Purpose**: Complete analysis of Grok's reasoning capabilities.

**Usage**:
```bash
python experiments/scripts/test_grok_reasoning_comprehensive.py
```

**Features**:
- Tests simple vs. complex reasoning prompts
- Analyzes reasoning indicators and structure
- Compares default vs. system prompt reasoning
- Provides definitive answer about reasoning mode

### 3. `test_grok_api.py`

**Purpose**: Advanced testing of Grok API with raw HTTP requests.

**Usage**:
```bash
python experiments/scripts/test_grok_api.py
```

**Features**:
- Uses raw requests instead of OpenAI client
- Shows complete response structure
- Tests both with and without logprobs
- Useful for debugging API issues

## Requirements

### API Keys

Create a `.env` file in the project root with these keys:

```env
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
XAI_API_KEY=xai-...
GEMINI_API_KEY=...
MOONSHOT_API_KEY=...
MISTRAL_API_KEY=...
```

### Python Dependencies

```bash
pip install openai python-dotenv requests
```

## Temperature Experiment Requirements

The temperature experiment requires:

1. **Logprobs Support**: For calibration metrics (ECE, Brier score)
2. **5 Models**: For meaningful comparisons
3. **Diverse Characteristics**: Size, thinking mode, architecture differences

### Target Models for Experiment

| Model                   | Provider  | Role                    |
|-------------------------|-----------|-------------------------|
| gpt-4.1                 | OpenAI    | Standard LLM (large)    |
| gpt-4.1-mini            | OpenAI    | Standard LLM (small)    |
| deepseek-chat           | DeepSeek  | Open-weight baseline    |
| deepseek-reasoner       | DeepSeek  | Same model, thinking ON |
| grok-4.1-fast-reasoning | xAI       | Reasoning architecture  |

## Expected Results

Based on our research:

✅ **Should support logprobs**:
- gpt-4o, gpt-4.1, gpt-4.1-mini
- deepseek-chat, deepseek-reasoner (via beta API)
- grok-4.1-fast-reasoning

❌ **Should NOT support logprobs**:
- o3, o4-mini (OpenAI reasoning models)
- gemini-2.5-flash (Google endpoint limitation)
- mistral-large-3 (official API blocks logprobs)

⚠️ **Complex cases**:
- kimi-k2-thinking (returns logprobs but includes reasoning tokens)

## Troubleshooting

### API Key Issues

If you get authentication errors:
1. Check your `.env` file has the correct keys
2. Verify keys are not expired
3. Check for typos in environment variable names

### Rate Limiting

If you get rate limit errors:
1. Add delays between requests
2. Use smaller test prompts
3. Check your API plan limits

### Connection Issues

If you get connection errors:
1. Check your internet connection
2. Verify API endpoints are correct
3. Test with a simple curl command first

## Output Interpretation

- ✅ **Logprobs supported**: Model can be used in temperature experiment
- ❌ **No logprobs**: Model cannot be used for calibration metrics
- ⚪ **N/A**: Could not test due to API errors
- 🧠 **Reasoning model**: Expected to not have logprobs (by design)

## Next Steps

After running verification:

1. **If all 5 models support logprobs**: Proceed with temperature experiment
2. **If some models fail**: Investigate API issues or consider alternatives
3. **If major issues**: Review model selection and experiment design

## 🧠 Grok Reasoning Analysis - Key Findings

**The user was correct** - I initially wasn't properly testing Grok's reasoning mode. However, comprehensive testing revealed:

### ✅ What Works
- **Default parameters**: Grok 4.1 fast-reasoning does reasoning automatically
- **Complex prompts**: Trigger detailed step-by-step reasoning responses
- **System prompts**: Can enhance reasoning but aren't required

### ❌ What Doesn't Work
- **`reasoning_mode=True`**: Parameter not supported by Grok API
- **`reasoning_effort="high"`**: Parameter not supported by this model variant
- **`prompt_mode="reasoning"`**: Parameter not supported
- **`chain_of_thought=True`**: Parameter not supported

### 🎯 Key Insight
The "fast-reasoning" in the model name indicates it's specifically designed for reasoning tasks. Reasoning capability is built into the model architecture and triggers automatically for complex prompts.

### 📋 Conclusion
**Grok 4.1 fast-reasoning is correctly configured for the temperature experiment.** No parameter changes are needed - the model automatically provides reasoning when the task requires it.

See `GROK_REASONING_ANALYSIS.md` for complete details.

## 🔥 DeepSeek Documentation Analysis - Critical Findings

**The DeepSeek documentation is COMPLETELY WRONG on critical points.** Comprehensive testing revealed:

### ✅ What Actually Works (Despite Documentation)
- **✅ logprobs DO work** - No errors, proper structure returned
- **✅ temperature DOES work** - Significantly affects output
- **✅ reasoning_content provided** - Separate from final answer
- **✅ Both models fully compatible** - Can be used in experiment

### ❌ What Documentation Claims (WRONG)
- **❌ "logprobs trigger errors"** → FALSE, works perfectly
- **❌ "temperature has no effect"** → FALSE, works perfectly
- **❌ Implies no logprobs for reasoning models** → FALSE, full support

### 🎯 Key Insight
DeepSeek provides **two separate fields** that are perfect for our task:
- **`content`**: Final classification (YES/NO/UNCERTAIN)
- **`reasoning_content`**: Chain-of-thought reasoning process

### 📋 Conclusion
**DeepSeek models are FULLY COMPATIBLE with temperature experiment requirements.** The documentation inaccuracies were significant but testing confirmed all required features work perfectly.

See `DEEPSEEK_ANALYSIS.md` for complete details.