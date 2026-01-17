"""Configuration for Checkworthiness pipeline."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelProvider(Enum):
    """Supported model providers."""

    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    MOONSHOT = "moonshot"
    TOGETHER_AI = "together_ai"


@dataclass
class SamplingDefaults:
    """Default sampling parameters by provider.

    These are the provider defaults when not specified.
    Documented here for reference.
    """

    # OpenAI defaults
    # https://platform.openai.com/docs/api-reference/chat/create
    OPENAI = {
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        # top_k, min_p, top_a not supported
    }

    # DeepSeek defaults
    # https://api-docs.deepseek.com/
    DEEPSEEK = {
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        # top_k, min_p, top_a not documented
    }

    # xAI/Grok defaults
    # https://docs.x.ai/api
    XAI = {
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        # top_k supported but default unknown
    }

    # Moonshot/Kimi defaults
    # https://platform.moonshot.cn/docs
    MOONSHOT = {
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }

    # Together AI defaults
    # https://docs.together.ai/reference/chat-completions
    TOGETHER_AI = {
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        # Together supports top_k but default varies by model
    }


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: ModelProvider
    model_name: str
    api_key_env: str
    api_base: str | None = None

    # Generation parameters
    max_tokens: int = 1024

    # Sampling parameters (provider defaults used if None)
    # top_p: nucleus sampling (default 1.0 for most providers)
    # top_k: not supported by OpenAI, varies by provider
    # min_p: not widely supported
    # top_a: not widely supported

    # Penalty parameters (provider defaults used)
    # repetition_penalty: not in OpenAI API, some providers support
    # presence_penalty: default 0.0
    # frequency_penalty: default 0.0

    # Response format
    response_format: str = "text"  # "text" or "json_object"

    # Logprobs settings
    supports_logprobs: bool = True
    logprobs: bool = True  # Request logprobs
    top_logprobs: int = 5  # Number of top logprobs to return (max 20 for OpenAI)

    # Reasoning model settings
    is_thinking_model: bool = False
    reasoning_effort: str = "medium"  # "low", "medium", "high"
    include_reasoning: bool = True  # Include reasoning in response

    # Prefill support (assistant message continuation)
    # Some models (especially older instruction-tuned models) don't follow assistant prefill
    supports_prefill: bool = True

    # API parameter naming (newer OpenAI models use max_completion_tokens instead of max_tokens)
    uses_max_completion_tokens: bool = False

    # Disabled features (always None/False)
    tools: None = None
    parallel_tool_calls: None = None
    web_search_options: None = None

    # Cost per 1M tokens (USD)
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

    def get_api_key(self) -> str | None:
        """Get the API key from environment."""
        return os.getenv(self.api_key_env)

    def get_provider_defaults(self) -> dict:
        """Get default sampling parameters for this provider."""
        defaults_map = {
            ModelProvider.OPENAI: SamplingDefaults.OPENAI,
            ModelProvider.DEEPSEEK: SamplingDefaults.DEEPSEEK,
            ModelProvider.XAI: SamplingDefaults.XAI,
            ModelProvider.MOONSHOT: SamplingDefaults.MOONSHOT,
            ModelProvider.TOGETHER_AI: SamplingDefaults.TOGETHER_AI,
        }
        return defaults_map.get(self.provider, SamplingDefaults.OPENAI)

    def get_api_params(self, temperature: float) -> dict[str, Any]:
        """Get parameters for API call."""
        params: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

        # Add logprobs if supported
        if self.supports_logprobs and self.logprobs:
            params["logprobs"] = True
            params["top_logprobs"] = self.top_logprobs

        # Reasoning model parameters
        if self.is_thinking_model:
            # Provider-specific reasoning params
            if self.provider == ModelProvider.DEEPSEEK:
                # DeepSeek uses different param names
                pass  # Handled in response parsing
            elif self.provider == ModelProvider.MOONSHOT:
                # Kimi K2 reasoning settings
                pass  # Handled in response parsing

        return params


# Pre-defined model configurations
MODELS: dict[str, ModelConfig] = {
    # ==========================================================================
    # OpenAI Models
    # ==========================================================================
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        api_base=None,
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        api_base=None,
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.15,  # GPT-4o-mini pricing
        cost_per_1m_output=0.60,
    ),
    "gpt-4.1": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4.1",
        api_key_env="OPENAI_API_KEY",
        api_base=None,
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=2.00,
        cost_per_1m_output=8.00,
    ),
    "gpt-5.2": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-5.2-2025-12-11",  # Correct model ID
        api_key_env="OPENAI_API_KEY",
        api_base=None,
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        uses_max_completion_tokens=True,  # GPT-5 uses new API parameter name
        cost_per_1m_input=3.00,  # Estimated pricing
        cost_per_1m_output=12.00,
    ),
    "gpt-4.1-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
        api_base=None,
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.40,
        cost_per_1m_output=1.60,
    ),
    "gpt-4-turbo-2024-04-09": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4-turbo-2024-04-09",
        api_key_env="OPENAI_API_KEY",
        api_base=None,
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=10.00,  # GPT-4 Turbo pricing
        cost_per_1m_output=30.00,
    ),
    # --- GPT-3.5 Turbo (cheaper, Sep 2021 cutoff - safe for CT24) ---
    "gpt-3.5-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo-0125",  # Latest 3.5, Sep 2021 cutoff
        api_key_env="OPENAI_API_KEY",
        api_base=None,
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.50,
        cost_per_1m_output=1.50,
    ),
    # ==========================================================================
    # DeepSeek Models
    # ==========================================================================
    "deepseek-v3.2": ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-chat",  # API model name
        api_key_env="DEEPSEEK_API_KEY",
        api_base="https://api.deepseek.com",
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=True,
        reasoning_effort="medium",
        include_reasoning=True,
        cost_per_1m_input=0.27,
        cost_per_1m_output=1.10,
    ),
    # ==========================================================================
    # xAI Grok Models
    # ==========================================================================
    "grok-4.1": ModelConfig(
        provider=ModelProvider.XAI,
        model_name="grok-4.1",
        api_key_env="XAI_API_KEY",
        api_base="https://api.x.ai/v1",
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
    ),
    # ==========================================================================
    # Moonshot Kimi Models
    # ==========================================================================
    "kimi-k2": ModelConfig(
        provider=ModelProvider.MOONSHOT,
        model_name="kimi-k2-thinking",  # K2 Long-term thinking model, 256k context
        api_key_env="MOONSHOT_API_KEY",
        api_base="https://api.moonshot.ai/v1",
        max_tokens=1024,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=True,
        reasoning_effort="medium",
        include_reasoning=True,
        cost_per_1m_input=0.60,
        cost_per_1m_output=2.40,
    ),
    # ==========================================================================
    # Together AI Models (Open-weight models via Together API)
    # https://docs.together.ai/docs/chat-models
    # Note: Only "Turbo" models are serverless. Non-Turbo models require dedicated endpoints.
    # Llama 2 and Falcon models are no longer available on Together AI.
    # ==========================================================================
    # --- Llama 3.1 Turbo Models (serverless, best performance) ---
    "llama-3.1-8b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # Serverless
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.18,
        cost_per_1m_output=0.18,
    ),
    "llama-3.1-70b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",  # Serverless
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.88,
        cost_per_1m_output=0.88,
    ),
    "llama-3.3-70b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Serverless, latest
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.88,
        cost_per_1m_output=0.88,
    ),
    # --- Llama 3.2 Small Models (serverless, for completeness) ---
    "llama-3.2-3b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="meta-llama/Llama-3.2-3B-Instruct-Turbo",  # Serverless, smallest
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.06,
        cost_per_1m_output=0.06,
    ),
    "llama-3.2-1b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="meta-llama/Llama-3.2-1B-Instruct",  # Smallest Llama
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.06,
        cost_per_1m_output=0.06,
    ),
    # --- Mistral Models ---
    # Note: Mistral instruction models don't reliably follow assistant prefill
    "mistral-7b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        supports_prefill=False,  # Mistral doesn't follow assistant prefill reliably
        cost_per_1m_input=0.20,
        cost_per_1m_output=0.20,
    ),
    "mixtral-8x7b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        supports_prefill=False,  # Mixtral doesn't follow assistant prefill reliably
        cost_per_1m_input=0.60,
        cost_per_1m_output=0.60,
    ),
    # --- Gemma Models ---
    # NOTE: gemma-3n (cutoff June 2024) excluded due to data contamination risk
    # CT24 dataset released January 2024 - model may have seen test data
    # --- Qwen Models ---
    "qwen-2.5-7b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.30,
        cost_per_1m_output=0.30,
    ),
    "qwen-2.5-14b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="Qwen/Qwen2.5-14B-Instruct",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.80,
        cost_per_1m_output=0.80,
    ),
    "qwen-2.5-72b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="Qwen/Qwen2.5-72B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=1.20,
        cost_per_1m_output=1.20,
    ),
    # --- DeepSeek Models via Together AI ---
    "deepseek-v3": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="deepseek-ai/DeepSeek-V3",  # Non-reasoning version
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=1.25,
        cost_per_1m_output=1.25,
    ),
    "deepseek-v3.1": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="deepseek-ai/DeepSeek-V3.1",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.90,
        cost_per_1m_output=0.90,
    ),
    # --- Additional Llama Models ---
    "llama-3.1-405b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=3.50,
        cost_per_1m_output=3.50,
    ),
    "llama-3-8b-lite": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.10,
    ),
    # --- Additional Mistral Models ---
    "mistral-7b-v0.3": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        supports_prefill=False,
        cost_per_1m_input=0.20,
        cost_per_1m_output=0.20,
    ),
    "mistral-small-24b": ModelConfig(
        provider=ModelProvider.TOGETHER_AI,
        model_name="mistralai/Mistral-Small-24B-Instruct-2501",
        api_key_env="TOGETHER_API_KEY",
        api_base="https://api.together.xyz/v1",
        max_tokens=2048,
        supports_logprobs=True,
        logprobs=True,
        top_logprobs=5,
        is_thinking_model=False,
        supports_prefill=False,
        cost_per_1m_input=0.80,
        cost_per_1m_output=0.80,
    ),
}


@dataclass
class TokenUsage:
    """Track token usage for a single call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def calculate_cost(self, config: ModelConfig) -> float:
        """Calculate cost in USD."""
        input_cost = (self.prompt_tokens / 1_000_000) * config.cost_per_1m_input
        output_cost = ((self.completion_tokens + self.reasoning_tokens) / 1_000_000) * config.cost_per_1m_output
        return input_cost + output_cost


@dataclass
class ExperimentStats:
    """Aggregate statistics for an experiment."""

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_usage(self, usage: TokenUsage, config: ModelConfig) -> None:
        """Add token usage from a call."""
        self.total_calls += 1
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_reasoning_tokens += usage.reasoning_tokens
        self.total_cost_usd += usage.calculate_cost(config)

    def summary(self) -> dict:
        """Return summary as dictionary."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens + self.total_reasoning_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
        }


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""

    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.2  # Can be overridden per experiment
    threshold: float = 50.0
    train_subset_size: int = 500
    use_baml_adapter: bool = True
    gepa_budget: str = "medium"
    random_seed: int = 42
    verbose: bool = True

    # Paths
    data_dir: str = "data/raw/CT24_checkworthy_english"
    results_dir: str = "experiments/results/checkworthiness"

    def get_model_config(self) -> ModelConfig:
        """Get the model configuration."""
        if self.model_name not in MODELS:
            raise ValueError(f"Unknown model: {self.model_name}. Available: {list(MODELS.keys())}")
        return MODELS[self.model_name]

    def print_config(self) -> None:
        """Print the experiment configuration."""
        model_config = self.get_model_config()
        defaults = model_config.get_provider_defaults()

        print("\n" + "=" * 70)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 70)

        print("\n[Model]")
        print(f"  Name: {self.model_name}")
        print(f"  Provider: {model_config.provider.value}")
        print(f"  API Base: {model_config.api_base or 'default (OpenAI)'}")
        print(f"  Is Reasoning Model: {model_config.is_thinking_model}")

        print("\n[Generation Parameters]")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max Tokens: {model_config.max_tokens}")
        print(f"  Response Format: {model_config.response_format}")

        print("\n[Sampling Defaults (provider)]")
        print(f"  top_p: {defaults.get('top_p', 'N/A')}")
        print(f"  presence_penalty: {defaults.get('presence_penalty', 'N/A')}")
        print(f"  frequency_penalty: {defaults.get('frequency_penalty', 'N/A')}")

        print("\n[Logprobs]")
        print(f"  Enabled: {model_config.logprobs}")
        print(f"  Top Logprobs: {model_config.top_logprobs}")

        if model_config.is_thinking_model:
            print("\n[Reasoning Settings]")
            print(f"  Reasoning Effort: {model_config.reasoning_effort}")
            print(f"  Include Reasoning: {model_config.include_reasoning}")

        print("\n[Disabled Features]")
        print("  Tools: None")
        print("  Parallel Tool Calls: None")
        print("  Web Search: None")

        print("\n[Cost]")
        print(f"  Input (per 1M tokens): ${model_config.cost_per_1m_input}")
        print(f"  Output (per 1M tokens): ${model_config.cost_per_1m_output}")

        print("\n[Experiment]")
        print(f"  Threshold: {self.threshold}")
        print(f"  BAML Adapter: {self.use_baml_adapter}")

        print("=" * 70 + "\n")
