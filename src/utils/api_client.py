"""
Unified API client supporting multiple providers (OpenAI, OpenRouter, DeepSeek, Hyperbolic, Mistral, xAI).

All providers use OpenAI-compatible API format with different base URLs.
"""

import os
import aiohttp
import asyncio
from typing import Optional, Dict, List, Any


# Provider configurations
PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "hyperbolic": {
        "base_url": "https://api.hyperbolic.xyz/v1",
        "env_key": "HYPERBOLIC_API_KEY",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
    },
}

# Model to provider mapping
MODEL_PROVIDERS = {
    # OpenAI models
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4-turbo": "openai",
    "gpt-4": "openai",
    "gpt-3.5-turbo": "openai",
    "o1": "openai",
    "o1-mini": "openai",
    "o1-preview": "openai",
    "o3-mini": "openai",
    "gpt-5": "openai",
    "gpt-5-2025-08-07": "openai",
    "gpt-5-pro": "openai",
    "gpt-5-pro-2025-10-06": "openai",
    "gpt-5.1": "openai",

    # DeepSeek models (direct)
    "deepseek-chat": "deepseek",
    "deepseek-reasoner": "deepseek",

    # Mistral models (direct)
    "mistral-large-latest": "mistral",
    "mistral-large-2412": "mistral",
    "pixtral-large-latest": "mistral",
    "ministral-3b-latest": "mistral",
    "ministral-8b-latest": "mistral",

    # xAI/Grok models
    "grok-3": "xai",
    "grok-3-mini": "xai",
    "grok-4": "xai",

    # Hyperbolic models
    "meta-llama/Meta-Llama-3.1-405B-Instruct": "hyperbolic",
    "meta-llama/Meta-Llama-3.1-405B": "hyperbolic",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "hyperbolic",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "hyperbolic",
    "Qwen/Qwen2.5-72B-Instruct": "hyperbolic",
    "deepseek-ai/DeepSeek-V3": "hyperbolic",
    "deepseek-ai/DeepSeek-V3-0324": "hyperbolic",
    "deepseek-ai/DeepSeek-R1-0528": "hyperbolic",
    "openai/gpt-oss-120b": "hyperbolic",

    # OpenRouter models - DeepSeek
    "deepseek/deepseek-r1-0528": "openrouter",
    "deepseek/deepseek-r1": "openrouter",
    "deepseek/deepseek-chat": "openrouter",
    "deepseek/deepseek-chat-v3-0324": "openrouter",
    "deepseek/deepseek-chat-v3.1": "openrouter",
    "deepseek/deepseek-v3.1-terminus": "openrouter",
    "deepseek/deepseek-v3.2": "openrouter",
    "deepseek/deepseek-prover-v2": "openrouter",

    # OpenRouter models - Mistral
    "mistralai/mistral-large-2512": "openrouter",
    "mistralai/mistral-large-2411": "openrouter",
    "mistralai/mistral-medium-3.1": "openrouter",
    "mistralai/mistral-small-3.2-24b-instruct": "openrouter",
    "mistralai/ministral-3b-2512": "openrouter",
    "mistralai/ministral-8b-2512": "openrouter",
    "mistralai/ministral-14b-2512": "openrouter",
    "mistralai/pixtral-large-2411": "openrouter",
    "mistralai/codestral-2508": "openrouter",

    # OpenRouter models - Qwen
    "qwen/qwen3-235b-a22b": "openrouter",
    "qwen/qwen3-235b-a22b-2507": "openrouter",
    "qwen/qwen3-max": "openrouter",
    "qwen/qwen3-next-80b-a3b-instruct": "openrouter",
    "qwen/qwen3-next-80b-a3b-thinking": "openrouter",
    "qwen/qwen3-coder": "openrouter",
    "qwen/qwen3-32b": "openrouter",
    "qwen/qwen-max": "openrouter",
    "qwen/qwq-32b": "openrouter",

    # OpenRouter models - Llama
    "meta-llama/llama-4-maverick": "openrouter",
    "meta-llama/llama-4-scout": "openrouter",
    "meta-llama/llama-3.3-70b-instruct": "openrouter",
    "meta-llama/llama-3.1-405b-instruct": "openrouter",

    # OpenRouter models - Claude
    "anthropic/claude-opus-4.5": "openrouter",
    "anthropic/claude-opus-4.1": "openrouter",
    "anthropic/claude-opus-4": "openrouter",
    "anthropic/claude-sonnet-4.5": "openrouter",
    "anthropic/claude-sonnet-4": "openrouter",
    "anthropic/claude-3.7-sonnet": "openrouter",
    "anthropic/claude-haiku-4.5": "openrouter",

    # OpenRouter models - Google
    "google/gemini-2.5-pro": "openrouter",
    "google/gemini-2.5-flash": "openrouter",
    "google/gemini-3-pro-preview": "openrouter",

    # OpenRouter models - Grok
    "x-ai/grok-4": "openrouter",
    "x-ai/grok-4-fast": "openrouter",
    "x-ai/grok-4.1-fast": "openrouter",
    "x-ai/grok-3": "openrouter",
    "x-ai/grok-3-mini": "openrouter",

    # Short aliases
    "llama-405b": "hyperbolic",
    "llama-70b": "hyperbolic",
    "llama-8b": "hyperbolic",
    "qwen-72b": "hyperbolic",
    "deepseek-v3": "hyperbolic",
    "deepseek-r1": "openrouter",
}

# Model name aliases (short -> full)
MODEL_ALIASES = {
    "llama-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "llama-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek/deepseek-r1-0528",
    # Latest models shortcuts
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "mistral-large": "mistralai/mistral-large-2512",
    "qwen3-235b": "qwen/qwen3-235b-a22b",
    "llama4-maverick": "meta-llama/llama-4-maverick",
    "claude-opus": "anthropic/claude-opus-4.5",
    "claude-sonnet": "anthropic/claude-sonnet-4.5",
    "gemini-pro": "google/gemini-2.5-pro",
    "grok4": "x-ai/grok-4",
}


def get_provider(model: str) -> str:
    """Get the provider for a model."""
    if model in MODEL_PROVIDERS:
        return MODEL_PROVIDERS[model]
    # Default to openai for unknown models
    return "openai"


def resolve_model_name(model: str) -> str:
    """Resolve model alias to full name."""
    return MODEL_ALIASES.get(model, model)


class UnifiedAsyncClient:
    """Async client that works with multiple OpenAI-compatible APIs."""

    def __init__(self, api_key: Optional[str] = None, provider: Optional[str] = None,
                 base_url: Optional[str] = None):
        """Initialize the client.

        Args:
            api_key: API key (if None, uses environment variable based on provider)
            provider: Force a specific provider ('openai', 'openrouter', 'deepseek', etc.)
            base_url: Override the base URL (takes precedence over provider)
        """
        self._api_keys = {}
        self._sessions = {}
        self._forced_provider = provider
        self._base_url_override = base_url

        # If api_key is provided, store it for the forced provider or openai by default
        if api_key:
            target_provider = provider or "openai"
            self._api_keys[target_provider] = api_key
            # Also store as default if base_url is provided
            if base_url:
                self._api_keys["_override"] = api_key

    def _get_api_key(self, provider: str) -> str:
        """Get API key for a provider."""
        if provider in self._api_keys:
            return self._api_keys[provider]

        env_key = PROVIDERS[provider]["env_key"]
        api_key = os.environ.get(env_key)
        if not api_key:
            raise ValueError(f"{env_key} not set. Please set environment variable.")

        self._api_keys[provider] = api_key
        return api_key

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if not hasattr(self, '_session') or self._session is None or self._session.closed:
            # Set long timeout for reasoning models like DeepSeek-R1 (can take 5+ minutes)
            timeout = aiohttp.ClientTimeout(total=600)  # 10 minutes
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the session."""
        if hasattr(self, '_session') and self._session and not self._session.closed:
            await self._session.close()

    @property
    def chat(self):
        """Return self for OpenAI-compatible interface (client.chat.completions.create)."""
        return self

    @property
    def completions(self):
        """Return self for OpenAI-compatible interface."""
        return self

    async def create(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Create a chat completion (OpenAI-compatible interface).

        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Response object with OpenAI-compatible structure
        """
        # Resolve model alias
        resolved_model = resolve_model_name(model)

        # Determine provider and base URL
        if self._base_url_override:
            # Use override base URL and api key
            base_url = self._base_url_override
            api_key = self._api_keys.get("_override") or self._api_keys.get(
                self._forced_provider or "openai"
            )
        else:
            provider = self._forced_provider or get_provider(resolved_model)
            api_key = self._get_api_key(provider)
            base_url = PROVIDERS[provider]["base_url"]

        # Build request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": resolved_model,
            "messages": messages,
        }

        # Add optional parameters
        for key in ['temperature', 'max_completion_tokens', 'top_p', 'reasoning_effort']:
            if key in kwargs and kwargs[key] is not None:
                data[key] = kwargs[key]

        url = f"{base_url}/chat/completions"

        session = await self._get_session()

        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API error {response.status}: {error_text}")

            result = await response.json()

        # Return OpenAI-compatible response object
        return ChatCompletionResponse(result)


class ChatCompletionResponse:
    """OpenAI-compatible response object."""

    def __init__(self, data: Dict):
        self._data = data
        self.choices = [ChatCompletionChoice(c) for c in data.get("choices", [])]
        self.model = data.get("model")
        self.usage = data.get("usage", {})

    def __repr__(self):
        return f"ChatCompletionResponse(model={self.model}, choices={len(self.choices)})"


class ChatCompletionChoice:
    """OpenAI-compatible choice object."""

    def __init__(self, data: Dict):
        self._data = data
        self.index = data.get("index", 0)
        self.message = ChatCompletionMessage(data.get("message", {}))
        self.finish_reason = data.get("finish_reason")


class ChatCompletionMessage:
    """OpenAI-compatible message object."""

    def __init__(self, data: Dict):
        self._data = data
        self.role = data.get("role", "assistant")
        self.content = data.get("content", "")
        # DeepSeek R1 reasoning content (thinking traces)
        # OpenRouter uses "reasoning", DeepSeek direct uses "reasoning_content"
        self.reasoning_content = data.get("reasoning_content") or data.get("reasoning", "")

    def __repr__(self):
        return f"ChatCompletionMessage(role={self.role}, content={self.content[:50]}...)"


def create_client(api_key: Optional[str] = None, model: Optional[str] = None,
                  base_url: Optional[str] = None) -> UnifiedAsyncClient:
    """Create a unified async client.

    Args:
        api_key: API key (optional, uses env vars if not provided)
        model: Model name to infer provider (optional)
        base_url: Override base URL (optional, takes precedence over provider detection)

    Returns:
        UnifiedAsyncClient instance
    """
    provider = None
    if model and not base_url:
        resolved_model = resolve_model_name(model)
        provider = get_provider(resolved_model)

    return UnifiedAsyncClient(api_key=api_key, provider=provider, base_url=base_url)


# List available models
def list_models():
    """Print available models and their providers."""
    print("Available models:")
    print("-" * 60)
    for model, provider in sorted(MODEL_PROVIDERS.items()):
        alias = ""
        for short, full in MODEL_ALIASES.items():
            if full == model:
                alias = f" (alias: {short})"
                break
        print(f"  {model:<50} [{provider}]{alias}")
