from .anthropic_client import AnthropicClient
from .base import LLMClient, LLMResponse, LLMToolCall
from .litellm_client import LiteLLMClient
from .openai_client import OpenAIClient


def make_client(provider: str, model: str) -> LLMClient:
    p = provider.lower().strip()
    if p == "openai":
        return OpenAIClient(model=model)
    if p == "anthropic":
        return AnthropicClient(model=model)
    if p == "litellm":
        return LiteLLMClient(model=model)
    raise ValueError(f"unknown provider {provider!r}")


__all__ = [
    "AnthropicClient",
    "LiteLLMClient",
    "LLMClient",
    "LLMResponse",
    "LLMToolCall",
    "OpenAIClient",
    "make_client",
]
