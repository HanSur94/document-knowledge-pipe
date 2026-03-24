"""LLM provider abstraction layer."""

from __future__ import annotations

from typing import Union

from docpipe.config import (
    AnthropicProviderConfig,
    ApiRetryConfig,
    AzureProviderConfig,
    OpenAIProviderConfig,
)
from docpipe.providers.anthropic import AnthropicProvider
from docpipe.providers.azure import AzureOpenAIProvider
from docpipe.providers.base import LLMProvider, ProviderError
from docpipe.providers.openai import OpenAIProvider

__all__ = ["LLMProvider", "ProviderError", "create_provider"]

ProviderConfig = Union[OpenAIProviderConfig, AzureProviderConfig, AnthropicProviderConfig]


def create_provider(
    provider: str,
    cfg: ProviderConfig,
    retry_cfg: ApiRetryConfig,
) -> LLMProvider:
    """Create the appropriate LLMProvider based on the provider name."""
    if provider == "openai":
        return OpenAIProvider(cfg, retry_cfg)  # type: ignore[arg-type]
    elif provider == "azure":
        return AzureOpenAIProvider(cfg, retry_cfg)  # type: ignore[arg-type]
    elif provider == "anthropic":
        return AnthropicProvider(cfg, retry_cfg)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Must be 'openai', 'azure', or 'anthropic'.")
