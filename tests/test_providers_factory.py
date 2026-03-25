from __future__ import annotations

import pytest

from docpipe.config import (
    AnthropicProviderConfig,
    ApiRetryConfig,
    AzureProviderConfig,
    OpenAIProviderConfig,
)
from docpipe.providers import create_provider
from docpipe.providers.anthropic import AnthropicProvider
from docpipe.providers.azure import AzureOpenAIProvider
from docpipe.providers.openai import OpenAIProvider


class TestCreateProvider:
    def test_creates_openai_provider(self) -> None:
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)
        openai_cfg = OpenAIProviderConfig(model="gpt-4o-mini")
        provider = create_provider("openai", openai_cfg, retry_cfg)
        assert isinstance(provider, OpenAIProvider)

    def test_creates_azure_provider(self) -> None:
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)
        azure_cfg = AzureProviderConfig(
            model="gpt-4o-mini",
            endpoint="https://test.openai.azure.com",
            deployment="my-deploy",
        )
        provider = create_provider("azure", azure_cfg, retry_cfg)
        assert isinstance(provider, AzureOpenAIProvider)

    def test_creates_anthropic_provider(self) -> None:
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)
        anthropic_cfg = AnthropicProviderConfig(model="claude-haiku-4-5-20251001")
        provider = create_provider("anthropic", anthropic_cfg, retry_cfg)
        assert isinstance(provider, AnthropicProvider)

    def test_raises_for_unknown_provider(self) -> None:
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)
        openai_cfg = OpenAIProviderConfig()
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("gemini", openai_cfg, retry_cfg)
