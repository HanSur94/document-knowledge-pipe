from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import ApiRetryConfig, AzureProviderConfig
from docpipe.providers.azure import AzureOpenAIProvider
from docpipe.providers.base import ProviderError


class TestAzureOpenAIProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_content(self) -> None:
        cfg = AzureProviderConfig(
            model="gpt-4o-mini",
            endpoint="https://test.openai.azure.com",
            deployment="my-deploy",
            api_version="2024-06-01",
        )
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "azure response"

        with patch("docpipe.providers.azure.openai.AsyncAzureOpenAI") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            provider = AzureOpenAIProvider(cfg, retry_cfg)
            result = await provider.complete("hello", 100)

        assert result == "azure response"

    @pytest.mark.asyncio
    async def test_complete_wraps_api_error(self) -> None:
        cfg = AzureProviderConfig(
            model="gpt-4o-mini",
            endpoint="https://test.openai.azure.com",
            deployment="my-deploy",
        )
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)

        import openai

        with patch("docpipe.providers.azure.openai.AsyncAzureOpenAI") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                side_effect=openai.APIError(
                    message="forbidden", request=MagicMock(), body=None
                )
            )

            provider = AzureOpenAIProvider(cfg, retry_cfg)
            with pytest.raises(ProviderError, match="forbidden"):
                await provider.complete("hello", 100)


class TestAzureOpenAIProviderVision:
    @pytest.mark.asyncio
    async def test_vision_uses_deployment_as_model(self) -> None:
        cfg = AzureProviderConfig(
            model="gpt-4o-mini",
            endpoint="https://test.openai.azure.com",
            deployment="my-vision-deploy",
        )
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "azure vision result"

        with patch("docpipe.providers.azure.openai.AsyncAzureOpenAI") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            provider = AzureOpenAIProvider(cfg, retry_cfg)
            result = await provider.vision("describe", "base64", "image/png", 300)

        assert result == "azure vision result"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-vision-deploy"
