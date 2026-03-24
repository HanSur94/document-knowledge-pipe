# tests/test_providers_anthropic.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import AnthropicProviderConfig, ApiRetryConfig
from docpipe.providers.anthropic import AnthropicProvider
from docpipe.providers.base import ProviderError


class TestAnthropicProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_content(self) -> None:
        cfg = AnthropicProviderConfig(model="claude-haiku-4-5-20251001")
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)

        mock_text_block = MagicMock()
        mock_text_block.text = "anthropic response"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        with patch("docpipe.providers.anthropic.anthropic_sdk.AsyncAnthropic") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            provider = AnthropicProvider(cfg, retry_cfg)
            result = await provider.complete("hello", 100)

        assert result == "anthropic response"

    @pytest.mark.asyncio
    async def test_complete_wraps_api_error(self) -> None:
        cfg = AnthropicProviderConfig(model="claude-haiku-4-5-20251001")
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)

        import anthropic as anthropic_sdk

        with patch("docpipe.providers.anthropic.anthropic_sdk.AsyncAnthropic") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic_sdk.APIError(
                    message="rate limit",
                    request=MagicMock(),
                    body=None,
                )
            )

            provider = AnthropicProvider(cfg, retry_cfg)
            with pytest.raises(ProviderError, match="rate limit"):
                await provider.complete("hello", 100)


class TestAnthropicProviderVision:
    @pytest.mark.asyncio
    async def test_vision_returns_content(self) -> None:
        cfg = AnthropicProviderConfig(model="claude-haiku-4-5-20251001")
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)

        mock_text_block = MagicMock()
        mock_text_block.text = "image of a chart"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        with patch("docpipe.providers.anthropic.anthropic_sdk.AsyncAnthropic") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            provider = AnthropicProvider(cfg, retry_cfg)
            result = await provider.vision("describe", "base64data", "image/png", 300)

        assert result == "image of a chart"
