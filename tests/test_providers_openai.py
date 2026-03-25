from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import ApiRetryConfig, OpenAIProviderConfig
from docpipe.providers.base import ProviderError
from docpipe.providers.openai import OpenAIProvider


class TestOpenAIProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_content(self) -> None:
        cfg = OpenAIProviderConfig(model="gpt-4o-mini")
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"

        with patch("docpipe.providers.openai.openai.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            provider = OpenAIProvider(cfg, retry_cfg)
            result = await provider.complete("hello", 100)

        assert result == "test response"

    @pytest.mark.asyncio
    async def test_complete_wraps_api_error(self) -> None:
        cfg = OpenAIProviderConfig(model="gpt-4o-mini")
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)

        import openai

        with patch("docpipe.providers.openai.openai.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                side_effect=openai.APIError(
                    message="rate limit", request=MagicMock(), body=None
                )
            )

            provider = OpenAIProvider(cfg, retry_cfg)
            with pytest.raises(ProviderError, match="rate limit"):
                await provider.complete("hello", 100)


class TestOpenAIProviderVision:
    @pytest.mark.asyncio
    async def test_vision_returns_content(self) -> None:
        cfg = OpenAIProviderConfig(model="gpt-4o-mini")
        retry_cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "image of a cat"

        with patch("docpipe.providers.openai.openai.AsyncOpenAI") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            provider = OpenAIProvider(cfg, retry_cfg)
            result = await provider.vision("describe", "base64data", "image/png", 300)

        assert result == "image of a cat"
