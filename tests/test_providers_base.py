from __future__ import annotations

import pytest

from docpipe.config import ApiRetryConfig
from docpipe.providers.base import LLMProvider, ProviderError


class TestProviderError:
    def test_is_exception(self) -> None:
        err = ProviderError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"


class _StubProvider(LLMProvider):
    """Concrete stub for testing the ABC."""

    def __init__(self, retry_cfg: ApiRetryConfig, responses: list[str | ProviderError]) -> None:
        super().__init__(retry_cfg)
        self._responses = list(responses)
        self._call_count = 0

    async def _complete_raw(self, prompt: str, max_tokens: int) -> str:
        return self._next()

    async def _vision_raw(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        return self._next()

    def _next(self) -> str:
        self._call_count += 1
        val = self._responses.pop(0)
        if isinstance(val, ProviderError):
            raise val
        return val


class TestLLMProviderRetry:
    @pytest.mark.asyncio
    async def test_complete_returns_on_first_success(self) -> None:
        cfg = ApiRetryConfig(max_retries=2, initial_delay_seconds=0, max_delay_seconds=0)
        provider = _StubProvider(cfg, ["hello"])
        result = await provider.complete("prompt", 100)
        assert result == "hello"
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_complete_retries_on_provider_error(self) -> None:
        cfg = ApiRetryConfig(max_retries=2, initial_delay_seconds=0, max_delay_seconds=0)
        provider = _StubProvider(cfg, [ProviderError("fail"), "ok"])
        result = await provider.complete("prompt", 100)
        assert result == "ok"
        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_complete_raises_after_exhausting_retries(self) -> None:
        cfg = ApiRetryConfig(max_retries=1, initial_delay_seconds=0, max_delay_seconds=0)
        provider = _StubProvider(cfg, [ProviderError("fail1"), ProviderError("fail2")])
        with pytest.raises(ProviderError, match="fail2"):
            await provider.complete("prompt", 100)

    @pytest.mark.asyncio
    async def test_vision_delegates_to_retry(self) -> None:
        cfg = ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0)
        provider = _StubProvider(cfg, ["desc"])
        result = await provider.vision("prompt", "base64data", "image/png", 300)
        assert result == "desc"
