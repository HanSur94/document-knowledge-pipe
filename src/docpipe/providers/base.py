"""Abstract base class for LLM providers with shared retry logic."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from docpipe.config import ApiRetryConfig

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Common exception wrapping provider-specific SDK errors."""


class LLMProvider(ABC):
    """Unified interface for LLM providers.

    Adapters receive their provider-specific config at construction time
    and store model/deployment names as instance attributes.

    Template method pattern: consumers call public complete()/vision()
    which handle retry. Adapters implement _complete_raw()/_vision_raw().
    """

    def __init__(self, retry_cfg: ApiRetryConfig) -> None:
        self._retry_cfg = retry_cfg

    async def complete(self, prompt: str, max_tokens: int) -> str:
        """Generate a text completion (with retry)."""
        return await self._retry(self._complete_raw, prompt, max_tokens)

    async def vision(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        """Generate a description of an image (with retry)."""
        return await self._retry(self._vision_raw, prompt, image_b64, media_type, max_tokens)

    @abstractmethod
    async def _complete_raw(self, prompt: str, max_tokens: int) -> str:
        """Raw completion call — implemented by each adapter."""
        ...

    @abstractmethod
    async def _vision_raw(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        """Raw vision call — implemented by each adapter."""
        ...

    async def _retry(self, fn: Callable[..., Any], *args: Any) -> str:
        """Exponential backoff retry. Catches ProviderError."""
        delay = self._retry_cfg.initial_delay_seconds
        for attempt in range(self._retry_cfg.max_retries + 1):
            try:
                return await fn(*args)
            except ProviderError:
                if attempt == self._retry_cfg.max_retries:
                    raise
                logger.warning(
                    "Provider call attempt %d failed, retrying in %.1fs...",
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._retry_cfg.max_delay_seconds)
        raise ProviderError("Exhausted retries")  # unreachable — loop re-raises on last attempt
