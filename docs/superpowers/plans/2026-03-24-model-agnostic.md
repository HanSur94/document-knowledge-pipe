# Model-Agnostic Provider Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scattered if/else provider dispatch with a clean ABC + adapter pattern supporting OpenAI, Azure OpenAI, and Anthropic.

**Architecture:** New `providers/` module with `LLMProvider` ABC, three adapter implementations, and a factory function. Consumers (`describer.py`, `registry.py`) receive a provider instance instead of making direct SDK calls. Config switches from flat `provider`/`model` fields to nested per-provider blocks. Graph module adds Azure support via env var configuration.

**Tech Stack:** Python 3.11+, openai SDK, anthropic SDK, dataclasses, asyncio

**Spec:** `docs/superpowers/specs/2026-03-24-model-agnostic-design.md`

---

### Task 1: Provider Base Class and Error Type

**Files:**
- Create: `src/docpipe/providers/__init__.py`
- Create: `src/docpipe/providers/base.py`
- Create: `tests/test_providers_base.py`

- [ ] **Step 1: Write the failing test for ProviderError and LLMProvider**

```python
# tests/test_providers_base.py
from __future__ import annotations

import asyncio

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docpipe.providers'`

- [ ] **Step 3: Write the implementation**

```python
# src/docpipe/providers/__init__.py
"""LLM provider abstraction layer."""

from docpipe.providers.base import LLMProvider, ProviderError

__all__ = ["LLMProvider", "ProviderError", "create_provider"]
```

```python
# src/docpipe/providers/base.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers_base.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/providers/__init__.py src/docpipe/providers/base.py tests/test_providers_base.py
git commit -m "feat: add LLMProvider ABC with retry logic and ProviderError"
```

---

### Task 2: OpenAI Adapter

**Files:**
- Create: `src/docpipe/providers/openai.py`
- Create: `tests/test_providers_openai.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_openai.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers_openai.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docpipe.providers.openai'` or `ImportError`

- [ ] **Step 3: Write the implementation**

```python
# src/docpipe/providers/openai.py
"""OpenAI provider adapter."""

from __future__ import annotations

import os

import openai

from docpipe.config import ApiRetryConfig, OpenAIProviderConfig
from docpipe.providers.base import LLMProvider, ProviderError


class OpenAIProvider(LLMProvider):
    """LLM provider using the OpenAI API."""

    def __init__(self, cfg: OpenAIProviderConfig, retry_cfg: ApiRetryConfig) -> None:
        super().__init__(retry_cfg)
        self._model = cfg.model
        self._client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )

    async def _complete_raw(self, prompt: str, max_tokens: int) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except openai.APIError as e:
            raise ProviderError(str(e)) from e

    async def _vision_raw(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content or ""
        except openai.APIError as e:
            raise ProviderError(str(e)) from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers_openai.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/providers/openai.py tests/test_providers_openai.py
git commit -m "feat: add OpenAI provider adapter"
```

---

### Task 3: Azure OpenAI Adapter

**Files:**
- Create: `src/docpipe/providers/azure.py`
- Create: `tests/test_providers_azure.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_azure.py
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
        MockClient.assert_called_once_with(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-06-01",
        )

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers_azure.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/docpipe/providers/azure.py
"""Azure OpenAI provider adapter."""

from __future__ import annotations

import os

import openai

from docpipe.config import ApiRetryConfig, AzureProviderConfig
from docpipe.providers.base import LLMProvider, ProviderError


class AzureOpenAIProvider(LLMProvider):
    """LLM provider using Azure OpenAI."""

    def __init__(self, cfg: AzureProviderConfig, retry_cfg: ApiRetryConfig) -> None:
        super().__init__(retry_cfg)
        self._model = cfg.deployment  # Azure uses deployment name as model
        self._client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=cfg.endpoint,
            api_version=cfg.api_version,
        )

    async def _complete_raw(self, prompt: str, max_tokens: int) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except openai.APIError as e:
            raise ProviderError(str(e)) from e

    async def _vision_raw(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content or ""
        except openai.APIError as e:
            raise ProviderError(str(e)) from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers_azure.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/providers/azure.py tests/test_providers_azure.py
git commit -m "feat: add Azure OpenAI provider adapter"
```

---

### Task 4: Anthropic Adapter

**Files:**
- Create: `src/docpipe/providers/anthropic.py`
- Create: `tests/test_providers_anthropic.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers_anthropic.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/docpipe/providers/anthropic.py
"""Anthropic provider adapter."""

from __future__ import annotations

import os
from typing import Literal, cast

import anthropic as anthropic_sdk

from docpipe.config import AnthropicProviderConfig, ApiRetryConfig
from docpipe.providers.base import LLMProvider, ProviderError


class AnthropicProvider(LLMProvider):
    """LLM provider using the Anthropic API."""

    def __init__(self, cfg: AnthropicProviderConfig, retry_cfg: ApiRetryConfig) -> None:
        super().__init__(retry_cfg)
        self._model = cfg.model
        self._client = anthropic_sdk.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    async def _complete_raw(self, prompt: str, max_tokens: int) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            first_block = response.content[0]
            return first_block.text if hasattr(first_block, "text") else ""
        except anthropic_sdk.APIError as e:
            raise ProviderError(str(e)) from e

    async def _vision_raw(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            anthropic_sdk.types.ImageBlockParam(
                                type="image",
                                source=anthropic_sdk.types.Base64ImageSourceParam(
                                    type="base64",
                                    media_type=cast(
                                        Literal[
                                            "image/jpeg",
                                            "image/png",
                                            "image/gif",
                                            "image/webp",
                                        ],
                                        media_type,
                                    ),
                                    data=image_b64,
                                ),
                            ),
                            anthropic_sdk.types.TextBlockParam(type="text", text=prompt),
                        ],
                    }
                ],
            )
            first_block = response.content[0]
            return first_block.text if hasattr(first_block, "text") else ""
        except anthropic_sdk.APIError as e:
            raise ProviderError(str(e)) from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers_anthropic.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/providers/anthropic.py tests/test_providers_anthropic.py
git commit -m "feat: add Anthropic provider adapter"
```

---

### Task 5: Factory Function and Provider Config Dataclasses

**Files:**
- Modify: `src/docpipe/config.py`
- Modify: `src/docpipe/providers/__init__.py`
- Create: `tests/test_providers_factory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_factory.py
from __future__ import annotations

from unittest.mock import patch

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers_factory.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_provider'` or `cannot import name 'OpenAIProviderConfig'`

- [ ] **Step 3: Add config dataclasses to `config.py`**

Add these new dataclasses to `src/docpipe/config.py` (before `DescriberConfig`):

```python
@dataclass
class OpenAIProviderConfig:
    model: str = "gpt-4o-mini"


@dataclass
class AzureProviderConfig:
    model: str = "gpt-4o-mini"
    endpoint: str = ""
    deployment: str = ""
    api_version: str = "2024-06-01"


@dataclass
class AnthropicProviderConfig:
    model: str = "claude-haiku-4-5-20251001"
```

- [ ] **Step 4: Write the factory function in `providers/__init__.py`**

```python
# src/docpipe/providers/__init__.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_providers_factory.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/docpipe/config.py src/docpipe/providers/__init__.py tests/test_providers_factory.py
git commit -m "feat: add provider config dataclasses and factory function"
```

---

### Task 6: Update Config Dataclasses and Deserialization

**Files:**
- Modify: `src/docpipe/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test for nested config loading**

Add to `tests/test_config.py`:

```python
class TestNestedProviderConfig:
    def test_loads_nested_describer_openai_config(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "describer": {
                "provider": "openai",
                "max_tokens": 300,
                "openai": {"model": "gpt-4o-mini"},
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        cfg = load_config(cfg_path)
        assert cfg.describer.provider == "openai"
        assert cfg.describer.openai.model == "gpt-4o-mini"

    def test_loads_nested_describer_azure_config(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "describer": {
                "provider": "azure",
                "max_tokens": 300,
                "azure": {
                    "model": "gpt-4o-mini",
                    "endpoint": "https://test.openai.azure.com",
                    "deployment": "my-deploy",
                    "api_version": "2024-06-01",
                },
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        cfg = load_config(cfg_path)
        assert cfg.describer.azure.endpoint == "https://test.openai.azure.com"
        assert cfg.describer.azure.deployment == "my-deploy"

    def test_loads_nested_graph_config(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "graph": {
                "provider": "openai",
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        cfg = load_config(cfg_path)
        assert cfg.graph.openai.embedding_model == "text-embedding-3-small"
        assert cfg.graph.openai.embedding_dim == 1536

    def test_defaults_work_without_nested_blocks(self, tmp_path: Path) -> None:
        """Minimal config still works — nested provider blocks default."""
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        cfg = load_config(cfg_path)
        assert cfg.describer.openai.model == "gpt-4o-mini"
        assert cfg.graph.openai.model == "gpt-4o-mini"


class TestConfigValidation:
    def test_azure_describer_missing_endpoint_raises(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "describer": {
                "provider": "azure",
                "azure": {"model": "gpt-4o-mini"},  # missing endpoint and deployment
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        with pytest.raises(ValueError, match="endpoint"):
            load_config(cfg_path)

    def test_graph_anthropic_raises(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "graph": {"provider": "anthropic"},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        with pytest.raises(ValueError, match="[Aa]nthropic.*not supported"):
            load_config(cfg_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestNestedProviderConfig -v`
Expected: FAIL — `AttributeError: 'DescriberConfig' object has no attribute 'openai'`

- [ ] **Step 3: Update `config.py`**

Update `DescriberConfig` to use nested provider configs (remove flat `model` field):

```python
@dataclass
class DescriberConfig:
    provider: str = "openai"
    max_tokens: int = 300
    include_context: bool = True
    context_chars: int = 500
    batch_size: int = 5
    openai: OpenAIProviderConfig = field(default_factory=OpenAIProviderConfig)
    azure: AzureProviderConfig = field(default_factory=AzureProviderConfig)
    anthropic: AnthropicProviderConfig = field(default_factory=AnthropicProviderConfig)
```

Add graph provider configs and update `GraphConfig`:

```python
@dataclass
class GraphOpenAIConfig:
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536


@dataclass
class GraphAzureConfig:
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    endpoint: str = ""
    deployment: str = ""
    embedding_deployment: str = ""
    api_version: str = "2024-06-01"


@dataclass
class GraphConfig:
    provider: str = "openai"
    storage: str = "file"
    store_dir: str = "./output/lightrag_store"
    max_tokens: int = 4096
    chunk_size: int = 1200
    chunk_overlap: int = 100
    openai: GraphOpenAIConfig = field(default_factory=GraphOpenAIConfig)
    azure: GraphAzureConfig = field(default_factory=GraphAzureConfig)
```

Update `DocpipeConfig` — remove `openai_api_key` and `anthropic_api_key` fields.

Update `_merge_dataclass` to handle nested dataclasses:

```python
import typing

def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    """Create a dataclass instance, recursively deserializing nested dataclass fields."""
    if data is None:
        return cls()
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    filtered: dict[str, Any] = {}
    for k, v in data.items():
        if k not in valid_fields:
            continue
        field_type = hints.get(k)
        if field_type and dataclasses.is_dataclass(field_type) and isinstance(v, dict):
            filtered[k] = _merge_dataclass(field_type, v)
        else:
            filtered[k] = v
    return cls(**filtered)
```

Update `load_config` — remove `openai_api_key`/`anthropic_api_key` lines, add validation:

```python
def _validate_config(cfg: DocpipeConfig) -> None:
    """Validate provider-specific required fields."""
    if cfg.describer.provider == "azure":
        az = cfg.describer.azure
        if not az.endpoint or not az.deployment:
            raise ValueError(
                "Azure describer requires 'endpoint' and 'deployment' in describer.azure config"
            )
    if cfg.graph.provider == "azure":
        az = cfg.graph.azure
        if not az.endpoint or not az.deployment or not az.embedding_deployment:
            raise ValueError(
                "Azure graph requires 'endpoint', 'deployment', and 'embedding_deployment' in graph.azure config"
            )
    if cfg.graph.provider == "anthropic":
        raise ValueError("Anthropic is not supported for graph — no embedding API")
```

Call `_validate_config(cfg)` at the end of `load_config()` before returning.

- [ ] **Step 4: Add deprecation warning for old flat config**

In `_merge_dataclass`, log a warning when known deprecated flat keys are found:

```python
import logging
_config_logger = logging.getLogger(__name__)

_DEPRECATED_FLAT_KEYS = {
    "DescriberConfig": {"model"},
    "GraphConfig": {"model", "embedding_model"},
}
```

In `_merge_dataclass`, after the `for k, v` loop, add:

```python
deprecated = _DEPRECATED_FLAT_KEYS.get(cls.__name__, set())
for k in data:
    if k in deprecated and k not in valid_fields:
        _config_logger.warning(
            "Deprecated config: '%s' under '%s' is ignored. Use nested provider blocks instead.",
            k,
            cls.__name__,
        )
```

- [ ] **Step 5: Update existing tests in `test_config.py`**

Update `test_load_from_yaml_file`:

```python
def test_load_from_yaml_file(self, config_file: Path, sample_config: dict[str, Any]) -> None:
    cfg = load_config(config_file)
    assert cfg.input_dir == Path(sample_config["input_dir"])
    assert cfg.output_dir == Path(sample_config["output_dir"])
    assert cfg.watcher.debounce_seconds == 1
    assert cfg.describer.openai.model == "gpt-4o-mini"
```

Update `test_load_with_defaults`:

```python
def test_load_with_defaults(self, tmp_path: Path) -> None:
    """Minimal YAML should fill defaults for all missing fields."""
    minimal = {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(minimal))
    cfg = load_config(cfg_path)
    assert cfg.watcher.debounce_seconds == 60
    assert cfg.graph.openai.embedding_model == "text-embedding-3-small"
    assert cfg.api_retry.max_retries == 3
```

Remove `test_env_override_for_api_key` entirely (keys no longer stored in config).

Update `conftest.py`'s `sample_config` fixture — change to nested format:

```python
"describer": {
    "provider": "openai",
    "max_tokens": 300,
    "include_context": True,
    "context_chars": 500,
    "batch_size": 5,
    "openai": {"model": "gpt-4o-mini"},
},
"graph": {
    "provider": "openai",
    "storage": "file",
    "store_dir": str(tmp_dirs["output"] / "lightrag_store"),
    "max_tokens": 4096,
    "chunk_size": 1200,
    "chunk_overlap": 100,
    "openai": {
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
    },
},
```

- [ ] **Step 5: Run all config tests**

Run: `pytest tests/test_config.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/docpipe/config.py tests/test_config.py tests/conftest.py
git commit -m "feat: update config with nested provider blocks and validation"
```

---

### Task 7: Refactor `describer.py` to Use Provider

**Files:**
- Modify: `src/docpipe/describer.py`
- Modify: `tests/test_describer.py`

- [ ] **Step 1: Write the failing test with the new signature**

Replace `tests/test_describer.py` tests that use the old API:

```python
# Update imports at top of tests/test_describer.py
from unittest.mock import AsyncMock

from docpipe.providers.base import LLMProvider
from docpipe.config import ApiRetryConfig, DescriberConfig
```

Add a mock provider fixture and update tests:

```python
class _MockProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0))

    async def _complete_raw(self, prompt: str, max_tokens: int) -> str:
        return "mock complete"

    async def _vision_raw(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        return "A test image showing a simple graphic"


class TestDescribeImageWithProvider:
    @pytest.mark.asyncio
    async def test_returns_description_from_provider(self, tmp_dirs: dict[str, Path]) -> None:
        img = tmp_dirs["output"] / "images" / "test_img.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(_MINIMAL_PNG)

        provider = _MockProvider()
        result = await describe_image(img, "preceding", "following", provider, max_tokens=300)
        assert isinstance(result, str)
        assert len(result) > 5


class TestReplaceImageRefsWithProvider:
    @pytest.mark.asyncio
    async def test_replaces_image_markdown(self, tmp_dirs: dict[str, Path]) -> None:
        img = tmp_dirs["output"] / "images" / "test_img001.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(_MINIMAL_PNG)

        cfg = DescriberConfig(provider="openai", max_tokens=300)
        provider = _MockProvider()

        markdown = "Some text\n\n![](images/test_img001.png)\n\nMore text"
        result = await replace_image_refs(markdown, tmp_dirs["output"], cfg, provider, doc_title="Test")
        assert "**[Image:" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_describer.py::TestDescribeImageWithProvider -v`
Expected: FAIL — signature mismatch

- [ ] **Step 3: Refactor `describer.py`**

Remove `import anthropic`, `import openai`, `_call_openai_vision_api`, `_call_anthropic_vision_api`. Update signatures:

```python
"""Replace image references in markdown with vision LLM descriptions."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

from docpipe.config import DescriberConfig
from docpipe.providers.base import LLMProvider, ProviderError

logger = logging.getLogger(__name__)

_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def get_surrounding_context(text: str, position: int, context_chars: int) -> tuple[str, str]:
    # unchanged
    ...


async def describe_image(
    image_path: Path,
    context_before: str,
    context_after: str,
    provider: LLMProvider,
    max_tokens: int = 300,
) -> str:
    """Send an image to the configured vision LLM and get a description."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    image_format = image_path.suffix.lstrip(".").lower()
    if image_format == "jpg":
        image_format = "jpeg"
    media_type = f"image/{image_format}"

    prompt = (
        "Describe this image concisely for a document knowledge base. "
        "Focus on what the image shows and its significance."
    )
    if context_before:
        prompt += f"\n\nPreceding text: {context_before}"
    if context_after:
        prompt += f"\n\nFollowing text: {context_after}"

    try:
        return await provider.vision(prompt, image_b64, media_type, max_tokens)
    except ProviderError:
        logger.error("Vision API failed for %s", image_path.name)
        return "[image description unavailable]"


async def replace_image_refs(
    markdown: str,
    output_dir: Path,
    cfg: DescriberConfig,
    provider: LLMProvider,
    doc_title: str = "",
) -> str:
    """Find image references in markdown and add AI descriptions."""
    matches = list(_IMAGE_PATTERN.finditer(markdown))
    if not matches:
        return markdown

    logger.info("Found %d images to describe", len(matches))

    result = markdown
    offset = 0

    for i, match in enumerate(matches):
        img_path_str = match.group(2)
        img_path = output_dir / img_path_str

        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        pos = match.start() + offset
        context_before, context_after = get_surrounding_context(result, pos, cfg.context_chars)
        if not context_before and doc_title:
            context_before = f"Document: {doc_title}"

        description = await describe_image(
            img_path, context_before, context_after, provider, cfg.max_tokens
        )

        desc_block = f"\n\n**[Image: {description}]**\n\n"
        insert_pos = match.start() + offset
        result = result[:insert_pos] + desc_block + result[insert_pos:]
        offset += len(desc_block)

        logger.debug("Described image %d/%d: %s", i + 1, len(matches), img_path.name)

    return result
```

- [ ] **Step 4: Remove old test classes that used the old API**

Remove `TestDescribeImage` and `TestReplaceImageRefs` classes (which used `_ANTHROPIC_CFG` and `_RETRY_CFG` directly). Remove the `_ANTHROPIC_CFG` and `_RETRY_CFG` module-level variables. Keep `TestGetSurroundingContext` unchanged.

- [ ] **Step 5: Run all describer tests**

Run: `pytest tests/test_describer.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/docpipe/describer.py tests/test_describer.py
git commit -m "refactor: describer uses LLMProvider instead of direct SDK calls"
```

---

### Task 8: Refactor `registry.py` to Use Provider

**Files:**
- Modify: `src/docpipe/registry.py`
- Modify: `tests/test_registry.py` (if needed — existing tests don't test `generate_summary`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_registry.py`:

```python
import pytest

from docpipe.config import ApiRetryConfig, RegistryConfig
from docpipe.providers.base import LLMProvider


class _MockProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0))

    async def _complete_raw(self, prompt: str, max_tokens: int) -> str:
        return "SUMMARY: A test document\nTOPICS: testing, docs"

    async def _vision_raw(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        return ""


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_returns_summary_and_topics(self) -> None:
        from docpipe.registry import generate_summary

        cfg = RegistryConfig()
        provider = _MockProvider()
        summary, topics = await generate_summary("# Test\nSome content", cfg, provider)
        assert summary == "A test document"
        assert "testing" in topics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::TestGenerateSummary -v`
Expected: FAIL — signature mismatch

- [ ] **Step 3: Refactor `registry.py`**

Remove `import openai` at top. Update `generate_summary`:

```python
async def generate_summary(
    markdown: str,
    cfg: RegistryConfig,
    provider: LLMProvider,
) -> tuple[str, str]:
    """Use an LLM to generate a summary and topic tags."""
    prompt = (
        f"Summarize this document in at most {cfg.summary_max_words} words. "
        "Then list 2-5 topic tags (comma-separated). "
        "Respond in exactly this format:\n"
        "SUMMARY: <your summary>\n"
        "TOPICS: <tag1, tag2, tag3>\n\n"
        f"Document:\n{markdown[:3000]}"
    )

    try:
        content = await provider.complete(prompt, max_tokens=200)
    except Exception:
        logger.error("Summary generation failed")
        return "Summary unavailable", "-"

    summary = ""
    topics = ""
    for line in content.splitlines():
        if line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
        elif line.startswith("TOPICS:"):
            topics = line.replace("TOPICS:", "").strip()
    return summary or "No summary available", topics or "-"
```

Add import at top:

```python
from docpipe.providers.base import LLMProvider
```

Remove the `import os` if no longer used (it's only used for API keys currently).

- [ ] **Step 4: Run all registry tests**

Run: `pytest tests/test_registry.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/registry.py tests/test_registry.py
git commit -m "refactor: registry uses LLMProvider instead of direct SDK calls"
```

---

### Task 9: Update `pipeline.py` to Create and Pass Provider

**Files:**
- Modify: `src/docpipe/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/conftest.py` (update `sample_config` fixture)

- [ ] **Step 1: Write the failing test**

Update `tests/test_pipeline.py` — the `api_config` fixture needs the new nested format:

```python
@pytest.fixture
def api_config(self, tmp_dirs: dict[str, Path]) -> Path:
    """Config with Anthropic provider and real API retry settings."""
    config = {
        "input_dir": str(tmp_dirs["input"]),
        "output_dir": str(tmp_dirs["output"]),
        "describer": {
            "provider": "anthropic",
            "max_tokens": 300,
            "include_context": True,
            "context_chars": 500,
            "batch_size": 5,
            "anthropic": {
                "model": "claude-haiku-4-5-20251001",
            },
        },
        "graph": {
            "provider": "openai",
            "storage": "file",
            "store_dir": str(tmp_dirs["output"] / "lightrag_store"),
            "openai": {
                "model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "embedding_dim": 1536,
            },
        },
        "api_retry": {
            "max_retries": 2,
            "initial_delay_seconds": 1,
            "max_delay_seconds": 10,
        },
        "logging": {
            "level": "DEBUG",
            "file": str(tmp_dirs["root"] / "logs" / "docpipe.log"),
            "max_size_mb": 1,
            "backup_count": 1,
        },
    }
    cfg_path = tmp_dirs["root"] / "config.yaml"
    cfg_path.write_text(yaml.dump(config))
    return cfg_path
```

- [ ] **Step 2: Update `pipeline.py`**

Add import and create provider at top of `process_file`:

```python
from docpipe.providers import create_provider
```

In `process_file`, after the `try:` block starts, create the provider:

```python
# Create LLM provider (shared by describer and registry)
provider_name = cfg.describer.provider
provider_cfg = getattr(cfg.describer, provider_name)
provider = create_provider(provider_name, provider_cfg, cfg.api_retry)
```

Update the `replace_image_refs` call:

```python
markdown = await replace_image_refs(
    result.markdown,
    cfg.output_dir,
    cfg.describer,
    provider,
    doc_title=doc_stem,
)
```

Update the `generate_summary` call:

```python
summary, topics = await generate_summary(markdown, cfg.registry, provider)
```

- [ ] **Step 3: Update `conftest.py` sample_config fixture**

Update the `sample_config` fixture to use nested format:

```python
"describer": {
    "provider": "openai",
    "max_tokens": 300,
    "include_context": True,
    "context_chars": 500,
    "batch_size": 5,
    "openai": {"model": "gpt-4o-mini"},
},
"graph": {
    "provider": "openai",
    "storage": "file",
    "store_dir": str(tmp_dirs["output"] / "lightrag_store"),
    "max_tokens": 4096,
    "chunk_size": 1200,
    "chunk_overlap": 100,
    "openai": {
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
    },
},
```

- [ ] **Step 4: Run pipeline tests**

Run: `pytest tests/test_pipeline.py -v -k "not test_processes_pdf"`
Expected: All non-e2e tests PASS (e2e tests require API keys)

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/pipeline.py tests/test_pipeline.py tests/conftest.py
git commit -m "refactor: pipeline creates LLMProvider and passes to describer/registry"
```

---

### Task 10: Update `graph.py` for Azure Support

**Files:**
- Modify: `src/docpipe/graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graph.py`:

```python
class TestGraphConfigProvider:
    def test_graph_config_has_provider_field(self) -> None:
        cfg = GraphConfig()
        assert cfg.provider == "openai"

    def test_graph_config_has_nested_openai(self) -> None:
        cfg = GraphConfig()
        assert cfg.openai.model == "gpt-4o-mini"
        assert cfg.openai.embedding_dim == 1536
```

- [ ] **Step 2: Run test to verify it passes** (should already pass from Task 6 config changes)

Run: `pytest tests/test_graph.py::TestGraphConfigProvider -v`
Expected: PASS

- [ ] **Step 3: Update `graph.py` to read from nested config**

Update `_get_rag_instance` to read model/embedding from the provider-specific nested config:

The full updated `graph.py` (note `import os` added at top):

```python
"""LightRAG knowledge graph ingestion."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from docpipe.config import GraphConfig

logger = logging.getLogger(__name__)


async def _get_rag_instance(cfg: GraphConfig) -> Any:  # noqa: ANN401
    """Create and initialize a LightRAG instance."""
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import finalize_share_data
    from lightrag.llm.openai import openai_complete, openai_embed
    from lightrag.utils import EmbeddingFunc

    finalize_share_data()

    store_dir = Path(cfg.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    # Get provider-specific config
    if cfg.provider == "azure":
        pcfg = cfg.azure
        # Set env vars for LightRAG's internal Azure OpenAI client
        os.environ["AZURE_OPENAI_ENDPOINT"] = pcfg.endpoint
        os.environ["OPENAI_API_VERSION"] = pcfg.api_version
        # AZURE_OPENAI_API_KEY should already be in env
        model_name = pcfg.deployment
        embedding_model = pcfg.embedding_deployment
    else:
        pcfg = cfg.openai
        model_name = pcfg.model
        embedding_model = pcfg.embedding_model

    embedding = EmbeddingFunc(
        embedding_dim=pcfg.embedding_dim,
        func=openai_embed.func,
        max_token_size=8192,
        model_name=embedding_model,
    )

    rag = LightRAG(
        working_dir=str(store_dir),
        llm_model_func=openai_complete,
        llm_model_name=model_name,
        embedding_func=embedding,
        chunk_token_size=cfg.chunk_size,
        chunk_overlap_token_size=cfg.chunk_overlap,
    )

    await rag.initialize_storages()
    return rag
```

The rest of `graph.py` (`ingest_document`, `rebuild_graph`) stays unchanged.

- [ ] **Step 4: Run graph tests**

Run: `pytest tests/test_graph.py -v -k "not needs_openai_key"`
Expected: All non-API tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/graph.py tests/test_graph.py
git commit -m "feat: graph.py reads nested provider config, supports Azure OpenAI"
```

---

### Task 11: Update Config YAML Files and Examples

**Files:**
- Modify: `config.yaml`
- Modify: `examples/config.yaml`

- [ ] **Step 1: Update `config.yaml`**

```yaml
# DocPipe Configuration
# Paths
input_dir: "./input"
output_dir: "./output"

# Watcher
watcher:
  enabled: true
  debounce_seconds: 60
  max_wait_seconds: 300
  poll_interval_seconds: 1
  watch_subdirectories: false

# Converter (LibreOffice)
converter:
  libreoffice_path: null
  timeout_seconds: 120
  supported_extensions:
    - .doc
    - .docx
    - .xls
    - .xlsx
    - .ppt
    - .pptx
    - .odt
    - .ods
    - .odp
    - .rtf
    - .html
    - .epub

# Extractor (PyMuPDF4LLM)
extractor:
  write_images: true
  image_format: "png"
  dpi: 150

# Describer (vision LLM)
describer:
  provider: "openai"            # "openai", "azure", or "anthropic"
  max_tokens: 300
  include_context: true
  context_chars: 500
  batch_size: 5

  openai:
    model: "gpt-4o-mini"
  azure:
    model: "gpt-4o-mini"
    endpoint: ""                # Set to your Azure endpoint
    deployment: ""              # Set to your Azure deployment name
    api_version: "2024-06-01"
  anthropic:
    model: "claude-haiku-4-5-20251001"

# Registry
registry:
  filename: "registry.md"
  summary_max_words: 30
  include_fields:
    - filename
    - author
    - date
    - summary
    - topics
    - pages
    - images

# Knowledge Graph (LightRAG) — OpenAI or Azure only
graph:
  provider: "openai"            # "openai" or "azure" (no anthropic — no embedding API)
  storage: "file"
  store_dir: "./output/lightrag_store"
  max_tokens: 4096
  chunk_size: 1200
  chunk_overlap: 100

  openai:
    model: "gpt-4o-mini"
    embedding_model: "text-embedding-3-small"
    embedding_dim: 1536
  azure:
    model: "gpt-4o-mini"
    embedding_model: "text-embedding-3-small"
    embedding_dim: 1536
    endpoint: ""
    deployment: ""
    embedding_deployment: ""
    api_version: "2024-06-01"

# API retry (applies to all provider calls)
api_retry:
  max_retries: 3
  initial_delay_seconds: 1
  max_delay_seconds: 30

# Logging
logging:
  level: "INFO"
  file: "./logs/docpipe.log"
  max_size_mb: 50
  backup_count: 3
```

- [ ] **Step 2: Update `examples/config.yaml`**

```yaml
# Example DocPipe Configuration
# Copy this file and adjust paths for your setup

input_dir: ./input
output_dir: ./output

# Watcher
watcher:
  enabled: true
  debounce_seconds: 60
  max_wait_seconds: 300

# Describer — choose your LLM provider
describer:
  # Set provider to "openai", "azure", or "anthropic"
  provider: "openai"
  max_tokens: 300
  include_context: true
  context_chars: 500

  openai:
    model: "gpt-4o-mini"

  # azure:
  #   model: "gpt-4o-mini"
  #   endpoint: "https://YOUR-INSTANCE.openai.azure.com"
  #   deployment: "YOUR-DEPLOYMENT"
  #   api_version: "2024-06-01"

  # anthropic:
  #   model: "claude-haiku-4-5-20251001"

# Knowledge Graph (OpenAI or Azure only)
graph:
  provider: "openai"
  storage: "file"
  store_dir: "./output/lightrag_store"

  openai:
    model: "gpt-4o-mini"
    embedding_model: "text-embedding-3-small"
    embedding_dim: 1536

# Logging
logging:
  level: "INFO"
  file: "./logs/docpipe.log"
```

- [ ] **Step 3: Commit**

```bash
git add config.yaml examples/config.yaml
git commit -m "docs: update config files with nested provider blocks"
```

---

### Task 12: Update Remaining Tests and Final Verification

**Files:**
- Modify: `tests/test_e2e_anthropic.py` (if it references old config format)
- Modify: `tests/test_integration.py` (if it references old config format)

- [ ] **Step 1: Update `tests/test_e2e_anthropic.py` fixture**

The `anthropic_config` fixture in `TestE2EAnthropic` uses old flat format. Update to nested:

```python
config = {
    "input_dir": str(FIXTURES),
    "output_dir": str(output_dir),
    "describer": {
        "provider": "anthropic",
        "max_tokens": 300,
        "include_context": True,
        "context_chars": 500,
        "batch_size": 5,
        "anthropic": {
            "model": "claude-haiku-4-5-20251001",
        },
    },
    "registry": {
        "filename": "registry.md",
        "summary_max_words": 30,
        "include_fields": [
            "filename", "author", "date", "summary", "topics", "pages", "images",
        ],
    },
    "graph": {
        "provider": "openai",
        "storage": "file",
        "store_dir": str(output_dir / "lightrag_store"),
        "max_tokens": 4096,
        "chunk_size": 1200,
        "chunk_overlap": 100,
        "openai": {
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
        },
    },
    "api_retry": {
        "max_retries": 2,
        "initial_delay_seconds": 1,
        "max_delay_seconds": 10,
    },
    "logging": {
        "level": "DEBUG",
        "file": str(tmp_path / "logs" / "docpipe.log"),
        "max_size_mb": 1,
        "backup_count": 1,
    },
}
```

- [ ] **Step 2: Update `tests/test_integration.py` fixture**

The `api_config` fixture in `TestEndToEnd` uses old flat format. Update to nested:

```python
config = {
    "input_dir": str(tmp_dirs["input"]),
    "output_dir": str(tmp_dirs["output"]),
    "describer": {
        "provider": "anthropic",
        "max_tokens": 300,
        "include_context": True,
        "context_chars": 500,
        "batch_size": 5,
        "anthropic": {
            "model": "claude-haiku-4-5-20251001",
        },
    },
    "registry": {
        "filename": "registry.md",
        "summary_max_words": 30,
        "include_fields": [
            "filename", "author", "date", "summary", "topics", "pages", "images",
        ],
    },
    "graph": {
        "provider": "openai",
        "storage": "file",
        "store_dir": str(tmp_dirs["output"] / "lightrag_store"),
        "openai": {
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
        },
    },
    "api_retry": {
        "max_retries": 2,
        "initial_delay_seconds": 1,
        "max_delay_seconds": 10,
    },
    "logging": {
        "level": "DEBUG",
        "file": str(tmp_dirs["root"] / "logs" / "docpipe.log"),
        "max_size_mb": 1,
        "backup_count": 1,
    },
}
```

- [ ] **Step 3: Search for any other remaining flat references**

Run: `grep -rn "\"model\".*gpt\|'model'.*gpt\|model.*claude" tests/`

Fix any remaining occurrences to use the new nested structure.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/test_e2e_anthropic.py --ignore=tests/test_full_system.py -k "not needs_openai_key and not needs_anthropic_key and not needs_api_keys and not needs_libreoffice"`
Expected: All tests PASS

- [ ] **Step 5: Run type check**

Run: `python -m mypy src/docpipe/providers/ src/docpipe/config.py src/docpipe/describer.py src/docpipe/registry.py src/docpipe/pipeline.py src/docpipe/graph.py --ignore-missing-imports`
Expected: No errors (or only pre-existing ones from lightrag stubs)

- [ ] **Step 6: Commit any remaining fixes**

```bash
git add -A
git commit -m "fix: update remaining tests for nested provider config"
```
