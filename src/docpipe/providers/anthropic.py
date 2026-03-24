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
