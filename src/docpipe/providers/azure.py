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
