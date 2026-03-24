from __future__ import annotations

from pathlib import Path

import pytest

from docpipe.config import ApiRetryConfig, DescriberConfig
from docpipe.describer import (
    describe_image,
    get_surrounding_context,
    replace_image_refs,
)
from docpipe.providers.base import LLMProvider

# Minimal valid 1x1 PNG
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _MockProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(ApiRetryConfig(max_retries=0, initial_delay_seconds=0, max_delay_seconds=0))

    async def _complete_raw(self, prompt: str, max_tokens: int) -> str:
        return "mock complete"

    async def _vision_raw(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        return "A test image showing a simple graphic"


class TestGetSurroundingContext:
    def test_extracts_context_around_position(self) -> None:
        text = "AAAA" * 50 + "![image](img.png)" + "BBBB" * 50
        pos = text.index("![image]")
        before, after = get_surrounding_context(text, pos, context_chars=20)
        assert len(before) <= 20
        assert len(after) <= 20

    def test_handles_start_of_document(self) -> None:
        text = "![image](img.png) followed by text"
        before, after = get_surrounding_context(text, 0, context_chars=100)
        assert before == ""
        assert "followed by text" in after

    def test_handles_end_of_document(self) -> None:
        text = "Some text ![image](img.png)"
        pos = text.index("![image]")
        before, after = get_surrounding_context(text, pos, context_chars=100)
        assert "Some text" in before
        assert after == ""


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
