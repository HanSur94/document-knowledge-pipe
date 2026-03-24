from __future__ import annotations

from pathlib import Path

import pytest
from conftest import needs_anthropic_key

from docpipe.config import AnthropicProviderConfig, ApiRetryConfig, DescriberConfig
from docpipe.describer import (
    describe_image,
    get_surrounding_context,
    replace_image_refs,
)

# Minimal valid 1x1 PNG
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

_ANTHROPIC_CFG = DescriberConfig(
    provider="anthropic",
    anthropic=AnthropicProviderConfig(model="claude-haiku-4-5-20251001"),
)
_RETRY_CFG = ApiRetryConfig(max_retries=2, initial_delay_seconds=1, max_delay_seconds=10)


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


@needs_anthropic_key
class TestDescribeImage:
    @pytest.mark.asyncio
    async def test_returns_description(self, tmp_dirs: dict[str, Path]) -> None:
        img = tmp_dirs["output"] / "images" / "test_img.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(_MINIMAL_PNG)

        result = await describe_image(
            img,
            "preceding text",
            "following text",
            _ANTHROPIC_CFG,
            _RETRY_CFG,
        )
        assert isinstance(result, str)
        assert len(result) > 10


# TestDescribeImageAnthropic removed — was identical to TestDescribeImage
# after both switched to Anthropic provider. One test covers the path.


@needs_anthropic_key
class TestReplaceImageRefs:
    @pytest.mark.asyncio
    async def test_replaces_image_markdown(self, tmp_dirs: dict[str, Path]) -> None:
        img = tmp_dirs["output"] / "images" / "test_img001.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(_MINIMAL_PNG)

        markdown = "Some text\n\n![](images/test_img001.png)\n\nMore text"
        result = await replace_image_refs(
            markdown,
            tmp_dirs["output"],
            _ANTHROPIC_CFG,
            _RETRY_CFG,
            doc_title="Test Document",
        )
        assert "**[Image:" in result
        assert "![](images/test_img001.png)" in result
