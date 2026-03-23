from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import ApiRetryConfig, DescriberConfig
from docpipe.describer import (
    describe_image,
    get_surrounding_context,
    replace_image_refs,
)


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


class TestDescribeImage:
    @pytest.mark.asyncio
    @patch("docpipe.describer._call_vision_api", new_callable=AsyncMock)
    async def test_returns_description(self, mock_api: AsyncMock, tmp_dirs: dict[str, Path]) -> None:
        mock_api.return_value = "A bar chart showing quarterly revenue."
        img = tmp_dirs["output"] / "images" / "test_img.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        # Create a minimal PNG (1x1 pixel)
        img.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = await describe_image(
            img, "preceding text", "following text",
            DescriberConfig(), ApiRetryConfig(),
        )
        assert result == "A bar chart showing quarterly revenue."


class TestReplaceImageRefs:
    @pytest.mark.asyncio
    @patch("docpipe.describer.describe_image", new_callable=AsyncMock)
    async def test_replaces_image_markdown(
        self, mock_describe: AsyncMock, tmp_dirs: dict[str, Path]
    ) -> None:
        mock_describe.return_value = "A photo of a cat."
        img = tmp_dirs["output"] / "images" / "test_img001.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"fake png")

        markdown = "Some text\n\n![](images/test_img001.png)\n\nMore text"
        result = await replace_image_refs(
            markdown,
            tmp_dirs["output"],
            DescriberConfig(),
            ApiRetryConfig(),
            doc_title="Test Document",
        )
        assert "A photo of a cat." in result
        assert "![](images/test_img001.png)" in result  # original ref preserved
