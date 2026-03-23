"""End-to-end integration test with mocked API calls."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from docpipe.config import load_config
from docpipe.pipeline import process_file


class TestEndToEnd:
    @pytest.mark.asyncio
    @patch("docpipe.pipeline.ingest_document", new_callable=AsyncMock, return_value=True)
    @patch(
        "docpipe.pipeline.generate_summary",
        new_callable=AsyncMock,
        return_value=("Test summary", "test, integration"),
    )
    @patch(
        "docpipe.pipeline.replace_image_refs",
        new_callable=AsyncMock,
        side_effect=lambda md, *a, **kw: md,
    )
    async def test_full_pipeline_pdf(
        self,
        mock_replace: AsyncMock,
        mock_summary: AsyncMock,
        mock_ingest: AsyncMock,
        sample_pdf: Path,
        config_file: Path,
    ) -> None:
        cfg = load_config(config_file)

        # Process the file
        success = await process_file(sample_pdf, cfg)
        assert success

        # Verify markdown output
        md_path = cfg.output_dir / "markdown" / "test_doc.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Hello World" in content

        # Verify registry
        reg_path = cfg.output_dir / "registry.md"
        assert reg_path.exists()
        reg_content = reg_path.read_text()
        assert "test_doc.md" in reg_content
        assert "Test summary" in reg_content

        # Verify status.json
        status_path = cfg.output_dir / "status.json"
        assert status_path.exists()

        # Verify API calls
        mock_ingest.assert_called_once()
        mock_summary.assert_called_once()
