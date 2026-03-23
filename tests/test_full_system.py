"""Full system test with real-world documents and real API calls.

Tests the complete pipeline: convert → extract → describe → registry → graph
on real business documents (Kaggle Company Documents) and real office files
(OpenPreserve format-corpus).

Requires:
    - ANTHROPIC_API_KEY in environment
    - LibreOffice installed (for non-PDF formats)

Run with: pytest tests/test_full_system.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from docpipe.config import ConverterConfig, load_config
from docpipe.converter import find_libreoffice
from docpipe.pipeline import process_file

REAL_WORLD = Path(__file__).parent / "fixtures" / "real-world"

needs_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

_lo_available = False
try:
    find_libreoffice(ConverterConfig())
    _lo_available = True
except FileNotFoundError:
    pass

needs_libreoffice = pytest.mark.skipif(
    not _lo_available,
    reason="LibreOffice not installed",
)


@pytest.fixture
def pipeline_config(tmp_path: Path) -> Path:
    """Config using Anthropic for the full system test."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "markdown").mkdir()
    (output_dir / "images").mkdir()

    config = {
        "input_dir": str(REAL_WORLD),
        "output_dir": str(output_dir),
        "describer": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 300,
            "include_context": True,
            "context_chars": 500,
            "batch_size": 5,
        },
        "registry": {
            "filename": "registry.md",
            "summary_max_words": 30,
            "include_fields": [
                "filename",
                "author",
                "date",
                "summary",
                "topics",
                "pages",
                "images",
            ],
        },
        "graph": {
            "storage": "file",
            "store_dir": str(output_dir / "lightrag_store"),
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_tokens": 4096,
            "chunk_size": 1200,
            "chunk_overlap": 100,
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
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config))
    return cfg_path


async def _run_and_verify(
    file_path: Path,
    cfg_path: Path,
    expect_pages_gt: int = 0,
) -> None:
    """Run pipeline on a file and verify outputs exist."""
    cfg = load_config(cfg_path)

    # Mock only graph ingestion (needs OpenAI embedding key)
    with patch(
        "docpipe.pipeline.ingest_document",
        new_callable=AsyncMock,
        return_value=True,
    ):
        success = await process_file(file_path, cfg)

    assert success, f"Pipeline failed for {file_path.name}"

    stem = file_path.stem
    md_path = cfg.output_dir / "markdown" / f"{stem}.md"
    assert md_path.exists(), f"Markdown not created for {file_path.name}"
    content = md_path.read_text()
    assert len(content) > 10, f"Markdown too short for {file_path.name}"

    reg_path = cfg.output_dir / "registry.md"
    assert reg_path.exists()
    assert f"{stem}.md" in reg_path.read_text()

    status_path = cfg.output_dir / "status.json"
    assert status_path.exists()


# --- Real-world PDF tests (Kaggle Company Documents) ---


@needs_anthropic_key
class TestKaggleBusinessDocs:
    """Full pipeline on real business PDFs from Kaggle."""

    @pytest.mark.asyncio
    async def test_invoice(self, pipeline_config: Path) -> None:
        f = REAL_WORLD / "invoice.pdf"
        if not f.exists():
            pytest.skip("invoice.pdf not found")
        await _run_and_verify(f, pipeline_config)

    @pytest.mark.asyncio
    async def test_inventory_report(self, pipeline_config: Path) -> None:
        f = REAL_WORLD / "inventory_report.pdf"
        if not f.exists():
            pytest.skip("inventory_report.pdf not found")
        await _run_and_verify(f, pipeline_config)

    @pytest.mark.asyncio
    async def test_purchase_order(self, pipeline_config: Path) -> None:
        f = REAL_WORLD / "purchase_order.pdf"
        if not f.exists():
            pytest.skip("purchase_order.pdf not found")
        await _run_and_verify(f, pipeline_config)

    @pytest.mark.asyncio
    async def test_shipping_order(self, pipeline_config: Path) -> None:
        f = REAL_WORLD / "shipping_order.pdf"
        if not f.exists():
            pytest.skip("shipping_order.pdf not found")
        await _run_and_verify(f, pipeline_config)


# --- Real-world non-PDF tests (OpenPreserve format-corpus) ---


@needs_anthropic_key
@needs_libreoffice
class TestFormatCorpusDocs:
    """Full pipeline on real office docs from OpenPreserve format-corpus."""

    @pytest.mark.asyncio
    async def test_odt_with_embedded_image(self, pipeline_config: Path) -> None:
        f = REAL_WORLD / "embedded_image.odt"
        if not f.exists():
            pytest.skip("embedded_image.odt not found")
        await _run_and_verify(f, pipeline_config)

    @pytest.mark.asyncio
    async def test_odp_presentation(self, pipeline_config: Path) -> None:
        f = REAL_WORLD / "presentation.odp"
        if not f.exists():
            pytest.skip("presentation.odp not found")
        await _run_and_verify(f, pipeline_config)

    @pytest.mark.asyncio
    async def test_ods_spreadsheet(self, pipeline_config: Path) -> None:
        f = REAL_WORLD / "spreadsheet.ods"
        if not f.exists():
            pytest.skip("spreadsheet.ods not found")
        await _run_and_verify(f, pipeline_config)

    @pytest.mark.asyncio
    async def test_legacy_ppt(self, pipeline_config: Path) -> None:
        f = REAL_WORLD / "legacy_slides.ppt"
        if not f.exists():
            pytest.skip("legacy_slides.ppt not found")
        await _run_and_verify(f, pipeline_config)

    @pytest.mark.asyncio
    async def test_xls_spreadsheet(self, pipeline_config: Path) -> None:
        f = REAL_WORLD / "reviews.xls"
        if not f.exists():
            pytest.skip("reviews.xls not found")
        await _run_and_verify(f, pipeline_config)
