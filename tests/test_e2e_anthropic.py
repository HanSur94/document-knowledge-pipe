"""End-to-end test with real Anthropic API calls.

Requires ANTHROPIC_API_KEY in environment. Skipped if not set.
Run with: pytest tests/test_e2e_anthropic.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from docpipe.config import load_config
from docpipe.pipeline import process_file

needs_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

FIXTURES = Path(__file__).parent / "fixtures"


@needs_anthropic_key
class TestE2EAnthropic:
    @pytest.fixture
    def anthropic_config(self, tmp_path: Path) -> Path:
        """Create a config using Anthropic as the LLM provider."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "markdown").mkdir()
        (output_dir / "images").mkdir()

        config = {
            "input_dir": str(FIXTURES),
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

    @pytest.mark.asyncio
    async def test_pdf_with_anthropic_describer(
        self, anthropic_config: Path, tmp_path: Path
    ) -> None:
        """Full pipeline: PDF → extract → describe (Anthropic) → registry."""
        cfg = load_config(anthropic_config)

        pdf = FIXTURES / "sample.pdf"
        if not pdf.exists():
            pytest.skip("sample.pdf fixture not found")

        # Mock only the graph ingestion (needs OpenAI embedding)
        from unittest.mock import AsyncMock, patch

        with patch(
            "docpipe.pipeline.ingest_document",
            new_callable=AsyncMock,
            return_value=True,
        ):
            success = await process_file(pdf, cfg)

        assert success

        # Verify markdown was written
        md_path = cfg.output_dir / "markdown" / "sample.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert len(content) > 100

        # Verify registry
        reg_path = cfg.output_dir / "registry.md"
        assert reg_path.exists()
        reg_content = reg_path.read_text()
        assert "sample.md" in reg_content

        # Verify status
        status_path = cfg.output_dir / "status.json"
        assert status_path.exists()
