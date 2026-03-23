"""End-to-end integration test with real API calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import needs_api_keys

from docpipe.config import load_config
from docpipe.pipeline import process_file


@needs_api_keys
class TestEndToEnd:
    @pytest.fixture
    def api_config(self, tmp_dirs: dict[str, Path]) -> Path:
        """Config with Anthropic provider for integration testing."""
        config = {
            "input_dir": str(tmp_dirs["input"]),
            "output_dir": str(tmp_dirs["output"]),
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
                "store_dir": str(tmp_dirs["output"] / "lightrag_store"),
                "model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
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

    @pytest.mark.asyncio
    async def test_full_pipeline_pdf(
        self,
        sample_pdf: Path,
        api_config: Path,
    ) -> None:
        cfg = load_config(api_config)

        success = await process_file(sample_pdf, cfg)
        assert success

        # Verify markdown output
        md_path = cfg.output_dir / "markdown" / "test_doc.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Hello World" in content

        # Verify registry has real summary
        reg_path = cfg.output_dir / "registry.md"
        assert reg_path.exists()
        reg_content = reg_path.read_text()
        assert "test_doc.md" in reg_content
        # Summary should be real, not a failure placeholder
        assert "Summary unavailable" not in reg_content
        assert "No summary available" not in reg_content

        # Verify status.json
        status_path = cfg.output_dir / "status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        file_status = status.get("files", {}).get("test_doc.pdf", {})
        assert file_status.get("status") == "done"
