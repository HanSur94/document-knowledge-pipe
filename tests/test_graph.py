from __future__ import annotations

from pathlib import Path

import pytest
from conftest import needs_openai_key

from docpipe.config import GraphConfig
from docpipe.graph import ingest_document, rebuild_graph


@needs_openai_key
class TestIngestDocument:
    @pytest.mark.asyncio
    async def test_inserts_markdown_into_lightrag(self, tmp_dirs: dict[str, Path]) -> None:
        cfg = GraphConfig(store_dir=str(tmp_dirs["output"] / "lightrag_store"))
        content = "# Test\nHello world. This is test content."
        result = await ingest_document(content, "test_doc", cfg)
        assert result is True


class TestIngestDocumentFailure:
    """Failure path — no API key needed since error occurs before any API call."""

    @pytest.mark.asyncio
    async def test_handles_ingestion_failure(self) -> None:
        # /dev/null/impossible triggers error at mkdir() before any API call
        cfg = GraphConfig(store_dir="/dev/null/impossible/path")
        result = await ingest_document("# Test", "test_doc", cfg)
        assert result is False


@needs_openai_key
class TestRebuildGraph:
    @pytest.mark.asyncio
    async def test_rebuild_deletes_store_first(self, tmp_dirs: dict[str, Path]) -> None:
        store_dir = tmp_dirs["output"] / "lightrag_store"
        store_dir.mkdir()
        (store_dir / "old_data.json").write_text("{}")

        cfg = GraphConfig(store_dir=str(store_dir))
        md_dir = tmp_dirs["output"] / "markdown"
        (md_dir / "doc1.md").write_text("# Doc 1\nThis is document content for testing.")

        count = await rebuild_graph(md_dir, cfg)

        assert not (store_dir / "old_data.json").exists()
        assert count == 1
