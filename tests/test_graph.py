from __future__ import annotations

from pathlib import Path

import pytest
from conftest import needs_openai_key

from docpipe.config import GraphConfig
from docpipe.graph import ingest_document, rebuild_graph


class TestGraphConfigProvider:
    def test_graph_config_has_provider_field(self) -> None:
        cfg = GraphConfig()
        assert cfg.provider == "openai"

    def test_graph_config_has_nested_openai(self) -> None:
        cfg = GraphConfig()
        assert cfg.openai.model == "gpt-4o-mini"
        assert cfg.openai.embedding_dim == 1536


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
    async def test_handles_ingestion_failure(self, tmp_path: Path) -> None:
        # Create a file where the store directory would need to be created.
        # mkdir(parents=True) fails on all platforms when a parent is a file.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        cfg = GraphConfig(store_dir=str(blocker / "nested" / "store"))
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
