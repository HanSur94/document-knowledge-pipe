# tests/test_graph.py
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import GraphConfig
from docpipe.graph import ingest_document, rebuild_graph


class TestIngestDocument:
    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_inserts_markdown_into_lightrag(
        self, mock_get_rag: MagicMock, tmp_dirs: dict[str, Path]
    ) -> None:
        mock_rag = AsyncMock()
        mock_get_rag.return_value = mock_rag
        cfg = GraphConfig(store_dir=str(tmp_dirs["output"] / "lightrag_store"))

        await ingest_document("# Test\nHello world", "test_doc", cfg)
        mock_rag.ainsert.assert_called_once()

    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_handles_ingestion_failure(
        self, mock_get_rag: MagicMock, tmp_dirs: dict[str, Path]
    ) -> None:
        mock_rag = AsyncMock()
        mock_rag.ainsert.side_effect = Exception("LightRAG error")
        mock_get_rag.return_value = mock_rag
        cfg = GraphConfig(store_dir=str(tmp_dirs["output"] / "lightrag_store"))

        result = await ingest_document("# Test", "test_doc", cfg)
        assert result is False


class TestRebuildGraph:
    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_rebuild_deletes_store_first(
        self, mock_get_rag: MagicMock, tmp_dirs: dict[str, Path]
    ) -> None:
        mock_rag = AsyncMock()
        mock_get_rag.return_value = mock_rag
        store_dir = tmp_dirs["output"] / "lightrag_store"
        store_dir.mkdir()
        (store_dir / "old_data.json").write_text("{}")

        cfg = GraphConfig(store_dir=str(store_dir))
        md_dir = tmp_dirs["output"] / "markdown"
        (md_dir / "doc1.md").write_text("# Doc 1\nContent")

        await rebuild_graph(md_dir, cfg)

        assert not (store_dir / "old_data.json").exists()
        mock_rag.ainsert.assert_called_once()
