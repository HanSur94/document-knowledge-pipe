"""Unit tests for query_graph function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import GraphConfig, GraphOpenAIConfig


@pytest.fixture
def graph_cfg(tmp_path):
    return GraphConfig(
        provider="openai",
        store_dir=str(tmp_path / "lightrag_store"),
        openai=GraphOpenAIConfig(
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
        ),
    )


class TestQueryGraph:
    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_returns_answer_string(self, mock_rag_factory, graph_cfg):
        mock_rag = AsyncMock()
        mock_rag.aquery = AsyncMock(return_value="Paris is the capital of France.")
        mock_rag_factory.return_value = mock_rag

        from docpipe.graph import query_graph

        result = await query_graph("What is the capital of France?", graph_cfg)

        assert result == "Paris is the capital of France."
        mock_rag.aquery.assert_called_once()

    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_returns_none_on_exception(self, mock_rag_factory, graph_cfg):
        mock_rag_factory.side_effect = RuntimeError("connection failed")

        from docpipe.graph import query_graph

        result = await query_graph("test question", graph_cfg)

        assert result is None

    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_raises_type_error_on_non_string(self, mock_rag_factory, graph_cfg):
        mock_rag = AsyncMock()
        mock_rag.aquery = AsyncMock(return_value=MagicMock(__aiter__=True))
        mock_rag_factory.return_value = mock_rag

        from docpipe.graph import query_graph

        with pytest.raises(TypeError, match="Expected str"):
            await query_graph("test question", graph_cfg)

    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_passes_mode_to_query_param(self, mock_rag_factory, graph_cfg):
        mock_rag = AsyncMock()
        mock_rag.aquery = AsyncMock(return_value="answer")
        mock_rag_factory.return_value = mock_rag

        from docpipe.graph import query_graph

        await query_graph("question", graph_cfg, mode="local")

        call_args = mock_rag.aquery.call_args
        assert call_args[0][0] == "question"
        param = call_args[0][1]
        assert param.mode == "local"
