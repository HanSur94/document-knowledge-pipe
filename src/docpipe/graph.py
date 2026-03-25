"""LightRAG knowledge graph ingestion."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from docpipe.config import GraphConfig

logger = logging.getLogger(__name__)


async def _get_rag_instance(cfg: GraphConfig) -> Any:  # noqa: ANN401
    """Create and initialize a LightRAG instance."""
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import finalize_share_data
    from lightrag.llm.openai import openai_complete, openai_embed
    from lightrag.utils import EmbeddingFunc

    finalize_share_data()

    store_dir = Path(cfg.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    # Get provider-specific config
    if cfg.provider == "azure":
        pcfg = cfg.azure
        # Set env vars for LightRAG's internal Azure OpenAI client
        os.environ["AZURE_OPENAI_ENDPOINT"] = pcfg.endpoint
        os.environ["OPENAI_API_VERSION"] = pcfg.api_version
        # AZURE_OPENAI_API_KEY should already be in env
        model_name = pcfg.deployment
        embedding_model = pcfg.embedding_deployment
    else:
        pcfg = cfg.openai
        model_name = pcfg.model
        embedding_model = pcfg.embedding_model

    embedding = EmbeddingFunc(
        embedding_dim=pcfg.embedding_dim,
        func=openai_embed.func,
        max_token_size=8192,
        model_name=embedding_model,
    )

    rag = LightRAG(
        working_dir=str(store_dir),
        llm_model_func=openai_complete,
        llm_model_name=model_name,
        embedding_func=embedding,
        chunk_token_size=cfg.chunk_size,
        chunk_overlap_token_size=cfg.chunk_overlap,
    )

    await rag.initialize_storages()
    return rag


async def ingest_document(
    markdown: str,
    doc_id: str,
    cfg: GraphConfig,
) -> bool:
    """Ingest a markdown document into the LightRAG knowledge graph.

    Returns True on success, False on failure (non-blocking).
    """
    try:
        rag = await _get_rag_instance(cfg)
        await rag.ainsert(markdown)
        logger.info("Ingested into knowledge graph: %s", doc_id)
        return True
    except Exception as e:
        logger.error("LightRAG ingestion failed for %s: %s", doc_id, e)
        return False


async def rebuild_graph(markdown_dir: Path, cfg: GraphConfig) -> int:
    """Delete the graph store and re-ingest all markdown files.

    Returns the number of documents ingested.
    """
    store_dir = Path(cfg.store_dir)
    if store_dir.exists():
        logger.info("Deleting existing graph store: %s", store_dir)
        shutil.rmtree(store_dir)

    md_files = sorted(markdown_dir.glob("*.md"))
    logger.info("Rebuilding graph from %d markdown files", len(md_files))

    count = 0
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        success = await ingest_document(content, md_file.stem, cfg)
        if success:
            count += 1

    logger.info("Graph rebuilt: %d documents ingested", count)
    return count


async def query_graph(
    question: str,
    cfg: GraphConfig,
    mode: str = "mix",
) -> str | None:
    """Query the LightRAG knowledge graph.

    Returns the answer string on success, None on failure.
    """
    try:
        from lightrag import QueryParam

        rag = await _get_rag_instance(cfg)
        result = await rag.aquery(question, QueryParam(mode=mode))

        if not isinstance(result, str):
            raise TypeError(f"Expected str from aquery, got {type(result).__name__}")

        return result
    except TypeError:
        raise
    except Exception as e:
        logger.error("LightRAG query failed: %s", e)
        return None
