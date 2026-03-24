"""LightRAG knowledge graph ingestion."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from docpipe.config import GraphConfig

logger = logging.getLogger(__name__)


async def _get_rag_instance(cfg: GraphConfig) -> Any:  # noqa: ANN401
    """Create and initialize a LightRAG instance."""
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete, openai_embed
    from lightrag.utils import EmbeddingFunc

    store_dir = Path(cfg.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    embedding = EmbeddingFunc(
        embedding_dim=1536,
        func=openai_embed.func,
        max_token_size=8192,
        model_name=cfg.embedding_model,
    )

    rag = LightRAG(
        working_dir=str(store_dir),
        llm_model_func=openai_complete,
        llm_model_name=cfg.model,
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
