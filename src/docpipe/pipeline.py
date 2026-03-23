"""Pipeline orchestrator — full flow per file."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from docpipe.config import DocpipeConfig
from docpipe.converter import convert_to_pdf
from docpipe.describer import replace_image_refs
from docpipe.extractor import extract_markdown
from docpipe.graph import ingest_document
from docpipe.registry import RegistryEntry, generate_summary, update_registry
from docpipe.status import StatusTracker

logger = logging.getLogger(__name__)


class Lockfile:
    """Simple file-based lock to prevent concurrent processing."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._acquired = False

    def acquire(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._acquired = True
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if self._acquired and self._path.exists():
            self._path.unlink()
            self._acquired = False

    def __enter__(self) -> Lockfile:
        if not self.acquire():
            raise RuntimeError(
                "Error: docpipe is already running (lockfile held). "
                "Stop the watcher first or wait for it to finish."
            )
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


def cleanup_orphans(doc_stem: str, output_dir: Path) -> None:
    """Remove orphaned output files from a previous partial run."""
    img_dir = output_dir / "images"
    if img_dir.exists():
        for img in img_dir.iterdir():
            if img.stem.startswith(doc_stem):
                img.unlink()
                logger.debug("Removed orphaned image: %s", img.name)

    md_path = output_dir / "markdown" / f"{doc_stem}.md"
    if md_path.exists():
        md_path.unlink()
        logger.debug("Removed orphaned markdown: %s", md_path.name)


async def process_file(file_path: Path, cfg: DocpipeConfig) -> bool:
    """Run the full pipeline for a single file. Returns True on success."""
    doc_stem = file_path.stem
    tracker = StatusTracker(cfg.output_dir)

    try:
        tracker.update_file(file_path.name, status="processing")
        tracker.save()

        # Stage 0: Cleanup orphans
        cleanup_orphans(doc_stem, cfg.output_dir)

        # Stage 1: Convert to PDF
        try:
            pdf_path = convert_to_pdf(file_path, cfg.output_dir, cfg.converter)
        except FileNotFoundError:
            logger.warning("File not found (possibly renamed): %s", file_path)
            tracker.update_file(file_path.name, status="failed", error="File not found")
            tracker.save()
            return False
        except Exception as e:
            logger.error("Conversion failed for %s: %s", file_path.name, e)
            tracker.update_file(file_path.name, status="failed", error=str(e))
            tracker.save()
            reg_entry = RegistryEntry.failed(file_path.name, str(e))
            update_registry(cfg.output_dir / cfg.registry.filename, reg_entry)
            return False

        # Stage 2: Extract markdown + images
        md_dir = cfg.output_dir / "markdown"
        md_dir.mkdir(parents=True, exist_ok=True)
        img_dir = cfg.output_dir / "images"

        result = extract_markdown(pdf_path, img_dir, cfg.extractor)

        if not result.markdown.strip():
            logger.warning("Empty extraction for %s, skipping", file_path.name)
            tracker.update_file(file_path.name, status="skipped", error="Empty document")
            tracker.save()
            reg_entry = RegistryEntry.failed(file_path.name, "Empty document")
            update_registry(cfg.output_dir / cfg.registry.filename, reg_entry)
            return False

        # Stage 3: Describe images with VLM
        markdown = await replace_image_refs(
            result.markdown,
            cfg.output_dir,
            cfg.describer,
            cfg.api_retry,
            doc_title=doc_stem,
        )

        # Write final markdown
        md_path = md_dir / f"{doc_stem}.md"
        md_path.write_text(markdown, encoding="utf-8")

        # Stage 4: Update registry
        summary, topics = await generate_summary(markdown, cfg.registry, cfg.api_retry)
        reg_entry = RegistryEntry(
            filename=f"{doc_stem}.md",
            author="-",
            date="-",
            summary=summary,
            topics=topics,
            pages=result.page_count,
            images=len(result.image_paths),
        )
        update_registry(cfg.output_dir / cfg.registry.filename, reg_entry)

        # Stage 5: Ingest into knowledge graph
        graph_ok = await ingest_document(markdown, doc_stem, cfg.graph)

        # Stage 6: Update status
        tracker.update_file(
            file_path.name,
            status="done",
            pages=result.page_count,
            images=len(result.image_paths),
            graph_ingested=graph_ok,
            md_path=f"markdown/{doc_stem}.md",
        )
        tracker.save()

        # Clean up converted PDF if it was a temp conversion
        if pdf_path != file_path and pdf_path.exists():
            pdf_path.unlink()

        logger.info("Successfully processed: %s", file_path.name)
        return True

    except Exception as e:
        logger.error("Pipeline failed for %s: %s", file_path.name, e, exc_info=True)
        tracker.update_file(file_path.name, status="failed", error=str(e))
        tracker.save()
        reg_entry = RegistryEntry.failed(file_path.name, str(e))
        update_registry(cfg.output_dir / cfg.registry.filename, reg_entry)
        return False
