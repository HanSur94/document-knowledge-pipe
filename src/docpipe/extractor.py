"""Extract structured markdown and images from PDF via PyMuPDF4LLM."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pymupdf4llm

from docpipe.config import ExtractorConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    markdown: str
    image_paths: list[Path] = field(default_factory=list)
    page_count: int = 0


def extract_markdown(
    pdf_path: Path,
    image_dir: Path,
    cfg: ExtractorConfig,
) -> ExtractionResult:
    """Extract markdown with images in reading order from a PDF."""
    image_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    if page_count == 0:
        logger.warning("PDF has 0 pages: %s", pdf_path.name)
        return ExtractionResult(markdown="", page_count=0)

    logger.info("Extracting %d pages from %s", page_count, pdf_path.name)

    md_text = pymupdf4llm.to_markdown(
        doc=str(pdf_path),
        write_images=cfg.write_images,
        image_path=str(image_dir),
        image_format=cfg.image_format,
        dpi=cfg.dpi,
    )

    # Collect any images that were written — only match this document's stem
    image_paths: list[Path] = []
    if cfg.write_images:
        stem = pdf_path.stem
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.stem.startswith(stem)
        )

    logger.info(
        "Extracted %d chars, %d images from %s",
        len(md_text),
        len(image_paths),
        pdf_path.name,
    )

    return ExtractionResult(
        markdown=md_text,
        image_paths=image_paths,
        page_count=page_count,
    )
