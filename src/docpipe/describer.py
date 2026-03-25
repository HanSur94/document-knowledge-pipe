"""Replace image references in markdown with vision LLM descriptions."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

from docpipe.config import DescriberConfig
from docpipe.providers.base import LLMProvider, ProviderError

logger = logging.getLogger(__name__)

_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def get_surrounding_context(text: str, position: int, context_chars: int) -> tuple[str, str]:
    """Extract text before and after a position in the document."""
    start = max(0, position - context_chars)
    end_of_ref = text.find(")", position)
    if end_of_ref == -1:
        end_of_ref = position
    else:
        end_of_ref += 1
    end = min(len(text), end_of_ref + context_chars)

    before = text[start:position].strip()
    after = text[end_of_ref:end].strip()
    return before, after


async def describe_image(
    image_path: Path,
    context_before: str,
    context_after: str,
    provider: LLMProvider,
    max_tokens: int = 300,
) -> str:
    """Send an image to the configured vision LLM and get a description."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    image_format = image_path.suffix.lstrip(".").lower()
    if image_format == "jpg":
        image_format = "jpeg"
    media_type = f"image/{image_format}"

    prompt = (
        "Describe this image concisely for a document knowledge base. "
        "Focus on what the image shows and its significance."
    )
    if context_before:
        prompt += f"\n\nPreceding text: {context_before}"
    if context_after:
        prompt += f"\n\nFollowing text: {context_after}"

    try:
        return await provider.vision(prompt, image_b64, media_type, max_tokens)
    except ProviderError:
        logger.error("Vision API failed for %s", image_path.name)
        return "[image description unavailable]"


async def replace_image_refs(
    markdown: str,
    output_dir: Path,
    cfg: DescriberConfig,
    provider: LLMProvider,
    doc_title: str = "",
) -> str:
    """Find image references in markdown and add AI descriptions."""
    matches = list(_IMAGE_PATTERN.finditer(markdown))
    if not matches:
        return markdown

    logger.info("Found %d images to describe", len(matches))

    result = markdown
    offset = 0

    for i, match in enumerate(matches):
        img_path_str = match.group(2)
        img_path = output_dir / img_path_str

        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        pos = match.start() + offset
        context_before, context_after = get_surrounding_context(result, pos, cfg.context_chars)
        if not context_before and doc_title:
            context_before = f"Document: {doc_title}"

        description = await describe_image(
            img_path, context_before, context_after, provider, cfg.max_tokens
        )

        desc_block = f"\n\n**[Image: {description}]**\n\n"
        insert_pos = match.start() + offset
        result = result[:insert_pos] + desc_block + result[insert_pos:]
        offset += len(desc_block)

        logger.debug("Described image %d/%d: %s", i + 1, len(matches), img_path.name)

    return result
