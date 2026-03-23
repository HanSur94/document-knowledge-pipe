"""Replace image references in markdown with vision LLM descriptions."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from pathlib import Path
from typing import Literal, cast

import anthropic
import openai

from docpipe.config import ApiRetryConfig, DescriberConfig

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


async def _call_openai_vision_api(
    client: openai.AsyncOpenAI,
    image_b64: str,
    image_format: str,
    context_before: str,
    context_after: str,
    cfg: DescriberConfig,
    retry_cfg: ApiRetryConfig,
) -> str:
    """Call OpenAI vision API with retry."""
    prompt = (
        "Describe this image concisely for a document knowledge base. "
        "Focus on what the image shows and its significance."
    )
    if context_before:
        prompt += f"\n\nPreceding text: {context_before}"
    if context_after:
        prompt += f"\n\nFollowing text: {context_after}"

    delay = retry_cfg.initial_delay_seconds
    for attempt in range(retry_cfg.max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
            )
            content = response.choices[0].message.content
            return content or "[no description generated]"
        except openai.APIError as e:
            if attempt == retry_cfg.max_retries:
                logger.error("Vision API failed after %d retries: %s", attempt + 1, e)
                return "[image description unavailable]"
            logger.warning("Vision API attempt %d failed: %s. Retrying...", attempt + 1, e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, retry_cfg.max_delay_seconds)

    return "[image description unavailable]"


async def _call_anthropic_vision_api(
    client: anthropic.AsyncAnthropic,
    image_b64: str,
    media_type: str,
    context_before: str,
    context_after: str,
    cfg: DescriberConfig,
    retry_cfg: ApiRetryConfig,
) -> str:
    """Call Anthropic vision API with retry."""
    prompt = (
        "Describe this image concisely for a document knowledge base. "
        "Focus on what the image shows and its significance."
    )
    if context_before:
        prompt += f"\n\nPreceding text: {context_before}"
    if context_after:
        prompt += f"\n\nFollowing text: {context_after}"

    delay = retry_cfg.initial_delay_seconds
    for attempt in range(retry_cfg.max_retries + 1):
        try:
            response = await client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            anthropic.types.ImageBlockParam(
                                type="image",
                                source=anthropic.types.Base64ImageSourceParam(
                                    type="base64",
                                    media_type=cast(
                                        Literal[
                                            "image/jpeg",
                                            "image/png",
                                            "image/gif",
                                            "image/webp",
                                        ],
                                        media_type,
                                    ),
                                    data=image_b64,
                                ),
                            ),
                            anthropic.types.TextBlockParam(type="text", text=prompt),
                        ],
                    }
                ],
            )
            first_block = response.content[0]
            if not isinstance(first_block, anthropic.types.TextBlock):
                return "[no description generated]"
            content = first_block.text
            return content or "[no description generated]"
        except anthropic.APIError as e:
            if attempt == retry_cfg.max_retries:
                logger.error("Anthropic vision API failed after %d retries: %s", attempt + 1, e)
                return "[image description unavailable]"
            logger.warning("Anthropic vision attempt %d failed: %s. Retrying...", attempt + 1, e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, retry_cfg.max_delay_seconds)

    return "[image description unavailable]"


async def describe_image(
    image_path: Path,
    context_before: str,
    context_after: str,
    cfg: DescriberConfig,
    retry_cfg: ApiRetryConfig,
) -> str:
    """Send an image to the configured vision LLM and get a description."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    image_format = image_path.suffix.lstrip(".").lower()
    if image_format == "jpg":
        image_format = "jpeg"
    media_type = f"image/{image_format}"

    if cfg.provider == "anthropic":
        anthropic_client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "test-key")
        )
        return await _call_anthropic_vision_api(
            anthropic_client, image_b64, media_type,
            context_before, context_after,
            cfg, retry_cfg,
        )
    else:
        openai_client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "test-key")
        )
        return await _call_openai_vision_api(
            openai_client, image_b64, image_format,
            context_before, context_after,
            cfg, retry_cfg,
        )


async def replace_image_refs(
    markdown: str,
    output_dir: Path,
    cfg: DescriberConfig,
    retry_cfg: ApiRetryConfig,
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

        description = await describe_image(img_path, context_before, context_after, cfg, retry_cfg)

        desc_block = f"\n\n**[Image: {description}]**\n\n"
        insert_pos = match.start() + offset
        result = result[:insert_pos] + desc_block + result[insert_pos:]
        offset += len(desc_block)

        logger.debug("Described image %d/%d: %s", i + 1, len(matches), img_path.name)

    return result
