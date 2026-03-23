"""End-to-end test with real Anthropic API calls + LLM-as-judge verification.

Requires ANTHROPIC_API_KEY in environment. Skipped if not set.
Run with: pytest tests/test_e2e_anthropic.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
import pytest
import yaml

from docpipe.config import load_config
from docpipe.pipeline import process_file

needs_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

FIXTURES = Path(__file__).parent / "fixtures"

_JUDGE_PROMPT = (
    "You are a strict quality judge for a document ingestion pipeline.\n"
    "You will receive two outputs produced by the pipeline from a PDF.\n\n"
    "## Output 1: Generated Markdown\n{markdown}\n\n"
    "## Output 2: Registry Entry\n{registry}\n\n"
    "## Evaluation Criteria\n\n"
    "Score each criterion as PASS or FAIL:\n\n"
    "1. **markdown_has_content**: The markdown contains extracted text\n"
    "   (at least several paragraphs). PASS even if the text is Lorem\n"
    "   Ipsum or placeholder — what matters is that extraction worked.\n"
    "   FAIL only if the markdown is empty or contains only error markers.\n"
    "2. **markdown_preserves_structure**: The markdown has recognizable\n"
    "   structure (paragraphs, headings, or sections).\n"
    "3. **image_descriptions_meaningful**: If the markdown contains image\n"
    '   descriptions (lines with "**[Image:"), they describe what the\n'
    "   image shows. PASS if descriptions exist. If no images, PASS.\n"
    "4. **registry_has_summary**: The registry entry contains a summary\n"
    "   that describes the document content (even if content is Lorem\n"
    '   Ipsum). FAIL only if it says "Summary unavailable".\n'
    "5. **registry_has_topics**: The registry has topic tags (not just -).\n"
    "6. **no_error_markers**: No error placeholders like\n"
    '   "[image description unavailable]" or "Processing failed".\n\n'
    "## Response Format\n\n"
    "Respond with ONLY valid JSON, no other text:\n"
    "{{\n"
    '  "markdown_has_content": "PASS" or "FAIL",\n'
    '  "markdown_preserves_structure": "PASS" or "FAIL",\n'
    '  "image_descriptions_meaningful": "PASS" or "FAIL",\n'
    '  "registry_has_summary": "PASS" or "FAIL",\n'
    '  "registry_has_topics": "PASS" or "FAIL",\n'
    '  "no_error_markers": "PASS" or "FAIL",\n'
    '  "overall": "PASS" or "FAIL",\n'
    '  "reasoning": "Brief explanation of any FAILs"\n'
    "}}\n\n"
    '"overall" is PASS only if ALL criteria pass.'
)


async def _judge_output(markdown: str, registry: str) -> dict[str, str]:
    """Use Claude as a judge to evaluate pipeline output quality."""
    client = anthropic.AsyncAnthropic()

    # Truncate to avoid token limits
    md_truncated = markdown[:4000] + ("..." if len(markdown) > 4000 else "")
    reg_truncated = registry[:1000]

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    markdown=md_truncated,
                    registry=reg_truncated,
                ),
            }
        ],
    )

    content = response.content[0].text
    # Parse JSON from response
    try:
        return json.loads(content)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        if "```" in content:
            json_str = content.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            return json.loads(json_str.strip())  # type: ignore[no-any-return]
        raise


@needs_anthropic_key
class TestE2EAnthropic:
    @pytest.fixture
    def anthropic_config(self, tmp_path: Path) -> Path:
        """Create a config using Anthropic as the LLM provider."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "markdown").mkdir()
        (output_dir / "images").mkdir()

        config = {
            "input_dir": str(FIXTURES),
            "output_dir": str(output_dir),
            "describer": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "include_context": True,
                "context_chars": 500,
                "batch_size": 5,
            },
            "registry": {
                "filename": "registry.md",
                "summary_max_words": 30,
                "include_fields": [
                    "filename",
                    "author",
                    "date",
                    "summary",
                    "topics",
                    "pages",
                    "images",
                ],
            },
            "graph": {
                "storage": "file",
                "store_dir": str(output_dir / "lightrag_store"),
                "model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "max_tokens": 4096,
                "chunk_size": 1200,
                "chunk_overlap": 100,
            },
            "api_retry": {
                "max_retries": 2,
                "initial_delay_seconds": 1,
                "max_delay_seconds": 10,
            },
            "logging": {
                "level": "DEBUG",
                "file": str(tmp_path / "logs" / "docpipe.log"),
                "max_size_mb": 1,
                "backup_count": 1,
            },
        }

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))
        return cfg_path

    @pytest.mark.asyncio
    async def test_pdf_with_anthropic_describer(
        self, anthropic_config: Path, tmp_path: Path
    ) -> None:
        """Full pipeline: PDF → extract → describe (Anthropic) → registry → judge."""
        cfg = load_config(anthropic_config)

        pdf = FIXTURES / "sample.pdf"
        if not pdf.exists():
            pytest.skip("sample.pdf fixture not found")

        # Mock only the graph ingestion (needs OpenAI embedding)
        from unittest.mock import AsyncMock, patch

        with patch(
            "docpipe.pipeline.ingest_document",
            new_callable=AsyncMock,
            return_value=True,
        ):
            success = await process_file(pdf, cfg)

        assert success, "Pipeline failed to process PDF"

        # Read outputs
        md_path = cfg.output_dir / "markdown" / "sample.md"
        assert md_path.exists(), "Markdown file not created"
        markdown = md_path.read_text()

        reg_path = cfg.output_dir / "registry.md"
        assert reg_path.exists(), "Registry file not created"
        registry = reg_path.read_text()

        status_path = cfg.output_dir / "status.json"
        assert status_path.exists(), "Status file not created"

        # LLM-as-judge: evaluate output quality
        verdict = await _judge_output(markdown, registry)

        # Report individual criteria for debugging
        for criterion, result in verdict.items():
            if criterion not in ("overall", "reasoning"):
                assert result == "PASS", (
                    f"LLM judge FAILED on '{criterion}': {verdict.get('reasoning', 'no reason')}"
                )

        assert verdict["overall"] == "PASS", (
            f"LLM judge overall FAIL: {verdict.get('reasoning', 'no reason')}"
        )
