"""E2E test: ingest -> query graph -> LLM-as-judge verification.

Requires ANTHROPIC_API_KEY and OPENAI_API_KEY in environment.
Run with: pytest tests/test_graph_query.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import pytest
import yaml
from conftest import needs_anthropic_key, needs_openai_key

from docpipe.config import load_config
from docpipe.graph import query_graph
from docpipe.pipeline import process_file

FIXTURES = Path(__file__).parent / "fixtures"

_JUDGE_PROMPT = (
    "You are a strict quality judge for a knowledge graph query system.\n\n"
    "## QUESTION\n{question}\n\n"
    "## ANSWER (from knowledge graph)\n{answer}\n\n"
    "## SOURCE MARKDOWN (ingested into the graph)\n{markdown}\n\n"
    "## Evaluation Criteria\n\n"
    "Score each criterion as PASS or FAIL:\n\n"
    "1. **answer_is_relevant**: The answer addresses the question asked,\n"
    "   not some unrelated topic.\n"
    "2. **answer_uses_document_content**: The answer references specific\n"
    "   information from the source document, not generic or hallucinated facts.\n"
    "3. **answer_is_coherent**: The answer is well-formed, readable, and\n"
    "   not garbled or truncated.\n\n"
    "## Response Format\n\n"
    "Respond with ONLY valid JSON, no other text:\n"
    "{{\n"
    '  "answer_is_relevant": "PASS" or "FAIL",\n'
    '  "answer_uses_document_content": "PASS" or "FAIL",\n'
    '  "answer_is_coherent": "PASS" or "FAIL",\n'
    '  "overall": "PASS" or "FAIL",\n'
    '  "reasoning": "Brief explanation of any FAILs"\n'
    "}}\n\n"
    '"overall" is PASS only if ALL criteria pass.'
)


async def _judge_graph_query(question: str, answer: str, markdown: str) -> dict[str, str]:
    """Use Claude as a judge to evaluate graph query quality."""
    client = anthropic.AsyncAnthropic()

    # Truncate markdown safely
    max_chars = 6000
    if len(markdown) > max_chars:
        truncated = markdown[:max_chars]
        last_img_open = truncated.rfind("**[Image:")
        if last_img_open != -1:
            last_img_close = truncated.find("]**", last_img_open)
            if last_img_close == -1:
                truncated = truncated[:last_img_open].rstrip()
        md_truncated = truncated
    else:
        md_truncated = markdown

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    question=question,
                    answer=answer,
                    markdown=md_truncated,
                ),
            }
        ],
    )

    content = response.content[0].text
    try:
        return json.loads(content)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        if "```" in content:
            json_str = content.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            return json.loads(json_str.strip())  # type: ignore[no-any-return]
        raise


@needs_anthropic_key
@needs_openai_key
class TestGraphQuery:
    @pytest.fixture
    def graph_query_config(self, tmp_path: Path) -> Path:
        """Config for graph query E2E test."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "markdown").mkdir()
        (output_dir / "images").mkdir()

        config = {
            "input_dir": str(FIXTURES),
            "output_dir": str(output_dir),
            "describer": {
                "provider": "anthropic",
                "max_tokens": 300,
                "include_context": True,
                "context_chars": 500,
                "batch_size": 5,
                "anthropic": {
                    "model": "claude-haiku-4-5-20251001",
                },
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
                "provider": "openai",
                "storage": "file",
                "store_dir": str(output_dir / "lightrag_store"),
                "max_tokens": 4096,
                "chunk_size": 1200,
                "chunk_overlap": 100,
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
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
    async def test_graph_query_with_llm_judge(
        self, graph_query_config: Path, tmp_path: Path
    ) -> None:
        """Ingest PDF -> query graph -> LLM judge evaluates answer quality."""
        cfg = load_config(graph_query_config)

        pdf = FIXTURES / "real-world" / "invoice.pdf"
        if not pdf.exists():
            pytest.skip("invoice.pdf fixture not found")

        # Stage 1: Ingest document into graph
        success = await process_file(pdf, cfg)
        assert success, "Pipeline failed to process invoice.pdf"

        # Read generated markdown for judge context
        md_path = cfg.output_dir / "markdown" / "invoice.md"
        assert md_path.exists(), "Markdown file not created"
        markdown = md_path.read_text()

        # Stage 2: Query the graph
        question = "What products and customer details are in the invoice?"
        answer = await query_graph(question, cfg.graph)

        assert answer is not None, "query_graph returned None (error)"
        assert len(answer) > 0, "query_graph returned empty string"

        # Stage 3: LLM judge
        verdict = await _judge_graph_query(question, answer, markdown)

        for criterion, result in verdict.items():
            if criterion not in ("overall", "reasoning"):
                assert result == "PASS", (
                    f"LLM judge FAILED on '{criterion}': {verdict.get('reasoning', 'no reason')}"
                )

        assert verdict["overall"] == "PASS", (
            f"LLM judge overall FAIL: {verdict.get('reasoning', 'no reason')}"
        )
