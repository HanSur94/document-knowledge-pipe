# Full Pipeline Demo & Graph Query — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add graph querying (function + CLI), rewrite the example demo script to cover every pipeline stage, and add CI coverage for graph query quality via LLM-as-judge.

**Architecture:** Four independent components — `query_graph()` in the graph module, a `docpipe query` CLI command, a rewritten `examples/run_demo.sh`, and a new `tests/test_graph_query.py` with CI job. Components 2–4 depend on Component 1.

**Tech Stack:** Python 3.11+, LightRAG (QueryParam), Click, Anthropic SDK (judge), bash, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-03-25-full-pipeline-demo-design.md`

---

### Task 1: Add `query_graph()` to `graph.py` — test

**Files:**
- Create: `tests/test_graph_query_unit.py`
- Reference: `src/docpipe/graph.py:62-78` (pattern from `ingest_document`)

- [ ] **Step 1: Write failing unit test for `query_graph`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_query_unit.py -v`
Expected: FAIL — `ImportError` or `AttributeError` because `query_graph` doesn't exist yet.

- [ ] **Step 3: Commit test**

```bash
git add tests/test_graph_query_unit.py
git commit -m "test: add unit tests for query_graph function"
```

---

### Task 2: Add `query_graph()` to `graph.py` — implementation

**Files:**
- Modify: `src/docpipe/graph.py` (append after `rebuild_graph`, currently ends at line 103)

- [ ] **Step 1: Implement `query_graph`**

Add at the end of `src/docpipe/graph.py` (after line 103):

```python
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
```

- [ ] **Step 2: Run unit tests to verify they pass**

Run: `uv run pytest tests/test_graph_query_unit.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 3: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -v --ignore=tests/test_e2e_anthropic.py --ignore=tests/test_full_system.py --ignore=tests/test_graph_query.py -x`
Expected: All existing tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/docpipe/graph.py
git commit -m "feat: add query_graph() function to graph module"
```

---

### Task 3: Add `docpipe query` CLI command — test

**Files:**
- Create: `tests/test_cli_query.py`
- Reference: `src/docpipe/cli.py:234-241` (pattern from `status` command)

- [ ] **Step 1: Write failing CLI test**

```python
"""Tests for docpipe query CLI command."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import yaml
from click.testing import CliRunner

from docpipe.cli import main


class TestQueryCommand:
    def test_query_prints_answer(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (tmp_path / "logs").mkdir()

        config = {
            "input_dir": str(tmp_path / "input"),
            "output_dir": str(output_dir),
            "graph": {
                "provider": "openai",
                "store_dir": str(output_dir / "lightrag_store"),
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
            },
            "logging": {
                "level": "DEBUG",
                "file": str(tmp_path / "logs" / "docpipe.log"),
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))

        runner = CliRunner()

        with patch("docpipe.cli.query_graph", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "The documents cover data science."
            result = runner.invoke(
                main, ["query", "--config", str(cfg_path), "What topics?"]
            )

        assert result.exit_code == 0
        assert "data science" in result.output

    def test_query_exits_1_on_failure(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (tmp_path / "logs").mkdir()

        config = {
            "input_dir": str(tmp_path / "input"),
            "output_dir": str(output_dir),
            "graph": {
                "provider": "openai",
                "store_dir": str(output_dir / "lightrag_store"),
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
            },
            "logging": {
                "level": "DEBUG",
                "file": str(tmp_path / "logs" / "docpipe.log"),
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))

        runner = CliRunner()

        with patch("docpipe.cli.query_graph", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = None
            result = runner.invoke(
                main, ["query", "--config", str(cfg_path), "What topics?"]
            )

        assert result.exit_code == 1

    def test_query_accepts_mode_flag(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (tmp_path / "logs").mkdir()

        config = {
            "input_dir": str(tmp_path / "input"),
            "output_dir": str(output_dir),
            "graph": {
                "provider": "openai",
                "store_dir": str(output_dir / "lightrag_store"),
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
            },
            "logging": {
                "level": "DEBUG",
                "file": str(tmp_path / "logs" / "docpipe.log"),
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))

        runner = CliRunner()

        with patch("docpipe.cli.query_graph", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "answer"
            result = runner.invoke(
                main,
                ["query", "--config", str(cfg_path), "--mode", "local", "Question?"],
            )

        assert result.exit_code == 0
        mock_query.assert_called_once()
        call_args = mock_query.call_args
        assert call_args.args[2] == "local"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_query.py -v`
Expected: FAIL — `query` command not found in Click group.

- [ ] **Step 3: Commit test**

```bash
git add tests/test_cli_query.py
git commit -m "test: add CLI tests for docpipe query command"
```

---

### Task 4: Add `docpipe query` CLI command — implementation

**Files:**
- Modify: `src/docpipe/cli.py:18` (update import)
- Modify: `src/docpipe/cli.py:242` (append new command)

- [ ] **Step 1: Update import on line 18**

Change `src/docpipe/cli.py` line 18:

```python
# Before:
from docpipe.graph import rebuild_graph

# After:
from docpipe.graph import query_graph, rebuild_graph
```

- [ ] **Step 2: Add query command after the `status` command (after line 241)**

```python
@main.command()
@click.option("--config", "config_path", default=DEFAULT_CONFIG, help="Path to config.yaml")
@click.option(
    "--mode",
    default="mix",
    type=click.Choice(["local", "global", "hybrid", "naive", "mix", "bypass"]),
    help="Query mode",
)
@click.argument("question")
def query(config_path: str, mode: str, question: str) -> None:
    """Query the knowledge graph."""
    cfg = load_config(Path(config_path))
    _setup_logging(cfg)

    result = asyncio.run(query_graph(question, cfg.graph, mode))
    if result is None:
        console.print("[red]Query failed. Check logs for details.[/red]")
        sys.exit(1)

    console.print(result)
```

- [ ] **Step 3: Run CLI tests**

Run: `uv run pytest tests/test_cli_query.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/test_e2e_anthropic.py --ignore=tests/test_full_system.py --ignore=tests/test_graph_query.py -x`
Expected: All tests PASS.

- [ ] **Step 5: Lint check**

Run: `uv run ruff check src/docpipe/cli.py src/docpipe/graph.py`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add src/docpipe/cli.py
git commit -m "feat: add docpipe query CLI command"
```

---

### Task 5: Rewrite `examples/run_demo.sh`

**Files:**
- Modify: `examples/run_demo.sh` (full rewrite)
- Reference: `src/docpipe/cli.py` (available CLI commands)

- [ ] **Step 1: Write the new demo script**

Replace `examples/run_demo.sh` with:

```bash
#!/usr/bin/env bash
# =============================================================================
# DocPipe Demo — full pipeline walkthrough
#
# Demonstrates EVERY stage of the pipeline:
#   1. Single file ingestion (convert → extract → describe → register → graph)
#   2. Batch ingestion of all documents
#   3. File watcher (automatic processing on file drop)
#   4. Knowledge graph querying
#   5. Pipeline status
#
# Prerequisites:
#   - Python 3.11+
#   - An API key: OPENAI_API_KEY (required for graph)
#     and optionally ANTHROPIC_API_KEY
#   - LibreOffice (optional, needed for non-PDF formats)
#
# Usage:
#   cd examples
#   export OPENAI_API_KEY=sk-...
#   bash run_demo.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

banner() { echo -e "\n${CYAN}${BOLD}═══ $1 ═══${NC}\n"; }
info()   { echo -e "${GREEN}→${NC} $1"; }
warn()   { echo -e "${YELLOW}⚠${NC} $1"; }

# ---------------------------------------------------------------------------
# 0. Check prerequisites
# ---------------------------------------------------------------------------
banner "DocPipe Demo"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY must be set (required for knowledge graph).${NC}"
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

if command -v soffice &>/dev/null; then
    info "LibreOffice found — all formats supported"
else
    warn "LibreOffice not found — only PDF files will be processed"
fi

echo ""
info "Input directory: ./input  ($(ls input | wc -l | tr -d ' ') files)"
info "Config file:     ./config.yaml"
echo ""

# ---------------------------------------------------------------------------
# Setup: hold back one file for the watcher demo
# ---------------------------------------------------------------------------
HELD_FILE="kaggle_medium.pdf"
HOLD_DIR="$(mktemp -d)"

if [ -f "input/${HELD_FILE}" ]; then
    mv "input/${HELD_FILE}" "${HOLD_DIR}/${HELD_FILE}"
    info "Held back ${HELD_FILE} for watcher demo"
fi

# ---------------------------------------------------------------------------
# 1. Ingest a single PDF
# ---------------------------------------------------------------------------
banner "Feature 1: Single File Ingestion"
info "Processing one PDF to show the basic pipeline..."
info "  docpipe ingest --config config.yaml kaggle_small.pdf"
echo ""

docpipe ingest --config config.yaml kaggle_small.pdf

echo ""
if [ -f output/markdown/kaggle_small.md ]; then
    info "Markdown output (first 20 lines):"
    echo "---"
    head -20 output/markdown/kaggle_small.md
    echo "..."
    echo "---"
fi

echo ""
img_count=$(find output/images/ -name "kaggle_small*" -type f 2>/dev/null | wc -l | tr -d ' ')
info "Images extracted: ${img_count}"

# ---------------------------------------------------------------------------
# 2. Batch ingest all documents
# ---------------------------------------------------------------------------
banner "Feature 2: Batch Ingestion (all formats)"
info "Processing all remaining documents in input/..."
info "  docpipe ingest --config config.yaml"
echo ""

docpipe ingest --config config.yaml

# ---------------------------------------------------------------------------
# 3. Pipeline status (mid-run)
# ---------------------------------------------------------------------------
banner "Feature 3: Pipeline Status"
info "  docpipe status --config config.yaml"
echo ""

docpipe status --config config.yaml

# ---------------------------------------------------------------------------
# 4. Watcher mode
# ---------------------------------------------------------------------------
banner "Feature 4: File Watcher"

if [ ! -f "${HOLD_DIR}/${HELD_FILE}" ]; then
    warn "Held-back file not available, skipping watcher demo"
else
    info "Starting watcher with fast debounce (5s)..."

    # Generate temp config with short debounce
    python3 -c '
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path("config.yaml").read_text())
cfg.setdefault("watcher", {})["debounce_seconds"] = 5
cfg.setdefault("watcher", {})["max_wait_seconds"] = 30
pathlib.Path("watcher_demo_config.yaml").write_text(yaml.dump(cfg))
'

    # Start watcher in background
    docpipe run --config watcher_demo_config.yaml &
    WATCHER_PID=$!
    trap 'kill $WATCHER_PID 2>/dev/null || true' EXIT

    sleep 2
    info "Watcher running (PID ${WATCHER_PID})"
    info "Dropping ${HELD_FILE} into input/..."
    cp "${HOLD_DIR}/${HELD_FILE}" "input/${HELD_FILE}"

    # Poll for completion
    info "Waiting for watcher to process ${HELD_FILE}..."
    TIMEOUT=120
    ELAPSED=0
    while [ $ELAPSED -lt $TIMEOUT ]; do
        if [ -f output/status.json ]; then
            STATUS=$(python3 -c "
import json, sys
data = json.load(open('output/status.json'))
files = data.get('files', {})
info = files.get('${HELD_FILE}', {})
print(info.get('status', 'unknown'))
" 2>/dev/null || echo "unknown")
            if [ "$STATUS" = "done" ]; then
                echo ""
                info "Watcher processed ${HELD_FILE} successfully!"
                break
            elif [ "$STATUS" = "failed" ]; then
                echo ""
                warn "Watcher reported failure for ${HELD_FILE}"
                break
            fi
        fi
        sleep 2
        ELAPSED=$((ELAPSED + 2))
        printf "."
    done

    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo ""
        warn "Timed out waiting for watcher (${TIMEOUT}s)"
    fi

    # Cleanup watcher
    kill $WATCHER_PID 2>/dev/null || true
    wait $WATCHER_PID 2>/dev/null || true
    trap - EXIT
    rm -f watcher_demo_config.yaml
    info "Watcher stopped"
fi

# Restore held file
if [ -f "${HOLD_DIR}/${HELD_FILE}" ]; then
    mv "${HOLD_DIR}/${HELD_FILE}" "input/${HELD_FILE}"
fi
rm -rf "${HOLD_DIR}"

# ---------------------------------------------------------------------------
# 5. Query the knowledge graph
# ---------------------------------------------------------------------------
banner "Feature 5: Knowledge Graph Query"
info "Querying the graph built from all ingested documents..."
echo ""

info "Query 1: What topics are covered across the documents?"
echo "---"
docpipe query --config config.yaml "What topics are covered across the documents?"
echo ""
echo "---"
echo ""

info "Query 2: What data or statistics are mentioned in the documents?"
echo "---"
docpipe query --config config.yaml "What data or statistics are mentioned in the documents?"
echo ""
echo "---"

# ---------------------------------------------------------------------------
# 6. Final status
# ---------------------------------------------------------------------------
banner "Feature 6: Final Status"
docpipe status --config config.yaml

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
banner "Demo Complete"
info "All outputs are in ./output/"
info ""
info "What to try next:"
info "  1. Start the live dashboard:  docpipe run --config config.yaml --dashboard"
info "  2. Query with a different mode:  docpipe query --mode local \"Your question\""
info "  3. Rebuild the knowledge graph:  docpipe ingest --config config.yaml --rebuild-graph"
info ""
info "To reset: rm -rf output/ logs/"
info ""
info "See README.md for more details."
```

- [ ] **Step 2: Verify script syntax**

Run: `bash -n examples/run_demo.sh`
Expected: No syntax errors.

- [ ] **Step 3: Commit**

```bash
git add examples/run_demo.sh
git commit -m "feat: rewrite run_demo.sh to cover entire pipeline

Demonstrates: single ingest, batch ingest, status, watcher mode
(with background process + file drop), knowledge graph querying,
and final status. Replaces the previous partial demo."
```

---

### Task 6: Add `tests/test_graph_query.py` — E2E test with LLM-as-judge

**Files:**
- Create: `tests/test_graph_query.py`
- Reference: `tests/test_e2e_anthropic.py:22-105` (judge pattern)
- Reference: `tests/conftest.py:10-17` (skip decorators)

- [ ] **Step 1: Write the E2E test file**

```python
"""E2E test: ingest → query graph → LLM-as-judge verification.

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


async def _judge_graph_query(
    question: str, answer: str, markdown: str
) -> dict[str, str]:
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
        """Ingest PDF → query graph → LLM judge evaluates answer quality."""
        cfg = load_config(graph_query_config)

        pdf = FIXTURES / "sample.pdf"
        if not pdf.exists():
            pytest.skip("sample.pdf fixture not found")

        # Stage 1: Ingest document into graph
        success = await process_file(pdf, cfg)
        assert success, "Pipeline failed to process sample.pdf"

        # Read generated markdown for judge context
        md_path = cfg.output_dir / "markdown" / "sample.md"
        assert md_path.exists(), "Markdown file not created"
        markdown = md_path.read_text()

        # Stage 2: Query the graph
        question = "What is this document about?"
        answer = await query_graph(question, cfg.graph)

        assert answer is not None, "query_graph returned None (error)"
        assert len(answer) > 0, "query_graph returned empty string"

        # Stage 3: LLM judge
        verdict = await _judge_graph_query(question, answer, markdown)

        for criterion, result in verdict.items():
            if criterion not in ("overall", "reasoning"):
                assert result == "PASS", (
                    f"LLM judge FAILED on '{criterion}': "
                    f"{verdict.get('reasoning', 'no reason')}"
                )

        assert verdict["overall"] == "PASS", (
            f"LLM judge overall FAIL: {verdict.get('reasoning', 'no reason')}"
        )
```

- [ ] **Step 2: Lint check**

Run: `uv run ruff check tests/test_graph_query.py`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add tests/test_graph_query.py
git commit -m "test: add graph query E2E test with LLM-as-judge"
```

---

### Task 7: Add `graph-query` CI job to `doc-tests.yml`

**Files:**
- Modify: `.github/workflows/doc-tests.yml:137` (append after `full-system` job)

- [ ] **Step 1: Add the new job**

Append at the end of `.github/workflows/doc-tests.yml`:

```yaml

  graph-query:
    needs: [e2e-anthropic]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4
        with:
          python-version: "3.12"

      - run: uv sync --dev

      # This test ingests sample.pdf (calls OpenAI embedding API),
      # then queries the graph and judges the answer with Claude Haiku.
      - name: Graph query E2E test with LLM judge
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: uv run pytest tests/test_graph_query.py -v --tb=short
```

- [ ] **Step 2: Validate YAML syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/doc-tests.yml'))"`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/doc-tests.yml
git commit -m "ci: add graph-query job with LLM-as-judge to doc-tests workflow"
```

---

### Task 8: Final verification

**Files:** All modified files from Tasks 1–7.

- [ ] **Step 1: Run linter on all changed files**

Run: `uv run ruff check src/docpipe/graph.py src/docpipe/cli.py tests/test_graph_query_unit.py tests/test_cli_query.py tests/test_graph_query.py`
Expected: No errors.

- [ ] **Step 2: Run formatter check**

Run: `uv run ruff format --check src/docpipe/graph.py src/docpipe/cli.py tests/test_graph_query_unit.py tests/test_cli_query.py tests/test_graph_query.py`
Expected: No reformatting needed.

- [ ] **Step 3: Run type checker on modified source**

Run: `uv run mypy src/docpipe/graph.py src/docpipe/cli.py`
Expected: No errors.

- [ ] **Step 4: Run all unit/integration tests (excluding E2E that need real API keys)**

Run: `uv run pytest tests/ -v --ignore=tests/test_e2e_anthropic.py --ignore=tests/test_full_system.py --ignore=tests/test_graph_query.py -x`
Expected: All tests PASS.

- [ ] **Step 5: Verify demo script syntax**

Run: `bash -n examples/run_demo.sh`
Expected: No errors.

- [ ] **Step 6: Verify CI YAML is valid**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/doc-tests.yml'))"`
Expected: No errors.
