# Full Pipeline Demo & Graph Query — Design Spec

**Date:** 2026-03-25
**Status:** Draft

## Problem

The existing `examples/run_demo.sh` demonstrates stages 1–4 of the pipeline (convert, extract, describe, register, graph build) plus status, but never shows the watcher mode or graph querying. There is no `query_graph()` function, no `docpipe query` CLI command, and no CI coverage for graph query quality.

## Goals

1. Users can run **one script** that demonstrates every pipeline feature end-to-end.
2. The knowledge graph is queryable via code (`query_graph()`) and CLI (`docpipe query`).
3. CI verifies graph query quality using an LLM-as-judge.

## Non-Goals

- Multi-provider switching demo at runtime.
- Error handling / corrupted file demos.
- Dashboard mode demo (requires a terminal — not scriptable).

---

## Component 1: `query_graph()` in `graph.py`

### Interface

```python
async def query_graph(
    question: str,
    cfg: GraphConfig,
    mode: str = "mix",
) -> str:
```

### Behavior

- Creates a LightRAG instance via the existing `_get_rag_instance(cfg)`.
- Calls `rag.aquery(question, QueryParam(mode=mode))`.
- Returns the answer string on success.
- On failure, logs the error and returns `""`.

### Parameters

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `question` | `str` | required | The natural-language query |
| `cfg` | `GraphConfig` | required | Same config used for ingestion |
| `mode` | `str` | `"mix"` | LightRAG query mode: `local`, `global`, `hybrid`, `mix` |

### Error handling

Wraps the entire call in try/except, logs to `docpipe.graph` logger, returns empty string. Matches the pattern used by `ingest_document()`.

---

## Component 2: `docpipe query` CLI command

### Usage

```
docpipe query [--config config.yaml] [--mode mix] "What topics are covered?"
```

### Click definition

- `@main.command()`
- `--config` option (default `config.yaml`, same as other commands)
- `--mode` option (default `"mix"`, type `click.Choice(["local", "global", "hybrid", "mix"])`)
- `QUESTION` argument (required, type `str`)

### Behavior

1. Load config via `load_config()`.
2. Set up logging via `_setup_logging()`.
3. Call `asyncio.run(query_graph(question, cfg.graph, mode))`.
4. Print the result to stdout via `console.print()`.
5. If result is empty, print an error message and exit with code 1.

### No lockfile

Graph queries are read-only. No lockfile needed.

---

## Component 3: `examples/run_demo.sh` (rewrite)

Replaces the existing script. Six sections:

### Feature 1: Prerequisites Check
- Same as current: check API keys, detect LibreOffice.
- Print file count in `input/`.

### Feature 2: Single File Ingestion
- `docpipe ingest --config config.yaml kaggle_small.pdf`
- Show first 20 lines of generated markdown.
- Show extracted images count.

### Feature 3: Batch Ingestion
- `docpipe ingest --config config.yaml` (all files).
- Print summary of files processed.

### Feature 4: Watcher Mode
- Write a temporary config override with `debounce_seconds: 5` (fast for demo).
- Move one file (`kaggle_medium.pdf`) out of `input/` into a holding area before batch ingest.
- Start `docpipe run --config watcher_config.yaml` in background (`&`, capture PID).
- Copy the held-back file into `input/`.
- Poll `output/status.json` for the file to appear (timeout 120s).
- Print confirmation that the watcher picked it up.
- Kill the watcher process (`kill $PID`), clean up temp config.

### Feature 5: Graph Query
- Run 2 queries via `docpipe query`:
  1. A broad question: `"What topics are covered across the documents?"`
  2. A specific question: `"What data or statistics are mentioned in the documents?"`
- Print both answers.

### Feature 6: Final Status & Cleanup
- `docpipe status --config config.yaml`
- Print hint: `rm -rf output/ logs/` to reset.

### Config changes for watcher demo

The script generates a temporary `watcher_demo_config.yaml` that is identical to `config.yaml` but with:
```yaml
watcher:
  debounce_seconds: 5
  max_wait_seconds: 30
```

This file is deleted at the end of the watcher section.

---

## Component 4: `tests/test_graph_query.py` + CI job

### Test file: `tests/test_graph_query.py`

Structure follows `test_e2e_anthropic.py` conventions.

#### Fixture: `graph_query_config`
- Creates a tmp_path config with Anthropic describer + OpenAI graph (same as existing e2e tests).
- Points input_dir at `tests/fixtures/`.

#### Test: `test_graph_query_with_llm_judge`
1. Ingest `sample.pdf` via `process_file()`.
2. Read the generated markdown for context.
3. Call `query_graph("What is this document about?", cfg.graph)`.
4. Assert the answer is non-empty.
5. Pass question + answer + markdown excerpt to LLM judge.

#### Judge prompt
```
You are a strict quality judge for a knowledge graph query system.

You will receive:
- A QUESTION that was asked
- An ANSWER returned by the knowledge graph
- The SOURCE MARKDOWN that was ingested into the graph

Score each criterion as PASS or FAIL:

1. **answer_is_relevant**: The answer addresses the question asked,
   not some unrelated topic.
2. **answer_uses_document_content**: The answer references specific
   information from the source document, not generic or hallucinated facts.
3. **answer_is_coherent**: The answer is well-formed, readable, and
   not garbled or truncated.

Respond with ONLY valid JSON:
{
  "answer_is_relevant": "PASS" or "FAIL",
  "answer_uses_document_content": "PASS" or "FAIL",
  "answer_is_coherent": "PASS" or "FAIL",
  "overall": "PASS" or "FAIL",
  "reasoning": "Brief explanation of any FAILs"
}

"overall" is PASS only if ALL criteria pass.
```

#### Judge implementation
Reuses the same pattern as `test_e2e_anthropic.py`:
- Claude Haiku as judge model.
- Truncate markdown to 6000 chars with safe image-block handling.
- Parse JSON response with code-block fallback.

### CI job in `doc-tests.yml`

New job `graph-query`:

```yaml
graph-query:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v4
      with:
        python-version: "3.12"
    - run: uv sync --dev
    - name: Graph query test with LLM judge
      env:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: uv run pytest tests/test_graph_query.py -v --tb=short
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/docpipe/graph.py` | Add `query_graph()` function |
| `src/docpipe/cli.py` | Add `docpipe query` command |
| `examples/run_demo.sh` | Rewrite with all 6 features |
| `tests/test_graph_query.py` | New test file with LLM-as-judge |
| `.github/workflows/doc-tests.yml` | Add `graph-query` job |

## Dependencies

No new dependencies. LightRAG's `QueryParam` is already installed as a transitive dependency of `lightrag`. `anthropic` is already a dev dependency for the judge.
