# Document Knowledge Pipe — Design Specification

## Overview

A standalone Python package that watches a folder for documents, converts them to structured markdown (preserving reading order with AI-generated image descriptions), builds a knowledge graph via LightRAG, and maintains a concise AI-readable registry. Designed for integration with a multi-agent system.

**Repository:** `document-knowledge-pipe`
**Location:** `~/document-knowledge-pipe/`
**Target platforms:** Windows 10, Windows 11 (also works on macOS/Linux)
**Python:** 3.10+
**Package manager:** uv

## Prerequisites

1. Python 3.10+
2. `uv` (package manager)
3. LibreOffice (for non-PDF document conversion)
4. OpenAI API key (GPT-4o mini / GPT-5)

## Architecture

### Approach: Linear Pipeline

Each stage is a separate Python module. The watcher triggers the full pipeline per file, **processing files sequentially** (never in parallel) to avoid concurrent write issues with `registry.md` and `status.json`. Simple, debuggable, easy to test each stage independently. The 60-second debounce means we never process more than a handful of files at once.

### Project Structure

```
document-knowledge-pipe/
├── src/
│   └── docpipe/
│       ├── __init__.py
│       ├── cli.py              # CLI: run, ingest, status, init
│       ├── config.py           # YAML config loading + defaults
│       ├── watcher.py          # Watchdog monitor + 60s debounce
│       ├── converter.py        # Non-PDF → PDF via LibreOffice headless
│       ├── extractor.py        # PDF → markdown + images via PyMuPDF4LLM
│       ├── describer.py        # Replace image refs with GPT-4o mini vision descriptions
│       ├── registry.py         # Build/update registry.md index
│       ├── graph.py            # LightRAG knowledge graph ingestion
│       ├── status.py           # Status tracking, dashboard, status.json
│       └── pipeline.py         # Orchestrates the full flow per file
├── tests/
├── docs/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint, typecheck, test, build
│       ├── release.yml         # Publish to PyPI on version tags
│       └── docs.yml            # Wiki generation via wiki-gen-action
├── pyproject.toml
├── config.yaml                 # Default configuration
├── .env                        # OPENAI_API_KEY (excluded from git and package build)
├── .gitignore
└── README.md
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `pymupdf4llm` | PDF → structured markdown + image extraction |
| `openai` | GPT-4o mini / GPT-5 vision API |
| `lightrag-hku` | Knowledge graph construction |
| `watchdog` | File system monitoring |
| `click` | CLI interface |
| `python-dotenv` | Environment variable loading |
| `pyyaml` | Configuration file parsing |
| `rich` | Terminal dashboard + status display |

Dev dependencies: `pytest`, `ruff`, `mypy`, `pytest-cov`

## Data Flow

```
INPUT FOLDER (watched)
    │
    ├── report.pdf ──────────────────────┐
    ├── slides.pptx ─→ [LibreOffice] ─→ PDF ─┐
    ├── data.xlsx ───→ [LibreOffice] ─→ PDF ─┤
    ├── memo.doc ────→ [LibreOffice] ─→ PDF ─┤
    │                                         │
    │              ┌──────────────────────────┘
    │              ▼
    │    [PyMuPDF4LLM] ─→ raw markdown + extracted images
    │              │
    │              ▼
    │    [GPT-4o mini vision] ─→ image descriptions
    │              │
    │              ▼
    │    [Post-processor] ─→ replaces image refs with
    │              │         contextual descriptions
    │              ▼
    │    final .md file ─→ OUTPUT FOLDER
    │              │         ├── markdown/
    │              │         │    ├── report.md
    │              │         │    ├── slides.md
    │              │         │    └── data.md
    │              │         ├── images/
    │              │         │    ├── report_img001.png
    │              │         │    └── slides_img001.png
    │              │         ├── registry.md
    │              │         └── status.json
    │              ▼
    │    [LightRAG] ─→ lightrag_store/
    │                    ├── graph_chunk_entity_relation.json
    │                    ├── kv_store_*.json
    │                    └── vdb_*.json

WATCHER (60s trailing-edge debounce, 5min max wait)
    │
    └── on file change/add → re-runs full pipeline for that file (restart from scratch)
        on file delete → removes .md + images + registry entry
                         (graph is NOT pruned — rebuilt on next `docpipe ingest --rebuild-graph`)
```

### Pipeline Stages (per file)

0. **Cleanup** (`pipeline.py`): Before processing, remove any orphaned output files from a previous partial run of this document (images, temp PDFs). This ensures a clean restart.
1. **Converter** (`converter.py`): If the file is not a PDF, convert it to PDF via LibreOffice headless (invoked with `--headless --norestore --safe-mode` to disable macros). PDFs pass through unchanged. **Timeout: 120 seconds** — if LibreOffice hangs, the subprocess is killed and the file is marked as `failed`.
2. **Extractor** (`extractor.py`): Use PyMuPDF4LLM to extract structured markdown with images in reading order. Images are saved to `output/images/`. Empty or 0-page PDFs are skipped with a warning.
3. **Describer** (`describer.py`): Find image references in the markdown. For each image, send it to GPT-4o mini vision along with surrounding text context (configurable, default 500 chars before/after). If an image has no preceding text (e.g., first element), the document title is used as context. Replace the image reference with a contextual description while keeping a reference to the original image file. Images are processed **sequentially in batches** (configurable `batch_size`, default 5).
4. **Registry** (`registry.py`): Use GPT-4o mini to generate a 1-2 sentence summary and topic tags. Update `registry.md` with the file entry. Failed files appear as: `| corrupt.pdf | - | - | Processing failed: [reason] | - | - | - |`
5. **Graph** (`graph.py`): Feed the final markdown into LightRAG for knowledge graph construction. On re-ingestion of a changed file, the document is inserted with the same document ID — LightRAG's `ainsert` deduplicates by content hash, so changed content creates new graph entries while unchanged content is skipped.
6. **Status** (`status.py`): Update `status.json` with processing results.

## Configuration

All settings are in `config.yaml` with sensible defaults. Users only need to set `input_dir` and their `OPENAI_API_KEY` in `.env` to get started.

```yaml
# Paths
input_dir: "./input"
output_dir: "./output"

# Watcher
watcher:
  enabled: true
  debounce_seconds: 60          # trailing-edge: wait 60s of silence after last event
  max_wait_seconds: 300         # maximum wait before processing even if events continue
  poll_interval_seconds: 1      # fallback polling for network drives
  watch_subdirectories: false   # flat input folder only (avoids filename collisions)

# Converter (LibreOffice)
converter:
  libreoffice_path: null      # auto-detect, or set manually
  timeout_seconds: 120        # kill LibreOffice if it hangs
  supported_extensions:
    - .doc
    - .docx
    - .xls
    - .xlsx
    - .ppt
    - .pptx
    - .odt
    - .ods
    - .odp
    - .rtf
    - .html
    - .epub

# Extractor (PyMuPDF4LLM)
extractor:
  write_images: true
  image_format: "png"
  dpi: 150

# Describer (GPT-4o mini vision)
describer:
  model: "gpt-4o-mini"
  max_tokens: 300
  include_context: true
  context_chars: 500
  batch_size: 5

# Registry
registry:
  filename: "registry.md"
  summary_max_words: 30
  include_fields:
    - filename
    - author
    - date
    - summary
    - topics
    - pages
    - images

# Knowledge Graph (LightRAG)
graph:
  storage: "file"
  store_dir: "./output/lightrag_store"
  model: "gpt-4o-mini"            # LLM for entity/relation extraction
  embedding_model: "text-embedding-3-small"  # OpenAI embedding model
  max_tokens: 4096
  chunk_size: 1200
  chunk_overlap: 100

# Describer API retry
describer_retry:
  max_retries: 3
  initial_delay_seconds: 1
  max_delay_seconds: 30          # exponential backoff cap

# Logging
logging:
  level: "INFO"
  file: "./logs/docpipe.log"
  max_size_mb: 50                # rotate log after 50MB
  backup_count: 3                # keep 3 rotated log files
```

## CLI Interface

```
docpipe init                          # Create default config.yaml + folder structure
docpipe run                           # Start watcher (foreground, blocking)
docpipe run --dashboard               # Start watcher with live rich dashboard
docpipe ingest                        # One-shot: process all files in input_dir
docpipe ingest report.pdf             # One-shot: process single file (resolved relative to input_dir)
docpipe ingest --rebuild-graph        # Re-process all files and rebuild LightRAG graph from scratch
docpipe status                        # Show rich status table
```

All commands accept `--config path/to/config.yaml` (defaults to `./config.yaml`).

**Note:** `docpipe run` is a **foreground blocking process** (not a daemon). This is the safest cross-platform behavior for Windows. To run in background on Windows, use `start /B docpipe run` or run it as a scheduled task. A lockfile (`output/.docpipe.lock`) prevents concurrent `docpipe run` and `docpipe ingest` from processing the same files simultaneously.

## Status & Monitoring

Three ways to check pipeline status:

### 1. Terminal Table (`docpipe status`)

```
╭─ DocPipe Status ──────────────────────────────────────────────╮
│ Watcher: ● RUNNING    │ Input: ./input (12 files)             │
│ Uptime:  2h 34m       │ Output: ./output (10 markdown files)  │
├───────────────────────────────────────────────────────────────┤
│ File              Status     Pages  Images  Graph   Updated   │
│ report_q4.pdf     ● done       12      3     ✓    10:23 AM   │
│ slides.pptx       ● done        8      5     ✓    10:24 AM   │
│ memo.doc          ⟳ processing  -      -     -       -        │
│ data.xlsx         ● done        3      0     ✓    10:25 AM   │
│ corrupt.pdf       ✗ failed      -      -     -    10:22 AM   │
├───────────────────────────────────────────────────────────────┤
│ Graph: 142 entities │ 89 relations │ 10 documents ingested    │
│ API usage: 1,247 tokens today │ Est. cost: $0.002             │
╰───────────────────────────────────────────────────────────────╯
```

### 2. Live Dashboard (`docpipe run --dashboard`)

Auto-refreshing version of the above, updates in real-time as files are processed.

### 3. Programmatic (`status.json`)

```json
{
  "watcher": "running",
  "started_at": "2026-03-23T10:00:00",
  "last_heartbeat_at": "2026-03-23T12:34:56",
  "files": {
    "report_q4.pdf": {
      "status": "done",
      "pages": 12,
      "images": 3,
      "graph_ingested": true,
      "last_processed": "2026-03-23T10:23:00",
      "md_path": "markdown/report_q4.md"
    }
  },
  "graph": {
    "entities": 142,
    "relations": 89,
    "documents": 10
  },
  "api_usage": {
    "tokens_today": 1247,
    "estimated_cost_usd": 0.002,
    "day_reset": "2026-03-23",
    "tokens_total": 45230
  }
}
```

## Registry Format

AI-readable index at `output/registry.md`:

```markdown
# Document Registry

| File | Author | Date | Summary | Topics | Pages | Images |
|------|--------|------|---------|--------|-------|--------|
| report_q4.md | J. Smith | 2026-01-15 | Q4 financial results and projections for FY2026 | finance, quarterly | 12 | 3 |
| slides.md | H. Suhr | 2026-03-01 | Technical architecture overview for new platform | architecture, platform | 8 | 5 |
```

Concise enough for an agent to scan without flooding its context, detailed enough to decide which files to read.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| LibreOffice not found | Clear error message with install link, exit |
| LibreOffice hangs | Kill subprocess after 120s (configurable), mark file as `failed` |
| GPT-4o mini API failure | Retry 3x with exponential backoff (1s → 30s cap), then insert `[image description unavailable]` placeholder |
| Corrupt/unreadable file | Log warning, skip file, mark as `failed` in registry and status.json |
| Empty/0-page PDF | Log warning, skip file, mark as `skipped` in registry |
| File locked (Windows) | Retry after 5s, up to 3 attempts |
| File renamed during processing | Catch `FileNotFoundError`, log warning, skip |
| LightRAG ingestion failure | Log error, markdown still gets written (graph is non-blocking) |
| Partial run crash | Next run cleans up orphaned outputs and restarts from scratch |
| Concurrent run/ingest | Lockfile prevents simultaneous processing |
| Unicode filenames (Windows) | Files are copied to a temp dir with ASCII-safe names for LibreOffice conversion |

The pipeline is **fault-tolerant per file** — one bad file never blocks the rest.

## CI/CD Pipeline

### CI (`ci.yml`) — Every push and PR

```yaml
matrix:
  os: [ubuntu-latest, windows-latest]
  python-version: ["3.10", "3.11", "3.12"]
```

Jobs:
1. **lint** — `ruff check` + `ruff format --check`
2. **typecheck** — `mypy --strict`
3. **test** — `pytest --cov` (Windows + Ubuntu matrix)
4. **build** — `uv build` to verify wheel packaging

### Release (`release.yml`) — On tag `v*`

1. Build wheel
2. Publish to PyPI via `uv publish`

### Docs (`docs.yml`) — On push to main

Uses `HanSur94/wiki-gen-action@v1` to auto-generate and publish GitHub wiki documentation. Note: the wiki action uses the Anthropic API (Claude) — the GPT-4o mini/GPT-5 constraint applies to the **runtime pipeline** only, not CI tooling.

```yaml
jobs:
  wiki:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: HanSur94/wiki-gen-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          wiki_pat: ${{ secrets.WIKI_PAT }}
```

## Integration with Multi-Agent System

This package is designed as a standalone GitHub repo that the `multi_agent_system/` project can consume via:

- **pip/uv dependency**: `uv add document-knowledge-pipe`
- **Git submodule**: for development
- **Programmatic API**: import `docpipe` and call pipeline functions directly

Agents interact with the pipeline outputs through:
- `registry.md` — discover available documents
- `markdown/` folder — read full document content
- `status.json` — check pipeline state programmatically
- LightRAG store — query the knowledge graph for entity/relation lookups

## Key Design Decisions

1. **PyMuPDF4LLM over custom extraction** — handles structure-preserving markdown conversion with image positioning out of the box
2. **VLM over OCR** — GPT-4o mini vision is lighter than running Tesseract/EasyOCR, produces richer descriptions with document context
3. **Normalize to PDF first** — one extraction pipeline instead of format-specific parsers
4. **File-based LightRAG storage** — no infrastructure dependencies, simple to deploy
5. **uv over pip** — handles native builds (hnswlib) more reliably on Windows
6. **YAML config** — all settings configurable, sensible defaults for quick start
7. **Linear pipeline** — simplest architecture, sufficient for 60s debounce workload
8. **Foreground watcher** — cross-platform safe, no daemon complexity on Windows
9. **Trailing-edge debounce** — waits for file writes to finish (important for large files on Windows), with 5min max wait cap
10. **Graph deletion is deferred** — deleting a source file removes markdown/registry entries immediately, but graph cleanup requires `docpipe ingest --rebuild-graph` (LightRAG has no per-document delete API for file-based storage)
11. **Sequential file processing** — avoids concurrent write issues with shared state (registry, status, graph)
