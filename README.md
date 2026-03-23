# Document Knowledge Pipe

[![CI](https://github.com/HanSur94/document-knowledge-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/HanSur94/document-knowledge-pipe/actions/workflows/ci.yml)

A document ingestion pipeline that watches a folder for documents, converts them to structured markdown with AI-generated image descriptions, builds a [LightRAG](https://github.com/HKUDS/LightRAG) knowledge graph, and maintains an AI-readable registry.

Designed for integration with multi-agent systems.

## How It Works

```
Input Folder (watched)
    |
    +-- report.pdf ----------+
    +-- slides.pptx -> PDF --+
    +-- data.xlsx ----> PDF --+
                              |
                    [PyMuPDF4LLM] -> markdown + images
                              |
                    [GPT-4o mini vision] -> image descriptions
                              |
                    Output Folder
                        +-- markdown/
                        +-- images/
                        +-- registry.md
                        +-- status.json
                        +-- lightrag_store/
```

1. **Convert** - Non-PDF documents are converted to PDF via LibreOffice headless
2. **Extract** - PyMuPDF4LLM extracts structured markdown with images in reading order
3. **Describe** - GPT-4o mini vision generates contextual descriptions for each image
4. **Register** - An AI-readable `registry.md` index is maintained with summaries and topics
5. **Graph** - LightRAG builds a knowledge graph from the markdown content
6. **Watch** - A file watcher with 60s trailing-edge debounce keeps everything up to date

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [LibreOffice](https://www.libreoffice.org/download/) 6.2+ (for non-PDF conversion)
- OpenAI API key (GPT-4o mini / GPT-5)

## Quick Start

```bash
# Install
git clone https://github.com/HanSur94/document-knowledge-pipe.git
cd document-knowledge-pipe
uv sync

# Initialize project
docpipe init

# Set your API key in .env
echo "OPENAI_API_KEY=sk-..." > .env

# Drop documents into ./input/ and run
docpipe ingest

# Or start the watcher
docpipe run
```

## CLI

```
docpipe init                        # Create config.yaml + folder structure
docpipe run                         # Start watcher (foreground, blocking)
docpipe run --dashboard             # Start watcher with live Rich dashboard
docpipe ingest                      # One-shot: process all files
docpipe ingest report.pdf           # One-shot: process single file
docpipe ingest --rebuild-graph      # Rebuild knowledge graph from scratch
docpipe status                      # Show pipeline status
```

All commands accept `--config path/to/config.yaml` (defaults to `./config.yaml`).

## Configuration

All settings in `config.yaml` with sensible defaults. Minimal config:

```yaml
input_dir: ./input
output_dir: ./output
```

See the full [default config](config.yaml) for all options including:
- Watcher debounce timing
- LibreOffice path and timeout
- Image extraction DPI and format
- GPT-4o mini vision settings (model, max tokens, context window)
- LightRAG graph settings (storage, chunking, embedding model)
- API retry configuration
- Logging with rotation

## Status & Monitoring

Three ways to check pipeline status:

**Terminal** - `docpipe status` shows a Rich table with file status, graph stats, and API usage

**Live Dashboard** - `docpipe run --dashboard` auto-refreshes in real-time

**Programmatic** - Read `output/status.json` from your agents:

```json
{
  "watcher": "running",
  "last_heartbeat_at": "2026-03-23T12:34:56",
  "files": {
    "report.pdf": {
      "status": "done",
      "pages": 12,
      "images": 3,
      "graph_ingested": true,
      "md_path": "markdown/report.md"
    }
  }
}
```

## Registry

AI-readable document index at `output/registry.md`:

```markdown
| File | Author | Date | Summary | Topics | Pages | Images |
|------|--------|------|---------|--------|-------|--------|
| report.md | J. Smith | 2026-01-15 | Q4 financial results | finance, quarterly | 12 | 3 |
```

Concise enough for an agent to scan without flooding its context.

## Supported Formats

PDF, DOC, DOCX, XLS, XLSX, PPT, PPTX, ODT, ODS, ODP, RTF, HTML, EPUB

Non-PDF formats are converted via LibreOffice headless before processing.

## Integration

Install as a dependency:

```bash
uv add document-knowledge-pipe
```

Or use the programmatic API:

```python
from docpipe.config import load_config
from docpipe.pipeline import process_file

cfg = load_config(Path("config.yaml"))
await process_file(Path("document.pdf"), cfg)
```

Agents interact with outputs through:
- `registry.md` - discover available documents
- `markdown/` folder - read full document content
- `status.json` - check pipeline state
- LightRAG store - query the knowledge graph

## License

MIT
