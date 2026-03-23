# Examples

Sample documents and configuration for testing the document knowledge pipe.

## Quick Start

```bash
cd examples

# Set your API key
export OPENAI_API_KEY=sk-...
# Or for Anthropic:
# export ANTHROPIC_API_KEY=sk-ant-...

# Run the pipeline on the example documents
docpipe ingest --config config.yaml
```

## Example Documents

| File | Format | Description |
|------|--------|-------------|
| `input/pipeline_guide.pdf` | PDF | 2-page guide with text, headings, and a flow diagram |
| `input/sample.pdf` | PDF | 3-page document with text content |
| `input/sample.docx` | DOCX | Word document (requires LibreOffice) |
| `input/sample.pptx` | PPTX | PowerPoint presentation (requires LibreOffice) |
| `input/sample.xlsx` | XLSX | Excel spreadsheet (requires LibreOffice) |

## Expected Output

After running `docpipe ingest`, you'll find:

```
output/
├── markdown/
│   ├── pipeline_guide.md    # Structured markdown with image descriptions
│   ├── sample.md
│   ├── sample_docx.md       # (if LibreOffice available)
│   ├── sample_pptx.md
│   └── sample_xlsx.md
├── images/
│   └── pipeline_guide_*.png # Extracted images
├── registry.md              # AI-readable document index
├── status.json              # Pipeline status
└── lightrag_store/          # Knowledge graph
```

## Using Anthropic Instead of OpenAI

Edit `config.yaml`:

```yaml
describer:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
```

Then set your Anthropic key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docpipe ingest --config config.yaml
```

## Watching for Changes

Start the watcher to automatically process new documents:

```bash
# Basic watcher
docpipe run --config config.yaml

# With live dashboard
docpipe run --config config.yaml --dashboard
```

Drop any supported file into `input/` and it will be processed within 60 seconds.
