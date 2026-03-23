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

| File | Format | Source | Description |
|------|--------|--------|-------------|
| `input/pipeline_guide.pdf` | PDF | Custom | 2-page guide with text, headings, and a flow diagram |
| `input/sample.pdf` | PDF | freetestdata.com | 3-page document with text content |
| `input/sample.docx` | DOCX | freetestdata.com | Word document (requires LibreOffice) |
| `input/sample.pptx` | PPTX | freetestdata.com | PowerPoint presentation (requires LibreOffice) |
| `input/sample.xlsx` | XLSX | freetestdata.com | Excel spreadsheet (requires LibreOffice) |
| `input/ffc.doc` | DOC | file-format-commons | Legacy Word document |
| `input/ffc.ppt` | PPT | file-format-commons | Legacy PowerPoint |
| `input/ffc.xls` | XLS | file-format-commons | Legacy Excel |
| `input/ffc.odt` | ODT | file-format-commons | OpenDocument Text |
| `input/ffc.rtf` | RTF | file-format-commons | Rich Text Format |
| `input/ffc.html` | HTML | file-format-commons | Web page |

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

## Bulk Testing with Kaggle Datasets

For large-scale testing, download Kaggle datasets:

```bash
# Install kaggle CLI and configure API key
pip install kaggle
# See: https://www.kaggle.com/docs/api

# List available datasets
python scripts/download_kaggle_datasets.py --list

# Download all (~1.4 GB)
python scripts/download_kaggle_datasets.py --output-dir ./data/kaggle
```

Available datasets:
- **manisha717/dataset-of-pdf-files** (~806 MB) — diverse PDFs
- **manisha717/dataset-for-doc-and-docx** (~571 MB) — Word documents
