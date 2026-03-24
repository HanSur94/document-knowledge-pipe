# Examples

Sample documents and configuration for testing the document knowledge pipe.

## Quick Start

```bash
cd examples

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
# Or: export OPENAI_API_KEY=sk-...

# Run the demo script (walks through all features)
bash run_demo.sh
```

The demo script (`run_demo.sh`) walks through each major feature:

1. **Single file ingestion** — process one PDF, inspect the markdown output
2. **Pipeline status** — check processing state with `docpipe status`
3. **Batch ingestion** — process all 28 example documents at once
4. **Output inspection** — browse markdown, images, registry, and knowledge graph
5. **Next steps** — watcher mode, dashboard, graph rebuild

You can also run individual commands directly:

```bash
# Process all documents
docpipe ingest --config config.yaml

# Process a single file
docpipe ingest --config config.yaml kaggle_small.pdf

# Start the file watcher with live dashboard
docpipe run --config config.yaml --dashboard

# Check status
docpipe status --config config.yaml

# Rebuild the knowledge graph from existing markdown
docpipe ingest --config config.yaml --rebuild-graph
```

## Example Documents

PDF, DOC, and DOCX files are sourced from Kaggle public datasets. Other formats
are generated sample documents with realistic content.

### PDFs (Kaggle)

| File | Size | Source |
|------|------|--------|
| `input/kaggle_pdf_4.pdf` | 8 KB | manisha717/dataset-of-pdf-files |
| `input/kaggle_pdf_5.pdf` | 9 KB | manisha717/dataset-of-pdf-files |
| `input/kaggle_small.pdf` | 17 KB | manisha717/dataset-of-pdf-files |
| `input/kaggle_pdf_7.pdf` | 64 KB | manisha717/dataset-of-pdf-files |
| `input/kaggle_medium.pdf` | 103 KB | manisha717/dataset-of-pdf-files |
| `input/kaggle_pdf_6.pdf` | 103 KB | manisha717/dataset-of-pdf-files |
| `input/kaggle_pdf_8.pdf` | 698 KB | manisha717/dataset-of-pdf-files |
| `input/kaggle_large.pdf` | 1.7 MB | manisha717/dataset-of-pdf-files |

### Word Documents (Kaggle)

| File | Size | Source |
|------|------|--------|
| `input/kaggle_small.doc` | 38 KB | manisha717/dataset-for-doc-and-docx |
| `input/kaggle_doc_3.doc` | 44 KB | manisha717/dataset-for-doc-and-docx |
| `input/kaggle_doc_4.doc` | 109 KB | manisha717/dataset-for-doc-and-docx |
| `input/kaggle_medium.doc` | 167 KB | manisha717/dataset-for-doc-and-docx |
| `input/kaggle_doc_5.doc` | 252 KB | manisha717/dataset-for-doc-and-docx |
| `input/kaggle_small.docx` | 37 KB | manisha717/dataset-for-doc-and-docx |
| `input/kaggle_docx_3.docx` | 42 KB | manisha717/dataset-for-doc-and-docx |
| `input/kaggle_docx_4.docx` | 99 KB | manisha717/dataset-for-doc-and-docx |
| `input/kaggle_medium.docx` | 116 KB | manisha717/dataset-for-doc-and-docx |
| `input/kaggle_docx_5.docx` | 137 KB | manisha717/dataset-for-doc-and-docx |

### Other Formats (generated)

| File | Format | Size | Description |
|------|--------|------|-------------|
| `input/legacy_slides.ppt` | PPT | 57 KB | Legacy PowerPoint (OpenPreserve format-corpus) |
| `input/quarterly_review.pptx` | PPTX | 31 KB | 4-slide business review presentation |
| `input/lab_results.xls` | XLS | 6 KB | Semiconductor metrology measurements |
| `input/sales_data.xlsx` | XLSX | 14 KB | 192-row sales dataset with summary sheet |
| `input/tech_spec.odt` | ODT | 2 KB | Technical specification document |
| `input/sprint_review.odp` | ODP | 2 KB | Sprint review presentation (5 slides) |
| `input/inventory.ods` | ODS | 2 KB | Lab parts inventory spreadsheet |
| `input/meeting_minutes.rtf` | RTF | 2 KB | Product planning meeting minutes |
| `input/api_docs.html` | HTML | 5 KB | REST API documentation page |
| `input/user_guide.epub` | EPUB | 3 KB | 3-chapter user guide |

All non-PDF formats require LibreOffice for conversion.

## Expected Output

After running `docpipe ingest`, you'll find:

```
output/
├── markdown/
│   ├── kaggle_small.md
│   ├── kaggle_medium.md
│   ├── kaggle_large.md
│   └── ...                  # One .md per input document
├── images/
│   └── *.png                # Extracted images (if any)
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
