#!/usr/bin/env bash
# =============================================================================
# DocPipe Demo — showcases the main features of the pipeline
#
# Prerequisites:
#   - Python 3.11+
#   - An API key: ANTHROPIC_API_KEY or OPENAI_API_KEY
#   - LibreOffice (optional, needed for non-PDF formats)
#
# Usage:
#   cd examples
#   export ANTHROPIC_API_KEY=sk-ant-...    # or OPENAI_API_KEY=sk-...
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

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -e "${RED}Error: Set ANTHROPIC_API_KEY or OPENAI_API_KEY before running.${NC}"
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
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

# ---------------------------------------------------------------------------
# 2. Check pipeline status
# ---------------------------------------------------------------------------
banner "Feature 2: Pipeline Status"
info "  docpipe status --config config.yaml"
echo ""

docpipe status --config config.yaml

# ---------------------------------------------------------------------------
# 3. Batch ingest all documents
# ---------------------------------------------------------------------------
banner "Feature 3: Batch Ingestion (all formats)"
info "Processing all documents in input/..."
info "  docpipe ingest --config config.yaml"
echo ""

docpipe ingest --config config.yaml

# ---------------------------------------------------------------------------
# 4. Inspect outputs
# ---------------------------------------------------------------------------
banner "Feature 4: Inspect Outputs"

info "Markdown files generated:"
ls -lh output/markdown/ 2>/dev/null || echo "  (none)"

echo ""
info "Images extracted:"
ls -lh output/images/ 2>/dev/null | head -10 || echo "  (none)"

echo ""
if [ -f output/registry.md ]; then
    info "Document registry (output/registry.md):"
    echo "---"
    head -40 output/registry.md
    echo "..."
    echo "---"
fi

echo ""
if [ -f output/status.json ]; then
    info "Status file: output/status.json"
fi

if [ -d output/lightrag_store ]; then
    info "Knowledge graph store: output/lightrag_store/"
    info "  $(find output/lightrag_store -type f | wc -l | tr -d ' ') files in graph store"
fi

# ---------------------------------------------------------------------------
# 5. Show status after full run
# ---------------------------------------------------------------------------
banner "Feature 5: Final Status"
docpipe status --config config.yaml

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
banner "Demo Complete"
info "All outputs are in ./output/"
info ""
info "What to try next:"
info "  1. Start the watcher:  docpipe run --config config.yaml --dashboard"
info "  2. Drop a file into input/ and watch it get processed automatically"
info "  3. Rebuild the knowledge graph:  docpipe ingest --config config.yaml --rebuild-graph"
info ""
info "See README.md for more details."
