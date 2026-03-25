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
