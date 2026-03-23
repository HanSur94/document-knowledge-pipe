# Document Knowledge Pipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package that watches a folder for documents, converts them to structured markdown with AI image descriptions, builds a LightRAG knowledge graph, and maintains an AI-readable registry.

**Architecture:** Linear pipeline — each stage is a separate module (converter → extractor → describer → registry → graph → status). Watchdog monitors the input folder with trailing-edge debounce. Files are processed sequentially. All config via YAML.

**Tech Stack:** Python 3.10+, uv, PyMuPDF4LLM, OpenAI (GPT-4o mini vision), LightRAG (HKUDS), Watchdog, Click, Rich, PyYAML

**Spec:** `docs/superpowers/specs/2026-03-23-document-knowledge-pipe-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package metadata, dependencies, CLI entry point |
| `.gitignore` | Exclude .env, __pycache__, output/, logs/, etc. |
| `config.yaml` | Default configuration template |
| `src/docpipe/__init__.py` | Package version, public API exports |
| `src/docpipe/config.py` | Load YAML config, merge defaults, validate |
| `src/docpipe/converter.py` | Non-PDF → PDF via LibreOffice headless |
| `src/docpipe/extractor.py` | PDF → markdown + images via PyMuPDF4LLM |
| `src/docpipe/describer.py` | Replace image refs with GPT-4o mini vision descriptions |
| `src/docpipe/registry.py` | Build/update registry.md |
| `src/docpipe/graph.py` | LightRAG knowledge graph ingestion |
| `src/docpipe/status.py` | Status tracking, status.json, heartbeat |
| `src/docpipe/pipeline.py` | Orchestrate full flow per file, cleanup, lockfile |
| `src/docpipe/watcher.py` | Watchdog monitor + trailing-edge debounce |
| `src/docpipe/cli.py` | Click CLI: init, run, ingest, status |
| `tests/conftest.py` | Shared fixtures (tmp dirs, sample PDFs, mock config) |
| `tests/test_config.py` | Config loading tests |
| `tests/test_converter.py` | Converter tests |
| `tests/test_extractor.py` | Extractor tests |
| `tests/test_describer.py` | Describer tests |
| `tests/test_registry.py` | Registry tests |
| `tests/test_graph.py` | Graph ingestion tests |
| `tests/test_status.py` | Status tracking tests |
| `tests/test_pipeline.py` | Pipeline orchestration tests |
| `tests/test_watcher.py` | Watcher debounce tests |
| `tests/test_cli.py` | CLI integration tests |
| `.github/workflows/ci.yml` | CI: lint, typecheck, test, build |
| `.github/workflows/release.yml` | Release: publish to PyPI |
| `.github/workflows/docs.yml` | Docs: wiki-gen-action |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `config.yaml`
- Create: `src/docpipe/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "document-knowledge-pipe"
version = "0.1.0"
description = "Document ingestion pipeline: folder → markdown → knowledge graph"
requires-python = ">=3.10"
license = "MIT"
authors = [{ name = "Hannes Suhr" }]
dependencies = [
    "pymupdf4llm>=0.0.17",
    "openai>=1.40.0",
    "lightrag-hku>=1.0.0",
    "watchdog>=4.0.0",
    "click>=8.1.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "rich>=13.0.0",
]

[project.scripts]
docpipe = "docpipe.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/docpipe"]

[tool.hatch.build.targets.sdist]
exclude = [".env", "output/", "logs/", "input/"]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true

[[tool.mypy.overrides]]
module = ["lightrag.*", "pymupdf4llm.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create .gitignore**

```
# Secrets
.env

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Project runtime
output/
input/
logs/
*.lock

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Create default config.yaml**

Copy the full config block from the spec (lines 130-209) into `config.yaml`.

- [ ] **Step 4: Create src/docpipe/__init__.py**

```python
"""Document Knowledge Pipe — folder → markdown → knowledge graph."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Create tests/conftest.py with shared fixtures**

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def tmp_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create input/output directory structure."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "markdown").mkdir()
    (output_dir / "images").mkdir()
    return {"input": input_dir, "output": output_dir, "root": tmp_path}


@pytest.fixture
def sample_config(tmp_dirs: dict[str, Path]) -> dict[str, Any]:
    """Return a minimal test config dict."""
    return {
        "input_dir": str(tmp_dirs["input"]),
        "output_dir": str(tmp_dirs["output"]),
        "watcher": {
            "enabled": False,
            "debounce_seconds": 1,
            "max_wait_seconds": 5,
            "poll_interval_seconds": 1,
            "watch_subdirectories": False,
        },
        "converter": {
            "libreoffice_path": None,
            "timeout_seconds": 30,
            "supported_extensions": [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"],
        },
        "extractor": {
            "write_images": True,
            "image_format": "png",
            "dpi": 150,
        },
        "describer": {
            "model": "gpt-4o-mini",
            "max_tokens": 300,
            "include_context": True,
            "context_chars": 500,
            "batch_size": 5,
        },
        "registry": {
            "filename": "registry.md",
            "summary_max_words": 30,
            "include_fields": ["filename", "author", "date", "summary", "topics", "pages", "images"],
        },
        "graph": {
            "storage": "file",
            "store_dir": str(tmp_dirs["output"] / "lightrag_store"),
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_tokens": 4096,
            "chunk_size": 1200,
            "chunk_overlap": 100,
        },
        "api_retry": {
            "max_retries": 3,
            "initial_delay_seconds": 0,
            "max_delay_seconds": 0,
        },
        "logging": {
            "level": "DEBUG",
            "file": str(tmp_dirs["root"] / "logs" / "docpipe.log"),
            "max_size_mb": 1,
            "backup_count": 1,
        },
    }


@pytest.fixture
def sample_pdf(tmp_dirs: dict[str, Path]) -> Path:
    """Create a minimal valid PDF for testing."""
    import fitz  # pymupdf

    pdf_path = tmp_dirs["input"] / "test_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World. This is a test document.")
    page.insert_text((72, 100), "It has multiple paragraphs for testing extraction.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def config_file(tmp_dirs: dict[str, Path], sample_config: dict[str, Any]) -> Path:
    """Write sample config to a YAML file and return its path."""
    config_path = tmp_dirs["root"] / "config.yaml"
    config_path.write_text(yaml.dump(sample_config))
    return config_path
```

- [ ] **Step 6: Install dependencies and verify**

Run: `cd ~/document-knowledge-pipe && uv sync --dev`
Expected: all deps install successfully

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore config.yaml src/ tests/conftest.py
git commit -m "feat: project scaffolding with deps, config template, test fixtures"
```

---

### Task 2: Config Module

**Files:**
- Create: `src/docpipe/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config loading**

```python
# tests/test_config.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from docpipe.config import DocpipeConfig, load_config


class TestLoadConfig:
    def test_load_from_yaml_file(self, config_file: Path, sample_config: dict[str, Any]) -> None:
        cfg = load_config(config_file)
        assert cfg.input_dir == Path(sample_config["input_dir"])
        assert cfg.output_dir == Path(sample_config["output_dir"])
        assert cfg.watcher.debounce_seconds == 1
        assert cfg.describer.model == "gpt-4o-mini"

    def test_load_with_defaults(self, tmp_path: Path) -> None:
        """Minimal YAML should fill defaults for all missing fields."""
        minimal = {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(minimal))
        cfg = load_config(cfg_path)
        assert cfg.watcher.debounce_seconds == 60
        assert cfg.graph.embedding_model == "text-embedding-3-small"
        assert cfg.api_retry.max_retries == 3

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_env_override_for_api_key(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        cfg = load_config(config_file)
        assert cfg.openai_api_key == "sk-test-key-123"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docpipe.config'`

- [ ] **Step 3: Implement config.py**

```python
# src/docpipe/config.py
"""Configuration loading and validation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class WatcherConfig:
    enabled: bool = True
    debounce_seconds: int = 60
    max_wait_seconds: int = 300
    poll_interval_seconds: int = 1
    watch_subdirectories: bool = False


@dataclass
class ConverterConfig:
    libreoffice_path: str | None = None
    timeout_seconds: int = 120
    supported_extensions: list[str] = field(
        default_factory=lambda: [
            ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".odt", ".ods", ".odp", ".rtf", ".html", ".epub",
        ]
    )


@dataclass
class ExtractorConfig:
    write_images: bool = True
    image_format: str = "png"
    dpi: int = 150


@dataclass
class DescriberConfig:
    model: str = "gpt-4o-mini"
    max_tokens: int = 300
    include_context: bool = True
    context_chars: int = 500
    batch_size: int = 5


@dataclass
class RegistryConfig:
    filename: str = "registry.md"
    summary_max_words: int = 30
    include_fields: list[str] = field(
        default_factory=lambda: [
            "filename", "author", "date", "summary", "topics", "pages", "images",
        ]
    )


@dataclass
class GraphConfig:
    storage: str = "file"
    store_dir: str = "./output/lightrag_store"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 4096
    chunk_size: int = 1200
    chunk_overlap: int = 100


@dataclass
class ApiRetryConfig:
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "./logs/docpipe.log"
    max_size_mb: int = 50
    backup_count: int = 3


@dataclass
class DocpipeConfig:
    input_dir: Path
    output_dir: Path
    openai_api_key: str = ""
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    converter: ConverterConfig = field(default_factory=ConverterConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    describer: DescriberConfig = field(default_factory=DescriberConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    api_retry: ApiRetryConfig = field(default_factory=ApiRetryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    """Create a dataclass instance, using defaults for missing keys."""
    if data is None:
        return cls()
    # Filter to only keys the dataclass accepts
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def load_config(path: Path) -> DocpipeConfig:
    """Load config from YAML file, merge with defaults, load .env."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load .env from same directory as config
    env_path = path.parent / ".env"
    load_dotenv(env_path)

    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

    return DocpipeConfig(
        input_dir=Path(raw.get("input_dir", "./input")),
        output_dir=Path(raw.get("output_dir", "./output")),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        watcher=_merge_dataclass(WatcherConfig, raw.get("watcher")),
        converter=_merge_dataclass(ConverterConfig, raw.get("converter")),
        extractor=_merge_dataclass(ExtractorConfig, raw.get("extractor")),
        describer=_merge_dataclass(DescriberConfig, raw.get("describer")),
        registry=_merge_dataclass(RegistryConfig, raw.get("registry")),
        graph=_merge_dataclass(GraphConfig, raw.get("graph")),
        api_retry=_merge_dataclass(ApiRetryConfig, raw.get("api_retry")),
        logging=_merge_dataclass(LoggingConfig, raw.get("logging")),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/config.py tests/test_config.py
git commit -m "feat: config module with YAML loading, defaults, .env support"
```

---

### Task 3: Converter Module

**Files:**
- Create: `src/docpipe/converter.py`
- Create: `tests/test_converter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_converter.py
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from docpipe.config import ConverterConfig
from docpipe.converter import convert_to_pdf, find_libreoffice


class TestFindLibreOffice:
    def test_explicit_path_returned_if_set(self) -> None:
        cfg = ConverterConfig(libreoffice_path="/usr/bin/soffice")
        assert find_libreoffice(cfg) == Path("/usr/bin/soffice")

    @patch("shutil.which", return_value="/usr/bin/soffice")
    def test_auto_detect_from_path(self, mock_which: MagicMock) -> None:
        cfg = ConverterConfig(libreoffice_path=None)
        result = find_libreoffice(cfg)
        assert result is not None

    @patch("shutil.which", return_value=None)
    def test_raises_if_not_found(self, mock_which: MagicMock) -> None:
        cfg = ConverterConfig(libreoffice_path=None)
        with pytest.raises(FileNotFoundError, match="LibreOffice"):
            find_libreoffice(cfg)


class TestConvertToPdf:
    def test_pdf_passes_through(self, sample_pdf: Path, tmp_dirs: dict[str, Path]) -> None:
        result = convert_to_pdf(
            sample_pdf,
            tmp_dirs["output"],
            ConverterConfig(),
        )
        assert result == sample_pdf  # PDF is returned unchanged

    def test_unsupported_extension_raises(self, tmp_dirs: dict[str, Path]) -> None:
        bad_file = tmp_dirs["input"] / "file.xyz"
        bad_file.write_text("not a real file")
        with pytest.raises(ValueError, match="Unsupported"):
            convert_to_pdf(bad_file, tmp_dirs["output"], ConverterConfig())

    @patch("subprocess.run")
    def test_calls_libreoffice_for_docx(
        self, mock_run: MagicMock, tmp_dirs: dict[str, Path]
    ) -> None:
        docx = tmp_dirs["input"] / "test.docx"
        docx.write_bytes(b"fake docx")
        out_pdf = tmp_dirs["output"] / "test.pdf"
        out_pdf.write_bytes(b"fake pdf")
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        cfg = ConverterConfig(libreoffice_path="/usr/bin/soffice")
        result = convert_to_pdf(docx, tmp_dirs["output"], cfg)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "--headless" in call_args
        assert "--norestore" in call_args
        assert "--safe-mode" in call_args

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="soffice", timeout=120))
    def test_timeout_raises(self, mock_run: MagicMock, tmp_dirs: dict[str, Path]) -> None:
        docx = tmp_dirs["input"] / "test.docx"
        docx.write_bytes(b"fake docx")
        cfg = ConverterConfig(libreoffice_path="/usr/bin/soffice", timeout_seconds=120)
        with pytest.raises(subprocess.TimeoutExpired):
            convert_to_pdf(docx, tmp_dirs["output"], cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_converter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docpipe.converter'`

- [ ] **Step 3: Implement converter.py**

```python
# src/docpipe/converter.py
"""Convert non-PDF documents to PDF via LibreOffice headless."""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from docpipe.config import ConverterConfig

logger = logging.getLogger(__name__)

_WINDOWS_PATHS = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]


def find_libreoffice(cfg: ConverterConfig) -> Path:
    """Find the LibreOffice executable."""
    if cfg.libreoffice_path:
        return Path(cfg.libreoffice_path)

    # Try PATH first
    found = shutil.which("soffice")
    if found:
        return Path(found)

    # Try common Windows locations
    if platform.system() == "Windows":
        for p in _WINDOWS_PATHS:
            path = Path(p)
            if path.exists():
                return path

    raise FileNotFoundError(
        "LibreOffice not found. Install from https://www.libreoffice.org/download/ "
        "or set converter.libreoffice_path in config.yaml"
    )


def _is_ascii_safe(name: str) -> bool:
    """Check if filename is safe for LibreOffice on Windows."""
    try:
        name.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def convert_to_pdf(file_path: Path, output_dir: Path, cfg: ConverterConfig) -> Path:
    """Convert a file to PDF. Returns path to the PDF (original if already PDF)."""
    if file_path.suffix.lower() == ".pdf":
        return file_path

    if file_path.suffix.lower() not in cfg.supported_extensions:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported: {cfg.supported_extensions}"
        )

    soffice = find_libreoffice(cfg)
    tmp_dir = None

    try:
        # Handle Unicode filenames on Windows
        source = file_path
        if platform.system() == "Windows" and not _is_ascii_safe(file_path.name):
            tmp_dir = tempfile.mkdtemp(prefix="docpipe_")
            safe_name = f"docpipe_convert{file_path.suffix}"
            source = Path(tmp_dir) / safe_name
            shutil.copy2(file_path, source)
            logger.debug("Copied Unicode filename to temp: %s", source)

        cmd = [
            str(soffice),
            "--headless",
            "--norestore",
            "--safe-mode",
            "--convert-to", "pdf",
            "--outdir", str(output_dir),
            str(source),
        ]

        logger.info("Converting %s to PDF", file_path.name)
        subprocess.run(cmd, timeout=cfg.timeout_seconds, check=True, capture_output=True)

        # Find the output PDF
        expected_name = source.stem + ".pdf"
        pdf_path = output_dir / expected_name

        # If we used a temp name, rename to match original
        if tmp_dir and pdf_path.exists():
            final_name = file_path.stem + ".pdf"
            final_path = output_dir / final_name
            pdf_path.rename(final_path)
            pdf_path = final_path

        if not pdf_path.exists():
            raise FileNotFoundError(f"LibreOffice did not produce expected PDF: {pdf_path}")

        logger.info("Converted to PDF: %s", pdf_path.name)
        return pdf_path

    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_converter.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/converter.py tests/test_converter.py
git commit -m "feat: converter module — LibreOffice headless PDF conversion"
```

---

### Task 4: Extractor Module

**Files:**
- Create: `src/docpipe/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extractor.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from docpipe.config import ExtractorConfig
from docpipe.extractor import ExtractionResult, extract_markdown


class TestExtractMarkdown:
    def test_extracts_text_from_pdf(self, sample_pdf: Path, tmp_dirs: dict[str, Path]) -> None:
        result = extract_markdown(
            sample_pdf,
            tmp_dirs["output"] / "images",
            ExtractorConfig(),
        )
        assert isinstance(result, ExtractionResult)
        assert "Hello World" in result.markdown
        assert result.page_count > 0

    def test_returns_image_paths(self, tmp_dirs: dict[str, Path]) -> None:
        """PDF with an image should list extracted image paths."""
        import fitz

        pdf_path = tmp_dirs["input"] / "with_image.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Text before image.")
        # Insert a small colored rectangle as a "graphic"
        rect = fitz.Rect(72, 100, 200, 200)
        page.draw_rect(rect, color=(1, 0, 0), fill=(0, 0, 1))
        page.insert_text((72, 230), "Text after image.")
        doc.save(str(pdf_path))
        doc.close()

        result = extract_markdown(
            pdf_path,
            tmp_dirs["output"] / "images",
            ExtractorConfig(write_images=True),
        )
        assert result.markdown  # Should have content

    def test_empty_pdf_returns_empty(self, tmp_dirs: dict[str, Path]) -> None:
        import fitz

        pdf_path = tmp_dirs["input"] / "empty.pdf"
        doc = fitz.open()
        doc.new_page()  # blank page
        doc.save(str(pdf_path))
        doc.close()

        result = extract_markdown(
            pdf_path,
            tmp_dirs["output"] / "images",
            ExtractorConfig(),
        )
        assert result.page_count == 1
        # Empty page produces minimal markdown
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extractor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement extractor.py**

```python
# src/docpipe/extractor.py
"""Extract structured markdown and images from PDF via PyMuPDF4LLM."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pymupdf4llm

from docpipe.config import ExtractorConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    markdown: str
    image_paths: list[Path] = field(default_factory=list)
    page_count: int = 0


def extract_markdown(
    pdf_path: Path,
    image_dir: Path,
    cfg: ExtractorConfig,
) -> ExtractionResult:
    """Extract markdown with images in reading order from a PDF."""
    image_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    if page_count == 0:
        logger.warning("PDF has 0 pages: %s", pdf_path.name)
        return ExtractionResult(markdown="", page_count=0)

    logger.info("Extracting %d pages from %s", page_count, pdf_path.name)

    md_text = pymupdf4llm.to_markdown(
        doc=str(pdf_path),
        write_images=cfg.write_images,
        image_path=str(image_dir),
        image_format=cfg.image_format,
        dpi=cfg.dpi,
    )

    # Collect any images that were written
    image_paths: list[Path] = []
    if cfg.write_images:
        stem = pdf_path.stem
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.stem.startswith(stem)
        )

    logger.info(
        "Extracted %d chars, %d images from %s",
        len(md_text),
        len(image_paths),
        pdf_path.name,
    )

    return ExtractionResult(
        markdown=md_text,
        image_paths=image_paths,
        page_count=page_count,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extractor.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/extractor.py tests/test_extractor.py
git commit -m "feat: extractor module — PDF to markdown via PyMuPDF4LLM"
```

---

### Task 5: Describer Module

**Files:**
- Create: `src/docpipe/describer.py`
- Create: `tests/test_describer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_describer.py
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import ApiRetryConfig, DescriberConfig
from docpipe.describer import (
    describe_image,
    get_surrounding_context,
    replace_image_refs,
)


class TestGetSurroundingContext:
    def test_extracts_context_around_position(self) -> None:
        text = "AAAA" * 50 + "![image](img.png)" + "BBBB" * 50
        pos = text.index("![image]")
        before, after = get_surrounding_context(text, pos, context_chars=20)
        assert len(before) <= 20
        assert len(after) <= 20

    def test_handles_start_of_document(self) -> None:
        text = "![image](img.png) followed by text"
        before, after = get_surrounding_context(text, 0, context_chars=100)
        assert before == ""
        assert "followed by text" in after

    def test_handles_end_of_document(self) -> None:
        text = "Some text ![image](img.png)"
        pos = text.index("![image]")
        before, after = get_surrounding_context(text, pos, context_chars=100)
        assert "Some text" in before
        assert after == ""


class TestDescribeImage:
    @pytest.mark.asyncio
    @patch("docpipe.describer._call_vision_api", new_callable=AsyncMock)
    async def test_returns_description(self, mock_api: AsyncMock, tmp_dirs: dict[str, Path]) -> None:
        mock_api.return_value = "A bar chart showing quarterly revenue."
        img = tmp_dirs["output"] / "images" / "test_img.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        # Create a minimal PNG (1x1 pixel)
        img.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = await describe_image(
            img, "preceding text", "following text",
            DescriberConfig(), ApiRetryConfig(),
        )
        assert result == "A bar chart showing quarterly revenue."


class TestReplaceImageRefs:
    @pytest.mark.asyncio
    @patch("docpipe.describer.describe_image")
    async def test_replaces_image_markdown(
        self, mock_describe: AsyncMock, tmp_dirs: dict[str, Path]
    ) -> None:
        mock_describe.return_value = "A photo of a cat."
        img = tmp_dirs["output"] / "images" / "test_img001.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"fake png")

        markdown = "Some text\n\n![](images/test_img001.png)\n\nMore text"
        result = await replace_image_refs(
            markdown,
            tmp_dirs["output"],
            DescriberConfig(),
            ApiRetryConfig(),
            doc_title="Test Document",
        )
        assert "A photo of a cat." in result
        assert "![](images/test_img001.png)" in result  # original ref preserved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_describer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement describer.py**

```python
# src/docpipe/describer.py
"""Replace image references in markdown with GPT-4o mini vision descriptions."""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from pathlib import Path

import openai

from docpipe.config import ApiRetryConfig, DescriberConfig

logger = logging.getLogger(__name__)

_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def get_surrounding_context(
    text: str, position: int, context_chars: int
) -> tuple[str, str]:
    """Extract text before and after a position in the document."""
    start = max(0, position - context_chars)
    # Find the end of the image markdown reference
    end_of_ref = text.find(")", position)
    if end_of_ref == -1:
        end_of_ref = position
    else:
        end_of_ref += 1
    end = min(len(text), end_of_ref + context_chars)

    before = text[start:position].strip()
    after = text[end_of_ref:end].strip()
    return before, after


async def _call_vision_api(
    client: openai.AsyncOpenAI,
    image_b64: str,
    image_format: str,
    context_before: str,
    context_after: str,
    cfg: DescriberConfig,
    retry_cfg: ApiRetryConfig,
) -> str:
    """Call OpenAI vision API with retry."""
    prompt = (
        "Describe this image concisely for a document knowledge base. "
        "Focus on what the image shows and its significance."
    )
    if context_before:
        prompt += f"\n\nPreceding text: {context_before}"
    if context_after:
        prompt += f"\n\nFollowing text: {context_after}"

    delay = retry_cfg.initial_delay_seconds
    for attempt in range(retry_cfg.max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
            )
            content = response.choices[0].message.content
            return content or "[no description generated]"
        except openai.APIError as e:
            if attempt == retry_cfg.max_retries:
                logger.error("Vision API failed after %d retries: %s", attempt + 1, e)
                return "[image description unavailable]"
            logger.warning("Vision API attempt %d failed: %s. Retrying...", attempt + 1, e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, retry_cfg.max_delay_seconds)

    return "[image description unavailable]"


async def describe_image(
    image_path: Path,
    context_before: str,
    context_after: str,
    cfg: DescriberConfig,
    retry_cfg: ApiRetryConfig,
) -> str:
    """Send an image to GPT-4o mini vision and get a description."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    image_format = image_path.suffix.lstrip(".").lower()
    if image_format == "jpg":
        image_format = "jpeg"

    client = openai.AsyncOpenAI()
    return await _call_vision_api(
        client, image_b64, image_format,
        context_before, context_after,
        cfg, retry_cfg,
    )


async def replace_image_refs(
    markdown: str,
    output_dir: Path,
    cfg: DescriberConfig,
    retry_cfg: ApiRetryConfig,
    doc_title: str = "",
) -> str:
    """Find image references in markdown and add AI descriptions."""
    matches = list(_IMAGE_PATTERN.finditer(markdown))
    if not matches:
        return markdown

    logger.info("Found %d images to describe", len(matches))

    # Process in batches
    result = markdown
    offset = 0

    for i, match in enumerate(matches):
        img_path_str = match.group(2)
        img_path = output_dir / img_path_str

        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        # Get surrounding context
        pos = match.start() + offset
        context_before, context_after = get_surrounding_context(
            result, pos, cfg.context_chars
        )
        if not context_before and doc_title:
            context_before = f"Document: {doc_title}"

        description = await describe_image(
            img_path, context_before, context_after, cfg, retry_cfg
        )

        # Insert description before the image reference
        desc_block = f"\n\n**[Image: {description}]**\n\n"
        insert_pos = match.start() + offset
        result = result[:insert_pos] + desc_block + result[insert_pos:]
        offset += len(desc_block)

        logger.debug("Described image %d/%d: %s", i + 1, len(matches), img_path.name)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_describer.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/describer.py tests/test_describer.py
git commit -m "feat: describer module — GPT-4o mini vision image descriptions"
```

---

### Task 6: Registry Module

**Files:**
- Create: `src/docpipe/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_registry.py
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from docpipe.config import ApiRetryConfig, RegistryConfig
from docpipe.registry import RegistryEntry, build_registry, update_registry


class TestRegistryEntry:
    def test_to_table_row(self) -> None:
        entry = RegistryEntry(
            filename="report.md",
            author="J. Smith",
            date="2026-01-15",
            summary="Q4 results",
            topics="finance, quarterly",
            pages=12,
            images=3,
        )
        row = entry.to_table_row()
        assert "report.md" in row
        assert "J. Smith" in row
        assert "| 12 |" in row

    def test_failed_entry(self) -> None:
        entry = RegistryEntry.failed("corrupt.pdf", "File is corrupted")
        row = entry.to_table_row()
        assert "corrupt.pdf" in row
        assert "Processing failed" in row


class TestUpdateRegistry:
    def test_creates_registry_file(self, tmp_dirs: dict[str, Path]) -> None:
        entry = RegistryEntry(
            filename="test.md", author="-", date="-",
            summary="Test doc", topics="test", pages=1, images=0,
        )
        reg_path = tmp_dirs["output"] / "registry.md"
        update_registry(reg_path, entry)
        assert reg_path.exists()
        content = reg_path.read_text()
        assert "| test.md |" in content
        assert "# Document Registry" in content

    def test_updates_existing_entry(self, tmp_dirs: dict[str, Path]) -> None:
        reg_path = tmp_dirs["output"] / "registry.md"
        entry1 = RegistryEntry(
            filename="test.md", author="-", date="-",
            summary="Version 1", topics="test", pages=1, images=0,
        )
        update_registry(reg_path, entry1)

        entry2 = RegistryEntry(
            filename="test.md", author="-", date="-",
            summary="Version 2", topics="test", pages=2, images=1,
        )
        update_registry(reg_path, entry2)

        content = reg_path.read_text()
        assert content.count("test.md") == 1  # no duplicates
        assert "Version 2" in content

    def test_removes_entry(self, tmp_dirs: dict[str, Path]) -> None:
        reg_path = tmp_dirs["output"] / "registry.md"
        entry = RegistryEntry(
            filename="test.md", author="-", date="-",
            summary="Will be removed", topics="test", pages=1, images=0,
        )
        update_registry(reg_path, entry)
        update_registry(reg_path, entry, remove=True)
        content = reg_path.read_text()
        assert "test.md" not in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Implement registry.py**

```python
# src/docpipe/registry.py
"""Build and update the AI-readable registry.md index."""
from __future__ import annotations

import asyncio
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import openai

from docpipe.config import ApiRetryConfig, RegistryConfig

logger = logging.getLogger(__name__)

_HEADER = """# Document Registry

| File | Author | Date | Summary | Topics | Pages | Images |
|------|--------|------|---------|--------|-------|--------|
"""


@dataclass
class RegistryEntry:
    filename: str
    author: str
    date: str
    summary: str
    topics: str
    pages: int
    images: int

    def to_table_row(self) -> str:
        return (
            f"| {self.filename} | {self.author} | {self.date} "
            f"| {self.summary} | {self.topics} | {self.pages} | {self.images} |"
        )

    @classmethod
    def failed(cls, filename: str, reason: str) -> RegistryEntry:
        return cls(
            filename=filename,
            author="-",
            date="-",
            summary=f"Processing failed: {reason}",
            topics="-",
            pages=0,
            images=0,
        )


def update_registry(
    registry_path: Path,
    entry: RegistryEntry,
    remove: bool = False,
) -> None:
    """Add, update, or remove an entry in registry.md."""
    # Read existing entries
    entries: list[str] = []
    if registry_path.exists():
        lines = registry_path.read_text().splitlines()
        for line in lines:
            if line.startswith("| ") and not line.startswith("| File") and "---" not in line:
                entries.append(line)

    # Remove existing entry for this filename
    entries = [e for e in entries if f"| {entry.filename} |" not in e]

    # Add new entry unless removing
    if not remove:
        entries.append(entry.to_table_row())

    # Sort entries by filename
    entries.sort(key=lambda e: e.split("|")[1].strip())

    # Write the file
    content = _HEADER + "\n".join(entries) + "\n"
    registry_path.write_text(content)
    logger.info("Registry updated: %s (%s)", entry.filename, "removed" if remove else "updated")


async def generate_summary(
    markdown: str,
    cfg: RegistryConfig,
    retry_cfg: ApiRetryConfig,
) -> tuple[str, str]:
    """Use GPT-4o mini to generate a summary and topic tags."""
    client = openai.AsyncOpenAI()
    prompt = (
        f"Summarize this document in at most {cfg.summary_max_words} words. "
        "Then list 2-5 topic tags (comma-separated). "
        "Respond in exactly this format:\n"
        "SUMMARY: <your summary>\n"
        "TOPICS: <tag1, tag2, tag3>\n\n"
        f"Document:\n{markdown[:3000]}"
    )

    delay = retry_cfg.initial_delay_seconds
    for attempt in range(retry_cfg.max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or ""
            summary = ""
            topics = ""
            for line in content.splitlines():
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                elif line.startswith("TOPICS:"):
                    topics = line.replace("TOPICS:", "").strip()
            return summary or "No summary available", topics or "-"
        except openai.APIError as e:
            if attempt == retry_cfg.max_retries:
                logger.error("Summary generation failed after %d retries: %s", attempt + 1, e)
                return "Summary unavailable", "-"
            logger.warning("Summary API attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, retry_cfg.max_delay_seconds)


def build_registry(output_dir: Path) -> str:
    """Read the current registry and return it as a string."""
    reg_path = output_dir / "registry.md"
    if reg_path.exists():
        return reg_path.read_text()
    return _HEADER
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/registry.py tests/test_registry.py
git commit -m "feat: registry module — AI-readable document index"
```

---

### Task 7: Status Module

**Files:**
- Create: `src/docpipe/status.py`
- Create: `tests/test_status.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_status.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from docpipe.status import StatusTracker


class TestStatusTracker:
    def test_init_creates_status(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        data = tracker.to_dict()
        assert data["watcher"] == "stopped"
        assert data["files"] == {}

    def test_set_watcher_running(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.set_watcher_running()
        assert tracker.to_dict()["watcher"] == "running"

    def test_update_file_status(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.update_file("test.pdf", status="done", pages=5, images=2, graph_ingested=True)
        files = tracker.to_dict()["files"]
        assert "test.pdf" in files
        assert files["test.pdf"]["status"] == "done"
        assert files["test.pdf"]["pages"] == 5

    def test_save_and_load(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.update_file("test.pdf", status="done", pages=3, images=1, graph_ingested=True)
        tracker.save()

        status_file = tmp_dirs["output"] / "status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert "test.pdf" in data["files"]

    def test_heartbeat_updates(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.set_watcher_running()
        tracker.heartbeat()
        data = tracker.to_dict()
        assert "last_heartbeat_at" in data

    def test_track_api_usage(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.add_api_usage(tokens=500, cost=0.001)
        data = tracker.to_dict()
        assert data["api_usage"]["tokens_today"] == 500

    def test_remove_file(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.update_file("test.pdf", status="done", pages=1, images=0, graph_ingested=True)
        tracker.remove_file("test.pdf")
        assert "test.pdf" not in tracker.to_dict()["files"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_status.py -v`
Expected: FAIL

- [ ] **Step 3: Implement status.py**

```python
# src/docpipe/status.py
"""Status tracking, status.json management, and heartbeat."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StatusTracker:
    """Track pipeline status and persist to status.json."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._status_file = output_dir / "status.json"
        self._watcher_state = "stopped"
        self._started_at: str | None = None
        self._last_heartbeat: str | None = None
        self._files: dict[str, dict[str, Any]] = {}
        self._graph: dict[str, int] = {"entities": 0, "relations": 0, "documents": 0}
        self._api_usage: dict[str, Any] = {
            "tokens_today": 0,
            "estimated_cost_usd": 0.0,
            "day_reset": str(date.today()),
            "tokens_total": 0,
        }
        self._load()

    def _load(self) -> None:
        """Load existing status from disk."""
        if self._status_file.exists():
            try:
                data = json.loads(self._status_file.read_text())
                self._watcher_state = data.get("watcher", "stopped")
                self._started_at = data.get("started_at")
                self._last_heartbeat = data.get("last_heartbeat_at")
                self._files = data.get("files", {})
                self._graph = data.get("graph", self._graph)
                self._api_usage = data.get("api_usage", self._api_usage)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupted status.json, starting fresh")

    def set_watcher_running(self) -> None:
        self._watcher_state = "running"
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self.heartbeat()

    def set_watcher_stopped(self) -> None:
        self._watcher_state = "stopped"

    def heartbeat(self) -> None:
        self._last_heartbeat = datetime.now().isoformat(timespec="seconds")

    def update_file(
        self,
        filename: str,
        status: str,
        pages: int = 0,
        images: int = 0,
        graph_ingested: bool = False,
        md_path: str = "",
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "status": status,
            "pages": pages,
            "images": images,
            "graph_ingested": graph_ingested,
            "last_processed": datetime.now().isoformat(timespec="seconds"),
        }
        if md_path:
            entry["md_path"] = md_path
        if error:
            entry["error"] = error
        self._files[filename] = entry

    def remove_file(self, filename: str) -> None:
        self._files.pop(filename, None)

    def update_graph_stats(self, entities: int, relations: int, documents: int) -> None:
        self._graph = {
            "entities": entities,
            "relations": relations,
            "documents": documents,
        }

    def add_api_usage(self, tokens: int, cost: float) -> None:
        today = str(date.today())
        if self._api_usage.get("day_reset") != today:
            self._api_usage["tokens_today"] = 0
            self._api_usage["estimated_cost_usd"] = 0.0
            self._api_usage["day_reset"] = today
        self._api_usage["tokens_today"] += tokens
        self._api_usage["estimated_cost_usd"] += cost
        self._api_usage["tokens_total"] += tokens

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "watcher": self._watcher_state,
            "files": self._files,
            "graph": self._graph,
            "api_usage": self._api_usage,
        }
        if self._started_at:
            result["started_at"] = self._started_at
        if self._last_heartbeat:
            result["last_heartbeat_at"] = self._last_heartbeat
        return result

    def save(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._status_file.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_status.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/status.py tests/test_status.py
git commit -m "feat: status module — status.json tracking and heartbeat"
```

---

### Task 8: Graph Module

**Files:**
- Create: `src/docpipe/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_graph.py
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import GraphConfig
from docpipe.graph import ingest_document, rebuild_graph


class TestIngestDocument:
    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_inserts_markdown_into_lightrag(
        self, mock_get_rag: MagicMock, tmp_dirs: dict[str, Path]
    ) -> None:
        mock_rag = AsyncMock()
        mock_get_rag.return_value = mock_rag
        cfg = GraphConfig(store_dir=str(tmp_dirs["output"] / "lightrag_store"))

        await ingest_document("# Test\nHello world", "test_doc", cfg)
        mock_rag.ainsert.assert_called_once()

    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_handles_ingestion_failure(
        self, mock_get_rag: MagicMock, tmp_dirs: dict[str, Path]
    ) -> None:
        mock_rag = AsyncMock()
        mock_rag.ainsert.side_effect = Exception("LightRAG error")
        mock_get_rag.return_value = mock_rag
        cfg = GraphConfig(store_dir=str(tmp_dirs["output"] / "lightrag_store"))

        # Should not raise — graph failures are non-blocking
        result = await ingest_document("# Test", "test_doc", cfg)
        assert result is False


class TestRebuildGraph:
    @pytest.mark.asyncio
    @patch("docpipe.graph._get_rag_instance")
    async def test_rebuild_deletes_store_first(
        self, mock_get_rag: MagicMock, tmp_dirs: dict[str, Path]
    ) -> None:
        mock_rag = AsyncMock()
        mock_get_rag.return_value = mock_rag
        store_dir = tmp_dirs["output"] / "lightrag_store"
        store_dir.mkdir()
        (store_dir / "old_data.json").write_text("{}")

        cfg = GraphConfig(store_dir=str(store_dir))
        md_dir = tmp_dirs["output"] / "markdown"
        (md_dir / "doc1.md").write_text("# Doc 1\nContent")

        await rebuild_graph(md_dir, cfg)

        assert not (store_dir / "old_data.json").exists()
        mock_rag.ainsert.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL

- [ ] **Step 3: Implement graph.py**

```python
# src/docpipe/graph.py
"""LightRAG knowledge graph ingestion."""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from docpipe.config import GraphConfig

logger = logging.getLogger(__name__)


async def _get_rag_instance(cfg: GraphConfig) -> Any:  # noqa: ANN401
    """Create and initialize a LightRAG instance."""
    from lightrag import LightRAG  # type: ignore[import-untyped]
    from lightrag.llm.openai import openai_complete, openai_embed  # type: ignore[import-untyped]

    store_dir = Path(cfg.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(store_dir),
        llm_model_func=openai_complete,
        llm_model_name=cfg.model,
        llm_model_max_tokens=cfg.max_tokens,
        embedding_func=openai_embed,
        embedding_model_name=cfg.embedding_model,
        chunk_size=cfg.chunk_size,
        chunk_overlap_size=cfg.chunk_overlap,
    )

    await rag.initialize_storages()
    await rag.initialize_pipeline_status()
    return rag


async def ingest_document(
    markdown: str,
    doc_id: str,
    cfg: GraphConfig,
) -> bool:
    """Ingest a markdown document into the LightRAG knowledge graph.

    Returns True on success, False on failure (non-blocking).
    """
    try:
        rag = await _get_rag_instance(cfg)
        await rag.ainsert(markdown)
        logger.info("Ingested into knowledge graph: %s", doc_id)
        return True
    except Exception as e:
        logger.error("LightRAG ingestion failed for %s: %s", doc_id, e)
        return False


async def rebuild_graph(markdown_dir: Path, cfg: GraphConfig) -> int:
    """Delete the graph store and re-ingest all markdown files.

    Returns the number of documents ingested.
    """
    store_dir = Path(cfg.store_dir)
    if store_dir.exists():
        logger.info("Deleting existing graph store: %s", store_dir)
        shutil.rmtree(store_dir)

    md_files = sorted(markdown_dir.glob("*.md"))
    logger.info("Rebuilding graph from %d markdown files", len(md_files))

    count = 0
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        success = await ingest_document(content, md_file.stem, cfg)
        if success:
            count += 1

    logger.info("Graph rebuilt: %d documents ingested", count)
    return count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/graph.py tests/test_graph.py
git commit -m "feat: graph module — LightRAG knowledge graph ingestion"
```

---

### Task 9: Pipeline Orchestrator

**Files:**
- Create: `src/docpipe/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import DocpipeConfig, load_config
from docpipe.pipeline import Lockfile, cleanup_orphans, process_file


class TestLockfile:
    def test_acquire_and_release(self, tmp_dirs: dict[str, Path]) -> None:
        lock_path = tmp_dirs["output"] / ".docpipe.lock"
        lock = Lockfile(lock_path)
        assert lock.acquire()
        assert lock_path.exists()
        lock.release()
        assert not lock_path.exists()

    def test_fails_if_already_locked(self, tmp_dirs: dict[str, Path]) -> None:
        lock_path = tmp_dirs["output"] / ".docpipe.lock"
        lock1 = Lockfile(lock_path)
        lock2 = Lockfile(lock_path)
        assert lock1.acquire()
        assert not lock2.acquire()
        lock1.release()

    def test_context_manager(self, tmp_dirs: dict[str, Path]) -> None:
        lock_path = tmp_dirs["output"] / ".docpipe.lock"
        lock = Lockfile(lock_path)
        with lock:
            assert lock_path.exists()
        assert not lock_path.exists()


class TestCleanupOrphans:
    def test_removes_orphaned_images(self, tmp_dirs: dict[str, Path]) -> None:
        img_dir = tmp_dirs["output"] / "images"
        (img_dir / "test_doc_img001.png").write_bytes(b"fake")
        (img_dir / "test_doc_img002.png").write_bytes(b"fake")
        (img_dir / "other_img001.png").write_bytes(b"fake")

        cleanup_orphans("test_doc", tmp_dirs["output"])

        assert not (img_dir / "test_doc_img001.png").exists()
        assert not (img_dir / "test_doc_img002.png").exists()
        assert (img_dir / "other_img001.png").exists()  # untouched


class TestProcessFile:
    @pytest.mark.asyncio
    @patch("docpipe.pipeline.ingest_document", new_callable=AsyncMock, return_value=True)
    @patch("docpipe.pipeline.generate_summary", new_callable=AsyncMock, return_value=("Summary", "topics"))
    @patch("docpipe.pipeline.replace_image_refs", new_callable=AsyncMock, side_effect=lambda md, *a, **kw: md)
    async def test_processes_pdf_end_to_end(
        self,
        mock_replace: AsyncMock,
        mock_summary: AsyncMock,
        mock_ingest: AsyncMock,
        sample_pdf: Path,
        config_file: Path,
    ) -> None:
        cfg = load_config(config_file)
        result = await process_file(sample_pdf, cfg)
        assert result is True

        # Check markdown was written
        md_path = cfg.output_dir / "markdown" / "test_doc.md"
        assert md_path.exists()
        assert "Hello World" in md_path.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement pipeline.py**

```python
# src/docpipe/pipeline.py
"""Pipeline orchestrator — full flow per file."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from docpipe.config import DocpipeConfig
from docpipe.converter import convert_to_pdf
from docpipe.describer import replace_image_refs
from docpipe.extractor import extract_markdown
from docpipe.graph import ingest_document
from docpipe.registry import RegistryEntry, generate_summary, update_registry
from docpipe.status import StatusTracker

logger = logging.getLogger(__name__)


class Lockfile:
    """Simple file-based lock to prevent concurrent processing."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._acquired = False

    def acquire(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._acquired = True
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if self._acquired and self._path.exists():
            self._path.unlink()
            self._acquired = False

    def __enter__(self) -> Lockfile:
        if not self.acquire():
            raise RuntimeError(
                "Error: docpipe is already running (lockfile held). "
                "Stop the watcher first or wait for it to finish."
            )
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


def cleanup_orphans(doc_stem: str, output_dir: Path) -> None:
    """Remove orphaned output files from a previous partial run."""
    # Clean images
    img_dir = output_dir / "images"
    if img_dir.exists():
        for img in img_dir.iterdir():
            if img.stem.startswith(doc_stem):
                img.unlink()
                logger.debug("Removed orphaned image: %s", img.name)

    # Clean markdown
    md_path = output_dir / "markdown" / f"{doc_stem}.md"
    if md_path.exists():
        md_path.unlink()
        logger.debug("Removed orphaned markdown: %s", md_path.name)


async def process_file(file_path: Path, cfg: DocpipeConfig) -> bool:
    """Run the full pipeline for a single file. Returns True on success."""
    doc_stem = file_path.stem
    tracker = StatusTracker(cfg.output_dir)

    try:
        tracker.update_file(file_path.name, status="processing")
        tracker.save()

        # Stage 0: Cleanup orphans
        cleanup_orphans(doc_stem, cfg.output_dir)

        # Stage 1: Convert to PDF
        try:
            pdf_path = convert_to_pdf(file_path, cfg.output_dir, cfg.converter)
        except FileNotFoundError:
            logger.warning("File not found (possibly renamed): %s", file_path)
            tracker.update_file(file_path.name, status="failed", error="File not found")
            tracker.save()
            return False
        except Exception as e:
            logger.error("Conversion failed for %s: %s", file_path.name, e)
            tracker.update_file(file_path.name, status="failed", error=str(e))
            tracker.save()
            reg_entry = RegistryEntry.failed(file_path.name, str(e))
            update_registry(cfg.output_dir / cfg.registry.filename, reg_entry)
            return False

        # Stage 2: Extract markdown + images
        md_dir = cfg.output_dir / "markdown"
        md_dir.mkdir(parents=True, exist_ok=True)
        img_dir = cfg.output_dir / "images"

        result = extract_markdown(pdf_path, img_dir, cfg.extractor)

        if not result.markdown.strip():
            logger.warning("Empty extraction for %s, skipping", file_path.name)
            tracker.update_file(file_path.name, status="skipped", error="Empty document")
            tracker.save()
            reg_entry = RegistryEntry.failed(file_path.name, "Empty document")
            update_registry(cfg.output_dir / cfg.registry.filename, reg_entry)
            return False

        # Stage 3: Describe images with VLM
        markdown = await replace_image_refs(
            result.markdown,
            cfg.output_dir,
            cfg.describer,
            cfg.api_retry,
            doc_title=doc_stem,
        )

        # Write final markdown
        md_path = md_dir / f"{doc_stem}.md"
        md_path.write_text(markdown, encoding="utf-8")

        # Stage 4: Update registry
        summary, topics = await generate_summary(markdown, cfg.registry, cfg.api_retry)
        reg_entry = RegistryEntry(
            filename=f"{doc_stem}.md",
            author="-",  # Could be extracted from PDF metadata
            date="-",
            summary=summary,
            topics=topics,
            pages=result.page_count,
            images=len(result.image_paths),
        )
        update_registry(cfg.output_dir / cfg.registry.filename, reg_entry)

        # Stage 5: Ingest into knowledge graph
        graph_ok = await ingest_document(markdown, doc_stem, cfg.graph)

        # Stage 6: Update status
        tracker.update_file(
            file_path.name,
            status="done",
            pages=result.page_count,
            images=len(result.image_paths),
            graph_ingested=graph_ok,
            md_path=f"markdown/{doc_stem}.md",
        )
        tracker.save()

        # Clean up converted PDF if it was a temp conversion
        if pdf_path != file_path and pdf_path.exists():
            pdf_path.unlink()

        logger.info("Successfully processed: %s", file_path.name)
        return True

    except Exception as e:
        logger.error("Pipeline failed for %s: %s", file_path.name, e, exc_info=True)
        tracker.update_file(file_path.name, status="failed", error=str(e))
        tracker.save()
        reg_entry = RegistryEntry.failed(file_path.name, str(e))
        update_registry(cfg.output_dir / cfg.registry.filename, reg_entry)
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrator with lockfile and cleanup"
```

---

### Task 10: Watcher Module

**Files:**
- Create: `src/docpipe/watcher.py`
- Create: `tests/test_watcher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_watcher.py
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import WatcherConfig
from docpipe.watcher import DebouncedHandler


class TestDebouncedHandler:
    def test_accumulates_events_within_debounce(self) -> None:
        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=1,
            max_wait_seconds=10,
            supported_extensions={".pdf", ".docx"},
        )
        handler._on_relevant_event(Path("/fake/test.pdf"))
        handler._on_relevant_event(Path("/fake/test.pdf"))

        # Callback should not have been called yet (debounce not expired)
        assert "test.pdf" in handler._pending

    def test_ignores_unsupported_extensions(self) -> None:
        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=1,
            max_wait_seconds=10,
            supported_extensions={".pdf"},
        )
        handler._on_relevant_event(Path("/fake/test.xyz"))
        assert len(handler._pending) == 0

    def test_tracks_deletions_separately(self) -> None:
        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=1,
            max_wait_seconds=10,
            supported_extensions={".pdf"},
        )
        handler._on_delete_event(Path("/fake/test.pdf"))
        assert "test.pdf" in handler._deleted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_watcher.py -v`
Expected: FAIL

- [ ] **Step 3: Implement watcher.py**

```python
# src/docpipe/watcher.py
"""Watchdog-based folder monitor with trailing-edge debounce."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from docpipe.config import ConverterConfig, WatcherConfig

logger = logging.getLogger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """Debounce file events and batch-trigger processing."""

    def __init__(
        self,
        callback: Callable[[list[Path], list[str]], None],
        debounce_seconds: int,
        max_wait_seconds: int,
        supported_extensions: set[str],
    ) -> None:
        super().__init__()
        self._callback = callback
        self._debounce = debounce_seconds
        self._max_wait = max_wait_seconds
        self._supported = supported_extensions | {".pdf"}
        self._pending: dict[str, Path] = {}
        self._deleted: set[str] = set()
        self._last_event_time: float = 0
        self._first_event_time: float = 0
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None

    def _is_relevant(self, path: Path) -> bool:
        return path.suffix.lower() in self._supported

    def _on_relevant_event(self, path: Path) -> None:
        if not self._is_relevant(path):
            return

        with self._lock:
            now = time.time()
            if not self._pending and not self._deleted:
                self._first_event_time = now
            self._last_event_time = now
            self._pending[path.name] = path
            self._deleted.discard(path.name)
            self._schedule_flush()

    def _on_delete_event(self, path: Path) -> None:
        if not self._is_relevant(path):
            return

        with self._lock:
            now = time.time()
            if not self._pending and not self._deleted:
                self._first_event_time = now
            self._last_event_time = now
            self._deleted.add(path.name)
            self._pending.pop(path.name, None)
            self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._timer:
            self._timer.cancel()

        now = time.time()
        time_since_first = now - self._first_event_time

        if time_since_first >= self._max_wait:
            # Max wait exceeded — flush immediately
            self._flush()
        else:
            # Schedule flush after debounce period
            delay = min(self._debounce, self._max_wait - time_since_first)
            self._timer = threading.Timer(delay, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            if not self._pending and not self._deleted:
                return

            files = list(self._pending.values())
            deleted = list(self._deleted)
            self._pending.clear()
            self._deleted.clear()

        if files or deleted:
            logger.info(
                "Debounce flush: %d files to process, %d deletions",
                len(files), len(deleted),
            )
            self._callback(files, deleted)

    # Watchdog event handlers
    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_relevant_event(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_relevant_event(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_delete_event(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_delete_event(Path(event.src_path))
            if hasattr(event, "dest_path"):
                self._on_relevant_event(Path(event.dest_path))


def start_watcher(
    input_dir: Path,
    watcher_cfg: WatcherConfig,
    converter_cfg: ConverterConfig,
    callback: Callable[[list[Path], list[str]], None],
) -> Observer:
    """Start the file system watcher. Returns the Observer (call .stop() to stop)."""
    supported = set(converter_cfg.supported_extensions) | {".pdf"}
    handler = DebouncedHandler(
        callback=callback,
        debounce_seconds=watcher_cfg.debounce_seconds,
        max_wait_seconds=watcher_cfg.max_wait_seconds,
        supported_extensions=supported,
    )

    observer = Observer()
    observer.schedule(handler, str(input_dir), recursive=watcher_cfg.watch_subdirectories)
    observer.start()
    logger.info("Watching %s for changes (debounce: %ds)", input_dir, watcher_cfg.debounce_seconds)
    return observer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_watcher.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/watcher.py tests/test_watcher.py
git commit -m "feat: watcher module — trailing-edge debounce file monitor"
```

---

### Task 11: CLI Module

**Files:**
- Create: `src/docpipe/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from docpipe.cli import main


class TestCLI:
    def test_init_creates_structure(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert Path("config.yaml").exists()
            assert Path("input").exists()
            assert Path("output").exists()

    def test_status_without_data(self, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["status", "--config", str(config_file)])
        assert result.exit_code == 0

    @patch("docpipe.cli._run_ingest", new_callable=AsyncMock)
    def test_ingest_calls_pipeline(self, mock_ingest: AsyncMock, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["ingest", "--config", str(config_file)])
        assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement cli.py**

```python
# src/docpipe/cli.py
"""CLI interface for docpipe."""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

from docpipe.config import DocpipeConfig, load_config
from docpipe.graph import rebuild_graph
from docpipe.pipeline import Lockfile, cleanup_orphans, process_file
from docpipe.registry import RegistryEntry, update_registry
from docpipe.status import StatusTracker
from docpipe.watcher import start_watcher

console = Console()
DEFAULT_CONFIG = "config.yaml"


def _setup_logging(cfg: DocpipeConfig) -> None:
    log_dir = Path(cfg.logging.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        cfg.logging.file,
        maxBytes=cfg.logging.max_size_mb * 1024 * 1024,
        backupCount=cfg.logging.backup_count,
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    root = logging.getLogger("docpipe")
    root.setLevel(getattr(logging, cfg.logging.level.upper()))
    root.addHandler(handler)


def _build_status_table(tracker: StatusTracker, cfg: DocpipeConfig) -> Table:
    data = tracker.to_dict()
    table = Table(title="DocPipe Status", show_lines=True)
    table.add_column("File")
    table.add_column("Status")
    table.add_column("Pages")
    table.add_column("Images")
    table.add_column("Graph")
    table.add_column("Updated")

    for fname, info in data.get("files", {}).items():
        status = info.get("status", "unknown")
        style = {"done": "green", "failed": "red", "processing": "yellow"}.get(status, "")
        table.add_row(
            fname,
            status,
            str(info.get("pages", "-")),
            str(info.get("images", "-")),
            "yes" if info.get("graph_ingested") else "no",
            info.get("last_processed", "-"),
            style=style,
        )

    graph = data.get("graph", {})
    api = data.get("api_usage", {})
    table.caption = (
        f"Graph: {graph.get('entities', 0)} entities, {graph.get('relations', 0)} relations | "
        f"API: {api.get('tokens_today', 0)} tokens today (~${api.get('estimated_cost_usd', 0):.4f})"
    )
    return table


async def _run_ingest(
    cfg: DocpipeConfig,
    file_path: str | None = None,
    rebuild: bool = False,
) -> None:
    if rebuild:
        md_dir = cfg.output_dir / "markdown"
        console.print(f"Rebuilding graph from {md_dir}...")
        count = await rebuild_graph(md_dir, cfg.graph)
        console.print(f"Graph rebuilt: {count} documents ingested.")
        return

    input_dir = cfg.input_dir
    if file_path:
        target = input_dir / file_path
        if not target.exists():
            console.print(f"[red]File not found: {target}[/red]")
            return
        files = [target]
    else:
        supported = set(cfg.converter.supported_extensions) | {".pdf"}
        files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        ]

    console.print(f"Processing {len(files)} file(s)...")
    for f in files:
        console.print(f"  Processing: {f.name}")
        success = await process_file(f, cfg)
        status = "[green]done[/green]" if success else "[red]failed[/red]"
        console.print(f"  {f.name}: {status}")


def _watcher_callback(cfg: DocpipeConfig) -> Any:
    """Create the callback for the watcher."""

    def callback(files: list[Path], deleted: list[str]) -> None:
        # Handle deletions
        for name in deleted:
            stem = Path(name).stem
            cleanup_orphans(stem, cfg.output_dir)
            reg_path = cfg.output_dir / cfg.registry.filename
            entry = RegistryEntry.failed(name, "deleted")
            update_registry(reg_path, entry, remove=True)
            tracker = StatusTracker(cfg.output_dir)
            tracker.remove_file(name)
            tracker.save()
            console.print(f"  Removed: {name}")

        # Process new/changed files
        for f in files:
            console.print(f"  Processing: {f.name}")
            success = asyncio.run(process_file(f, cfg))
            status = "[green]done[/green]" if success else "[red]failed[/red]"
            console.print(f"  {f.name}: {status}")

    return callback


@click.group()
def main() -> None:
    """DocPipe — document ingestion pipeline."""
    pass


@main.command()
def init() -> None:
    """Create default config.yaml and folder structure."""
    cfg_path = Path(DEFAULT_CONFIG)
    if cfg_path.exists():
        console.print("[yellow]config.yaml already exists, skipping[/yellow]")
    else:
        # Copy from package or create minimal
        cfg_path.write_text(
            "# DocPipe Configuration\n"
            "input_dir: ./input\n"
            "output_dir: ./output\n"
        )
        console.print("Created config.yaml")

    for d in ["input", "output", "output/markdown", "output/images", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text("OPENAI_API_KEY=your-key-here\n")
        console.print("Created .env (set your OPENAI_API_KEY)")

    console.print("[green]Project initialized.[/green]")


@main.command()
@click.option("--config", "config_path", default=DEFAULT_CONFIG, help="Path to config.yaml")
@click.option("--dashboard", is_flag=True, help="Show live dashboard")
def run(config_path: str, dashboard: bool) -> None:
    """Start the file watcher (foreground, blocking)."""
    cfg = load_config(Path(config_path))
    _setup_logging(cfg)

    lock = Lockfile(cfg.output_dir / ".docpipe.lock")
    if not lock.acquire():
        console.print(
            "[red]Error: docpipe is already running (lockfile held). "
            "Stop the watcher first or wait for it to finish.[/red]"
        )
        sys.exit(1)

    try:
        tracker = StatusTracker(cfg.output_dir)
        tracker.set_watcher_running()
        tracker.save()

        callback = _watcher_callback(cfg)
        observer = start_watcher(cfg.input_dir, cfg.watcher, cfg.converter, callback)

        console.print(f"[green]Watching {cfg.input_dir} (Ctrl+C to stop)[/green]")

        if dashboard:
            with Live(_build_status_table(tracker, cfg), refresh_per_second=0.5, console=console) as live:
                while observer.is_alive():
                    tracker.heartbeat()
                    tracker.save()
                    live.update(_build_status_table(tracker, cfg))
                    observer.join(timeout=2)
        else:
            while observer.is_alive():
                tracker.heartbeat()
                tracker.save()
                observer.join(timeout=5)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping watcher...[/yellow]")
        observer.stop()
        observer.join()
    finally:
        tracker = StatusTracker(cfg.output_dir)
        tracker.set_watcher_stopped()
        tracker.save()
        lock.release()

    console.print("[green]Watcher stopped.[/green]")


@main.command()
@click.option("--config", "config_path", default=DEFAULT_CONFIG, help="Path to config.yaml")
@click.option("--rebuild-graph", "rebuild", is_flag=True, help="Rebuild LightRAG graph from scratch")
@click.argument("file", required=False)
def ingest(config_path: str, rebuild: bool, file: str | None) -> None:
    """One-shot: process files in input_dir."""
    cfg = load_config(Path(config_path))
    _setup_logging(cfg)

    lock = Lockfile(cfg.output_dir / ".docpipe.lock")
    if not lock.acquire():
        console.print(
            "[red]Error: docpipe is already running (lockfile held). "
            "Stop the watcher first or wait for it to finish.[/red]"
        )
        sys.exit(1)

    try:
        asyncio.run(_run_ingest(cfg, file, rebuild))
    finally:
        lock.release()


@main.command()
@click.option("--config", "config_path", default=DEFAULT_CONFIG, help="Path to config.yaml")
def status(config_path: str) -> None:
    """Show pipeline status."""
    cfg = load_config(Path(config_path))
    tracker = StatusTracker(cfg.output_dir)
    table = _build_status_table(tracker, cfg)
    console.print(table)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/docpipe/cli.py tests/test_cli.py
git commit -m "feat: CLI module — init, run, ingest, status commands"
```

---

### Task 12: CI/CD Workflows

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/release.yml`
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Create CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --dev
      - run: uv run ruff check src/ tests/
      - run: uv run ruff format --check src/ tests/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --dev
      - run: uv run mypy src/docpipe/

  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --dev
      - run: uv run pytest --cov=docpipe --cov-report=xml -v
      - uses: codecov/codecov-action@v4
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
        with:
          file: coverage.xml

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

- [ ] **Step 2: Create release workflow**

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv build
      - run: uv publish --trusted-publishing always
```

- [ ] **Step 3: Create docs workflow**

```yaml
# .github/workflows/docs.yml
name: Docs

on:
  push:
    branches: [main]

concurrency:
  group: wiki-gen
  cancel-in-progress: true

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

- [ ] **Step 4: Commit**

```bash
git add .github/
git commit -m "ci: add CI, release, and docs workflows"
```

---

### Task 13: Integration Test & Final Verification

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test with mocked API calls."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from docpipe.config import load_config
from docpipe.pipeline import process_file


class TestEndToEnd:
    @pytest.mark.asyncio
    @patch("docpipe.graph.ingest_document", new_callable=AsyncMock, return_value=True)
    @patch("docpipe.registry.generate_summary", new_callable=AsyncMock, return_value=("Test summary", "test, integration"))
    @patch("docpipe.describer.replace_image_refs", new_callable=AsyncMock, side_effect=lambda md, *a, **kw: md)
    async def test_full_pipeline_pdf(
        self,
        mock_replace: AsyncMock,
        mock_summary: AsyncMock,
        mock_ingest: AsyncMock,
        sample_pdf: Path,
        config_file: Path,
    ) -> None:
        cfg = load_config(config_file)

        # Process the file
        success = await process_file(sample_pdf, cfg)
        assert success

        # Verify markdown output
        md_path = cfg.output_dir / "markdown" / "test_doc.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Hello World" in content

        # Verify registry
        reg_path = cfg.output_dir / "registry.md"
        assert reg_path.exists()
        reg_content = reg_path.read_text()
        assert "test_doc.md" in reg_content
        assert "Test summary" in reg_content

        # Verify status.json
        status_path = cfg.output_dir / "status.json"
        assert status_path.exists()

        # Verify API calls
        mock_ingest.assert_called_once()
        mock_summary.assert_called_once()
```

- [ ] **Step 2: Run integration test**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: All tests pass

- [ ] **Step 4: Run linter and type checker**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: No issues

Run: `uv run mypy src/docpipe/`
Expected: No errors (with lightrag/pymupdf4llm ignored)

- [ ] **Step 5: Verify build**

Run: `uv build`
Expected: Wheel and sdist created in `dist/`

- [ ] **Step 6: Commit integration test**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test"
```

- [ ] **Step 7: Final commit — README**

Create a brief README.md with quickstart (init, configure, run). Then:

```bash
git add README.md
git commit -m "docs: add README with quickstart"
```
