from __future__ import annotations

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
    import fitz

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
