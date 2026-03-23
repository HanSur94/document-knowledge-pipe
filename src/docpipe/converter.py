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

    found = shutil.which("soffice")
    if found:
        return Path(found)

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

        expected_name = source.stem + ".pdf"
        pdf_path = output_dir / expected_name

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
