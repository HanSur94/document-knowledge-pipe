from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import needs_libreoffice

from docpipe.config import ConverterConfig
from docpipe.converter import convert_to_pdf, find_libreoffice


class TestFindLibreOffice:
    def test_explicit_path_returned_if_set(self) -> None:
        cfg = ConverterConfig(libreoffice_path="/usr/bin/soffice")
        assert find_libreoffice(cfg) == Path("/usr/bin/soffice")

    @needs_libreoffice
    def test_auto_detect_from_path(self) -> None:
        cfg = ConverterConfig(libreoffice_path=None)
        result = find_libreoffice(cfg)
        assert result is not None
        assert result.exists()


class TestConvertToPdf:
    def test_pdf_passes_through(self, sample_pdf: Path, tmp_dirs: dict[str, Path]) -> None:
        result = convert_to_pdf(
            sample_pdf,
            tmp_dirs["output"],
            ConverterConfig(),
        )
        assert result == sample_pdf

    def test_unsupported_extension_raises(self, tmp_dirs: dict[str, Path]) -> None:
        bad_file = tmp_dirs["input"] / "file.xyz"
        bad_file.write_text("not a real file")
        with pytest.raises(ValueError, match="Unsupported"):
            convert_to_pdf(bad_file, tmp_dirs["output"], ConverterConfig())

    @needs_libreoffice
    def test_converts_docx_to_pdf(self, tmp_dirs: dict[str, Path]) -> None:
        """Convert a real .docx file using LibreOffice."""
        fixtures = Path(__file__).parent / "fixtures"
        docx_src = fixtures / "sample.docx"
        if not docx_src.exists():
            pytest.skip("sample.docx fixture not found")

        docx = tmp_dirs["input"] / "sample.docx"
        shutil.copy2(docx_src, docx)

        result = convert_to_pdf(docx, tmp_dirs["output"], ConverterConfig())
        assert result.exists()
        assert result.suffix == ".pdf"
