from __future__ import annotations

import subprocess
from pathlib import Path
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
        assert result == sample_pdf

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
        convert_to_pdf(docx, tmp_dirs["output"], cfg)

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
