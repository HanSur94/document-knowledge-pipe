"""Document type tests — verify extraction works for each supported format.

Tests marked with @pytest.mark.doctypes require LibreOffice and are run
in CI via the doc-tests workflow, not in the standard test suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docpipe.config import ConverterConfig, ExtractorConfig
from docpipe.converter import convert_to_pdf, find_libreoffice
from docpipe.extractor import ExtractionResult, extract_markdown

FIXTURES = Path(__file__).parent / "fixtures"

# Check if LibreOffice is available
_lo_available = False
try:
    find_libreoffice(ConverterConfig())
    _lo_available = True
except FileNotFoundError:
    pass

needs_libreoffice = pytest.mark.skipif(not _lo_available, reason="LibreOffice not installed")


class TestPdfExtraction:
    """PDF extraction works without LibreOffice."""

    def test_extract_sample_pdf(self, tmp_path: Path) -> None:
        pdf = FIXTURES / "sample.pdf"
        if not pdf.exists():
            pytest.skip("sample.pdf fixture not found")

        img_dir = tmp_path / "images"
        result = extract_markdown(pdf, img_dir, ExtractorConfig())

        assert isinstance(result, ExtractionResult)
        assert result.page_count > 0
        assert len(result.markdown) > 100


@needs_libreoffice
class TestDocxConversion:
    def test_convert_and_extract_docx(self, tmp_path: Path) -> None:
        docx = FIXTURES / "sample.docx"
        if not docx.exists():
            pytest.skip("sample.docx fixture not found")

        cfg = ConverterConfig()
        pdf_path = convert_to_pdf(docx, tmp_path, cfg)
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"

        result = extract_markdown(pdf_path, tmp_path / "images", ExtractorConfig())
        assert result.page_count > 0
        assert len(result.markdown) > 50


@needs_libreoffice
class TestDocConversion:
    def test_convert_and_extract_doc(self, tmp_path: Path) -> None:
        doc = FIXTURES / "sample.doc"
        if not doc.exists():
            pytest.skip("sample.doc fixture not found")

        cfg = ConverterConfig()
        pdf_path = convert_to_pdf(doc, tmp_path, cfg)
        assert pdf_path.exists()

        result = extract_markdown(pdf_path, tmp_path / "images", ExtractorConfig())
        assert result.page_count > 0
        assert len(result.markdown) > 50


@needs_libreoffice
class TestPptxConversion:
    def test_convert_and_extract_pptx(self, tmp_path: Path) -> None:
        pptx = FIXTURES / "sample.pptx"
        if not pptx.exists():
            pytest.skip("sample.pptx fixture not found")

        cfg = ConverterConfig()
        pdf_path = convert_to_pdf(pptx, tmp_path, cfg)
        assert pdf_path.exists()

        result = extract_markdown(pdf_path, tmp_path / "images", ExtractorConfig())
        assert result.page_count > 0


@needs_libreoffice
class TestXlsxConversion:
    def test_convert_and_extract_xlsx(self, tmp_path: Path) -> None:
        xlsx = FIXTURES / "sample.xlsx"
        if not xlsx.exists():
            pytest.skip("sample.xlsx fixture not found")

        cfg = ConverterConfig()
        pdf_path = convert_to_pdf(xlsx, tmp_path, cfg)
        assert pdf_path.exists()

        result = extract_markdown(pdf_path, tmp_path / "images", ExtractorConfig())
        assert result.page_count > 0


@needs_libreoffice
class TestOdtConversion:
    def test_convert_and_extract_odt(self, tmp_path: Path) -> None:
        odt = FIXTURES / "sample.odt"
        if not odt.exists():
            pytest.skip("sample.odt fixture not found")

        cfg = ConverterConfig()
        pdf_path = convert_to_pdf(odt, tmp_path, cfg)
        assert pdf_path.exists()

        result = extract_markdown(pdf_path, tmp_path / "images", ExtractorConfig())
        assert result.page_count > 0


@needs_libreoffice
class TestRtfConversion:
    def test_convert_and_extract_rtf(self, tmp_path: Path) -> None:
        rtf = FIXTURES / "sample.rtf"
        if not rtf.exists():
            pytest.skip("sample.rtf fixture not found")

        cfg = ConverterConfig()
        pdf_path = convert_to_pdf(rtf, tmp_path, cfg)
        assert pdf_path.exists()

        result = extract_markdown(pdf_path, tmp_path / "images", ExtractorConfig())
        assert result.page_count > 0


@needs_libreoffice
class TestHtmlConversion:
    def test_convert_and_extract_html(self, tmp_path: Path) -> None:
        html = FIXTURES / "sample.html"
        if not html.exists():
            pytest.skip("sample.html fixture not found")

        cfg = ConverterConfig()
        pdf_path = convert_to_pdf(html, tmp_path, cfg)
        assert pdf_path.exists()

        result = extract_markdown(pdf_path, tmp_path / "images", ExtractorConfig())
        assert result.page_count > 0
        assert len(result.markdown) > 50
