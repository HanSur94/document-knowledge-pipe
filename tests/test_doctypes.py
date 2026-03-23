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
FFC_FIXTURES = Path(__file__).parent / "fixtures" / "file-format-commons"

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


# --- file-format-commons tests (second fixture set) ---


class TestFfcPdfExtraction:
    """PDF from file-format-commons."""

    def test_extract_ffc_pdf(self, tmp_path: Path) -> None:
        pdf = FFC_FIXTURES / "ffc.pdf"
        if not pdf.exists():
            pytest.skip("ffc.pdf fixture not found")

        result = extract_markdown(pdf, tmp_path / "images", ExtractorConfig())
        assert result.page_count > 0
        assert len(result.markdown) > 10


_FFC_CONVERT_CASES = [
    ("ffc.docx", ".docx"),
    ("ffc.doc", ".doc"),
    ("ffc_6.doc", ".doc"),
    ("ffc_95.doc", ".doc"),
    ("ffc_97_2000_xp.doc", ".doc"),
    ("ffc.pptx", ".pptx"),
    ("ffc.ppt", ".ppt"),
    ("ffc.xlsx", ".xlsx"),
    ("ffc.xls", ".xls"),
    ("ffc.odt", ".odt"),
    ("ffc.ods", ".ods"),
    ("ffc.rtf", ".rtf"),
    ("ffc.html", ".html"),
]


@needs_libreoffice
class TestFfcConversions:
    """Convert and extract all file-format-commons documents."""

    @pytest.mark.parametrize("filename,ext", _FFC_CONVERT_CASES)
    def test_convert_and_extract(self, filename: str, ext: str, tmp_path: Path) -> None:
        src = FFC_FIXTURES / filename
        if not src.exists():
            pytest.skip(f"{filename} fixture not found")

        cfg = ConverterConfig()
        pdf_path = convert_to_pdf(src, tmp_path, cfg)
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"

        result = extract_markdown(pdf_path, tmp_path / "images", ExtractorConfig())
        assert result.page_count > 0
