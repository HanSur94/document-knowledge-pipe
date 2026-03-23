from __future__ import annotations

from pathlib import Path

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
        assert result.markdown

    def test_empty_pdf_returns_empty(self, tmp_dirs: dict[str, Path]) -> None:
        import fitz

        pdf_path = tmp_dirs["input"] / "empty.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        result = extract_markdown(
            pdf_path,
            tmp_dirs["output"] / "images",
            ExtractorConfig(),
        )
        assert result.page_count == 1
