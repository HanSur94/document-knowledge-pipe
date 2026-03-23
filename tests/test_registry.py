from __future__ import annotations

from pathlib import Path

from docpipe.registry import RegistryEntry, update_registry


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
            filename="test.md",
            author="-",
            date="-",
            summary="Test doc",
            topics="test",
            pages=1,
            images=0,
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
            filename="test.md",
            author="-",
            date="-",
            summary="Version 1",
            topics="test",
            pages=1,
            images=0,
        )
        update_registry(reg_path, entry1)

        entry2 = RegistryEntry(
            filename="test.md",
            author="-",
            date="-",
            summary="Version 2",
            topics="test",
            pages=2,
            images=1,
        )
        update_registry(reg_path, entry2)

        content = reg_path.read_text()
        assert content.count("test.md") == 1
        assert "Version 2" in content

    def test_removes_entry(self, tmp_dirs: dict[str, Path]) -> None:
        reg_path = tmp_dirs["output"] / "registry.md"
        entry = RegistryEntry(
            filename="test.md",
            author="-",
            date="-",
            summary="Will be removed",
            topics="test",
            pages=1,
            images=0,
        )
        update_registry(reg_path, entry)
        update_registry(reg_path, entry, remove=True)
        content = reg_path.read_text()
        assert "test.md" not in content
