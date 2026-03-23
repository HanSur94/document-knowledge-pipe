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
        assert (img_dir / "other_img001.png").exists()


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

        md_path = cfg.output_dir / "markdown" / "test_doc.md"
        assert md_path.exists()
        assert "Hello World" in md_path.read_text()
