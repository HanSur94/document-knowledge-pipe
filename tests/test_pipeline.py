from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import needs_api_keys

from docpipe.config import load_config
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


@needs_api_keys
class TestProcessFile:
    @pytest.fixture
    def api_config(self, tmp_dirs: dict[str, Path]) -> Path:
        """Config with Anthropic provider and real API retry settings."""
        config = {
            "input_dir": str(tmp_dirs["input"]),
            "output_dir": str(tmp_dirs["output"]),
            "describer": {
                "provider": "anthropic",
                "max_tokens": 300,
                "include_context": True,
                "context_chars": 500,
                "batch_size": 5,
                "anthropic": {
                    "model": "claude-haiku-4-5-20251001",
                },
            },
            "graph": {
                "provider": "openai",
                "storage": "file",
                "store_dir": str(tmp_dirs["output"] / "lightrag_store"),
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
            },
            "api_retry": {
                "max_retries": 2,
                "initial_delay_seconds": 1,
                "max_delay_seconds": 10,
            },
            "logging": {
                "level": "DEBUG",
                "file": str(tmp_dirs["root"] / "logs" / "docpipe.log"),
                "max_size_mb": 1,
                "backup_count": 1,
            },
        }
        cfg_path = tmp_dirs["root"] / "config.yaml"
        cfg_path.write_text(yaml.dump(config))
        return cfg_path

    @pytest.mark.asyncio
    async def test_processes_pdf_end_to_end(
        self,
        sample_pdf: Path,
        api_config: Path,
    ) -> None:
        cfg = load_config(api_config)
        result = await process_file(sample_pdf, cfg)
        assert result is True

        md_path = cfg.output_dir / "markdown" / "test_doc.md"
        assert md_path.exists()
        assert "Hello World" in md_path.read_text()
