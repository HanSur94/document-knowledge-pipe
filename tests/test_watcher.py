from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docpipe.config import WatcherConfig
from docpipe.watcher import DebouncedHandler


class TestDebouncedHandler:
    def test_accumulates_events_within_debounce(self) -> None:
        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=1,
            max_wait_seconds=10,
            supported_extensions={".pdf", ".docx"},
        )
        handler._on_relevant_event(Path("/fake/test.pdf"))
        handler._on_relevant_event(Path("/fake/test.pdf"))

        assert "test.pdf" in handler._pending

    def test_ignores_unsupported_extensions(self) -> None:
        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=1,
            max_wait_seconds=10,
            supported_extensions={".pdf"},
        )
        handler._on_relevant_event(Path("/fake/test.xyz"))
        assert len(handler._pending) == 0

    def test_tracks_deletions_separately(self) -> None:
        callback = MagicMock()
        handler = DebouncedHandler(
            callback=callback,
            debounce_seconds=1,
            max_wait_seconds=10,
            supported_extensions={".pdf"},
        )
        handler._on_delete_event(Path("/fake/test.pdf"))
        assert "test.pdf" in handler._deleted
