"""Watchdog-based folder monitor with trailing-edge debounce."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from docpipe.config import ConverterConfig, WatcherConfig

logger = logging.getLogger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """Debounce file events and batch-trigger processing."""

    def __init__(
        self,
        callback: Callable[[list[Path], list[str]], None],
        debounce_seconds: int,
        max_wait_seconds: int,
        supported_extensions: set[str],
    ) -> None:
        super().__init__()
        self._callback = callback
        self._debounce = debounce_seconds
        self._max_wait = max_wait_seconds
        self._supported = supported_extensions | {".pdf"}
        self._pending: dict[str, Path] = {}
        self._deleted: set[str] = set()
        self._last_event_time: float = 0
        self._first_event_time: float = 0
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None

    def _is_relevant(self, path: Path) -> bool:
        return path.suffix.lower() in self._supported

    def _on_relevant_event(self, path: Path) -> None:
        if not self._is_relevant(path):
            return

        with self._lock:
            now = time.time()
            if not self._pending and not self._deleted:
                self._first_event_time = now
            self._last_event_time = now
            self._pending[path.name] = path
            self._deleted.discard(path.name)
            self._schedule_flush()

    def _on_delete_event(self, path: Path) -> None:
        if not self._is_relevant(path):
            return

        with self._lock:
            now = time.time()
            if not self._pending and not self._deleted:
                self._first_event_time = now
            self._last_event_time = now
            self._deleted.add(path.name)
            self._pending.pop(path.name, None)
            self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._timer:
            self._timer.cancel()

        now = time.time()
        time_since_first = now - self._first_event_time

        if time_since_first >= self._max_wait:
            self._flush()
        else:
            delay = min(self._debounce, self._max_wait - time_since_first)
            self._timer = threading.Timer(delay, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            if not self._pending and not self._deleted:
                return

            files = list(self._pending.values())
            deleted = list(self._deleted)
            self._pending.clear()
            self._deleted.clear()

        if files or deleted:
            logger.info(
                "Debounce flush: %d files to process, %d deletions",
                len(files),
                len(deleted),
            )
            self._callback(files, deleted)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_relevant_event(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_relevant_event(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_delete_event(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._on_delete_event(Path(event.src_path))
            if hasattr(event, "dest_path"):
                self._on_relevant_event(Path(event.dest_path))


def start_watcher(
    input_dir: Path,
    watcher_cfg: WatcherConfig,
    converter_cfg: ConverterConfig,
    callback: Callable[[list[Path], list[str]], None],
) -> Observer:
    """Start the file system watcher. Returns the Observer (call .stop() to stop)."""
    supported = set(converter_cfg.supported_extensions) | {".pdf"}
    handler = DebouncedHandler(
        callback=callback,
        debounce_seconds=watcher_cfg.debounce_seconds,
        max_wait_seconds=watcher_cfg.max_wait_seconds,
        supported_extensions=supported,
    )

    observer = Observer()
    observer.schedule(handler, str(input_dir), recursive=watcher_cfg.watch_subdirectories)
    observer.start()
    logger.info("Watching %s for changes (debounce: %ds)", input_dir, watcher_cfg.debounce_seconds)
    return observer
