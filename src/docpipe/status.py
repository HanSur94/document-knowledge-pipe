"""Status tracking, status.json management, and heartbeat."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StatusTracker:
    """Track pipeline status and persist to status.json."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._status_file = output_dir / "status.json"
        self._watcher_state = "stopped"
        self._started_at: str | None = None
        self._last_heartbeat: str | None = None
        self._files: dict[str, dict[str, Any]] = {}
        self._graph: dict[str, int] = {"entities": 0, "relations": 0, "documents": 0}
        self._api_usage: dict[str, Any] = {
            "tokens_today": 0,
            "estimated_cost_usd": 0.0,
            "day_reset": str(date.today()),
            "tokens_total": 0,
        }
        self._load()

    def _load(self) -> None:
        """Load existing status from disk."""
        if self._status_file.exists():
            try:
                data = json.loads(self._status_file.read_text())
                self._watcher_state = data.get("watcher", "stopped")
                self._started_at = data.get("started_at")
                self._last_heartbeat = data.get("last_heartbeat_at")
                self._files = data.get("files", {})
                self._graph = data.get("graph", self._graph)
                self._api_usage = data.get("api_usage", self._api_usage)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupted status.json, starting fresh")

    def set_watcher_running(self) -> None:
        self._watcher_state = "running"
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self.heartbeat()

    def set_watcher_stopped(self) -> None:
        self._watcher_state = "stopped"

    def heartbeat(self) -> None:
        self._last_heartbeat = datetime.now().isoformat(timespec="seconds")

    def update_file(
        self,
        filename: str,
        status: str,
        pages: int = 0,
        images: int = 0,
        graph_ingested: bool = False,
        md_path: str = "",
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "status": status,
            "pages": pages,
            "images": images,
            "graph_ingested": graph_ingested,
            "last_processed": datetime.now().isoformat(timespec="seconds"),
        }
        if md_path:
            entry["md_path"] = md_path
        if error:
            entry["error"] = error
        self._files[filename] = entry

    def remove_file(self, filename: str) -> None:
        self._files.pop(filename, None)

    def update_graph_stats(self, entities: int, relations: int, documents: int) -> None:
        self._graph = {
            "entities": entities,
            "relations": relations,
            "documents": documents,
        }

    def add_api_usage(self, tokens: int, cost: float) -> None:
        today = str(date.today())
        if self._api_usage.get("day_reset") != today:
            self._api_usage["tokens_today"] = 0
            self._api_usage["estimated_cost_usd"] = 0.0
            self._api_usage["day_reset"] = today
        self._api_usage["tokens_today"] += tokens
        self._api_usage["estimated_cost_usd"] += cost
        self._api_usage["tokens_total"] += tokens

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "watcher": self._watcher_state,
            "files": self._files,
            "graph": self._graph,
            "api_usage": self._api_usage,
        }
        if self._started_at:
            result["started_at"] = self._started_at
        if self._last_heartbeat:
            result["last_heartbeat_at"] = self._last_heartbeat
        return result

    def save(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._status_file.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
