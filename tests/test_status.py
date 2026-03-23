from __future__ import annotations

import json
from pathlib import Path

from docpipe.status import StatusTracker


class TestStatusTracker:
    def test_init_creates_status(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        data = tracker.to_dict()
        assert data["watcher"] == "stopped"
        assert data["files"] == {}

    def test_set_watcher_running(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.set_watcher_running()
        assert tracker.to_dict()["watcher"] == "running"

    def test_update_file_status(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.update_file("test.pdf", status="done", pages=5, images=2, graph_ingested=True)
        files = tracker.to_dict()["files"]
        assert "test.pdf" in files
        assert files["test.pdf"]["status"] == "done"
        assert files["test.pdf"]["pages"] == 5

    def test_save_and_load(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.update_file("test.pdf", status="done", pages=3, images=1, graph_ingested=True)
        tracker.save()

        status_file = tmp_dirs["output"] / "status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert "test.pdf" in data["files"]

    def test_heartbeat_updates(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.set_watcher_running()
        tracker.heartbeat()
        data = tracker.to_dict()
        assert "last_heartbeat_at" in data

    def test_track_api_usage(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.add_api_usage(tokens=500, cost=0.001)
        data = tracker.to_dict()
        assert data["api_usage"]["tokens_today"] == 500

    def test_remove_file(self, tmp_dirs: dict[str, Path]) -> None:
        tracker = StatusTracker(tmp_dirs["output"])
        tracker.update_file("test.pdf", status="done", pages=1, images=0, graph_ingested=True)
        tracker.remove_file("test.pdf")
        assert "test.pdf" not in tracker.to_dict()["files"]
