from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from docpipe.config import DocpipeConfig, load_config


class TestLoadConfig:
    def test_load_from_yaml_file(self, config_file: Path, sample_config: dict[str, Any]) -> None:
        cfg = load_config(config_file)
        assert cfg.input_dir == Path(sample_config["input_dir"])
        assert cfg.output_dir == Path(sample_config["output_dir"])
        assert cfg.watcher.debounce_seconds == 1
        assert cfg.describer.model == "gpt-4o-mini"

    def test_load_with_defaults(self, tmp_path: Path) -> None:
        """Minimal YAML should fill defaults for all missing fields."""
        minimal = {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(minimal))
        cfg = load_config(cfg_path)
        assert cfg.watcher.debounce_seconds == 60
        assert cfg.graph.embedding_model == "text-embedding-3-small"
        assert cfg.api_retry.max_retries == 3

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_env_override_for_api_key(
        self, config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        cfg = load_config(config_file)
        assert cfg.openai_api_key == "sk-test-key-123"
