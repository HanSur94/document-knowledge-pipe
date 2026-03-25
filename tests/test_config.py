from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from docpipe.config import load_config


class TestLoadConfig:
    def test_load_from_yaml_file(self, config_file: Path, sample_config: dict[str, Any]) -> None:
        cfg = load_config(config_file)
        assert cfg.input_dir == Path(sample_config["input_dir"])
        assert cfg.output_dir == Path(sample_config["output_dir"])
        assert cfg.watcher.debounce_seconds == 1
        assert cfg.describer.openai.model == "gpt-4o-mini"

    def test_load_with_defaults(self, tmp_path: Path) -> None:
        """Minimal YAML should fill defaults for all missing fields."""
        minimal = {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(minimal))
        cfg = load_config(cfg_path)
        assert cfg.watcher.debounce_seconds == 60
        assert cfg.graph.openai.embedding_model == "text-embedding-3-small"
        assert cfg.api_retry.max_retries == 3

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestNestedProviderConfig:
    def test_loads_nested_describer_openai_config(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "describer": {
                "provider": "openai",
                "max_tokens": 300,
                "openai": {"model": "gpt-4o-mini"},
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        cfg = load_config(cfg_path)
        assert cfg.describer.provider == "openai"
        assert cfg.describer.openai.model == "gpt-4o-mini"

    def test_loads_nested_describer_azure_config(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "describer": {
                "provider": "azure",
                "max_tokens": 300,
                "azure": {
                    "model": "gpt-4o-mini",
                    "endpoint": "https://test.openai.azure.com",
                    "deployment": "my-deploy",
                    "api_version": "2024-06-01",
                },
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        cfg = load_config(cfg_path)
        assert cfg.describer.azure.endpoint == "https://test.openai.azure.com"
        assert cfg.describer.azure.deployment == "my-deploy"

    def test_loads_nested_graph_config(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "graph": {
                "provider": "openai",
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        cfg = load_config(cfg_path)
        assert cfg.graph.openai.embedding_model == "text-embedding-3-small"
        assert cfg.graph.openai.embedding_dim == 1536

    def test_defaults_work_without_nested_blocks(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        cfg = load_config(cfg_path)
        assert cfg.describer.openai.model == "gpt-4o-mini"
        assert cfg.graph.openai.model == "gpt-4o-mini"


class TestConfigValidation:
    def test_azure_describer_missing_endpoint_raises(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "describer": {
                "provider": "azure",
                "azure": {"model": "gpt-4o-mini"},
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        with pytest.raises(ValueError, match="endpoint"):
            load_config(cfg_path)

    def test_graph_anthropic_raises(self, tmp_path: Path) -> None:
        config_data = {
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "graph": {"provider": "anthropic"},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))
        with pytest.raises(ValueError, match="[Aa]nthropic.*not supported"):
            load_config(cfg_path)
