"""Tests for docpipe query CLI command."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import yaml
from click.testing import CliRunner

from docpipe.cli import main


class TestQueryCommand:
    def test_query_prints_answer(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (tmp_path / "logs").mkdir()

        config = {
            "input_dir": str(tmp_path / "input"),
            "output_dir": str(output_dir),
            "graph": {
                "provider": "openai",
                "store_dir": str(output_dir / "lightrag_store"),
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
            },
            "logging": {
                "level": "DEBUG",
                "file": str(tmp_path / "logs" / "docpipe.log"),
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))

        runner = CliRunner()

        with patch("docpipe.cli.query_graph", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "The documents cover data science."
            result = runner.invoke(
                main, ["query", "--config", str(cfg_path), "What topics?"]
            )

        assert result.exit_code == 0
        assert "data science" in result.output

    def test_query_exits_1_on_failure(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (tmp_path / "logs").mkdir()

        config = {
            "input_dir": str(tmp_path / "input"),
            "output_dir": str(output_dir),
            "graph": {
                "provider": "openai",
                "store_dir": str(output_dir / "lightrag_store"),
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
            },
            "logging": {
                "level": "DEBUG",
                "file": str(tmp_path / "logs" / "docpipe.log"),
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))

        runner = CliRunner()

        with patch("docpipe.cli.query_graph", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = None
            result = runner.invoke(
                main, ["query", "--config", str(cfg_path), "What topics?"]
            )

        assert result.exit_code == 1

    def test_query_accepts_mode_flag(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (tmp_path / "logs").mkdir()

        config = {
            "input_dir": str(tmp_path / "input"),
            "output_dir": str(output_dir),
            "graph": {
                "provider": "openai",
                "store_dir": str(output_dir / "lightrag_store"),
                "openai": {
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dim": 1536,
                },
            },
            "logging": {
                "level": "DEBUG",
                "file": str(tmp_path / "logs" / "docpipe.log"),
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))

        runner = CliRunner()

        with patch("docpipe.cli.query_graph", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "answer"
            result = runner.invoke(
                main,
                ["query", "--config", str(cfg_path), "--mode", "local", "Question?"],
            )

        assert result.exit_code == 0
        mock_query.assert_called_once()
        call_args = mock_query.call_args
        assert call_args.args[2] == "local"
