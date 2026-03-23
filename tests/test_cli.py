from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from docpipe.cli import main


class TestCLI:
    def test_init_creates_structure(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert Path("config.yaml").exists()
            assert Path("input").exists()
            assert Path("output").exists()

    def test_status_without_data(self, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["status", "--config", str(config_file)])
        assert result.exit_code == 0

    def test_ingest_with_empty_input(self, config_file: Path) -> None:
        """CLI ingest with empty input dir processes zero files and exits 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["ingest", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "Processing 0 file(s)" in result.output
