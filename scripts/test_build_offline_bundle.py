#!/usr/bin/env python3
"""Unit tests for build_offline_bundle.py (non-network functions only)."""

import sys
from pathlib import Path

# Ensure scripts/ is on the path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_offline_bundle import (
    copy_application_source,
    create_staging,
    generate_docpipe_bat,
    generate_install_bat,
    generate_readme,
    generate_windows_requirements,
)


class TestCreateStaging:
    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        assert staging == tmp_path / "docpipe-win64"
        assert staging.is_dir()
        assert (staging / "runtime" / "python").is_dir()
        assert (staging / "runtime" / "libreoffice").is_dir()
        assert (staging / "vendor" / "pip-wheels").is_dir()
        assert (staging / "src" / "docpipe").is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        """Calling create_staging twice should not raise."""
        create_staging(tmp_path)
        staging = create_staging(tmp_path)
        assert staging.is_dir()


class TestGenerateWindowsRequirements:
    def test_reads_deps_from_pyproject(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        req_file = generate_windows_requirements(staging)
        assert req_file.exists()
        content = req_file.read_text()
        # Check that key deps from pyproject.toml are present
        assert "click" in content
        assert "pyyaml" in content or "PyYAML" in content
        assert "anthropic" in content
        assert "openai" in content
        assert "rich" in content

    def test_no_duplicates(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        req_file = generate_windows_requirements(staging)
        lines = [line.strip() for line in req_file.read_text().splitlines() if line.strip()]
        pkg_names = [
            line.split(">=")[0].split("==")[0].split("[")[0].strip().lower() for line in lines
        ]
        assert len(pkg_names) == len(set(pkg_names))

    def test_output_path(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        req_file = generate_windows_requirements(staging)
        assert req_file.name == "requirements-windows.txt"
        assert req_file.parent == staging


class TestGenerateInstallBat:
    def test_creates_bat_file(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_install_bat(staging)
        bat = staging / "install.bat"
        assert bat.exists()

    def test_contains_key_strings(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_install_bat(staging)
        content = (staging / "install.bat").read_text()
        assert "DocPipe" in content
        assert "install" in content.lower()
        assert "pip" in content
        assert "PYTHON" in content
        assert "get-pip.py" in content
        assert "requirements-windows.txt" in content
        assert "vendor\\pip-wheels" in content or "vendor\\\\pip-wheels" in content
        assert ".env" in content
        assert "docpipe" in content.lower()
        assert "anthropic" in content

    def test_crlf_line_endings(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_install_bat(staging)
        raw = (staging / "install.bat").read_bytes()
        assert b"\r\n" in raw


class TestGenerateDocpipeBat:
    def test_creates_bat_file(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_docpipe_bat(staging)
        bat = staging / "docpipe.bat"
        assert bat.exists()

    def test_contains_key_strings(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_docpipe_bat(staging)
        content = (staging / "docpipe.bat").read_text()
        assert "PYTHON" in content
        assert "PYTHONPATH" in content
        assert "docpipe.cli" in content
        assert "libreoffice" in content.lower()
        assert ".env" in content

    def test_crlf_line_endings(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_docpipe_bat(staging)
        raw = (staging / "docpipe.bat").read_bytes()
        assert b"\r\n" in raw

    def test_loads_env_vars(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_docpipe_bat(staging)
        content = (staging / "docpipe.bat").read_text()
        # Should load .env file
        assert "for /f" in content.lower()
        assert ".env" in content


class TestGenerateReadme:
    def test_creates_readme(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_readme(staging, "v1.0.0")
        readme = staging / "README.txt"
        assert readme.exists()

    def test_contains_version(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_readme(staging, "v2.3.4")
        content = (staging / "README.txt").read_text()
        assert "v2.3.4" in content

    def test_contains_key_sections(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        generate_readme(staging, "dev")
        content = (staging / "README.txt").read_text()
        assert "Quick Start" in content
        assert "install.bat" in content
        assert "docpipe.bat" in content
        assert "OPENAI_API_KEY" in content
        assert "ANTHROPIC_API_KEY" in content
        assert "LibreOffice" in content


class TestCopyApplicationSource:
    def test_copies_source_files(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        result = copy_application_source(staging)
        assert result is True
        # Check that docpipe source was copied
        assert (staging / "src" / "docpipe").is_dir()
        assert (staging / "src" / "docpipe" / "__init__.py").exists()
        assert (staging / "src" / "docpipe" / "cli.py").exists()

    def test_copies_config(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        copy_application_source(staging)
        assert (staging / "config.yaml").exists()

    def test_creates_env_example(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        copy_application_source(staging)
        env_example = staging / ".env.example"
        assert env_example.exists()
        content = env_example.read_text()
        assert "OPENAI_API_KEY" in content
        assert "ANTHROPIC_API_KEY" in content

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        staging = create_staging(tmp_path)
        copy_application_source(staging)
        pycache_dirs = list((staging / "src").rglob("__pycache__"))
        assert len(pycache_dirs) == 0
