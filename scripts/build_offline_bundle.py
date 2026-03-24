#!/usr/bin/env python3
"""
Build an offline Windows 10 installer bundle for docpipe.

Runs on any OS with Python 3.11+ and internet.
Downloads portable runtimes, pre-fetches all dependencies,
and packages everything into a self-contained zip.

Usage:
    python scripts/build_offline_bundle.py                     # full build
    python scripts/build_offline_bundle.py --version v1.0.0    # tag the bundle
    python scripts/build_offline_bundle.py --dry-run            # create dirs only
    python scripts/build_offline_bundle.py --skip-download      # use cached downloads
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
if sys.stdout.isatty():
    G, R, Y, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[0m"
else:
    G = R = Y = B = X = ""


def ok(m: str) -> None:
    print(f"  {G}OK{X}   {m}")


def fail(m: str) -> None:
    print(f"  {R}FAIL{X} {m}")


def info(m: str) -> None:
    print(f"  {B}...{X}  {m}")


def warn(m: str) -> None:
    print(f"  {Y}WARN{X} {m}")


def hdr(m: str) -> None:
    print(f"\n{B}{'=' * 60}{X}")
    print(f"{B}  {m}{X}")
    print(f"{B}{'=' * 60}{X}\n")


# ---------------------------------------------------------------------------
# Download URLs and checksums
# ---------------------------------------------------------------------------
PYTHON_VERSION = "3.11.9"
PYTHON_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
)
PYTHON_SHA256 = ""  # Set to known-good hash from python.org when pinning

GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
GET_PIP_SHA256 = ""  # Set to known-good hash from bootstrap.pypa.io when pinning

LIBREOFFICE_VERSION = "25.8.5"
LIBREOFFICE_URL = (
    f"https://download.documentfoundation.org/libreoffice/stable/{LIBREOFFICE_VERSION}/"
    f"win/x86_64/LibreOffice_{LIBREOFFICE_VERSION}_Win_x86-64.msi"
)
LIBREOFFICE_SHA256 = ""  # Set to known-good hash when pinning


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def create_staging(output_dir: Path) -> Path:
    """Create the staging directory structure."""
    staging = output_dir / "docpipe-win64"
    for d in [
        staging / "runtime" / "python",
        staging / "runtime" / "libreoffice",
        staging / "vendor" / "pip-wheels",
        staging / "src" / "docpipe",
    ]:
        d.mkdir(parents=True, exist_ok=True)
    return staging


def download_file(url: str, dest: Path, expected_sha256: str = "") -> bool:
    """Download a file with optional SHA256 verification."""
    info(f"Downloading {url.split('/')[-1]} ...")
    try:
        resp = urllib.request.urlopen(url, timeout=300)
        data = resp.read()
    except Exception as e:
        fail(f"Download failed: {e}")
        return False

    if expected_sha256:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            fail(f"SHA256 mismatch: expected {expected_sha256}, got {actual}")
            return False
        ok(f"SHA256 verified: {actual[:16]}...")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    ok(f"Downloaded {dest.name} ({len(data) / 1024 / 1024:.1f} MB)")
    return True


def extract_zip(zip_path: Path, dest: Path, strip_top: bool = True) -> bool:
    """Extract a zip, optionally stripping the top-level directory."""
    info(f"Extracting {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if strip_top:
                    parts = member.split("/")
                    rel = "/".join(parts[1:])
                    if not rel:
                        continue
                else:
                    rel = member

                out = (dest / rel).resolve()
                # Zip Slip guard
                if not out.is_relative_to(dest.resolve()):
                    warn(f"Skipping suspicious path: {member}")
                    continue

                if member.endswith("/"):
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src:
                        out.write_bytes(src.read())
        ok(f"Extracted to {dest}")
        return True
    except Exception as e:
        fail(f"Extraction failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------
def generate_windows_requirements(staging: Path) -> Path:
    """Read deps from pyproject.toml and write requirements-windows.txt."""
    pyproject_path = ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        fail("pyproject.toml not found")
        return staging / "requirements-windows.txt"

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    deps = pyproject.get("project", {}).get("dependencies", [])

    lines: list[str] = []
    seen: set[str] = set()
    for dep in deps:
        pkg_name = dep.split(">=")[0].split("==")[0].split("[")[0].strip()
        if pkg_name.lower() not in seen:
            seen.add(pkg_name.lower())
            lines.append(dep)

    out = staging / "requirements-windows.txt"
    out.write_text("\n".join(lines) + "\n")
    ok(f"Generated {out.name} ({len(lines)} packages)")
    return out


def _ensure_uv() -> str:
    """Ensure uv is available, install if needed. Return path to uv."""
    uv = shutil.which("uv")
    if uv:
        return uv
    info("Installing uv ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "uv"],
        capture_output=True,
        check=True,
    )
    uv = shutil.which("uv")
    if not uv:
        uv = str(Path(sys.executable).parent / "uv")
    return uv


def download_pip_wheels(staging: Path, req_file: Path) -> bool:
    """Download pip wheels for Windows x64 / CPython 3.11.

    Uses uv to resolve all dependencies (including transitive) for Windows,
    then pip to download the exact pinned versions as wheels.
    """
    wheels_dir = staging / "vendor" / "pip-wheels"
    uv = _ensure_uv()

    # Step 1: Use uv to resolve all deps for Windows into a pinned file
    resolved_file = staging / "requirements-resolved.txt"
    info("Resolving all dependencies for Windows (via uv pip compile) ...")
    r = subprocess.run(
        [
            uv,
            "pip",
            "compile",
            str(req_file),
            "--python-platform",
            "windows",
            "--python-version",
            "3.11",
            "-o",
            str(resolved_file),
            "--no-header",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        fail(f"uv pip compile failed:\n{r.stderr[-1000:]}")
        return False

    resolved_count = sum(
        1
        for line in resolved_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    ok(f"Resolved {resolved_count} packages (including transitive deps)")

    # Step 2: Download all resolved packages as wheels for win_amd64
    # Bootstrap first: pip/setuptools/wheel
    info("Downloading pip/setuptools/wheel ...")
    pip_download_args = [
        "--platform",
        "win_amd64",
        "--python-version",
        "3.11",
        "--implementation",
        "cp",
        "--only-binary=:all:",
    ]
    for pkg in ["pip", "setuptools", "wheel"]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                pkg,
                *pip_download_args,
                "-d",
                str(wheels_dir),
            ],
            capture_output=True,
        )

    # Download all resolved packages (fully pinned, no resolution needed)
    info(f"Downloading {resolved_count} resolved packages ...")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "-r",
            str(resolved_file),
            *pip_download_args,
            "-d",
            str(wheels_dir),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        warn(f"pip download had errors (trying fallback):\n{r.stderr[-500:]}")
        # Fallback: download individually, skip failures
        for line in resolved_file.read_text().splitlines():
            pkg = line.strip()
            if not pkg or pkg.startswith("#"):
                continue
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    pkg,
                    "--no-deps",
                    *pip_download_args,
                    "-d",
                    str(wheels_dir),
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                # Try without platform constraint (pure-Python)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "download",
                        pkg,
                        "--no-deps",
                        "-d",
                        str(wheels_dir),
                    ],
                    capture_output=True,
                )

    wheel_count = len(list(wheels_dir.glob("*.whl"))) + len(list(wheels_dir.glob("*.tar.gz")))
    if wheel_count == 0:
        fail("No wheels downloaded")
        return False

    if wheel_count < 10:
        warn(f"Only {wheel_count} packages downloaded — some may be missing")

    ok(f"Downloaded {wheel_count} packages to vendor/pip-wheels/")
    return True


# ---------------------------------------------------------------------------
# LibreOffice extraction
# ---------------------------------------------------------------------------
def download_libreoffice(staging: Path, cache_dir: Path, skip_download: bool = False) -> bool:
    """Download and extract LibreOffice portable from MSI via 7z."""
    msi_path = cache_dir / f"LibreOffice_{LIBREOFFICE_VERSION}_Win_x86-64.msi"
    lo_dest = staging / "runtime" / "libreoffice"

    # Download MSI
    if not skip_download or not msi_path.exists():
        if not download_file(LIBREOFFICE_URL, msi_path, LIBREOFFICE_SHA256):
            return False
    else:
        ok(f"Using cached {msi_path.name}")

    # Check for 7z
    sevenz = shutil.which("7z") or shutil.which("7za")
    if not sevenz:
        warn(
            "7z not found — cannot extract LibreOffice MSI. "
            "Install p7zip-full (apt) or 7zip (brew) and re-run."
        )
        return False

    # Extract MSI to a temp directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        info(f"Extracting LibreOffice MSI with 7z ({msi_path.name}) ...")
        r = subprocess.run(
            [sevenz, "x", str(msi_path), f"-o{tmp_dir}"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            fail(f"7z extraction failed:\n{r.stderr[-500:]}")
            return False

        # Find soffice.exe in extracted tree
        soffice_files = list(tmp_dir.rglob("soffice.exe"))
        if not soffice_files:
            # Try soffice.bin as alternative marker
            soffice_files = list(tmp_dir.rglob("soffice.bin"))

        if not soffice_files:
            fail("Could not find soffice.exe in extracted MSI")
            return False

        # The program/ dir containing soffice.exe
        program_dir = soffice_files[0].parent
        info(f"Found LibreOffice program dir: {program_dir.name}")

        # Copy program/ directory to runtime/libreoffice/program/
        dest_program = lo_dest / "program"
        if dest_program.exists():
            shutil.rmtree(dest_program)
        shutil.copytree(program_dir, dest_program, dirs_exist_ok=True)

    ok("LibreOffice extracted to runtime/libreoffice/")
    return True


# ---------------------------------------------------------------------------
# Source copy and launcher scripts
# ---------------------------------------------------------------------------
def copy_application_source(staging: Path) -> bool:
    """Copy application source and config files."""
    info("Copying application source ...")

    # Copy src/docpipe/
    shutil.copytree(
        ROOT / "src" / "docpipe",
        staging / "src" / "docpipe",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
        ),
    )

    # Copy config.yaml
    config_src = ROOT / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, staging / "config.yaml")

    # Create .env.example
    env_example = staging / ".env.example"
    env_example.write_text(
        "# DocPipe Environment Configuration\n"
        "# Copy this file to .env and fill in your API keys.\n"
        "\n"
        "OPENAI_API_KEY=your-openai-api-key-here\n"
        "ANTHROPIC_API_KEY=your-anthropic-api-key-here\n"
    )

    ok("Application source copied")
    return True


def generate_install_bat(staging: Path) -> None:
    """Generate the offline install.bat."""
    content = r"""@echo off
setlocal enabledelayedexpansion
set ROOT=%~dp0
set ERRORS=0

echo ========================================================
echo   DocPipe — Offline Installer
echo ========================================================
echo.
echo   Install directory: %ROOT%
echo.

REM -------------------------------------------------------
REM Step 1: Bootstrap pip into embeddable Python
REM -------------------------------------------------------
echo [Step 1/4] Setting up Python...

set PYTHON=%ROOT%runtime\python\python.exe
if not exist "%PYTHON%" (
    echo [ERROR] Portable Python not found at %PYTHON%
    exit /b 1
)

REM Check if pip is already installed
"%PYTHON%" -m pip --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   pip already installed, skipping bootstrap.
    goto :step2
)

set WHEELS=%ROOT%vendor\pip-wheels
"%PYTHON%" "%ROOT%runtime\python\get-pip.py" ^
    --no-index --find-links "%WHEELS%" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to bootstrap pip
    exit /b 1
)
echo   [OK] pip installed.

:step2
REM -------------------------------------------------------
REM Step 2: Install Python packages offline
REM -------------------------------------------------------
echo [Step 2/4] Installing Python packages (offline)...

set REQS=%ROOT%requirements-windows.txt
"%PYTHON%" -m pip install --no-index ^
    --find-links "%WHEELS%" -r "%REQS%" --quiet
if %ERRORLEVEL% neq 0 (
    echo [WARN] Some packages may have failed. Retrying...
    "%PYTHON%" -m pip install --no-index ^
        --find-links "%WHEELS%" -r "%REQS%"
    if !ERRORLEVEL! neq 0 (
        echo   [FAIL] Python package installation failed.
        set /a ERRORS+=1
        goto :step3
    )
)
echo   [OK] Python packages installed.

:step3
REM -------------------------------------------------------
REM Step 3: Configure environment
REM -------------------------------------------------------
echo [Step 3/4] Configuring environment...

if not exist "%ROOT%.env" (
    copy "%ROOT%.env.example" "%ROOT%.env" >nul
    echo   [OK] Created .env from template.
) else (
    echo   .env already exists, preserving.
)

if not exist "%ROOT%input\" mkdir "%ROOT%input"
if not exist "%ROOT%output\" mkdir "%ROOT%output"
if not exist "%ROOT%logs\" mkdir "%ROOT%logs"
echo   [OK] Directories created.

REM -------------------------------------------------------
REM Step 4: Verification
REM -------------------------------------------------------
echo [Step 4/4] Verifying installation...
echo.

set PYTHONPATH=%ROOT%src
"%PYTHON%" --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "delims=" %%v in ('"%PYTHON%" --version') do echo   [OK] %%v
) else (
    echo   [FAIL] Python not working
    set /a ERRORS+=1
)

"%PYTHON%" -c "import docpipe; print('  [OK] docpipe package')" 2>nul
if %ERRORLEVEL% neq 0 ( echo   [FAIL] docpipe import & set /a ERRORS+=1 )

"%PYTHON%" -c "import anthropic; print(f'  [OK] anthropic {anthropic.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 ( echo   [FAIL] anthropic & set /a ERRORS+=1 )

if exist "%ROOT%runtime\libreoffice\program\soffice.exe" (
    echo   [OK] LibreOffice portable
) else (
    echo   [WARN] LibreOffice not found (document conversion will be limited)
)

echo.
if %ERRORS% equ 0 (
    echo ========================================================
    echo   Installation successful!
    echo ========================================================
    echo.
    echo   IMPORTANT: Edit .env and add your API keys:
    echo     - OPENAI_API_KEY (required for LLM + embeddings)
    echo     - ANTHROPIC_API_KEY (optional, for Claude provider)
    echo.
    echo   Then run docpipe.bat to launch the pipeline.
    echo.
) else (
    echo ========================================================
    echo   Installation completed with %ERRORS% error(s).
    echo ========================================================
    echo   Please review the output above.
)

pause
"""
    (staging / "install.bat").write_text(content.replace("\n", "\r\n"), encoding="utf-8")
    ok("Generated install.bat")


def generate_docpipe_bat(staging: Path) -> None:
    """Generate the docpipe.bat CLI wrapper."""
    content = r"""@echo off
setlocal enabledelayedexpansion
set ROOT=%~dp0
set PYTHON=%ROOT%runtime\python\python.exe
set PYTHONPATH=%ROOT%src;%PYTHONPATH%
set PATH=%ROOT%runtime\libreoffice\program;%PATH%

REM Load .env variables
if exist "%ROOT%.env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%ROOT%.env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%a"=="" set "%%a=%%b"
        )
    )
)

if not exist "%PYTHON%" (
    echo [ERROR] Not installed. Run install.bat first.
    pause
    exit /b 1
)

"%PYTHON%" -m docpipe.cli %*
"""
    (staging / "docpipe.bat").write_text(content.replace("\n", "\r\n"), encoding="utf-8")
    ok("Generated docpipe.bat")


def generate_readme(staging: Path, version: str) -> None:
    """Generate a quick-start README.txt."""
    content = f"""DocPipe {version} — Offline Windows Bundle
{"=" * 50}

Document ingestion pipeline: folder -> markdown -> knowledge graph

Quick Start
-----------
1. Extract this zip to a folder (e.g. C:\\docpipe)
2. Run install.bat (one-time setup, no internet needed)
3. Edit .env and add your API keys:
   - OPENAI_API_KEY (required)
   - ANTHROPIC_API_KEY (optional)
4. Run docpipe.bat to use the pipeline

Usage Examples
--------------
  docpipe.bat run              Process documents in input/ folder
  docpipe.bat watch            Watch input/ for new documents
  docpipe.bat --help           Show all commands

Bundle Contents
---------------
  runtime/python/       Python {PYTHON_VERSION} embeddable
  runtime/libreoffice/  LibreOffice portable (document conversion)
  vendor/pip-wheels/    Pre-fetched Python packages
  src/docpipe/          Application source code
  config.yaml           Pipeline configuration
  install.bat           One-time offline installer
  docpipe.bat           CLI launcher

System Requirements
-------------------
  - Windows 10/11 x64
  - No internet connection needed after extraction
  - ~500 MB disk space

Troubleshooting
---------------
  - If install.bat fails, check that you extracted to a path
    without spaces or special characters.
  - If docpipe.bat fails, ensure you ran install.bat first
    and that .env contains valid API keys.
  - LibreOffice is used for .doc/.docx/.ppt/.xls conversion.
    PDF files work without LibreOffice.
"""
    (staging / "README.txt").write_text(content, encoding="utf-8")
    ok("Generated README.txt")


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------
def build(args: argparse.Namespace) -> int:
    """Run the full build pipeline."""
    output_dir = Path(args.output).resolve()
    version = args.version
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create staging directory
    hdr("Step 1/6: Create staging directory")
    staging = create_staging(output_dir)
    ok(f"Staging at {staging}")

    if args.dry_run:
        info("Dry-run mode — skipping downloads and build steps")
        generate_install_bat(staging)
        generate_docpipe_bat(staging)
        generate_readme(staging, version)
        ok("Dry-run complete")
        return 0

    # Step 2: Download + extract Python embeddable, patch ._pth, get-pip.py
    hdr("Step 2/6: Python embeddable")
    python_dir = staging / "runtime" / "python"
    python_zip = cache_dir / f"python-{PYTHON_VERSION}-embed-amd64.zip"

    if not args.skip_download or not python_zip.exists():
        if not download_file(PYTHON_URL, python_zip, PYTHON_SHA256):
            return 1
    else:
        ok(f"Using cached {python_zip.name}")

    if not extract_zip(python_zip, python_dir, strip_top=False):
        return 1

    # Patch python311._pth to uncomment 'import site'
    pth_file = python_dir / f"python{PYTHON_VERSION.replace('.', '')[:3]}._pth"
    if not pth_file.exists():
        # Try common pattern
        pth_file = python_dir / "python311._pth"
    if pth_file.exists():
        text = pth_file.read_text()
        text = text.replace("#import site", "import site")
        pth_file.write_text(text)
        ok("Patched python311._pth (uncommented import site)")
    else:
        warn("python311._pth not found — pip may not work correctly")

    # Download get-pip.py
    get_pip_dest = python_dir / "get-pip.py"
    if not args.skip_download or not get_pip_dest.exists():
        if not download_file(GET_PIP_URL, get_pip_dest, GET_PIP_SHA256):
            return 1
    else:
        ok(f"Using cached {get_pip_dest.name}")

    # Step 3: Download + extract LibreOffice
    hdr("Step 3/6: LibreOffice portable")
    if not download_libreoffice(staging, cache_dir, args.skip_download):
        warn("LibreOffice extraction failed — bundle will work without it")

    # Step 4: Generate requirements + download wheels
    hdr("Step 4/6: Python dependencies")
    req_file = generate_windows_requirements(staging)
    if not download_pip_wheels(staging, req_file):
        return 1

    # Step 5: Copy source + generate launchers
    hdr("Step 5/6: Application source and launchers")
    if not copy_application_source(staging):
        return 1
    generate_install_bat(staging)
    generate_docpipe_bat(staging)
    generate_readme(staging, version)

    # Step 6: Create zip
    hdr("Step 6/6: Create distributable zip")
    zip_name = f"docpipe-{version}-win64"
    zip_path = output_dir / f"{zip_name}.zip"
    info(f"Creating {zip_path.name} ...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(staging.rglob("*")):
            if file.is_file():
                arcname = f"docpipe-win64/{file.relative_to(staging)}"
                zf.write(file, arcname)

    size_mb = zip_path.stat().st_size / 1024 / 1024
    ok(f"Created {zip_path.name} ({size_mb:.1f} MB)")

    hdr("Build complete!")
    info(f"Bundle: {zip_path}")
    info("To test: extract zip and run install.bat on Windows")
    return 0


def main() -> None:
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Build an offline Windows bundle for docpipe",
    )
    parser.add_argument(
        "--version",
        default="dev",
        help="Version string for the bundle (default: dev)",
    )
    parser.add_argument(
        "--output",
        default=str(DIST_DIR),
        help=f"Output directory (default: {DIST_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create directory structure only, skip downloads",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use cached downloads if available",
    )
    args = parser.parse_args()

    hdr("DocPipe Offline Bundle Builder")
    info(f"Version: {args.version}")
    info(f"Output:  {args.output}")
    info(f"Dry-run: {args.dry_run}")

    rc = build(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
