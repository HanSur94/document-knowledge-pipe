#!/usr/bin/env python3
"""Download Kaggle datasets for bulk document testing.

Prerequisites:
    pip install kaggle
    # Set up ~/.kaggle/kaggle.json with your API credentials
    # See: https://www.kaggle.com/docs/api

Usage:
    python scripts/download_kaggle_datasets.py [--output-dir ./data/kaggle]

Datasets downloaded:
    - manisha717/dataset-of-pdf-files (~806 MB) — diverse PDFs
    - manisha717/dataset-for-doc-and-docx (~571 MB) — Word documents
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASETS = [
    {
        "slug": "manisha717/dataset-of-pdf-files",
        "description": "Diverse PDF files (reports, articles, manuals) ~806 MB",
        "subdir": "pdf-files",
    },
    {
        "slug": "manisha717/dataset-for-doc-and-docx",
        "description": "DOC and DOCX files (various topics) ~571 MB",
        "subdir": "doc-docx",
    },
]


def check_kaggle_cli() -> bool:
    """Check if kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_dataset(slug: str, output_dir: Path) -> bool:
    """Download and unzip a Kaggle dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {slug}...")
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(output_dir),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False

    # Count files
    files = list(output_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    print(f"  Downloaded {file_count} files to {output_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Kaggle datasets for bulk document testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/kaggle"),
        help="Output directory (default: ./data/kaggle)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets without downloading",
    )
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for ds in DATASETS:
            print(f"  {ds['slug']} — {ds['description']}")
        return

    if not check_kaggle_cli():
        print("ERROR: kaggle CLI not found.")
        print("Install: pip install kaggle")
        print("Configure: https://www.kaggle.com/docs/api")
        sys.exit(1)

    print(f"Downloading {len(DATASETS)} datasets to {args.output_dir}")
    print()

    success = 0
    for ds in DATASETS:
        print(f"[{ds['subdir']}] {ds['description']}")
        target = args.output_dir / ds["subdir"]
        if download_dataset(ds["slug"], target):
            success += 1
        print()

    print(f"Done: {success}/{len(DATASETS)} datasets downloaded.")

    if success > 0:
        print()
        print("To test with these documents:")
        print("  docpipe ingest --config config.yaml")
        print(f"  # (set input_dir to {args.output_dir}/<dataset> in config)")


if __name__ == "__main__":
    main()
