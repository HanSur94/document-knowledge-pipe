"""Build and update the AI-readable registry.md index."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from docpipe.config import RegistryConfig
from docpipe.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_HEADER = """# Document Registry

| File | Author | Date | Summary | Topics | Pages | Images |
|------|--------|------|---------|--------|-------|--------|
"""


@dataclass
class RegistryEntry:
    filename: str
    author: str
    date: str
    summary: str
    topics: str
    pages: int
    images: int

    def to_table_row(self) -> str:
        return (
            f"| {self.filename} | {self.author} | {self.date} "
            f"| {self.summary} | {self.topics} | {self.pages} | {self.images} |"
        )

    @classmethod
    def failed(cls, filename: str, reason: str) -> RegistryEntry:
        return cls(
            filename=filename,
            author="-",
            date="-",
            summary=f"Processing failed: {reason}",
            topics="-",
            pages=0,
            images=0,
        )


def update_registry(
    registry_path: Path,
    entry: RegistryEntry,
    remove: bool = False,
) -> None:
    """Add, update, or remove an entry in registry.md."""
    entries: list[str] = []
    if registry_path.exists():
        lines = registry_path.read_text().splitlines()
        for line in lines:
            if line.startswith("| ") and not line.startswith("| File") and "---" not in line:
                entries.append(line)

    entries = [e for e in entries if f"| {entry.filename} |" not in e]

    if not remove:
        entries.append(entry.to_table_row())

    entries.sort(key=lambda e: e.split("|")[1].strip())

    content = _HEADER + "\n".join(entries) + "\n"
    registry_path.write_text(content)
    logger.info("Registry updated: %s (%s)", entry.filename, "removed" if remove else "updated")


async def generate_summary(
    markdown: str,
    cfg: RegistryConfig,
    provider: LLMProvider,
) -> tuple[str, str]:
    """Use an LLM to generate a summary and topic tags."""
    prompt = (
        f"Summarize this document in at most {cfg.summary_max_words} words. "
        "Then list 2-5 topic tags (comma-separated). "
        "Respond in exactly this format:\n"
        "SUMMARY: <your summary>\n"
        "TOPICS: <tag1, tag2, tag3>\n\n"
        f"Document:\n{markdown[:3000]}"
    )

    try:
        content = await provider.complete(prompt, max_tokens=200)
    except Exception:
        logger.error("Summary generation failed")
        return "Summary unavailable", "-"

    summary = ""
    topics = ""
    for line in content.splitlines():
        if line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
        elif line.startswith("TOPICS:"):
            topics = line.replace("TOPICS:", "").strip()
    return summary or "No summary available", topics or "-"


def build_registry(output_dir: Path) -> str:
    """Read the current registry and return it as a string."""
    reg_path = output_dir / "registry.md"
    if reg_path.exists():
        return reg_path.read_text()
    return _HEADER
