"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class WatcherConfig:
    enabled: bool = True
    debounce_seconds: int = 60
    max_wait_seconds: int = 300
    poll_interval_seconds: int = 1
    watch_subdirectories: bool = False


@dataclass
class ConverterConfig:
    libreoffice_path: str | None = None
    timeout_seconds: int = 120
    supported_extensions: list[str] = field(
        default_factory=lambda: [
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".odt",
            ".ods",
            ".odp",
            ".rtf",
            ".html",
            ".epub",
        ]
    )


@dataclass
class ExtractorConfig:
    write_images: bool = True
    image_format: str = "png"
    dpi: int = 150


@dataclass
class DescriberConfig:
    model: str = "gpt-4o-mini"
    max_tokens: int = 300
    include_context: bool = True
    context_chars: int = 500
    batch_size: int = 5


@dataclass
class RegistryConfig:
    filename: str = "registry.md"
    summary_max_words: int = 30
    include_fields: list[str] = field(
        default_factory=lambda: [
            "filename",
            "author",
            "date",
            "summary",
            "topics",
            "pages",
            "images",
        ]
    )


@dataclass
class GraphConfig:
    storage: str = "file"
    store_dir: str = "./output/lightrag_store"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 4096
    chunk_size: int = 1200
    chunk_overlap: int = 100


@dataclass
class ApiRetryConfig:
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "./logs/docpipe.log"
    max_size_mb: int = 50
    backup_count: int = 3


@dataclass
class DocpipeConfig:
    input_dir: Path
    output_dir: Path
    openai_api_key: str = ""
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    converter: ConverterConfig = field(default_factory=ConverterConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    describer: DescriberConfig = field(default_factory=DescriberConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    api_retry: ApiRetryConfig = field(default_factory=ApiRetryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    """Create a dataclass instance, using defaults for missing keys."""
    if data is None:
        return cls()
    import dataclasses

    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def load_config(path: Path) -> DocpipeConfig:
    """Load config from YAML file, merge with defaults, load .env."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    env_path = path.parent / ".env"
    load_dotenv(env_path)

    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

    return DocpipeConfig(
        input_dir=Path(raw.get("input_dir", "./input")),
        output_dir=Path(raw.get("output_dir", "./output")),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        watcher=_merge_dataclass(WatcherConfig, raw.get("watcher")),
        converter=_merge_dataclass(ConverterConfig, raw.get("converter")),
        extractor=_merge_dataclass(ExtractorConfig, raw.get("extractor")),
        describer=_merge_dataclass(DescriberConfig, raw.get("describer")),
        registry=_merge_dataclass(RegistryConfig, raw.get("registry")),
        graph=_merge_dataclass(GraphConfig, raw.get("graph")),
        api_retry=_merge_dataclass(ApiRetryConfig, raw.get("api_retry")),
        logging=_merge_dataclass(LoggingConfig, raw.get("logging")),
    )
