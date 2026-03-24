"""Configuration loading and validation."""

from __future__ import annotations

import dataclasses
import logging as _logging
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_config_logger = _logging.getLogger(__name__)

_DEPRECATED_FLAT_KEYS = {
    "DescriberConfig": {"model"},
    "GraphConfig": {"model", "embedding_model"},
}


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
class OpenAIProviderConfig:
    model: str = "gpt-4o-mini"


@dataclass
class AzureProviderConfig:
    model: str = "gpt-4o-mini"
    endpoint: str = ""
    deployment: str = ""
    api_version: str = "2024-06-01"


@dataclass
class AnthropicProviderConfig:
    model: str = "claude-haiku-4-5-20251001"


@dataclass
class GraphOpenAIConfig:
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536


@dataclass
class GraphAzureConfig:
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    endpoint: str = ""
    deployment: str = ""
    embedding_deployment: str = ""
    api_version: str = "2024-06-01"


@dataclass
class DescriberConfig:
    provider: str = "openai"
    max_tokens: int = 300
    include_context: bool = True
    context_chars: int = 500
    batch_size: int = 5
    openai: OpenAIProviderConfig = field(default_factory=OpenAIProviderConfig)
    azure: AzureProviderConfig = field(default_factory=AzureProviderConfig)
    anthropic: AnthropicProviderConfig = field(default_factory=AnthropicProviderConfig)


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
    provider: str = "openai"
    storage: str = "file"
    store_dir: str = "./output/lightrag_store"
    max_tokens: int = 4096
    chunk_size: int = 1200
    chunk_overlap: int = 100
    openai: GraphOpenAIConfig = field(default_factory=GraphOpenAIConfig)
    azure: GraphAzureConfig = field(default_factory=GraphAzureConfig)


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
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    converter: ConverterConfig = field(default_factory=ConverterConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    describer: DescriberConfig = field(default_factory=DescriberConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    api_retry: ApiRetryConfig = field(default_factory=ApiRetryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    """Create a dataclass instance, recursively deserializing nested dataclass fields."""
    if data is None:
        return cls()
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    filtered: dict[str, Any] = {}
    for k, v in data.items():
        if k not in valid_fields:
            continue
        field_type = hints.get(k)
        if field_type and dataclasses.is_dataclass(field_type) and isinstance(v, dict):
            filtered[k] = _merge_dataclass(field_type, v)
        else:
            filtered[k] = v

    deprecated = _DEPRECATED_FLAT_KEYS.get(cls.__name__, set())
    for k in data:
        if k in deprecated and k not in valid_fields:
            _config_logger.warning(
                "Deprecated config: '%s' under '%s' is ignored. Use nested provider blocks instead.",
                k,
                cls.__name__,
            )

    return cls(**filtered)


def _validate_config(cfg: DocpipeConfig) -> None:
    """Validate provider-specific required fields."""
    if cfg.describer.provider == "azure":
        az = cfg.describer.azure
        if not az.endpoint or not az.deployment:
            raise ValueError(
                "Azure describer requires 'endpoint' and 'deployment' in describer.azure config"
            )
    if cfg.graph.provider == "azure":
        az = cfg.graph.azure
        if not az.endpoint or not az.deployment or not az.embedding_deployment:
            raise ValueError(
                "Azure graph requires 'endpoint', 'deployment', and 'embedding_deployment' in graph.azure config"
            )
    if cfg.graph.provider == "anthropic":
        raise ValueError("Anthropic is not supported for graph — no embedding API")


def load_config(path: Path) -> DocpipeConfig:
    """Load config from YAML file, merge with defaults, load .env."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    env_path = path.parent / ".env"
    load_dotenv(env_path)

    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

    cfg = DocpipeConfig(
        input_dir=Path(raw.get("input_dir", "./input")),
        output_dir=Path(raw.get("output_dir", "./output")),
        watcher=_merge_dataclass(WatcherConfig, raw.get("watcher")),
        converter=_merge_dataclass(ConverterConfig, raw.get("converter")),
        extractor=_merge_dataclass(ExtractorConfig, raw.get("extractor")),
        describer=_merge_dataclass(DescriberConfig, raw.get("describer")),
        registry=_merge_dataclass(RegistryConfig, raw.get("registry")),
        graph=_merge_dataclass(GraphConfig, raw.get("graph")),
        api_retry=_merge_dataclass(ApiRetryConfig, raw.get("api_retry")),
        logging=_merge_dataclass(LoggingConfig, raw.get("logging")),
    )

    _validate_config(cfg)
    return cfg
