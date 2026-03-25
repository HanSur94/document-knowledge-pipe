# Model-Agnostic Provider Abstraction

**Date:** 2026-03-24
**Status:** Draft
**Goal:** Make the system support OpenAI, Azure OpenAI, and Anthropic as interchangeable providers with a clean abstraction layer.

## Provider Support Matrix

| Use case | OpenAI | Azure OpenAI | Anthropic |
|----------|--------|--------------|-----------|
| Describer (vision) | Yes | Yes | Yes |
| Registry (summaries) | Yes | Yes | Yes |
| Graph (LightRAG) | Yes | Yes | No |

Anthropic is excluded from Graph because it does not offer an embedding API, and LightRAG requires embeddings.

## Architecture

### Provider Abstraction Layer

New module `src/docpipe/providers/`:

```
src/docpipe/providers/
├── __init__.py       # Factory function + re-exports
├── base.py           # Abstract base class with shared retry logic
├── openai.py         # OpenAI adapter
├── azure.py          # Azure OpenAI adapter
└── anthropic.py      # Anthropic adapter
```

#### Abstract Base Class (`base.py`)

```python
from abc import ABC, abstractmethod

class ProviderError(Exception):
    """Common exception wrapping provider-specific SDK errors.
    Defined in providers/base.py, re-exported from providers/__init__.py."""
    pass

class LLMProvider(ABC):
    """Unified interface for LLM providers.

    Adapters receive their provider-specific config at construction time
    and store model/deployment names as instance attributes.
    """

    def __init__(self, retry_cfg: ApiRetryConfig) -> None:
        self._retry_cfg = retry_cfg

    async def complete(self, prompt: str, max_tokens: int) -> str:
        """Generate a text completion (with retry)."""
        return await self._retry(self._complete_raw, prompt, max_tokens)

    async def vision(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        """Generate a description of an image (with retry)."""
        return await self._retry(self._vision_raw, prompt, image_b64, media_type, max_tokens)

    @abstractmethod
    async def _complete_raw(self, prompt: str, max_tokens: int) -> str:
        """Raw completion call — implemented by each adapter."""
        ...

    @abstractmethod
    async def _vision_raw(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        """Raw vision call — implemented by each adapter.
        Note: Anthropic adapter must cast media_type to its Literal type internally."""
        ...

    async def _retry(self, fn, *args) -> str:
        """Exponential backoff retry. Catches ProviderError."""
        delay = self._retry_cfg.initial_delay_seconds
        for attempt in range(self._retry_cfg.max_retries + 1):
            try:
                return await fn(*args)
            except ProviderError as e:
                if attempt == self._retry_cfg.max_retries:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._retry_cfg.max_delay_seconds)
        raise ProviderError("Exhausted retries")  # unreachable — loop re-raises on last attempt
```

**Template method pattern:** Consumers call the public `complete()` / `vision()` methods which handle retry. Adapters implement `_complete_raw()` / `_vision_raw()` which make a single API call and raise `ProviderError` on failure. This prevents duplicate retry loops.

**`ProviderError`** lives in `providers/base.py` and is re-exported from `providers/__init__.py`.

#### Factory Function (`__init__.py`)

```python
def create_provider(provider: str, provider_configs: dict, retry_cfg: ApiRetryConfig) -> LLMProvider:
    """Create the appropriate LLMProvider based on the provider name."""
```

Maps `"openai"` / `"azure"` / `"anthropic"` to the corresponding adapter class. Raises `ValueError` for unknown providers.

#### Adapter Implementations

**OpenAI (`openai.py`):**
- Wraps `openai.AsyncOpenAI`
- `complete()` uses `client.chat.completions.create()`
- `vision()` uses the same endpoint with `image_url` content blocks
- API key from `OPENAI_API_KEY` env var

**Azure OpenAI (`azure.py`):**
- Wraps `openai.AsyncAzureOpenAI` (same SDK, different client class)
- Requires: `endpoint`, `deployment`, `api_version` from config
- `complete()` and `vision()` use the same methods as OpenAI, but the client routes to Azure
- API key from `AZURE_OPENAI_API_KEY` env var

**Anthropic (`anthropic.py`):**
- Wraps `anthropic.AsyncAnthropic`
- `complete()` uses `client.messages.create()`
- `vision()` uses `ImageBlockParam` with base64 source
- API key from `ANTHROPIC_API_KEY` env var

### Configuration Changes

Replace flat `provider`/`model` fields with nested per-provider config blocks.

#### New Config YAML Structure

```yaml
describer:
  provider: "openai"          # "openai", "azure", or "anthropic"
  max_tokens: 300
  include_context: true
  context_chars: 500
  batch_size: 5

  openai:
    model: "gpt-4o-mini"
  azure:
    model: "gpt-4o-mini"
    endpoint: "https://myinstance.openai.azure.com"
    deployment: "my-gpt4o-mini"
    api_version: "2024-06-01"
  anthropic:
    model: "claude-haiku-4-5-20251001"

graph:
  provider: "openai"          # "openai" or "azure" only
  storage: "file"
  store_dir: "./output/lightrag_store"
  chunk_size: 1200
  chunk_overlap: 100

  openai:
    model: "gpt-4o-mini"
    embedding_model: "text-embedding-3-small"
    embedding_dim: 1536
  azure:
    model: "gpt-4o-mini"
    embedding_model: "text-embedding-3-small"
    embedding_dim: 1536
    endpoint: "https://myinstance.openai.azure.com"
    deployment: "my-gpt4o-mini"
    embedding_deployment: "my-embedding"
    api_version: "2024-06-01"
```

#### New Config Dataclasses

```python
@dataclass
class OpenAIProviderConfig:
    model: str = "gpt-4o-mini"

@dataclass
class AzureProviderConfig:
    model: str = "gpt-4o-mini"
    endpoint: str = ""        # Required when provider=="azure"; validated at startup
    deployment: str = ""      # Required when provider=="azure"; validated at startup
    api_version: str = "2024-06-01"

@dataclass
class AnthropicProviderConfig:
    model: str = "claude-haiku-4-5-20251001"

@dataclass
class GraphOpenAIConfig:
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # Must match the embedding model's output dimension

@dataclass
class GraphAzureConfig:
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # Must match the embedding model's output dimension
    endpoint: str = ""         # Required when provider=="azure"; validated at startup
    deployment: str = ""       # Required when provider=="azure"; validated at startup
    embedding_deployment: str = ""  # Required when provider=="azure"; validated at startup
    api_version: str = "2024-06-01"
```

#### Updated Top-Level Config Dataclasses

The flat `model` and `embedding_model` fields are **removed** from `DescriberConfig` and `GraphConfig`. They are replaced by nested provider config objects.

```python
@dataclass
class DescriberConfig:
    provider: str = "openai"  # "openai", "azure", or "anthropic"
    max_tokens: int = 300
    include_context: bool = True
    context_chars: int = 500
    batch_size: int = 5
    openai: OpenAIProviderConfig = field(default_factory=OpenAIProviderConfig)
    azure: AzureProviderConfig = field(default_factory=AzureProviderConfig)
    anthropic: AnthropicProviderConfig = field(default_factory=AnthropicProviderConfig)

@dataclass
class GraphConfig:
    provider: str = "openai"  # "openai" or "azure" only
    storage: str = "file"
    store_dir: str = "./output/lightrag_store"
    max_tokens: int = 4096
    chunk_size: int = 1200
    chunk_overlap: int = 100
    openai: GraphOpenAIConfig = field(default_factory=GraphOpenAIConfig)
    azure: GraphAzureConfig = field(default_factory=GraphAzureConfig)

@dataclass
class DocpipeConfig:
    input_dir: Path
    output_dir: Path
    # openai_api_key and anthropic_api_key are REMOVED — providers read keys from os.environ
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    converter: ConverterConfig = field(default_factory=ConverterConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    describer: DescriberConfig = field(default_factory=DescriberConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    api_retry: ApiRetryConfig = field(default_factory=ApiRetryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
```

`load_config()` must also remove the `openai_api_key=os.environ.get(...)` and `anthropic_api_key=os.environ.get(...)` lines from the `DocpipeConfig(...)` constructor call.

#### Config Validation

`load_config()` validates after loading:
- If `describer.provider == "azure"`: `AzureProviderConfig.endpoint` and `AzureProviderConfig.deployment` must be non-empty, else `ValueError`.
- If `graph.provider == "azure"`: `GraphAzureConfig.endpoint`, `deployment`, and `embedding_deployment` must be non-empty, else `ValueError`.
- If `graph.provider == "anthropic"`: `ValueError("Anthropic is not supported for graph — no embedding API")`.

#### Nested Config Deserialization

The current `_merge_dataclass()` is flat — it passes raw dicts for nested keys. It must be extended to handle nested dataclass fields recursively:

```python
import typing

def _merge_dataclass(cls, data):
    """Create a dataclass instance, recursively deserializing nested dataclass fields.

    Uses typing.get_type_hints() to resolve forward references, which is
    necessary when 'from __future__ import annotations' is active (annotations
    are stored as strings, so dataclasses.is_dataclass() would fail on them).
    """
    if data is None:
        return cls()
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    filtered = {}
    for k, v in data.items():
        if k not in valid_fields:
            continue
        field_type = hints.get(k)
        if field_type and dataclasses.is_dataclass(field_type) and isinstance(v, dict):
            filtered[k] = _merge_dataclass(field_type, v)
        else:
            filtered[k] = v
    return cls(**filtered)
```

This allows `DescriberConfig` and `GraphConfig` to hold nested provider config fields that are automatically deserialized from YAML dicts.

#### Config Migration

Existing YAML configs using the old flat format (`provider: "openai"`, `model: "gpt-4o-mini"` directly under `describer:`) will still load — the `model` key will be ignored by `_merge_dataclass` since it no longer exists on `DescriberConfig`. A startup warning is logged: `"Deprecated config: 'model' under 'describer' is ignored. Use nested provider blocks instead."` This gives users a clear signal to migrate.

#### API Key Handling

API keys remain in `.env` / environment variables:
- `OPENAI_API_KEY` — used by OpenAI provider
- `AZURE_OPENAI_API_KEY` — used by Azure provider
- `ANTHROPIC_API_KEY` — used by Anthropic provider

The top-level `openai_api_key` and `anthropic_api_key` fields are removed from `DocpipeConfig`. Provider adapters read keys directly from `os.environ` at construction time.

**`.env` loading:** `load_dotenv()` remains in `load_config()` so that `.env` files are loaded before any provider adapter reads `os.environ`.

### Consumer Changes

#### `describer.py`
- Remove all direct `openai`/`anthropic` SDK imports
- Remove `_call_openai_vision_api()` and `_call_anthropic_vision_api()` functions
- Remove `retry_cfg` parameter — retry is now handled inside `LLMProvider._retry()`
- New signatures:

```python
async def describe_image(
    image_path: Path,
    context_before: str,
    context_after: str,
    provider: LLMProvider,
    max_tokens: int = 300,
) -> str:

async def replace_image_refs(
    markdown: str,
    output_dir: Path,
    cfg: DescriberConfig,   # still needed for include_context, context_chars, batch_size
    provider: LLMProvider,
    doc_title: str = "",
) -> str:
```

- Prompt building and image loading logic unchanged

#### `registry.py`
- Remove direct `openai`/`anthropic` SDK imports and the if/else provider dispatch in `generate_summary()`
- Remove hardcoded model names (`"gpt-4o-mini"`, `"claude-haiku-4-5-20251001"`) — models come from provider config
- Remove `retry_cfg` parameter — retry handled by provider
- `generate_summary()` new signature:

```python
async def generate_summary(
    markdown: str,
    cfg: RegistryConfig,
    provider: LLMProvider,
) -> tuple[str, str]:
```

- `max_tokens` for summary calls uses a fixed value of `200` (matching current behavior). This is not configurable — the prompt is tightly coupled to this token budget via `summary_max_words`.
- Response parsing (SUMMARY:/TOPICS: format) unchanged
- **Registry shares the describer's provider.** There is no separate `registry.provider` config — `pipeline.py` creates one `LLMProvider` from `cfg.describer` and passes it to both `replace_image_refs()` and `generate_summary()`

#### `pipeline.py`
- Creates the `LLMProvider` once at the start of `process_file()` via `create_provider()`
- Passes it to `replace_image_refs()` and `generate_summary()`
- Graph code unchanged aside from passing provider config

#### `graph.py`
- Adds `provider` field to `GraphConfig` (default: `"openai"`)
- When `provider == "openai"`, behavior is unchanged from today
- When `provider == "azure"`, `_get_rag_instance()` configures LightRAG for Azure by:
  1. Setting environment variables that LightRAG's `openai_complete`/`openai_embed` read internally:
     - `AZURE_OPENAI_ENDPOINT` = `cfg.azure.endpoint`
     - `AZURE_OPENAI_API_KEY` = `os.environ["AZURE_OPENAI_API_KEY"]`
     - `OPENAI_API_VERSION` = `cfg.azure.api_version`
  2. Passing deployment names as model names to LightRAG (Azure maps deployments → models)
  3. Using `cfg.azure.embedding_dim` instead of the hardcoded `1536`
  4. **Concurrency constraint:** `os.environ` is process-global, so `_get_rag_instance()` must not be called concurrently. The current pipeline processes files sequentially, so this is safe. If parallel file processing is added later, graph ingestion must be serialized (e.g., via an `asyncio.Lock`)
  5. For test isolation, env vars can be restored after LightRAG init via a context manager
- Config validation rejects `provider == "anthropic"` for graph
- **LightRAG version constraint:** The env var names above (`AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION`) are based on `lightrag-hku>=1.0.0` as pinned in `pyproject.toml`. If the LightRAG version changes, verify env var names still match

### Error Handling

A common `ProviderError` exception is introduced. Each adapter catches its SDK-specific errors (`openai.APIError`, `anthropic.APIError`) and wraps them in `ProviderError`. This keeps consumers SDK-agnostic.

The base class `_retry` helper catches `ProviderError` for retry logic, so retry behavior is identical across all providers.

### What Does NOT Change

- Document conversion (LibreOffice) — no LLM involvement
- Markdown extraction (PyMuPDF4LLM) — no LLM involvement
- File watching and status tracking
- Registry table format and file structure
- LightRAG's internal graph operations
- CLI commands and their interfaces
- The overall pipeline flow in `pipeline.py`
