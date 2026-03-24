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

class LLMProvider(ABC):
    """Unified interface for LLM providers."""

    @abstractmethod
    async def complete(self, prompt: str, max_tokens: int) -> str:
        """Generate a text completion."""
        ...

    @abstractmethod
    async def vision(self, prompt: str, image_b64: str, media_type: str, max_tokens: int) -> str:
        """Generate a description of an image."""
        ...
```

- Retry logic with exponential backoff lives as a `_retry` helper in the base class. Adapters implement only the raw API call; the base class wraps it with retry.
- Each adapter catches its provider-specific exceptions and re-raises a common `ProviderError` so consumers don't need to know which SDK is in use.

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
  azure:
    model: "gpt-4o-mini"
    embedding_model: "text-embedding-3-small"
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
    endpoint: str = ""
    deployment: str = ""
    api_version: str = "2024-06-01"

@dataclass
class AnthropicProviderConfig:
    model: str = "claude-haiku-4-5-20251001"

@dataclass
class GraphAzureConfig:
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    endpoint: str = ""
    deployment: str = ""
    embedding_deployment: str = ""
    api_version: str = "2024-06-01"

@dataclass
class GraphOpenAIConfig:
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
```

#### API Key Handling

API keys remain in `.env` / environment variables:
- `OPENAI_API_KEY` — used by OpenAI provider
- `AZURE_OPENAI_API_KEY` — used by Azure provider
- `ANTHROPIC_API_KEY` — used by Anthropic provider

The top-level `openai_api_key` and `anthropic_api_key` fields are removed from `DocpipeConfig`. Provider adapters read keys directly from `os.environ` at construction time.

### Consumer Changes

#### `describer.py`
- Remove all direct `openai`/`anthropic` SDK imports
- Remove `_call_openai_vision_api()` and `_call_anthropic_vision_api()` functions
- `describe_image()` receives an `LLMProvider` instance and calls `provider.vision()`
- `replace_image_refs()` accepts an `LLMProvider` parameter instead of `DescriberConfig.provider`
- Prompt building and image loading logic unchanged

#### `registry.py`
- Remove direct `openai`/`anthropic` SDK imports and the if/else provider dispatch in `generate_summary()`
- `generate_summary()` receives an `LLMProvider` instance and calls `provider.complete()`
- Response parsing (SUMMARY:/TOPICS: format) unchanged

#### `pipeline.py`
- Creates the `LLMProvider` once at the start of `process_file()` via `create_provider()`
- Passes it to `replace_image_refs()` and `generate_summary()`
- Graph code unchanged aside from passing provider config

#### `graph.py`
- Adds `provider` field to `GraphConfig` (default: `"openai"`)
- When `provider == "azure"`, configures LightRAG with Azure-specific environment variables and deployment names
- When `provider == "openai"`, behavior is unchanged from today
- Config validation rejects `provider == "anthropic"` for graph

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
