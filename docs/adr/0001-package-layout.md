# ADR 0001: Use one namespace package with RAG-oriented boundaries

- Status: Accepted
- Date: 2026-09-02

## Context

The upstream specification uses several generic top-level packages such as `core`, `libs`, and
`mcp_server`. They describe the original implementation, but generic names can collide with other
packages and make dependency direction less obvious when this project is later embedded in an
Agent system.

## Decision

All source code lives under the `modular_rag` namespace. The first stable boundaries are:

- `domain`: pure data models and provider contracts;
- `ingestion`: offline document processing workflows;
- `retrieval`: online search, fusion, and ranking workflows;
- `generation`: grounded answers and citations;
- `providers`: external LLM, embedding, loader, reranker, and storage implementations;
- `interfaces`: CLI, MCP, and future HTTP adapters;
- `observability`: logs, traces, metrics, and evaluation.

Runtime data directories are created only when a use case needs them. Provider-specific packages
are also added incrementally instead of creating a large empty directory tree in A1.

## Dependency direction

```text
interfaces ──> ingestion / retrieval / generation ──> domain
                         ^
                         │
                    providers

observability can be injected at workflow boundaries without owning business decisions.
```

## Consequences

- Imports are explicit and collision-resistant: `modular_rag.retrieval` instead of `core`.
- The code will not match upstream file-for-file, which is intentional.
- Later tasks must map the upstream specification to these boundaries before implementation.
