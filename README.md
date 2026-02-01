# Code Duplicate Analyzer (vLLM embeddings + Weaviate)

## Quick start

1) Put source code under `./input/` (this folder is mounted read-only into the analyzer).
2) Run:

```bash
docker compose up --build
```

3) Open:
- Local report: `output/report.html` (open in browser)
- Weaviate UI: http://localhost:8080
- vLLM: http://localhost:8000/v1
- MCP server (tools API): http://localhost:8090
- Web UI (Dedup Explorer): http://localhost:8091

4) Outputs are written to `./output/`:
- `report.html` — quick stats + inline duplicate code previews
- `stats.json` — summary stats
- `chunks.jsonl` — one JSON per extracted chunk
- `candidates_exact_dups.jsonl` — local exact-duplicate candidates (by fingerprint)
- `analysis_progress.sqlite` — per-session annotations (set via MCP)

## Example output
- See `examples/output/` for a sample run (generated from a small JS project).

## Notes
- Default Weaviate vectorizer is disabled; we insert vectors ourselves.
- If your model download needs auth, export `HF_TOKEN` before running.
- GPU: uncomment the `deploy.resources...` block for the `vllm` service (works on Docker with NVIDIA runtime).
- VRAM: the default `Octen/Octen-Embedding-4B` model needs ~8 GB VRAM. If you have less, swap in a lighter embedding model (~1B family) via `docker-compose.yml` and the `EMBEDDING_MODEL` / vLLM command args.
- Other resources: nothing special beyond GPU memory; if VRAM fits, it should start immediately.

## MCP tools (simple HTTP)
The MCP service is a separate container so you can restart it without losing progress. It prefers Weaviate when available, and falls back to `output/chunks.jsonl`. It reads `output/candidates_exact_dups.jsonl` and stores annotations in `output/analysis_progress.sqlite`.

Endpoints:
- `GET /health`
- `GET /tools/list`
- `POST /tools/call`

Example tool call:
```bash
curl -s http://localhost:8090/tools/call \\
  -H 'content-type: application/json' \\
  -d '{"name":"search_chunks","arguments":{"path_contains":"src/","limit":5}}'
```

Tools:
- `search_chunks` — filter by repo/path/language/node_type/fingerprint/text; returns chunk summaries
- `get_chunk_text` — get raw text for a chunk with `max_length`
- `list_duplicate_groups` — list fingerprint groups from `candidates_exact_dups.jsonl`
- `get_duplicate_group` — fetch a specific group; optional chunk text
- `set_annotation` / `get_annotation` / `list_annotations` — track status/priority/comments per chunk or group

Annotation priorities:
- Separate `human_priority` and `ai_priority` (bigger = more important).
- MCP updates only `ai_priority` by default. To allow human updates via MCP, set `ALLOW_HUMAN_PRIORITY_UPDATE=1` on the MCP service.

## Web UI
The web UI runs as a separate service (port 8091) and reuses the same data sources as MCP. It provides a single-page explorer with filters, chunk inspection, duplicate groups, and annotation editing.

Highlights:
- Status buttons (2do/skip/done) for duplicate groups; sets status for all chunks in a group.
- Status filter chips (2do/skip/done) to exclude specific statuses from the list.
- Group details show per‑chunk status pills and stay in sync after Apply or group status changes.

See `web/README.md` for UI-specific notes and API details.
