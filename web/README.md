# Web UI (Dedup Explorer)

The web UI is a lightweight FastAPI service that serves a single-page app for browsing chunks, duplicate groups, and annotations. It runs on port **8091** in `docker-compose.yml`.

## Run

From repo root:

```bash
docker compose up --build
```

Then open `http://localhost:8091`.

## What it shows

- **Chunks table** with search filters and paging.
- **Inspector** for selected chunk (raw text + annotation fields).
- **Duplicate groups** list with group actions.
- **Group details** with per‑chunk preview and status pills.

## Status workflow

- Group rows have **2do / skip / done** buttons that set status for **all chunks in the group**.
- Status buttons highlight the active status; the current Open group is visually emphasized.
- Filters include 3 toggle chips (2do/skip/done). All are ON by default; turning one OFF excludes that status.
- Group details show a status pill per chunk and keep in sync after Apply or group status changes.

## API (web service)

These are internal endpoints used by the UI. They are provided by `web/server.py`:

- `GET /api/chunks/search` — search chunks (supports filters + `exclude_statuses`)
- `GET /api/chunks/text` — fetch raw chunk text
- `GET /api/dups/list_filtered` — list groups under current filters
- `GET /api/dups/get` — fetch a group by fingerprint (ignores UI filters)
- `POST /api/annotations/set` — set annotation for a single target
- `POST /api/annotations/set_group_status?fingerprint=...` — set status for all chunks in a group
- `POST /api/annotations/bulk_get` — fetch annotations for multiple chunk IDs

## Config

The web service uses the same env vars as MCP for data paths:

- `OUTPUT_DIR`, `CHUNKS_PATH`, `DUPS_PATH`, `MCP_DB_PATH`
- `ALLOW_HUMAN_PRIORITY_UPDATE=1` to allow updating `human_priority`

