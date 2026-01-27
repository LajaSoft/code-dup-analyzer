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

4) Outputs are written to `./output/`:
- `report.html` — quick stats + inline duplicate code previews
- `stats.json` — summary stats
- `chunks.jsonl` — one JSON per extracted chunk
- `candidates_exact_dups.jsonl` — local exact-duplicate candidates (by fingerprint)

## Example output
- See `examples/output/` for a sample run (generated from a small JS project).
- Browsing on GitHub? Use the HTML preview helper: `https://htmlpreview.github.io/?https://raw.githubusercontent.com/<owner>/<repo>/main/examples/output/report.html` (replace `<owner>/<repo>` with your repo path).

## Notes
- Default Weaviate vectorizer is disabled; we insert vectors ourselves.
- If your model download needs auth, export `HF_TOKEN` before running.
- GPU: uncomment the `deploy.resources...` block for the `vllm` service (works on Docker with NVIDIA runtime).
