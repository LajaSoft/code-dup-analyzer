from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from core import (
    ALLOW_HUMAN_PRIORITY_UPDATE,
    OUTPUT_DIR,
    CHUNKS_PATH,
    DUPS_PATH,
    DB_PATH,
    SESSION_ID,
    DEFAULT_MAX_TEXT_LEN,
    SearchParams,
    DupListParams,
    DupGetParams,
    AnnotationSetParams,
    AnnotationGetParams,
    AnnotationListParams,
    init,
    close,
    search_chunks,
    get_chunk_text,
    list_dup_groups,
    get_dup_group,
    set_annotation,
    get_annotation,
    list_annotations,
    list_dup_groups_filtered,
    get_dup_group_filtered,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init()
    try:
        yield
    finally:
        close()


app = FastAPI(title="code-dup-web", lifespan=lifespan)


def _int(v: Optional[str]) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _bool(v: Optional[str]) -> Optional[bool]:
    if v is None or v == "":
        return None
    return v.lower() in ("1", "true", "yes", "y")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "output_dir": str(OUTPUT_DIR),
        "chunks_path": str(CHUNKS_PATH),
        "dups_path": str(DUPS_PATH),
        "db_path": str(DB_PATH),
        "session_id": SESSION_ID,
        "allow_human_priority_update": ALLOW_HUMAN_PRIORITY_UPDATE,
    }


@app.get("/api/stats")
def api_stats() -> Dict[str, Any]:
    stats_path = OUTPUT_DIR / "stats.json"
    if not stats_path.exists():
        return {"ok": False, "error": "stats.json not found"}
    try:
        data = orjson.loads(stats_path.read_bytes())
    except Exception:
        return {"ok": False, "error": "stats.json parse error"}
    return {"ok": True, "data": data}


def _parse_search_params(q) -> SearchParams:
    exclude_raw = q.get("exclude_statuses")
    exclude_statuses = None
    if exclude_raw:
        exclude_statuses = tuple(s.strip() for s in exclude_raw.split(",") if s.strip())
    return SearchParams(
        repo=q.get("repo"),
        path_contains=q.get("path_contains"),
        language=q.get("language"),
        node_type=q.get("node_type"),
        fingerprint=q.get("fingerprint"),
        text_contains=q.get("text_contains"),
        normalized_contains=q.get("normalized_contains"),
        exclude_statuses=exclude_statuses,
        min_tokens=_int(q.get("min_tokens")),
        max_tokens=_int(q.get("max_tokens")),
        min_lines=_int(q.get("min_lines")),
        max_lines=_int(q.get("max_lines")),
        min_dup_count=_int(q.get("min_dup_count")),
        max_dup_count=_int(q.get("max_dup_count")),
        sort_by=q.get("sort_by"),
        sort_order=q.get("sort_order") or "desc",
        limit=_int(q.get("limit")) or 50,
        offset=_int(q.get("offset")) or 0,
    )


@app.get("/api/chunks/search")
def api_chunks_search(request: Request) -> Dict[str, Any]:
    return search_chunks(_parse_search_params(request.query_params))


@app.get("/api/chunks/text")
def api_chunk_text(chunk_id: str, max_length: int = DEFAULT_MAX_TEXT_LEN) -> Dict[str, Any]:
    data = get_chunk_text(chunk_id, max_length)
    if not data:
        raise HTTPException(status_code=404, detail="chunk_id not found")
    return data


@app.get("/api/dups/list")
def api_dups_list(
    min_count: int = 2,
    limit: int = 50,
    offset: int = 0,
    max_chunk_ids: int = 50,
) -> Dict[str, Any]:
    return list_dup_groups(DupListParams(min_count=min_count, limit=limit, offset=offset, max_chunk_ids=max_chunk_ids))


@app.get("/api/dups/get")
def api_dups_get(
    fingerprint: str,
    include_chunks: bool = False,
    chunk_text_max: int = DEFAULT_MAX_TEXT_LEN,
) -> Dict[str, Any]:
    data = get_dup_group(DupGetParams(fingerprint=fingerprint, include_chunks=include_chunks, chunk_text_max=chunk_text_max))
    if not data:
        raise HTTPException(status_code=404, detail="fingerprint not found")
    return data


@app.get("/api/dups/list_filtered")
def api_dups_list_filtered(request: Request) -> Dict[str, Any]:
    q = request.query_params
    params = _parse_search_params(q)
    min_count = _int(q.get("min_count")) or 2
    limit = _int(q.get("limit")) or 50
    offset = _int(q.get("offset")) or 0
    return list_dup_groups_filtered(params, min_count=min_count, limit=limit, offset=offset)


@app.get("/api/dups/get_filtered")
def api_dups_get_filtered(
    request: Request,
    fingerprint: str,
    chunk_text_max: int = DEFAULT_MAX_TEXT_LEN,
) -> Dict[str, Any]:
    params = _parse_search_params(request.query_params)
    data = get_dup_group_filtered(DupGetParams(fingerprint=fingerprint, include_chunks=True, chunk_text_max=chunk_text_max), params)
    if not data:
        raise HTTPException(status_code=404, detail="fingerprint not found")
    return data


@app.get("/api/annotations/get")
def api_annotation_get(target_type: str, target_id: str) -> Dict[str, Any]:
    data = get_annotation(AnnotationGetParams(target_type=target_type, target_id=target_id))
    return {"item": data}


@app.get("/api/annotations/list")
def api_annotations_list(
    target_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    return list_annotations(AnnotationListParams(target_type=target_type, status=status, limit=limit, offset=offset))


@app.post("/api/annotations/set")
async def api_annotations_set(request: Request) -> Dict[str, Any]:
    payload = await request.json()
    params = AnnotationSetParams(
        target_type=payload.get("target_type", ""),
        target_id=payload.get("target_id", ""),
        status=payload.get("status"),
        priority=payload.get("priority"),
        ai_priority=payload.get("ai_priority"),
        human_priority=payload.get("human_priority"),
        comment=payload.get("comment"),
    )
    if not params.target_type or not params.target_id:
        raise HTTPException(status_code=400, detail="target_type and target_id are required")
    return set_annotation(params)


@app.post("/api/annotations/set_group_status")
async def api_annotations_set_group_status(request: Request, fingerprint: str) -> Dict[str, Any]:
    payload = await request.json()
    status = payload.get("status")
    if not status:
        raise HTTPException(status_code=400, detail="status is required")
    data = get_dup_group(DupGetParams(fingerprint=fingerprint, include_chunks=False, chunk_text_max=0))
    if not data:
        raise HTTPException(status_code=404, detail="fingerprint not found")
    updated = 0
    for cid in data.get("chunk_ids") or []:
        set_annotation(AnnotationSetParams(target_type="chunk", target_id=cid, status=status))
        updated += 1
    return {"ok": True, "updated": updated, "status": status}


@app.post("/api/annotations/bulk_get")
async def api_annotations_bulk_get(request: Request) -> Dict[str, Any]:
    payload = await request.json()
    target_type = payload.get("target_type")
    target_ids = payload.get("target_ids") or []
    if not target_type or not isinstance(target_ids, list):
        raise HTTPException(status_code=400, detail="target_type and target_ids are required")
    items = []
    for tid in target_ids:
        if not tid:
            continue
        item = get_annotation(AnnotationGetParams(target_type=target_type, target_id=tid))
        if item:
            items.append(item)
    return {"items": items, "count": len(items)}


@app.get("/")
def index() -> HTMLResponse:
    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Dedup Explorer</title>
  <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css\" />
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap');
    :root {
      --bg-1: #0b1020;
      --bg-2: #111827;
      --ink: #e2e8f0;
      --muted: #94a3b8;
      --accent: #22d3ee;
      --accent-2: #f97316;
      --card: #0f172a;
      --border: #1f2937;
      --shadow: 0 18px 40px rgba(2, 6, 23, 0.5);
      --radius: 14px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Space Grotesk', sans-serif;
      color: var(--ink);
      background: radial-gradient(1200px 700px at 10% -10%, #0f766e 0%, transparent 55%),
                  radial-gradient(900px 600px at 110% 0%, #1d4ed8 0%, transparent 60%),
                  linear-gradient(180deg, var(--bg-1), var(--bg-2));
      min-height: 100vh;
    }
    header {
      padding: 28px 32px 10px;
    }
    .title {
      font-family: 'Instrument Serif', serif;
      font-size: 36px;
      letter-spacing: 0.2px;
      margin: 0 0 6px 0;
    }
    .subtitle { color: var(--muted); margin: 0; }
    .container { padding: 18px 32px 42px; display: grid; gap: 18px; }
    .grid { display: grid; grid-template-columns: 1.1fr 1fr; gap: 18px; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .stat { background: #0b1224; border-radius: 12px; padding: 12px; border: 1px solid #1f2937; }
    .stat h4 { margin: 0 0 6px 0; font-size: 14px; color: var(--muted); }
    .stat p { margin: 0; font-size: 20px; font-weight: 600; }
    .filters { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 6px; }
    input, select, textarea {
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      font-family: 'Space Grotesk', sans-serif;
      font-size: 13px;
      background: #0b1224;
      color: var(--ink);
    }
    textarea { resize: vertical; min-height: 64px; }
    .btn {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 9px 14px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
    }
    .btn.secondary { background: #111827; }
    .btn.ghost { background: transparent; color: var(--accent); border: 1px solid var(--accent); }
    .table-wrap { overflow: auto; border-radius: 12px; border: 1px solid var(--border); }
    table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    th, td { padding: 9px 10px; border-bottom: 1px solid #1f2937; text-align: left; }
    th { background: #0b1224; font-size: 12px; color: var(--muted); position: sticky; top: 0; }
    tr:hover { background: #0b1224; cursor: pointer; }
    .pill { display: inline-block; background: #111827; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
    .muted { color: var(--muted); }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
    .panel-title { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .panel-title h3 { margin: 0; font-size: 16px; }
    .badge { background: #0ea5e9; color: #fff; padding: 2px 8px; border-radius: 6px; font-size: 11px; }
    .list { display: grid; gap: 8px; }
    .dup-item { border: 1px solid #1f2937; border-radius: 10px; padding: 10px; }
    .dup-item:hover { border-color: #334155; }
    .dup-item.is-open { border-color: #38bdf8; background: #0c152a; }
    .footer-actions { display: flex; gap: 10px; align-items: center; }
    .dup-actions { display: flex; gap: 10px; align-items: center; margin-top: 6px; }
    .status-actions { display: flex; gap: 6px; align-items: center; margin-left: auto; }
    .status-btn {
      border: none;
      border-radius: 8px;
      padding: 6px 10px;
      font-weight: 600;
      font-size: 12px;
      cursor: pointer;
    }
    .status-btn.is-active { box-shadow: 0 0 0 2px rgba(15,23,42,0.4), inset 0 1px 0 rgba(255,255,255,0.2); }
    .status-btn.is-inactive { opacity: 0.35; }
    .status-btn.todo { background: #38bdf8; color: #06243d; }
    .status-btn.skip { background: #64748b; color: #0b1020; }
    .status-btn.done { background: #22c55e; color: #052a14; }
    .status-btn.comment { background: #fbbf24; color: #3a2a08; }
    .group-comment {
      margin-top: 8px;
      background: #0b1224;
      border: 1px solid #1f2937;
      border-radius: 10px;
      padding: 10px;
    }
    .group-comment textarea {
      width: 100%;
      min-height: 70px;
      background: #0f172a;
      color: var(--ink);
      border: 1px solid #1f2937;
      border-radius: 8px;
      padding: 8px;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 12px;
    }
    .group-comment .actions {
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }
    .btn.open-active { box-shadow: 0 0 0 2px rgba(34,211,238,0.6); }
    .status-pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      background: #111827;
      color: var(--muted);
    }
    .status-pill.todo { background: #38bdf8; color: #06243d; }
    .status-pill.skip { background: #94a3b8; color: #0b1020; }
    .status-pill.done { background: #22c55e; color: #052a14; }
    .status-filters { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .toggle-chip { display: inline-flex; align-items: center; cursor: pointer; user-select: none; }
    .toggle-chip input { display: none; }
    .toggle-chip span {
      padding: 4px 8px;
      border-radius: 8px;
      border: 1px solid #1f2937;
      background: #0b1224;
      color: var(--muted);
      font-size: 11px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 2px 4px rgba(0,0,0,0.35);
      transition: transform .1s ease, background .15s ease, color .15s ease, border .15s ease;
    }
    .toggle-chip input:checked + span {
      color: #0b1020;
      border-color: transparent;
      transform: translateY(1px);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);
    }
    .toggle-chip.todo input:checked + span { background: #38bdf8; }
    .toggle-chip.skip input:checked + span { background: #94a3b8; }
    .toggle-chip.done input:checked + span { background: #22c55e; }
    .toggle-chip.new input:checked + span { background: #fbbf24; }
    @media (max-width: 1100px) {
      .grid { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .filters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 700px) {
      .filters { grid-template-columns: 1fr; }
      .stats { grid-template-columns: 1fr; }
      .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1 class=\"title\">Dedup Explorer</h1>
    <p class=\"subtitle\">Расширенный просмотр дубликатов, фильтры и приоритеты для работы по refactor‑исследованию.</p>
  </header>

  <main class=\"container\">
    <section class=\"card\">
      <div class=\"panel-title\"><h3>Общая статистика</h3><span class=\"badge\" id=\"repoBadge\">repo</span></div>
      <div class=\"stats\" id=\"statsGrid\"></div>
    </section>

    <section class=\"card\">
      <div class=\"panel-title\"><h3>Фильтры чанков</h3><span class=\"muted\" id=\"resultMeta\">0 результатов</span></div>
      <div class=\"filters\">
        <div>
          <label>Path contains</label>
          <input id=\"pathContains\" placeholder=\"src/\" />
        </div>
        <div>
          <label>Language</label>
          <select id=\"languageSelect\"><option value=\"\">Any</option></select>
        </div>
        <div>
          <label>Node type</label>
          <input id=\"nodeType\" placeholder=\"function_definition\" />
        </div>
        <div>
          <label>Text contains</label>
          <input id=\"textContains\" placeholder=\"substring in raw\" />
        </div>
        <div>
          <label>Normalized contains</label>
          <input id=\"normContains\" placeholder=\"ID return\" />
        </div>
        <div>
          <label>Fingerprint</label>
          <input id=\"fingerprint\" placeholder=\"hash\" />
        </div>
        <div>
          <label>Status</label>
          <div class=\"status-filters\">
            <label class=\"toggle-chip new\"><input id=\"statusNew\" type=\"checkbox\" checked /><span>new</span></label>
            <label class=\"toggle-chip todo\"><input id=\"statusTodo\" type=\"checkbox\" checked /><span>2do</span></label>
            <label class=\"toggle-chip skip\"><input id=\"statusSkip\" type=\"checkbox\" checked /><span>skip</span></label>
            <label class=\"toggle-chip done\"><input id=\"statusDone\" type=\"checkbox\" checked /><span>done</span></label>
          </div>
        </div>
        <div>
          <label>Min tokens</label>
          <input id=\"minTokens\" type=\"number\" />
        </div>
        <div>
          <label>Max tokens</label>
          <input id=\"maxTokens\" type=\"number\" />
        </div>
        <div>
          <label>Min lines</label>
          <input id=\"minLines\" type=\"number\" />
        </div>
        <div>
          <label>Max lines</label>
          <input id=\"maxLines\" type=\"number\" />
        </div>
        <div>
          <label>Min dup count</label>
          <input id=\"minDup\" type=\"number\" />
        </div>
        <div>
          <label>Max dup count</label>
          <input id=\"maxDup\" type=\"number\" />
        </div>
        <div>
          <label>Sort by</label>
          <select id=\"sortBy\">
            <option value=\"\">Default</option>
            <option value=\"dup_count\">Dup count</option>
            <option value=\"token_estimate\">Token estimate</option>
            <option value=\"line_count\">Line count</option>
            <option value=\"path\">Path</option>
          </select>
        </div>
        <div>
          <label>Sort order</label>
          <select id=\"sortOrder\"><option value=\"desc\">Desc</option><option value=\"asc\">Asc</option></select>
        </div>
      </div>
      <div class=\"footer-actions\" style=\"margin-top:12px;\">
        <button class=\"btn\" id=\"applyFilters\">Apply</button>
        <button class=\"btn ghost\" id=\"resetFilters\">Reset</button>
        <span class=\"muted\" id=\"statusLine\"></span>
      </div>
    </section>

    <section class=\"grid\">
      <div class=\"card\">
        <div class=\"panel-title\"><h3>Чанки</h3><span class=\"muted\">клик по строке открывает инспектор</span></div>
        <div class=\"table-wrap\">
          <table id=\"chunksTable\">
            <thead>
              <tr>
                <th>Path</th>
                <th>Lang</th>
                <th>Lines</th>
                <th>Tokens</th>
                <th>Dup count</th>
                <th>Fingerprint</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
        <div style=\"margin-top:10px; display:flex; gap:8px;\">
          <button class=\"btn secondary\" id=\"loadMore\">Load more</button>
          <span class=\"muted\" id=\"pageInfo\"></span>
        </div>
      </div>

      <div class=\"card\" id=\"inspector\">
        <div class=\"panel-title\"><h3>Инспектор</h3><span class=\"muted\" id=\"selectedMeta\">не выбран</span></div>
        <div class=\"split\">
          <div>
            <div class=\"muted\">Chunk preview</div>
            <pre class=\"mono\" style=\"white-space: pre-wrap; background:#0b1224; color:#e2e8f0; padding:10px; border-radius:10px; min-height:120px;\"><code id=\"chunkPreview\" class=\"language-plaintext\"></code></pre>
          </div>
          <div>
            <div class=\"muted\">Annotation</div>
            <label>Status</label>
            <input id=\"annStatus\" placeholder=\"todo/skip/done\" />
            <div class=\"split\" style=\"margin-top:8px;\">
              <div>
                <label>AI priority</label>
                <input id=\"annAi\" type=\"number\" />
              </div>
              <div>
                <label>Human priority</label>
                <input id=\"annHuman\" type=\"number\" />
              </div>
            </div>
            <label style=\"margin-top:8px;\">Comment</label>
            <textarea id=\"annComment\"></textarea>
            <div style=\"margin-top:8px; display:flex; gap:8px;\">
              <button class=\"btn\" id=\"saveAnnotation\">Save</button>
              <span class=\"muted\" id=\"annHint\"></span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class=\"grid\">
      <div class=\"card\">
        <div class=\"panel-title\"><h3>Duplicate groups</h3><span class=\"muted\" id=\"dupMeta\">0 групп</span></div>
        <div class=\"filters\" style=\"grid-template-columns: repeat(4, minmax(0, 1fr));\">
          <div>
            <label>Min group size</label>
            <input id=\"dupMin\" type=\"number\" value=\"2\" />
          </div>
          <div>
            <label>Limit</label>
            <input id=\"dupLimit\" type=\"number\" value=\"30\" />
          </div>
          <div>
            <label>Offset</label>
            <input id=\"dupOffset\" type=\"number\" value=\"0\" />
          </div>
          <div style=\"align-self:end;\">
            <button class=\"btn\" id=\"loadDups\">Load groups</button>
          </div>
        </div>
        <div class=\"list\" id=\"dupList\" style=\"margin-top:12px;\"></div>
      </div>

      <div class=\"card\">
        <div class=\"panel-title\"><h3>Group details</h3><span class=\"muted\" id=\"groupMeta\">нет выбора</span></div>
        <div id=\"groupDetails\" class=\"list\"></div>
      </div>
    </section>
  </main>

<script>
let offset = 0;
let lastQuery = null;
let selectedTarget = null;
let allowHumanPriority = false;
let lastDupParams = '';
let currentOpenFingerprint = null;
let currentGroupChunkIds = [];
const groupStatus = new Map();

function qs(id){ return document.getElementById(id); }
function escapeHtml(str){
  return (str ?? '').toString().replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));
}
function langClass(lang){
  const map = {
    'c_sharp': 'csharp',
    'c#': 'csharp',
    'cpp': 'cpp',
    'c++': 'cpp',
    'ts': 'typescript',
    'tsx': 'tsx',
    'jsx': 'javascript',
    'js': 'javascript',
    'py': 'python',
    'rb': 'ruby',
    'yml': 'yaml',
  };
  const key = (lang || '').toLowerCase();
  return map[key] || key || 'plaintext';
}

function getExcludedStatuses(){
  const excluded = [];
  if(!qs('statusNew').checked) excluded.push('new');
  if(!qs('statusTodo').checked) excluded.push('todo');
  if(!qs('statusSkip').checked) excluded.push('skip');
  if(!qs('statusDone').checked) excluded.push('done');
  return excluded;
}

async function fetchJSON(url, options){
  const res = await fetch(url, options);
  if(!res.ok){
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return await res.json();
}

async function fetchChunkStatuses(chunkIds){
  if(!chunkIds?.length){ return {}; }
  const data = await fetchJSON('/api/annotations/bulk_get', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({target_type: 'chunk', target_ids: chunkIds}),
  });
  const map = {};
  (data.items || []).forEach(item => {
    if(item?.target_id){ map[item.target_id] = item.status || ''; }
  });
  return map;
}

async function refreshOpenGroupStatuses(){
  if(!currentGroupChunkIds.length){ return; }
  const statusMap = await fetchChunkStatuses(currentGroupChunkIds);
  const container = qs('groupDetails');
  container.querySelectorAll('[data-chunk-id]').forEach(item => {
    const cid = item.dataset.chunkId;
    const status = statusMap[cid] || '';
    const pill = item.querySelector('.status-pill');
    if(pill){
      pill.textContent = status || '-';
      pill.className = `status-pill ${status || ''}`.trim();
    }
  });
  if(selectedTarget?.type === 'chunk'){
    const status = statusMap[selectedTarget.id];
    if(status !== undefined){
      qs('annStatus').value = status || '';
    }
  }
}

async function loadStats(){
  const health = await fetchJSON('/health');
  allowHumanPriority = !!health.allow_human_priority_update;
  qs('annHuman').disabled = !allowHumanPriority;
  if(!allowHumanPriority){ qs('annHint').textContent = 'human_priority disabled'; }

  const statsResp = await fetchJSON('/api/stats');
  if(!statsResp.ok){
    qs('statsGrid').innerHTML = '<div class="muted">No stats.json yet</div>';
    return;
  }
  const data = statsResp.data;
  qs('repoBadge').textContent = data.repo || 'repo';
  const stats = [
    {label:'Files scanned', value: data.files_scanned || 0},
    {label:'Chunks extracted', value: data.chunks_extracted || 0},
    {label:'Duration (s)', value: data.duration_seconds || 0},
  ];
  qs('statsGrid').innerHTML = stats.map(s => `
    <div class="stat"><h4>${s.label}</h4><p>${s.value}</p></div>
  `).join('');

  const langSelect = qs('languageSelect');
  const langs = Object.keys(data.by_language || {});
  langSelect.innerHTML = '<option value="">Any</option>' + langs.map(l => `<option value="${l}">${l}</option>`).join('');
}

function buildFilterParams(){
  const params = new URLSearchParams();
  const fields = {
    path_contains: qs('pathContains').value,
    language: qs('languageSelect').value,
    node_type: qs('nodeType').value,
    text_contains: qs('textContains').value,
    normalized_contains: qs('normContains').value,
    fingerprint: qs('fingerprint').value,
    min_tokens: qs('minTokens').value,
    max_tokens: qs('maxTokens').value,
    min_lines: qs('minLines').value,
    max_lines: qs('maxLines').value,
    min_dup_count: qs('minDup').value,
    max_dup_count: qs('maxDup').value,
    sort_by: qs('sortBy').value,
    sort_order: qs('sortOrder').value,
  };
  for(const [k,v] of Object.entries(fields)){
    if(v !== '' && v !== null && v !== undefined){ params.set(k, v); }
  }
  const excluded = getExcludedStatuses();
  if(excluded.length){ params.set('exclude_statuses', excluded.join(',')); }
  return params;
}

function buildQuery(resetOffset=false){
  if(resetOffset){ offset = 0; }
  const params = buildFilterParams();
  params.set('limit', 50);
  params.set('offset', offset);
  return params;
}

async function loadChunks(reset=false){
  const params = buildQuery(reset);
  lastQuery = params.toString();
  qs('statusLine').textContent = 'Loading...';
  const data = await fetchJSON('/api/chunks/search?' + params.toString());
  qs('statusLine').textContent = '';

  if(reset){ qs('chunksTable').querySelector('tbody').innerHTML = ''; }
  const tbody = qs('chunksTable').querySelector('tbody');
  for(const item of data.items){
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono">${escapeHtml(item.path)}:${item.start_line}-${item.end_line}</td>
      <td><span class="pill">${escapeHtml(item.language)}</span></td>
      <td>${item.line_count}</td>
      <td>${item.token_estimate}</td>
      <td>${item.dup_count}</td>
      <td class="mono">${escapeHtml(item.fingerprint.slice(0, 10))}…</td>
    `;
    tr.onclick = () => selectChunk(item.chunk_id, item);
    tbody.appendChild(tr);
  }
  offset += data.items.length;
  qs('resultMeta').textContent = `${offset} результатов`;
  qs('pageInfo').textContent = `offset ${offset}`;
}

async function selectChunk(chunkId, summary){
  selectedTarget = {type: 'chunk', id: chunkId};
  qs('selectedMeta').textContent = summary ? `${summary.path}:${summary.start_line}-${summary.end_line}` : chunkId;
  const text = await fetchJSON(`/api/chunks/text?chunk_id=${encodeURIComponent(chunkId)}&max_length=2400`);
  const codeEl = qs('chunkPreview');
  const lang = summary?.language || text.language || 'plaintext';
  codeEl.className = `language-${langClass(lang)}`;
  codeEl.textContent = text.raw_text || '';
  if (window.hljs) { hljs.highlightElement(codeEl); }
  await loadAnnotation();
}

async function loadAnnotation(){
  if(!selectedTarget) return;
  const data = await fetchJSON(`/api/annotations/get?target_type=${selectedTarget.type}&target_id=${encodeURIComponent(selectedTarget.id)}`);
  const item = data.item || {};
  qs('annStatus').value = item.status || '';
  qs('annAi').value = item.ai_priority ?? '';
  qs('annHuman').value = item.human_priority ?? '';
  qs('annComment').value = item.comment || '';
}

async function saveAnnotation(){
  if(!selectedTarget){ return; }
  const payload = {
    target_type: selectedTarget.type,
    target_id: selectedTarget.id,
    status: qs('annStatus').value || null,
    ai_priority: qs('annAi').value ? parseInt(qs('annAi').value) : null,
    human_priority: qs('annHuman').value ? parseInt(qs('annHuman').value) : null,
    comment: qs('annComment').value || null,
  };
  const res = await fetchJSON('/api/annotations/set', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify(payload),
  });
  qs('annHint').textContent = 'saved';
  setTimeout(() => qs('annHint').textContent = '', 1500);
  if(res.human_priority_allowed === false){
    qs('annHuman').value = res.human_priority ?? '';
  }
}

async function loadDupGroups(){
  const min = qs('dupMin').value || 2;
  const limit = qs('dupLimit').value || 30;
  const offsetLocal = qs('dupOffset').value || 0;
  const params = buildFilterParams();
  params.set('min_count', min);
  params.set('limit', limit);
  params.set('offset', offsetLocal);
  lastDupParams = params.toString();
  const data = await fetchJSON(`/api/dups/list_filtered?${params.toString()}`);
  qs('dupMeta').textContent = `${data.count} групп`;
  const list = qs('dupList');
  list.innerHTML = '';
  const chunkIds = [];
  const chunkToGroup = new Map();
  data.items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'dup-item';
    div.dataset.fp = item.fingerprint;
    div.innerHTML = `
      <div class="mono">${escapeHtml(item.fingerprint)}</div>
      <div class="muted">size: ${item.count}</div>
      <div class="dup-actions">
        <button class="btn" data-open>Open</button>
        <div class="status-actions">
          <button class="status-btn todo" data-status="todo">2do</button>
          <button class="status-btn skip" data-status="skip">skip</button>
          <button class="status-btn done" data-status="done">done</button>
          <button class="status-btn comment" data-comment>comment</button>
        </div>
      </div>
      <div class="group-comment" data-comment-editor style="display:none;">
        <textarea placeholder="comment for this group"></textarea>
        <div class="actions">
          <button class="btn" data-comment-save>Ok</button>
          <button class="btn ghost" data-comment-cancel>Cancel</button>
        </div>
      </div>
    `;
    div.querySelector('[data-open]').onclick = () => openGroup(item.fingerprint);
    div.querySelectorAll('.status-btn[data-status]').forEach(btn => {
      btn.onclick = () => setGroupStatus(item.fingerprint, btn.dataset.status);
    });
    const commentBtn = div.querySelector('[data-comment]');
    if(commentBtn){ commentBtn.onclick = () => toggleGroupComment(item.fingerprint, div); }
    updateGroupRow(div, groupStatus.get(item.fingerprint) || '', currentOpenFingerprint === item.fingerprint);
    list.appendChild(div);
    (item.chunk_ids || []).forEach(cid => {
      if(!cid) return;
      chunkIds.push(cid);
      chunkToGroup.set(cid, item.fingerprint);
    });
  });
  if(chunkIds.length){
    const statusMap = await fetchChunkStatuses(chunkIds);
    for(const [cid, status] of Object.entries(statusMap)){
      if(!status) continue;
      const fp = chunkToGroup.get(cid);
      if(!fp || groupStatus.has(fp)) continue;
      groupStatus.set(fp, status);
      const row = list.querySelector(`[data-fp="${CSS.escape(fp)}"]`);
      if(row){ updateGroupRow(row, status, currentOpenFingerprint === fp); }
    }
  }
}

function updateGroupRow(div, status, isOpen){
  div.classList.toggle('is-open', !!isOpen);
  const openBtn = div.querySelector('[data-open]');
  if(openBtn){ openBtn.classList.toggle('open-active', !!isOpen); }
  const buttons = Array.from(div.querySelectorAll('.status-btn[data-status]'));
  buttons.forEach(btn => {
    const isMatch = status && btn.dataset.status === status;
    btn.classList.toggle('is-active', !!isMatch);
    btn.classList.toggle('is-inactive', !!status && !isMatch);
  });
}

async function setGroupStatus(fp, status){
  if(!fp || !status){ return; }
  qs('statusLine').textContent = 'Updating group...';
  const params = new URLSearchParams();
  params.set('fingerprint', fp);
  await fetchJSON(`/api/annotations/set_group_status?${params.toString()}`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({status}),
  });
  groupStatus.set(fp, status);
  const row = qs('dupList').querySelector(`[data-fp="${CSS.escape(fp)}"]`);
  if(row){ updateGroupRow(row, status, currentOpenFingerprint === fp); }
  if(currentOpenFingerprint === fp){
    qs('groupDetails').querySelectorAll('[data-chunk-id]').forEach(item => {
      const pill = item.querySelector('.status-pill');
      if(pill){
        pill.textContent = status;
        pill.className = `status-pill ${status}`;
      }
    });
    if(selectedTarget?.type === 'chunk'){
      qs('annStatus').value = status;
    }
  }
  qs('statusLine').textContent = `group ${fp.slice(0, 8)}… -> ${status}`;
  setTimeout(() => qs('statusLine').textContent = '', 1600);
}

async function toggleGroupComment(fp, row){
  const editor = row.querySelector('[data-comment-editor]');
  if(!editor){ return; }
  const isOpen = editor.style.display !== 'none';
  if(isOpen){
    editor.style.display = 'none';
    return;
  }
  editor.style.display = 'block';
  let current = '';
  try{
    const data = await fetchJSON(`/api/annotations/get?target_type=dup_group&target_id=${encodeURIComponent(fp)}`);
    current = data?.comment || '';
  }catch(_){}
  const textarea = editor.querySelector('textarea');
  if(textarea){ textarea.value = current; textarea.focus(); }
  const saveBtn = editor.querySelector('[data-comment-save]');
  const cancelBtn = editor.querySelector('[data-comment-cancel]');
  if(saveBtn){
    saveBtn.onclick = async () => {
      const comment = textarea ? textarea.value : '';
      qs('statusLine').textContent = 'Updating comment...';
      await fetchJSON('/api/annotations/set', {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({target_type: 'dup_group', target_id: fp, comment: comment || null}),
      });
      qs('statusLine').textContent = `group ${fp.slice(0, 8)}… comment updated`;
      setTimeout(() => qs('statusLine').textContent = '', 1600);
      editor.style.display = 'none';
    };
  }
  if(cancelBtn){
    cancelBtn.onclick = () => { editor.style.display = 'none'; };
  }
}

async function openGroup(fp){
  const params = new URLSearchParams();
  params.set('fingerprint', fp);
  params.set('include_chunks', 'true');
  params.set('chunk_text_max', 1000);
  const data = await fetchJSON(`/api/dups/get?${params.toString()}`);
  selectedTarget = {type: 'dup_group', id: fp};
  currentOpenFingerprint = fp;
  qs('dupList').querySelectorAll('.dup-item').forEach(div => {
    updateGroupRow(div, groupStatus.get(div.dataset.fp) || '', div.dataset.fp === fp);
  });
  qs('groupMeta').textContent = `fingerprint ${fp} · size ${data.count}`;
  await loadAnnotation();
  const container = qs('groupDetails');
  container.innerHTML = '';
  currentGroupChunkIds = data.chunks.map(c => c.chunk_id);
  data.chunks.forEach(ch => {
    const item = document.createElement('div');
    item.className = 'dup-item';
    item.dataset.chunkId = ch.chunk_id;
    item.innerHTML = `
      <div class="mono">${escapeHtml(ch.path)}:${ch.start_line}-${ch.end_line}</div>
      <div class="muted">tokens: ${ch.token_estimate} · dup_count: ${ch.dup_count} · <span class="status-pill">?</span></div>
      <pre class="mono" style="white-space:pre-wrap; background:#0b1224; color:#e2e8f0; padding:8px; border-radius:8px;"><code class="language-plaintext"></code></pre>
      <button class="btn ghost">Inspect chunk</button>
    `;
    const code = item.querySelector('code');
    if (code) {
      code.className = `language-${langClass(ch.language || 'plaintext')}`;
      code.textContent = ch.raw_text || '';
      if (window.hljs) { hljs.highlightElement(code); }
    }
    item.querySelector('button').onclick = () => selectChunk(ch.chunk_id, ch);
    container.appendChild(item);
  });
  const statusMap = await fetchChunkStatuses(currentGroupChunkIds);
  container.querySelectorAll('[data-chunk-id]').forEach(item => {
    const cid = item.dataset.chunkId;
    const status = statusMap[cid] || '';
    const pill = item.querySelector('.status-pill');
    if(pill){
      pill.textContent = status || '-';
      pill.className = `status-pill ${status || ''}`.trim();
    }
  });
  if(data.chunks.length){
    await selectChunk(data.chunks[0].chunk_id, data.chunks[0]);
  }
}

qs('applyFilters').onclick = async () => {
  await loadChunks(true);
  await loadDupGroups();
  await refreshOpenGroupStatuses();
};
qs('resetFilters').onclick = () => {
  ['pathContains','nodeType','textContains','normContains','fingerprint','minTokens','maxTokens','minLines','maxLines','minDup','maxDup'].forEach(id => qs(id).value = '');
  qs('languageSelect').value = '';
  qs('sortBy').value = '';
  qs('sortOrder').value = 'desc';
  qs('statusNew').checked = true;
  qs('statusTodo').checked = true;
  qs('statusSkip').checked = true;
  qs('statusDone').checked = true;
  loadChunks(true);
};
qs('loadMore').onclick = () => loadChunks(false);
qs('saveAnnotation').onclick = saveAnnotation;
qs('loadDups').onclick = loadDupGroups;

loadStats().then(() => loadChunks(true)).then(loadDupGroups);
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
</body>
</html>"""
    return HTMLResponse(html)
