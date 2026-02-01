from __future__ import annotations

import os
import time
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import orjson
import requests

try:
    import weaviate
    from weaviate.classes.query import Filter
except Exception:  # pragma: no cover - optional dependency at runtime
    weaviate = None
    Filter = None


OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output")).resolve()
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", str(OUTPUT_DIR / "chunks.jsonl"))).resolve()
DUPS_PATH = Path(os.getenv("DUPS_PATH", str(OUTPUT_DIR / "candidates_exact_dups.jsonl"))).resolve()
DB_PATH = Path(os.getenv("MCP_DB_PATH", str(OUTPUT_DIR / "analysis_progress.sqlite"))).resolve()
SESSION_ID = os.getenv("SESSION_ID", "default")
ALLOW_HUMAN_PRIORITY_UPDATE = os.getenv("ALLOW_HUMAN_PRIORITY_UPDATE", "0").strip() in ("1", "true", "True")

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "weaviate")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_GRPC_SECURE = os.getenv("WEAVIATE_GRPC_SECURE", "0").strip() in ("1", "true", "True")
USE_WEAVIATE = os.getenv("USE_WEAVIATE", "1").strip() not in ("0", "false", "False")
WEAVIATE_FETCH_LIMIT = int(os.getenv("WEAVIATE_FETCH_LIMIT", "10000"))
DEFAULT_MAX_TEXT_LEN = int(os.getenv("DEFAULT_MAX_TEXT_LEN", "2000"))


@dataclass
class ChunkSummary:
    chunk_id: str
    repo: str
    path: str
    language: str
    node_type: str
    start_line: int
    end_line: int
    line_count: int
    token_estimate: int
    fingerprint: str
    dup_count: int


@dataclass
class SearchParams:
    repo: Optional[str] = None
    path_contains: Optional[str] = None
    language: Optional[str] = None
    node_type: Optional[str] = None
    fingerprint: Optional[str] = None
    text_contains: Optional[str] = None
    normalized_contains: Optional[str] = None
    exclude_statuses: Optional[Tuple[str, ...]] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    min_lines: Optional[int] = None
    max_lines: Optional[int] = None
    min_dup_count: Optional[int] = None
    max_dup_count: Optional[int] = None
    limit: int = 50
    offset: int = 0
    sort_by: Optional[str] = None
    sort_order: str = "desc"


@dataclass
class DupListParams:
    min_count: int = 2
    limit: int = 50
    offset: int = 0
    max_chunk_ids: int = 50


@dataclass
class DupGetParams:
    fingerprint: str
    include_chunks: bool = False
    chunk_text_max: int = DEFAULT_MAX_TEXT_LEN


@dataclass
class AnnotationSetParams:
    target_type: str
    target_id: str
    status: Optional[str] = None
    priority: Optional[int] = None
    ai_priority: Optional[int] = None
    human_priority: Optional[int] = None
    comment: Optional[str] = None


@dataclass
class AnnotationGetParams:
    target_type: str
    target_id: str


@dataclass
class AnnotationListParams:
    target_type: Optional[str] = None
    status: Optional[str] = None
    limit: int = 100
    offset: int = 0


# --- Utilities ---

def _json_loads(line: bytes) -> Dict[str, Any]:
    return orjson.loads(line)


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...[truncated]..."


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield _json_loads(line)
            except Exception:
                continue


# --- Duplicate counts cache ---

_dup_counts_cache: Tuple[float, Dict[str, int]] = (0.0, {})


def dup_counts() -> Dict[str, int]:
    global _dup_counts_cache
    try:
        mtime = DUPS_PATH.stat().st_mtime
    except OSError:
        return {}
    cached_mtime, cached = _dup_counts_cache
    if cached and cached_mtime == mtime:
        return cached
    counts: Dict[str, int] = {}
    for g in _iter_jsonl(DUPS_PATH):
        fp = g.get("fingerprint")
        if not fp:
            continue
        counts[fp] = int(g.get("count", 0) or 0)
    _dup_counts_cache = (mtime, counts)
    return counts


# --- SQLite annotations ---

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                session_id TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT,
                human_priority INTEGER,
                ai_priority INTEGER,
                comment TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (session_id, target_type, target_id)
            )
            """
        )
        cols = {row[1] for row in conn.execute("PRAGMA table_info(annotations)")}
        if "human_priority" not in cols:
            conn.execute("ALTER TABLE annotations ADD COLUMN human_priority INTEGER")
        if "ai_priority" not in cols:
            conn.execute("ALTER TABLE annotations ADD COLUMN ai_priority INTEGER")
        conn.commit()


def set_annotation(args: AnnotationSetParams) -> Dict[str, Any]:
    now = time.time()
    ai_priority = args.ai_priority if args.ai_priority is not None else args.priority
    human_priority = args.human_priority if ALLOW_HUMAN_PRIORITY_UPDATE else None
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO annotations (session_id, target_type, target_id, status, human_priority, ai_priority, comment, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, target_type, target_id)
            DO UPDATE SET status=excluded.status, human_priority=COALESCE(excluded.human_priority, annotations.human_priority),
                          ai_priority=excluded.ai_priority, comment=excluded.comment, updated_at=excluded.updated_at
            """,
            (SESSION_ID, args.target_type, args.target_id, args.status, human_priority, ai_priority, args.comment, now),
        )
        conn.commit()
    current = get_annotation(AnnotationGetParams(target_type=args.target_type, target_id=args.target_id)) or {}
    return {
        **current,
        "human_priority_allowed": ALLOW_HUMAN_PRIORITY_UPDATE,
    }


def get_annotation(args: AnnotationGetParams) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT session_id, target_type, target_id, status, human_priority, ai_priority, comment, updated_at
            FROM annotations
            WHERE session_id=? AND target_type=? AND target_id=?
            """,
            (SESSION_ID, args.target_type, args.target_id),
        ).fetchone()
    if not row:
        return None
    return {
        "session_id": row[0],
        "target_type": row[1],
        "target_id": row[2],
        "status": row[3],
        "human_priority": row[4],
        "ai_priority": row[5],
        "comment": row[6],
        "updated_at": row[7],
    }


def list_annotations(args: AnnotationListParams) -> Dict[str, Any]:
    q = "SELECT session_id, target_type, target_id, status, human_priority, ai_priority, comment, updated_at FROM annotations WHERE session_id=?"
    params: List[Any] = [SESSION_ID]
    if args.target_type:
        q += " AND target_type=?"
        params.append(args.target_type)
    if args.status:
        q += " AND status=?"
        params.append(args.status)
    q += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    params.extend([args.limit, args.offset])
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(q, params).fetchall()
    items = [
        {
            "session_id": r[0],
            "target_type": r[1],
            "target_id": r[2],
            "status": r[3],
            "human_priority": r[4],
            "ai_priority": r[5],
            "comment": r[6],
            "updated_at": r[7],
        }
        for r in rows
    ]
    return {"items": items, "count": len(items)}


def _load_status_map(exclude_statuses: Optional[Iterable[str]], target_type: str = "chunk") -> Dict[str, str]:
    if not exclude_statuses:
        return {}
    statuses = [s for s in exclude_statuses if s]
    if not statuses:
        return {}
    placeholders = ",".join("?" for _ in statuses)
    q = (
        "SELECT target_id, status FROM annotations "
        f"WHERE session_id=? AND target_type=? AND status IN ({placeholders})"
    )
    params: List[Any] = [SESSION_ID, target_type, *statuses]
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(q, params).fetchall()
    return {row[0]: row[1] for row in rows}


# --- Weaviate connector (optional) ---

class WeaviateSource:
    def __init__(self) -> None:
        self.client = None
        self.collection = None

    def connect(self) -> None:
        if not USE_WEAVIATE or weaviate is None:
            return
        try:
            parsed = requests.utils.urlparse(WEAVIATE_URL)
            http_host = parsed.hostname or "weaviate"
            http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            http_secure = parsed.scheme == "https"
            self.client = weaviate.connect_to_custom(
                http_host=http_host,
                http_port=http_port,
                http_secure=http_secure,
                grpc_host=WEAVIATE_GRPC_HOST,
                grpc_port=WEAVIATE_GRPC_PORT,
                grpc_secure=WEAVIATE_GRPC_SECURE,
                skip_init_checks=True,
            )
            self.collection = self.client.collections.get("CodeChunk")
        except Exception:
            self.client = None
            self.collection = None

    def close(self) -> None:
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass

    def fetch_by_repo(self, repo: Optional[str]) -> List[Dict[str, Any]]:
        if not self.collection:
            return []
        props = [
            "chunk_id",
            "repo",
            "path",
            "language",
            "node_type",
            "parent_id",
            "depth",
            "start_line",
            "end_line",
            "start_byte",
            "end_byte",
            "token_estimate",
            "fingerprint",
            "raw_text",
            "normalized_text",
        ]
        try:
            if repo:
                resp = self.collection.query.fetch_objects(
                    limit=WEAVIATE_FETCH_LIMIT,
                    return_properties=props,
                    filters=Filter.by_property("repo").equal(repo),
                )
            else:
                resp = self.collection.query.fetch_objects(
                    limit=WEAVIATE_FETCH_LIMIT,
                    return_properties=props,
                )
        except Exception:
            return []
        objs = getattr(resp, "objects", None) or []
        return [o.properties for o in objs]

    def fetch_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        if not self.collection:
            return None
        props = [
            "chunk_id",
            "repo",
            "path",
            "language",
            "node_type",
            "start_line",
            "end_line",
            "token_estimate",
            "fingerprint",
            "raw_text",
            "normalized_text",
        ]
        try:
            resp = self.collection.query.fetch_objects(
                limit=1,
                return_properties=props,
                filters=Filter.by_property("chunk_id").equal(chunk_id),
            )
        except Exception:
            return None
        objs = getattr(resp, "objects", None) or []
        if not objs:
            return None
        return objs[0].properties


WEAVIATE = WeaviateSource()


def init() -> None:
    init_db()
    WEAVIATE.connect()


def close() -> None:
    WEAVIATE.close()


# --- Chunk access ---

def _chunk_summary(obj: Dict[str, Any], dup_map: Dict[str, int]) -> ChunkSummary:
    start_line = int(obj.get("start_line", 0) or 0)
    end_line = int(obj.get("end_line", 0) or 0)
    line_count = max(0, end_line - start_line + 1)
    fp = obj.get("fingerprint", "")
    dup_count = dup_map.get(fp, 1)
    return ChunkSummary(
        chunk_id=obj.get("chunk_id", ""),
        repo=obj.get("repo", ""),
        path=obj.get("path", ""),
        language=obj.get("language", ""),
        node_type=obj.get("node_type", ""),
        start_line=start_line,
        end_line=end_line,
        line_count=line_count,
        token_estimate=int(obj.get("token_estimate", 0) or 0),
        fingerprint=fp,
        dup_count=dup_count,
    )


def _matches_search(
    obj: Dict[str, Any],
    args: SearchParams,
    dup_map: Dict[str, int],
    status_map: Optional[Dict[str, str]] = None,
) -> bool:
    if args.repo and obj.get("repo") != args.repo:
        return False
    if args.path_contains and args.path_contains not in (obj.get("path") or ""):
        return False
    if args.language and obj.get("language") != args.language:
        return False
    if args.node_type and obj.get("node_type") != args.node_type:
        return False
    if args.fingerprint and obj.get("fingerprint") != args.fingerprint:
        return False
    if args.min_tokens is not None and int(obj.get("token_estimate", 0) or 0) < args.min_tokens:
        return False
    if args.max_tokens is not None and int(obj.get("token_estimate", 0) or 0) > args.max_tokens:
        return False
    if args.text_contains:
        raw = obj.get("raw_text") or ""
        if args.text_contains.lower() not in raw.lower():
            return False
    if args.normalized_contains:
        norm = obj.get("normalized_text") or ""
        if args.normalized_contains.lower() not in norm.lower():
            return False
    start_line = int(obj.get("start_line", 0) or 0)
    end_line = int(obj.get("end_line", 0) or 0)
    line_count = max(0, end_line - start_line + 1)
    if args.min_lines is not None and line_count < args.min_lines:
        return False
    if args.max_lines is not None and line_count > args.max_lines:
        return False
    if args.min_dup_count is not None:
        fp = obj.get("fingerprint", "")
        if dup_map.get(fp, 1) < args.min_dup_count:
            return False
    if args.max_dup_count is not None:
        fp = obj.get("fingerprint", "")
        if dup_map.get(fp, 1) > args.max_dup_count:
            return False
    if status_map:
        chunk_id = obj.get("chunk_id", "")
        if chunk_id and chunk_id in status_map:
            return False
    return True


def _iter_chunks(repo: Optional[str]) -> Iterable[Dict[str, Any]]:
    if WEAVIATE.collection:
        for obj in WEAVIATE.fetch_by_repo(repo):
            yield obj
        return
    for obj in _iter_jsonl(CHUNKS_PATH):
        yield obj


def _find_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    if WEAVIATE.collection:
        hit = WEAVIATE.fetch_by_chunk_id(chunk_id)
        if hit:
            return hit
    for obj in _iter_jsonl(CHUNKS_PATH):
        if obj.get("chunk_id") == chunk_id:
            return obj
    return None


def search_chunks(args: SearchParams) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    dup_map = dup_counts()
    status_map = _load_status_map(args.exclude_statuses, "chunk")

    if args.sort_by:
        all_items: List[ChunkSummary] = []
        for obj in _iter_chunks(args.repo):
            if not _matches_search(obj, args, dup_map, status_map):
                continue
            all_items.append(_chunk_summary(obj, dup_map))
        key = args.sort_by
        reverse = args.sort_order.lower() != "asc"
        if key == "dup_count":
            all_items.sort(key=lambda x: x.dup_count, reverse=reverse)
        elif key == "token_estimate":
            all_items.sort(key=lambda x: x.token_estimate, reverse=reverse)
        elif key == "line_count":
            all_items.sort(key=lambda x: x.line_count, reverse=reverse)
        elif key == "path":
            all_items.sort(key=lambda x: x.path, reverse=reverse)
        else:
            all_items.sort(key=lambda x: x.path, reverse=reverse)
        sliced = all_items[args.offset: args.offset + args.limit]
        items = [c.__dict__ for c in sliced]
        return {"items": items, "count": len(items), "offset": args.offset}

    skipped = 0
    for obj in _iter_chunks(args.repo):
        if not _matches_search(obj, args, dup_map, status_map):
            continue
        if skipped < args.offset:
            skipped += 1
            continue
        items.append(_chunk_summary(obj, dup_map).__dict__)
        if len(items) >= args.limit:
            break
    return {"items": items, "count": len(items), "offset": args.offset}


def get_chunk_text(chunk_id: str, max_length: int = DEFAULT_MAX_TEXT_LEN) -> Optional[Dict[str, Any]]:
    obj = _find_chunk_by_id(chunk_id)
    if not obj:
        return None
    raw = obj.get("raw_text", "")
    return {
        "chunk_id": obj.get("chunk_id", ""),
        "repo": obj.get("repo", ""),
        "path": obj.get("path", ""),
        "language": obj.get("language", ""),
        "node_type": obj.get("node_type", ""),
        "start_line": obj.get("start_line", 0),
        "end_line": obj.get("end_line", 0),
        "raw_text": _truncate(raw, max_length),
        "raw_text_truncated": len(raw) > max_length,
    }


# --- Duplicates ---

def list_dup_groups(args: DupListParams) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    skipped = 0
    for g in _iter_jsonl(DUPS_PATH):
        count = int(g.get("count", 0) or 0)
        if count < args.min_count:
            continue
        if skipped < args.offset:
            skipped += 1
            continue
        items.append(
            {
                "fingerprint": g.get("fingerprint", ""),
                "count": count,
                "chunk_ids": (g.get("chunk_ids") or [])[: args.max_chunk_ids],
            }
        )
        if len(items) >= args.limit:
            break
    return {"items": items, "count": len(items), "offset": args.offset}


def get_dup_group(args: DupGetParams) -> Optional[Dict[str, Any]]:
    for g in _iter_jsonl(DUPS_PATH):
        if g.get("fingerprint") != args.fingerprint:
            continue
        out = {
            "fingerprint": g.get("fingerprint", ""),
            "count": int(g.get("count", 0) or 0),
            "chunk_ids": g.get("chunk_ids") or [],
        }
        if args.include_chunks:
            chunks: List[Dict[str, Any]] = []
            for cid in out["chunk_ids"]:
                obj = _find_chunk_by_id(cid)
                if not obj:
                    continue
                chunks.append(
                    {
                        **_chunk_summary(obj, dup_counts()).__dict__,
                        "raw_text": _truncate(obj.get("raw_text", ""), args.chunk_text_max),
                        "raw_text_truncated": len(obj.get("raw_text", "")) > args.chunk_text_max,
                    }
                )
            out["chunks"] = chunks
        return out
    return None


def _base_group_search(args: SearchParams) -> SearchParams:
    base = SearchParams(**args.__dict__)
    base.min_dup_count = None
    base.max_dup_count = None
    return base


def list_dup_groups_filtered(args: SearchParams, min_count: int = 2, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    sample_ids: Dict[str, List[str]] = {}
    base = _base_group_search(args)
    status_map = _load_status_map(base.exclude_statuses, "chunk")
    for obj in _iter_chunks(base.repo):
        if not _matches_search(obj, base, dup_counts(), status_map):
            continue
        fp = obj.get("fingerprint")
        if not fp:
            continue
        counts[fp] = counts.get(fp, 0) + 1
        if len(sample_ids.get(fp, [])) < 5:
            cid = obj.get("chunk_id")
            if cid:
                sample_ids.setdefault(fp, []).append(cid)
    items = [
        {"fingerprint": fp, "count": cnt, "chunk_ids": sample_ids.get(fp, [])}
        for fp, cnt in counts.items()
        if cnt >= min_count
    ]
    items.sort(key=lambda x: (x["count"], x["fingerprint"]), reverse=True)
    sliced = items[offset: offset + limit]
    return {"items": sliced, "count": len(sliced), "offset": offset}


def get_dup_group_filtered(args: DupGetParams, search: SearchParams) -> Optional[Dict[str, Any]]:
    base = _base_group_search(search)
    status_map = _load_status_map(base.exclude_statuses, "chunk")
    fp = args.fingerprint
    chunks: List[Dict[str, Any]] = []
    count = 0
    for obj in _iter_chunks(base.repo):
        if obj.get("fingerprint") != fp:
            continue
        if not _matches_search(obj, base, dup_counts(), status_map):
            continue
        count += 1
        chunks.append(
            {
                **_chunk_summary(obj, dup_counts()).__dict__,
                "raw_text": _truncate(obj.get("raw_text", ""), args.chunk_text_max),
                "raw_text_truncated": len(obj.get("raw_text", "")) > args.chunk_text_max,
            }
        )
    if count == 0:
        return None
    return {"fingerprint": fp, "count": count, "chunk_ids": [c["chunk_id"] for c in chunks], "chunks": chunks}
