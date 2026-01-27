from __future__ import annotations

import os
import re
import time
import json
import math
import urllib.parse
import hashlib
import pathspec
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import orjson
import requests
from tqdm import tqdm

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from tree_sitter_languages import get_parser


SUPPORTED_EXTS: Dict[str, str] = {
    # C-like / braces
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".cs": "c_sharp",
    ".php": "php",
    # Others
    ".py": "python",
    ".rb": "ruby",
    # YAML-like
    ".yml": "yaml",
    ".yaml": "yaml",
}

def load_gitignore_spec(root: Path) -> Optional[pathspec.PathSpec]:
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return None
    try:
        lines = gitignore_path.read_text().splitlines()
    except OSError:
        return None
    return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, lines)


CHUNK_NODE_TYPES_BY_LANG: Dict[str, Tuple[str, ...]] = {
    "c": ("function_definition", "compound_statement", "for_statement", "while_statement", "if_statement", "switch_statement", "do_statement"),
    "cpp": ("function_definition", "compound_statement", "for_statement", "while_statement", "if_statement", "switch_statement", "try_statement", "do_statement"),
    "java": ("method_declaration", "constructor_declaration", "block", "for_statement", "while_statement", "if_statement", "switch_expression", "switch_statement", "try_statement", "do_statement"),
    "javascript": ("function_declaration", "method_definition", "function", "statement_block", "for_statement", "while_statement", "if_statement", "switch_statement", "try_statement", "do_statement"),
    "typescript": ("function_declaration", "method_definition", "function", "statement_block", "for_statement", "while_statement", "if_statement", "switch_statement", "try_statement", "do_statement"),
    "tsx": ("function_declaration", "method_definition", "function", "statement_block", "for_statement", "while_statement", "if_statement", "switch_statement", "try_statement", "do_statement"),
    "go": ("function_declaration", "method_declaration", "block", "for_statement", "if_statement", "switch_statement", "type_switch_statement"),
    "rust": ("function_item", "block", "for_expression", "while_expression", "if_expression", "match_expression", "loop_expression"),
    "c_sharp": ("method_declaration", "constructor_declaration", "block", "for_statement", "foreach_statement", "while_statement", "if_statement", "switch_statement", "try_statement", "do_statement"),
    "php": ("function_definition", "method_declaration", "compound_statement", "for_statement", "foreach_statement", "while_statement", "if_statement", "switch_statement", "try_statement", "do_statement"),
    "python": ("function_definition", "class_definition", "for_statement", "while_statement", "if_statement", "try_statement", "with_statement"),
    "ruby": ("method", "class", "module", "if", "while", "until", "for", "begin"),
    "yaml": ("block_mapping_pair", "block_sequence_item", "flow_pair"),
}

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUM_RE = re.compile(r"(?<![A-Za-z_])[0-9]+(\.[0-9]+)?(?![A-Za-z_])")
WS_RE = re.compile(r"\s+")
STRING_RE = re.compile(r"""(?x)
    ("([^"\\]|\\.)*")
  | ('([^'\\]|\\.)*')
""")

# Very rough comment stripping for common languages.
LINE_COMMENT_RE = re.compile(r"//.*?$|#.*?$", re.M)
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    repo: str
    path: str
    language: str
    node_type: str
    parent_id: Optional[str]
    depth: int
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    raw_text: str
    normalized_text: str
    token_estimate: int
    fingerprint: str


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip() not in ("0", "false", "False", "no", "NO")


def strip_comments(text: str) -> str:
    text = BLOCK_COMMENT_RE.sub("", text)
    text = LINE_COMMENT_RE.sub("", text)
    return text


def normalize_code(text: str) -> str:
    text = strip_comments(text)

    # Replace strings first to avoid turning identifiers inside strings.
    text = STRING_RE.sub(" STR ", text)
    text = NUM_RE.sub(" NUM ", text)

    # Replace identifiers, but keep a few keywords to preserve structure a bit.
    keywords = {
        "for", "while", "if", "else", "switch", "case", "break", "continue", "return", "try", "catch", "finally",
        "class", "struct", "enum", "def", "fn", "func", "function", "import", "from", "package", "public", "private",
        "protected", "static", "const", "let", "var", "new", "throw", "await", "async", "match", "with",
    }

    def repl(m: re.Match[str]) -> str:
        s = m.group(0)
        if s in keywords:
            return s
        return " ID "

    text = IDENT_RE.sub(repl, text)
    text = WS_RE.sub(" ", text).strip()
    return text


def token_estimate(text: str) -> int:
    # Cheap estimate: whitespace tokens.
    if not text:
        return 0
    return max(1, len(text.split()))


def fingerprint_text(norm_text: str) -> str:
    # Stable fingerprint for exact/near-exact duplicates on normalized text.
    h = hashlib.blake2b(norm_text.encode("utf-8"), digest_size=16).hexdigest()
    return h


def fetch_chunks_from_weaviate(collection: weaviate.WeaviateClient, repo: str) -> List[Chunk]:
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
    resp = collection.query.fetch_objects(
        limit=10_000,
        return_properties=props,
        filters=Filter.by_property("repo").equal(repo),
    )
    objs = getattr(resp, "objects", None) or []
    results: List[Chunk] = []
    for obj in objs:
        p = obj.properties
        results.append(
            Chunk(
                chunk_id=p.get("chunk_id", ""),
                repo=p.get("repo", ""),
                path=p.get("path", ""),
                language=p.get("language", ""),
                node_type=p.get("node_type", ""),
                parent_id=p.get("parent_id"),
                depth=int(p.get("depth", 0) or 0),
                start_byte=int(p.get("start_byte", 0) or 0),
                end_byte=int(p.get("end_byte", 0) or 0),
                start_line=int(p.get("start_line", 0) or 0),
                end_line=int(p.get("end_line", 0) or 0),
                raw_text=p.get("raw_text", ""),
                normalized_text=p.get("normalized_text", ""),
                token_estimate=int(p.get("token_estimate", 0) or 0),
                fingerprint=p.get("fingerprint", ""),
            )
        )
    return results

def iter_files(root: Path, max_files: int) -> Iterable[Path]:
    ignore_spec = load_gitignore_spec(root)
    n = 0
    for p in root.rglob("*"):
        if n >= max_files:
            return
        if not p.is_file():
            continue
        rel_path = p.relative_to(root).as_posix()
        if ignore_spec and ignore_spec.match_file(rel_path):
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        n += 1
        yield p


def get_language_for_path(path: Path) -> Optional[str]:
    return SUPPORTED_EXTS.get(path.suffix.lower())


def bytes_to_line_map(src: bytes) -> List[int]:
    # map byte offset to line number by scanning newlines
    # returns list of newline byte positions
    newlines = [-1]
    for i, b in enumerate(src):
        if b == 10:  # \n
            newlines.append(i)
    return newlines


def byte_to_line(newlines: List[int], byte_offset: int) -> int:
    # binary search
    lo, hi = 0, len(newlines) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if newlines[mid] < byte_offset:
            lo = mid + 1
        else:
            hi = mid - 1
    # hi points to last newline before offset
    return max(1, hi + 1)


def extract_chunks_from_tree(
    src: bytes,
    path: str,
    repo: str,
    language: str,
    min_chars: int,
    max_chars: int,
) -> List[Chunk]:
    parser = get_parser(language)
    tree = parser.parse(src)
    root = tree.root_node
    newlines = bytes_to_line_map(src)

    wanted_types = set(CHUNK_NODE_TYPES_BY_LANG.get(language, ()))

    chunks: List[Chunk] = []

    def walk(node, parent_id: Optional[str], depth: int) -> None:
        node_type = node.type
        is_wanted = node_type in wanted_types
        if is_wanted:
            start_b = node.start_byte
            end_b = node.end_byte
            raw = src[start_b:end_b].decode("utf-8", errors="replace")
            if min_chars <= len(raw) <= max_chars:
                norm = normalize_code(raw)
                fp = fingerprint_text(norm)
                cid = hashlib.blake2b(f"{path}:{start_b}:{end_b}:{fp}".encode("utf-8"), digest_size=16).hexdigest()
                s_line = byte_to_line(newlines, start_b)
                e_line = byte_to_line(newlines, max(start_b, end_b - 1))
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        repo=repo,
                        path=path,
                        language=language,
                        node_type=node_type,
                        parent_id=parent_id,
                        depth=depth,
                        start_byte=start_b,
                        end_byte=end_b,
                        start_line=s_line,
                        end_line=e_line,
                        raw_text=raw,
                        normalized_text=norm,
                        token_estimate=token_estimate(norm),
                        fingerprint=fp,
                    )
                )
            # This chunk becomes parent for inner chunks
            next_parent = hashlib.blake2b(f"{path}:{node.start_byte}:{node.end_byte}".encode("utf-8"), digest_size=16).hexdigest()
        else:
            next_parent = parent_id

        for child in node.children:
            walk(child, next_parent, depth + 1)

    walk(root, None, 0)
    return chunks


def ensure_weaviate_schema(client: weaviate.WeaviateClient, vector_dim: int) -> None:
    if client.collections.exists("CodeChunk"):
        return

    client.collections.create(
        name="CodeChunk",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="repo", data_type=DataType.TEXT),
            Property(name="path", data_type=DataType.TEXT),
            Property(name="language", data_type=DataType.TEXT),
            Property(name="node_type", data_type=DataType.TEXT),
            Property(name="parent_id", data_type=DataType.TEXT),
            Property(name="depth", data_type=DataType.INT),
            Property(name="start_line", data_type=DataType.INT),
            Property(name="end_line", data_type=DataType.INT),
            Property(name="start_byte", data_type=DataType.INT),
            Property(name="end_byte", data_type=DataType.INT),
            Property(name="token_estimate", data_type=DataType.INT),
            Property(name="fingerprint", data_type=DataType.TEXT),
            Property(name="raw_text", data_type=DataType.TEXT),
            Property(name="normalized_text", data_type=DataType.TEXT),
        ],
    )


def wait_for_http(url: str, timeout_s: int = 120) -> None:
    t0 = time.time()
    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code < 500:
                return
        except requests.RequestException:
            pass
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"Timed out waiting for {url}")
        time.sleep(1)


def embed_texts(base_url: str, model: str, texts: List[str]) -> List[List[float]]:
    # OpenAI-compatible embeddings endpoint: POST /v1/embeddings
    url = base_url.rstrip("/") + "/embeddings"
    payload = {"model": model, "input": texts}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Expect: {"data":[{"embedding":[...], "index":0}, ...]}
    items = sorted(data["data"], key=lambda x: x["index"])
    return [it["embedding"] for it in items]


def chunks_to_jsonl(path: Path, chunks: List[Chunk]) -> None:
    with path.open("wb") as f:
        for c in chunks:
            obj = {
                "chunk_id": c.chunk_id,
                "repo": c.repo,
                "path": c.path,
                "language": c.language,
                "node_type": c.node_type,
                "parent_id": c.parent_id,
                "depth": c.depth,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "start_byte": c.start_byte,
                "end_byte": c.end_byte,
                "token_estimate": c.token_estimate,
                "fingerprint": c.fingerprint,
                "raw_text": c.raw_text,
                "normalized_text": c.normalized_text,
            }
            f.write(orjson.dumps(obj))
            f.write(b"\n")


def write_stats(output_dir: Path, chunks: List[Chunk], files_scanned: int, started_at: float) -> Dict[str, Any]:
    by_lang: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    tok_bins = {"<=50": 0, "51-150": 0, "151-400": 0, "401-1000": 0, ">1000": 0}

    for c in chunks:
        by_lang[c.language] = by_lang.get(c.language, 0) + 1
        by_type[c.node_type] = by_type.get(c.node_type, 0) + 1
        t = c.token_estimate
        if t <= 50:
            tok_bins["<=50"] += 1
        elif t <= 150:
            tok_bins["51-150"] += 1
        elif t <= 400:
            tok_bins["151-400"] += 1
        elif t <= 1000:
            tok_bins["401-1000"] += 1
        else:
            tok_bins[">1000"] += 1

    # exact dup candidates by fingerprint
    fp_map: Dict[str, List[str]] = {}
    for c in chunks:
        fp_map.setdefault(c.fingerprint, []).append(c.chunk_id)

    dup_groups = [{"fingerprint": fp, "count": len(ids), "chunk_ids": ids[:50]} for fp, ids in fp_map.items() if len(ids) >= 2]
    dup_groups.sort(key=lambda x: x["count"], reverse=True)

    duration_s = time.time() - started_at
    stats = {
        "files_scanned": files_scanned,
        "chunks_extracted": len(chunks),
        "by_language": dict(sorted(by_lang.items(), key=lambda x: x[1], reverse=True)),
        "by_node_type": dict(sorted(by_type.items(), key=lambda x: x[1], reverse=True)),
        "token_bins": tok_bins,
        "exact_duplicate_fingerprint_groups_top": dup_groups[:100],
        "duration_seconds": round(duration_s, 3),
    }

    (output_dir / "stats.json").write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    # dump exact dup candidates as jsonl (one group per line)
    with (output_dir / "candidates_exact_dups.jsonl").open("wb") as f:
        for g in dup_groups:
            f.write(orjson.dumps(g))
            f.write(b"\n")

    return stats


def write_report_html(output_dir: Path, stats: Dict[str, Any]) -> None:
    def html_escape(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    # Prepare lookup for top duplicate chunks to display code samples
    dup_groups = stats.get("exact_duplicate_fingerprint_groups_top", [])[:20]
    wanted_ids = {cid for g in dup_groups for cid in g.get("chunk_ids", [])[:5]}
    chunk_samples: Dict[str, Dict[str, Any]] = {}
    chunks_path = output_dir / "chunks.jsonl"
    if chunks_path.exists() and wanted_ids:
        with chunks_path.open("rb") as f:
            for line in f:
                obj = orjson.loads(line)
                cid = obj.get("chunk_id")
                if cid in wanted_ids:
                    # keep a short preview to avoid huge HTML
                    raw = obj.get("raw_text", "")
                    preview = raw if len(raw) <= 800 else raw[:800] + "\n...[truncated]..."
                    chunk_samples[cid] = {
                        "path": obj.get("path", ""),
                        "start_line": obj.get("start_line"),
                        "end_line": obj.get("end_line"),
                        "language": obj.get("language", ""),
                        "raw_preview": preview,
                    }
    def row_dict(d: Dict[str, Any]) -> str:
        rows = []
        for k, v in d.items():
            rows.append(f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>")
        return "\n".join(rows)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Code Duplicate Analyzer Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; background:#f5f7fb; color:#111827; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }}
    .card {{ border: 1px solid #dce2ec; background:#ffffff; border-radius: 12px; padding: 16px; box-shadow: 0 8px 22px rgba(17, 24, 39, 0.06); }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    td, th {{ border-bottom: 1px solid #e6ebf2; padding: 6px 8px; }}
    th {{ text-align: left; color:#0f172a; }}
    .muted {{ color: #475569; }}
    code {{ background: #e8ecf4; padding: 2px 6px; border-radius: 6px; color:#0f172a; }}
    .pathwrap {{ word-break: break-all; white-space: normal; }}
    .samples {{ table-layout: fixed; }}
    .samples th:nth-child(1), .samples td:nth-child(1) {{ width: 18%; }}
    .samples th:nth-child(2), .samples td:nth-child(2) {{ width: 22%; }}
    .samples th:nth-child(3), .samples td:nth-child(3) {{ width: 60%; }}
    pre {{ background:#0f172a; color:#e2e8f0; padding:10px; border-radius:8px; max-height:360px; overflow:auto; }}
  </style>
</head>
<body>
  <h1>Report</h1>
  <p class="muted">
    Files scanned: <b>{stats["files_scanned"]}</b> ·
    Chunks extracted: <b>{stats["chunks_extracted"]}</b> ·
    Duration: <b>{stats["duration_seconds"]}s</b>
  </p>

  <div class="grid">
    <div class="card">
      <h2>By language</h2>
      <table>
        <tr><th>Language</th><th style="text-align:right">Chunks</th></tr>
        {row_dict(stats["by_language"])}
      </table>
    </div>

    <div class="card">
      <h2>By node type</h2>
      <table>
        <tr><th>Node type</th><th style="text-align:right">Chunks</th></tr>
        {row_dict(stats["by_node_type"])}
      </table>
    </div>

    <div class="card">
      <h2>Token bins (normalized)</h2>
      <table>
        <tr><th>Bin</th><th style="text-align:right">Chunks</th></tr>
        {row_dict(stats["token_bins"])}
      </table>
    </div>

    <div class="card">
      <h2>Exact-dup candidates (top 20 groups)</h2>
      <p class="muted">Based on fingerprint of normalized text. Full list: <code>candidates_exact_dups.jsonl</code></p>
      <table>
        <tr><th style='text-align:right'>Group size</th><th>Fingerprint</th><th>Chunk IDs (first 5)</th></tr>
        {''.join(
            "<tr><td style='text-align:right'>{count}</td>"
            "<td><code>{fingerprint}</code></td>"
            "<td><code>{ids}</code></td></tr>".format(
                count=g["count"],
                fingerprint=g["fingerprint"],
                ids=", ".join(g["chunk_ids"][:5]),
            )
            for g in stats["exact_duplicate_fingerprint_groups_top"][:20]
        )}
      </table>
      <p class="muted">Use <code>chunk_ids</code> to lookup raw code in <code>chunks.jsonl</code> (fields: path, start_line, end_line, raw_text).</p>
    </div>

    <div class="card">
      <h2>Duplicate samples (top groups)</h2>
      <p class="muted">Showing up to 5 chunks per group with raw code preview (truncated at 800 chars).</p>
      {''.join(
        "<div style='margin-bottom:12px'>"
        f"<div class='muted' style='margin-bottom:6px'>Fingerprint: <code>{g['fingerprint']}</code> · size {g['count']}</div>"
        "<table class='samples'><tr><th>chunk_id</th><th>path:lines</th><th>code preview</th></tr>"
        + ''.join(
            "<tr><td><code>{cid}</code></td>"
            "<td class='pathwrap'><code>{path}:{start}-{end}</code></td>"
            "<td><pre style='white-space:pre-wrap'>{code}</pre></td></tr>".format(
                cid=cid,
                path=html_escape(chunk_samples.get(cid, {}).get("path", "")),
                start=chunk_samples.get(cid, {}).get("start_line", ""),
                end=chunk_samples.get(cid, {}).get("end_line", ""),
                code=html_escape(chunk_samples.get(cid, {}).get("raw_preview", "[chunk not found in chunks.jsonl]")),
            )
            for cid in g.get("chunk_ids", [])[:5]
        )
        + "</table></div>"
        for g in dup_groups
      )}
    </div>
  </div>

  <h2>Output files</h2>
  <ul>
    <li><code>stats.json</code> — summary</li>
    <li><code>chunks.jsonl</code> — extracted chunks (includes raw + normalized text)</li>
    <li><code>candidates_exact_dups.jsonl</code> — local exact-duplicate candidates</li>
  </ul>
</body>
</html>
"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def batch(iterable: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def main() -> None:
    input_dir = Path(os.getenv("INPUT_DIR", "/data/input")).resolve()
    output_dir = Path(os.getenv("OUTPUT_DIR", "/data/output")).resolve()
    weaviate_url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
    embedding_model = os.getenv("EMBEDDING_MODEL", "Octen/Octen-Embedding-4B")
    weaviate_grpc_host = os.getenv("WEAVIATE_GRPC_HOST", "weaviate")
    weaviate_grpc_port = env_int("WEAVIATE_GRPC_PORT", 50051)
    weaviate_grpc_secure = env_bool("WEAVIATE_GRPC_SECURE", False)

    use_embeddings = env_bool("USE_EMBEDDINGS", True)
    min_chunk_chars = env_int("MIN_CHUNK_CHARS", 120)
    max_chunk_chars = env_int("MAX_CHUNK_CHARS", 12000)
    max_files = env_int("MAX_FILES", 20000)

    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    if not input_dir.exists():
        raise RuntimeError(f"INPUT_DIR does not exist: {input_dir}")

    # Wait for dependencies
    wait_for_http(f"{weaviate_url}/v1/.well-known/ready")
    if use_embeddings:
        wait_for_http(vllm_base_url.rstrip("/") + "/models")

    # Connect Weaviate (HTTP + gRPC are both required by the v4 client)
    parsed = urllib.parse.urlparse(weaviate_url)
    http_host = parsed.hostname or "weaviate"
    http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    http_secure = parsed.scheme == "https"
    client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=http_secure,
        grpc_host=weaviate_grpc_host,
        grpc_port=weaviate_grpc_port,
        grpc_secure=weaviate_grpc_secure,
        skip_init_checks=True,
    )
    try:
        ensure_weaviate_schema(client, vector_dim=2560)
        collection = client.collections.get("CodeChunk")

        repo = input_dir.name
        all_chunks: List[Chunk] = []

        files = list(iter_files(input_dir, max_files=max_files))
        for fpath in tqdm(files, desc="Parsing"):
            lang = get_language_for_path(fpath)
            if lang is None:
                continue
            try:
                src = fpath.read_bytes()
            except OSError:
                continue

            rel_path = str(fpath.relative_to(input_dir))
            try:
                chunks = extract_chunks_from_tree(
                    src=src,
                    path=rel_path,
                    repo=repo,
                    language=lang,
                    min_chars=min_chunk_chars,
                    max_chars=max_chunk_chars,
                )
            except Exception:
                # Parser might fail on some files; keep going.
                continue

            all_chunks.extend(chunks)

        if not all_chunks:
            print("No chunks extracted. Done.")
            return

        print(f"Upserting {len(all_chunks)} chunks into Weaviate...")
        BATCH_SIZE = 48  # default batch size; adjust if your GPU/latency needs

        if not use_embeddings:
            # Upsert without vectors
            with collection.batch.dynamic() as b:
                for c in tqdm(all_chunks, desc="Weaviate upsert"):
                    b.add_object(
                        properties={
                            "chunk_id": c.chunk_id,
                            "repo": c.repo,
                            "path": c.path,
                            "language": c.language,
                            "node_type": c.node_type,
                            "parent_id": c.parent_id or "",
                            "depth": c.depth,
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                            "start_byte": c.start_byte,
                            "end_byte": c.end_byte,
                            "token_estimate": c.token_estimate,
                            "fingerprint": c.fingerprint,
                            "raw_text": c.raw_text,
                            "normalized_text": c.normalized_text,
                        },
                        uuid=c.chunk_id,
                    )
            stored_chunks = fetch_chunks_from_weaviate(collection, repo)
            chunks_to_jsonl(output_dir / "chunks.jsonl", stored_chunks)
            stats = write_stats(output_dir, stored_chunks, files_scanned=len(files), started_at=started)
            write_report_html(output_dir, stats)
            print("Done (without embeddings).")
            return

        # With embeddings
        for group in tqdm(list(batch(all_chunks, BATCH_SIZE)), desc="Embedding+upsert batches"):
            texts = [c.normalized_text for c in group]
            vectors = embed_texts(vllm_base_url, embedding_model, texts)

            with collection.batch.dynamic() as b:
                for c, v in zip(group, vectors):
                    b.add_object(
                        properties={
                            "chunk_id": c.chunk_id,
                            "repo": c.repo,
                            "path": c.path,
                            "language": c.language,
                            "node_type": c.node_type,
                            "parent_id": c.parent_id or "",
                            "depth": c.depth,
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                            "start_byte": c.start_byte,
                            "end_byte": c.end_byte,
                            "token_estimate": c.token_estimate,
                            "fingerprint": c.fingerprint,
                            "raw_text": c.raw_text,
                            "normalized_text": c.normalized_text,
                        },
                        vector=v,
                        uuid=c.chunk_id,
                    )

        # After successful upsert, read back from Weaviate so report reflects stored records
        stored_chunks = fetch_chunks_from_weaviate(collection, repo)
        chunks_to_jsonl(output_dir / "chunks.jsonl", stored_chunks)
        stats = write_stats(output_dir, stored_chunks, files_scanned=len(files), started_at=started)
        write_report_html(output_dir, stats)

        print("Done. See output/report.html and output/stats.json")
    finally:
        client.close()


if __name__ == "__main__":
    main()
