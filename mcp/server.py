from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import orjson
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from core import (
    DEFAULT_MAX_TEXT_LEN,
    ALLOW_HUMAN_PRIORITY_UPDATE,
    OUTPUT_DIR,
    CHUNKS_PATH,
    DUPS_PATH,
    DB_PATH,
    SESSION_ID,
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
)


APP_NAME = "code-dup-mcp"


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Any = None
    method: str
    params: Optional[Dict[str, Any]] = None


def _rpc_result(rpc_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _rpc_error(rpc_id: Any, code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": err}


class SearchChunksArgs(BaseModel):
    repo: Optional[str] = None
    path_contains: Optional[str] = None
    language: Optional[str] = None
    node_type: Optional[str] = None
    fingerprint: Optional[str] = None
    text_contains: Optional[str] = None
    normalized_contains: Optional[str] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    min_lines: Optional[int] = None
    max_lines: Optional[int] = None
    min_dup_count: Optional[int] = None
    max_dup_count: Optional[int] = None
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    limit: int = 50
    offset: int = 0


class GetChunkTextArgs(BaseModel):
    chunk_id: str
    max_length: int = DEFAULT_MAX_TEXT_LEN


class ListDupGroupsArgs(BaseModel):
    min_count: int = 2
    limit: int = 50
    offset: int = 0
    max_chunk_ids: int = 50


class GetDupGroupArgs(BaseModel):
    fingerprint: str
    include_chunks: bool = False
    chunk_text_max: int = DEFAULT_MAX_TEXT_LEN


class SetAnnotationArgs(BaseModel):
    target_type: str
    target_id: str
    status: Optional[str] = None
    priority: Optional[int] = None
    ai_priority: Optional[int] = None
    human_priority: Optional[int] = None
    comment: Optional[str] = None


class GetAnnotationArgs(BaseModel):
    target_type: str
    target_id: str


class ListAnnotationsArgs(BaseModel):
    target_type: Optional[str] = None
    status: Optional[str] = None
    limit: int = 100
    offset: int = 0


@asynccontextmanager
async def lifespan(_: FastAPI):
    init()
    try:
        yield
    finally:
        close()


app = FastAPI(title=APP_NAME, lifespan=lifespan)


TOOLS = [
    {
        "name": "search_chunks",
        "description": "Search code chunks by repo/path/language/node_type/fingerprint/text (substring match).",
        "input_schema": SearchChunksArgs.model_json_schema(),
    },
    {
        "name": "get_chunk_text",
        "description": "Get raw text for a chunk_id with optional truncation.",
        "input_schema": GetChunkTextArgs.model_json_schema(),
    },
    {
        "name": "list_duplicate_groups",
        "description": "List duplicate fingerprint groups (from candidates_exact_dups.jsonl).",
        "input_schema": ListDupGroupsArgs.model_json_schema(),
    },
    {
        "name": "get_duplicate_group",
        "description": "Get a duplicate group by fingerprint; optionally include chunk text.",
        "input_schema": GetDupGroupArgs.model_json_schema(),
    },
    {
        "name": "set_annotation",
        "description": "Upsert annotation for chunk or duplicate group (ai_priority writable; human_priority only if ALLOW_HUMAN_PRIORITY_UPDATE=1).",
        "input_schema": SetAnnotationArgs.model_json_schema(),
    },
    {
        "name": "get_annotation",
        "description": "Fetch annotation for a chunk or duplicate group.",
        "input_schema": GetAnnotationArgs.model_json_schema(),
    },
    {
        "name": "list_annotations",
        "description": "List annotations (filter by target_type/status).",
        "input_schema": ListAnnotationsArgs.model_json_schema(),
    },
]


def _mcp_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "inputSchema": t.get("input_schema", {}),
        }
        for t in TOOLS
    ]


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


@app.get("/.well-known/oauth-authorization-server")
def oauth_metadata(request: Request) -> Dict[str, Any]:
    base = str(request.base_url).rstrip("/")
    return {
        "issuer": base,
        "token_endpoint": f"{base}/oauth/token",
        "grant_types_supported": ["client_credentials"],
        "token_endpoint_auth_methods_supported": ["none"],
    }


@app.post("/oauth/token")
def oauth_token(request: Request) -> Dict[str, Any]:
    return {
        "access_token": "local-dev-token",
        "token_type": "bearer",
        "expires_in": 3600,
        "scope": "",
        "issuer": str(request.base_url).rstrip("/"),
    }


@app.get("/tools/list")
def list_tools() -> Dict[str, Any]:
    return {"tools": TOOLS}


@app.post("/tools/call")
def call_tool(call: ToolCall) -> Dict[str, Any]:
    name = call.name
    args = call.arguments or {}

    if name == "search_chunks":
        parsed = SearchChunksArgs(**args)
        data = search_chunks(SearchParams(**parsed.model_dump()))
    elif name == "get_chunk_text":
        parsed = GetChunkTextArgs(**args)
        data = get_chunk_text(parsed.chunk_id, parsed.max_length)
        if not data:
            raise HTTPException(status_code=404, detail="chunk_id not found")
    elif name == "list_duplicate_groups":
        parsed = ListDupGroupsArgs(**args)
        data = list_dup_groups(DupListParams(**parsed.model_dump()))
    elif name == "get_duplicate_group":
        parsed = GetDupGroupArgs(**args)
        data = get_dup_group(DupGetParams(**parsed.model_dump()))
        if not data:
            raise HTTPException(status_code=404, detail="fingerprint not found")
    elif name == "set_annotation":
        parsed = SetAnnotationArgs(**args)
        data = set_annotation(AnnotationSetParams(**parsed.model_dump()))
    elif name == "get_annotation":
        parsed = GetAnnotationArgs(**args)
        data = get_annotation(AnnotationGetParams(**parsed.model_dump()))
    elif name == "list_annotations":
        parsed = ListAnnotationsArgs(**args)
        data = list_annotations(AnnotationListParams(**parsed.model_dump()))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {name}")

    return {
        "content": [
            {
                "type": "text",
                "text": f"{name} ok",
            }
        ],
        "data": data,
    }


@app.post("/")
async def mcp_jsonrpc(req: Request) -> Any:
    raw = await req.body()
    try:
        payload = orjson.loads(raw) if raw else {}
    except Exception:
        payload = {}
    print(f"[mcp] request: {payload}")
    try:
        request = JsonRpcRequest(**payload)
    except Exception as exc:
        resp = _rpc_error(None, -32600, "Invalid Request", data=str(exc))
        print(f"[mcp] response: {resp}")
        return resp

    if request.jsonrpc != "2.0":
        resp = _rpc_error(request.id, -32600, "Invalid JSON-RPC version")
        print(f"[mcp] response: {resp}")
        return resp
    method = request.method
    params = request.params or {}
    if method.startswith("notifications/"):
        print(f"[mcp] notification: {method}")
        return Response(status_code=204)

    try:
        if method == "initialize":
            client_version = params.get("protocolVersion") or "2024-11-05"
            return _rpc_result(
                request.id,
                {
                    "protocolVersion": client_version,
                    "serverInfo": {"name": APP_NAME, "version": "0.1.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                },
            )
        if method == "tools/list":
            return _rpc_result(request.id, {"tools": _mcp_tools()})
        if method == "resources/list":
            return _rpc_result(request.id, {"resources": []})
        if method == "resources/templates/list":
            return _rpc_result(request.id, {"resourceTemplates": []})
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not name:
                return _rpc_error(request.id, -32602, "Missing tool name")
            data = call_tool(ToolCall(name=name, arguments=arguments))
            resp = _rpc_result(request.id, data)
            print(f"[mcp] response: {resp}")
            return resp
        if method == "ping":
            resp = _rpc_result(request.id, {"ok": True})
            print(f"[mcp] response: {resp}")
            return resp
    except HTTPException as exc:
        resp = _rpc_error(request.id, exc.status_code, str(exc.detail))
        print(f"[mcp] response: {resp}")
        return resp
    except Exception as exc:
        resp = _rpc_error(request.id, -32603, "Internal error", data=str(exc))
        print(f"[mcp] response: {resp}")
        return resp

    resp = _rpc_error(request.id, -32601, f"Method not found: {method}")
    print(f"[mcp] response: {resp}")
    return resp
