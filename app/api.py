"""FastAPI application for RAGtio."""
import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import AppConfig, load_config
from app.indexing import _get_embedding_dim, build_index
from app.ingest import UnsupportedFormatError, ingest
from app.query_enhancement import enhance_query
from app.rag_qa import build_context, _call_llm, _stream_llm
from app.retrieval import retrieve

logger = logging.getLogger(__name__)

# ── Response Models ──────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    status: str
    n_documents_ingested: int
    n_chunks: int
    n_documents_written: int
    elapsed_time_seconds: float
    warnings: List[str] = []


class QueryRequest(BaseModel):
    query: str
    retrieval_mode: Optional[Literal["dense", "sparse", "hybrid"]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    top_n_after_rerank: Optional[int] = Field(default=None, ge=1, le=100)
    filters: Optional[dict] = None
    use_expansion: Optional[bool] = None
    use_decomposition: Optional[bool] = None
    stream: bool = False


class SourceDocumentResponse(BaseModel):
    id: str
    content: str
    score: float
    source: str
    page: Optional[int] = None
    row_index: Optional[int] = None
    meta: dict = {}


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocumentResponse]
    query: str
    sub_queries: List[str] = []
    retrieval_mode: str
    n_sources_retrieved: int
    n_sources_used: int
    elapsed_time_seconds: float


class ServiceStatus(BaseModel):
    connected: bool
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class SystemStatusResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    qdrant: ServiceStatus
    ollama: ServiceStatus
    collection_name: str
    n_documents: Optional[int] = None
    embedder_model: str
    llm_model: str


# ── App lifecycle ─────────────────────────────────────────────────────────────

_config_path = os.getenv("CONFIG_PATH", "config.yaml")
_initial_cfg = load_config(_config_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cfg = load_config(os.getenv("CONFIG_PATH", "config.yaml"))
    logger.info(
        "Config caricata. Qdrant: %s:%s, LLM: %s",
        app.state.cfg.qdrant.host,
        app.state.cfg.qdrant.port,
        app.state.cfg.llm.model,
    )
    yield


app = FastAPI(title="RAGtio", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_initial_cfg.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (served at /static/...; root GET / serves index.html directly)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_cfg(request: Request) -> AppConfig:
    return request.app.state.cfg


def _apply_query_overrides(cfg: AppConfig, req: QueryRequest) -> AppConfig:
    """Return a deep-copy of cfg with per-request field overrides applied."""
    needs_copy = any(
        [
            req.retrieval_mode is not None,
            req.top_k is not None,
            req.top_n_after_rerank is not None,
            req.use_expansion is not None,
            req.use_decomposition is not None,
        ]
    )
    if not needs_copy:
        return cfg

    cfg = cfg.model_copy(deep=True)
    if req.retrieval_mode is not None:
        cfg.retrieval.mode = req.retrieval_mode
    if req.top_k is not None:
        cfg.retrieval.top_k = req.top_k
    if req.top_n_after_rerank is not None:
        cfg.retrieval.top_n_after_rerank = req.top_n_after_rerank
    if req.use_expansion is not None:
        if cfg.query_enhancement and cfg.query_enhancement.expansion:
            cfg.query_enhancement.expansion.enabled = req.use_expansion
    if req.use_decomposition is not None:
        if cfg.query_enhancement and cfg.query_enhancement.decomposition:
            cfg.query_enhancement.decomposition.enabled = req.use_decomposition
    return cfg


def _build_ollama_payload(
    context: str, question: str, cfg: AppConfig, stream: bool
) -> dict:
    prompt = cfg.llm.rag_prompt_template.format(context=context, question=question)
    return {
        "model": cfg.llm.model,
        "messages": [
            {"role": "system", "content": cfg.llm.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": cfg.llm.temperature,
            "top_p": cfg.llm.top_p,
            "num_predict": cfg.llm.max_tokens,
        },
        "stream": stream,
    }


def _collect_docs(query: str, cfg: AppConfig, filters: Optional[dict]) -> list:
    enhanced = enhance_query(query, cfg)
    all_docs = []
    seen_ids: set = set()
    for q in enhanced.all_queries:
        for d in retrieve(query=q, cfg=cfg, filters=filters):
            if d.id not in seen_ids:
                seen_ids.add(d.id)
                all_docs.append(d)
    return all_docs, enhanced


def _docs_to_response(docs: list) -> List[SourceDocumentResponse]:
    result = []
    for d in docs:
        meta = d.meta or {}
        result.append(
            SourceDocumentResponse(
                id=d.id or "",
                content=d.content or "",
                score=d.score or 0.0,
                source=meta.get("source", "sconosciuto"),
                page=meta.get("page"),
                row_index=meta.get("row_index"),
                meta=meta,
            )
        )
    return result


# ── SSE streaming ─────────────────────────────────────────────────────────────

_NO_DOCS_ANSWER = "Non ho trovato documenti rilevanti per rispondere alla domanda."


async def _rag_stream_generator(
    query: str,
    cfg: AppConfig,
    filters: Optional[dict],
) -> AsyncGenerator[str, None]:
    start = time.monotonic()
    all_docs, enhanced = _collect_docs(query, cfg, filters)
    n_retrieved = len(all_docs)

    if not all_docs:
        yield f"event: token\ndata: {json.dumps({'token': _NO_DOCS_ANSWER})}\n\n"
        done = {
            "sources": [],
            "sub_queries": [],
            "retrieval_mode": cfg.retrieval.mode,
            "n_sources_retrieved": 0,
            "n_sources_used": 0,
            "elapsed_time_seconds": time.monotonic() - start,
        }
        yield f"event: done\ndata: {json.dumps(done)}\n\n"
        return

    context, n_included = build_context(all_docs, max_context_length=cfg.llm.max_context_length)

    async for token in _stream_llm(context, query, cfg):
        yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

    sub_queries = enhanced.sub_queries + enhanced.expanded_queries
    sources_json = [s.model_dump() for s in _docs_to_response(all_docs[:n_included])]
    done = {
        "sources": sources_json,
        "sub_queries": sub_queries,
        "retrieval_mode": cfg.retrieval.mode,
        "n_sources_retrieved": n_retrieved,
        "n_sources_used": n_included,
        "elapsed_time_seconds": time.monotonic() - start,
    }
    yield f"event: done\ndata: {json.dumps(done)}\n\n"


# ── Error handlers ────────────────────────────────────────────────────────────


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail), "type": type(exc).__name__},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/api/ingest", response_model=IngestResponse)
async def api_ingest(
    request: Request,
    file: UploadFile = File(...),
    content_column: Optional[str] = Form(default=None),
    metadata_columns: Optional[str] = Form(default=None),
    extra_metadata: Optional[str] = Form(default=None),
    duplicate_policy: Optional[str] = Form(default=None),
):
    cfg = _get_cfg(request)

    contents = await file.read()
    max_bytes = cfg.api.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File supera la dimensione massima consentita di {cfg.api.max_upload_size_mb} MB",
        )

    # Parse optional JSON metadata
    extra_meta: dict = {}
    if extra_metadata:
        try:
            extra_meta = json.loads(extra_metadata)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"extra_metadata non è JSON valido: {e}")

    meta_cols = [c.strip() for c in metadata_columns.split(",")] if metadata_columns else None

    if duplicate_policy:
        if duplicate_policy not in ("SKIP", "OVERWRITE"):
            raise HTTPException(
                status_code=400, detail="duplicate_policy deve essere 'SKIP' o 'OVERWRITE'"
            )
        cfg = cfg.model_copy(deep=True)
        cfg.indexing.duplicate_policy = duplicate_policy  # type: ignore[assignment]

    suffix = Path(file.filename).suffix if file.filename else ""
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        start = time.monotonic()
        try:
            documents = ingest(
                file_path=tmp_path,
                content_column=content_column,
                metadata_columns=meta_cols,
                extra_metadata=extra_meta,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except UnsupportedFormatError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=422, detail=f"Encoding non riconoscibile: {e}")
        except (OSError, RuntimeError) as e:
            raise HTTPException(status_code=422, detail=str(e))

        try:
            result = build_index(documents, cfg)
        except (RuntimeError, OSError) as e:
            raise HTTPException(status_code=500, detail=f"Errore durante l'indicizzazione: {e}")

        return IngestResponse(
            status="success",
            n_documents_ingested=result.n_documents_ingested,
            n_chunks=result.n_chunks,
            n_documents_written=result.n_documents_written,
            elapsed_time_seconds=time.monotonic() - start,
            warnings=result.errors,
        )
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/api/query")
async def api_query(request: Request, body: QueryRequest):
    cfg = _apply_query_overrides(_get_cfg(request), body)

    if body.stream:
        return StreamingResponse(
            _rag_stream_generator(body.query, cfg, body.filters),
            media_type="text/event-stream",
        )

    start = time.monotonic()
    all_docs, enhanced = _collect_docs(body.query, cfg, body.filters)
    n_retrieved = len(all_docs)

    if not all_docs:
        return QueryResponse(
            answer=_NO_DOCS_ANSWER,
            sources=[],
            query=body.query,
            sub_queries=enhanced.sub_queries + enhanced.expanded_queries,
            retrieval_mode=cfg.retrieval.mode,
            n_sources_retrieved=0,
            n_sources_used=0,
            elapsed_time_seconds=time.monotonic() - start,
        )

    context, n_included = build_context(all_docs, max_context_length=cfg.llm.max_context_length)
    answer = await _call_llm(context, body.query, cfg)

    return QueryResponse(
        answer=answer,
        sources=_docs_to_response(all_docs[:n_included]),
        query=body.query,
        sub_queries=enhanced.sub_queries + enhanced.expanded_queries,
        retrieval_mode=cfg.retrieval.mode,
        n_sources_retrieved=n_retrieved,
        n_sources_used=n_included,
        elapsed_time_seconds=time.monotonic() - start,
    )


@app.get("/api/status", response_model=SystemStatusResponse)
async def api_status(request: Request):
    cfg = _get_cfg(request)
    qdrant_base = f"http://{cfg.qdrant.host}:{cfg.qdrant.port}"

    # ── Qdrant health ──
    qdrant_status = ServiceStatus(connected=False)
    n_documents: Optional[int] = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            t0 = time.monotonic()
            resp = await client.get(f"{qdrant_base}/healthz")
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
        qdrant_status = ServiceStatus(connected=resp.status_code == 200, latency_ms=latency_ms)

        if qdrant_status.connected:
            async with httpx.AsyncClient(timeout=5.0) as client:
                coll_resp = await client.get(
                    f"{qdrant_base}/collections/{cfg.qdrant.collection_name}"
                )
            if coll_resp.status_code == 200:
                n_documents = coll_resp.json().get("result", {}).get("points_count")
    except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException) as e:
        qdrant_status = ServiceStatus(connected=False, detail=str(e))

    # ── LLM health ──
    if cfg.llm.provider == "ollama":
        ollama_status = ServiceStatus(connected=False)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                t0 = time.monotonic()
                resp = await client.get(f"{cfg.llm.host}/api/tags")
                latency_ms = round((time.monotonic() - t0) * 1000, 2)
            ollama_status = ServiceStatus(connected=resp.status_code == 200, latency_ms=latency_ms)
        except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException) as e:
            ollama_status = ServiceStatus(connected=False, detail=str(e))
        llm_ok = ollama_status.connected
    else:
        ollama_status = ServiceStatus(connected=True, detail=f"n/a (provider={cfg.llm.provider})")
        llm_ok = True

    both_ok = qdrant_status.connected and llm_ok
    either_ok = qdrant_status.connected or llm_ok
    overall: Literal["ok", "degraded", "down"] = (
        "ok" if both_ok else ("degraded" if either_ok else "down")
    )

    return SystemStatusResponse(
        status=overall,
        qdrant=qdrant_status,
        ollama=ollama_status,
        collection_name=cfg.qdrant.collection_name,
        n_documents=n_documents,
        embedder_model=cfg.embedder.model,
        llm_model=cfg.llm.model,
    )


@app.get("/api/config")
async def api_get_config(request: Request):
    data = _get_cfg(request).model_dump()
    if data.get("qdrant", {}).get("api_key"):
        data["qdrant"]["api_key"] = "***"
    return data


# Fields that cannot be updated at runtime (infra-level)
_NON_UPDATABLE: dict = {
    "qdrant": {"host", "port", "grpc_port"},
}


@app.post("/api/config")
async def api_update_config(request: Request, body: dict):
    cfg = _get_cfg(request)

    for section, locked_fields in _NON_UPDATABLE.items():
        if section in body:
            for f in locked_fields:
                if f in body[section]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Il campo '{section}.{f}' non può essere modificato a runtime",
                    )

    current = cfg.model_dump()
    changed_fields: List[str] = []

    for section, values in body.items():
        if section not in current:
            raise HTTPException(status_code=422, detail=f"Sezione di config sconosciuta: '{section}'")
        if isinstance(values, dict) and isinstance(current.get(section), dict):
            for k, v in values.items():
                if current[section].get(k) != v:
                    changed_fields.append(f"{section}.{k}")
            current[section].update(values)
        else:
            if current.get(section) != values:
                changed_fields.append(section)
            current[section] = values

    try:
        from pydantic import ValidationError
        new_cfg = AppConfig.model_validate(current)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Configurazione non valida: {e}")

    request.app.state.cfg = new_cfg

    # Invalidate retrieval pipeline/store caches so they rebuild with updated config
    from app import retrieval as _retrieval
    _retrieval._pipelines.clear()
    _retrieval._stores.clear()
    _retrieval._reranker = None

    return {"status": "updated", "changed_fields": changed_fields}


@app.delete("/api/index")
async def api_delete_index(
    request: Request,
    confirm: bool = Query(default=False),
):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Parametro 'confirm=true' richiesto per eseguire l'operazione",
        )

    cfg = _get_cfg(request)
    qdrant_base = f"http://{cfg.qdrant.host}:{cfg.qdrant.port}"

    # Read current count before deletion
    n_deleted: int = 0
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{qdrant_base}/collections/{cfg.qdrant.collection_name}")
        if resp.status_code == 200:
            n_deleted = resp.json().get("result", {}).get("points_count", 0) or 0
    except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException):
        pass

    try:
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

        embedding_dim = _get_embedding_dim(cfg.embedder)
        QdrantDocumentStore(
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
            api_key=cfg.qdrant.api_key,
            index=cfg.qdrant.collection_name,
            embedding_dim=embedding_dim,
            use_sparse_embeddings=True,
            recreate_index=True,
        )
    except (RuntimeError, OSError) as e:
        raise HTTPException(
            status_code=500, detail=f"Errore durante la cancellazione dell'indice: {e}"
        )

    # Clear retrieval caches since the store was recreated
    from app import retrieval as _retrieval
    _retrieval._pipelines.clear()
    _retrieval._stores.clear()

    return {
        "status": "deleted",
        "collection_name": cfg.qdrant.collection_name,
        "n_documents_deleted": n_deleted,
    }


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Chunks browser endpoint ────────────────────────────────────────────────────


@app.get("/api/chunks")
async def api_chunks(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: Optional[str] = Query(default=None, description="Qdrant scroll offset (opaque token)"),
    search: Optional[str] = Query(default=None, description="Filtra per testo contenuto nel chunk"),
):
    """Restituisce un elenco paginato di chunk con ID e preview del testo."""
    cfg = _get_cfg(request)

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchText

        client = QdrantClient(
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
            api_key=cfg.qdrant.api_key,
            timeout=cfg.qdrant.timeout,
        )

        scroll_filter = None
        if search:
            scroll_filter = Filter(
                must=[FieldCondition(key="content", match=MatchText(text=search))]
            )

        records, next_offset = client.scroll(
            collection_name=cfg.qdrant.collection_name,
            limit=limit,
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore Qdrant: {exc}")

    chunks = []
    for r in records:
        payload = getattr(r, "payload", {}) or {}
        doc_id = payload.get("id") or payload.get("_id") or str(r.id)
        text = payload.get("content") or payload.get("text") or ""
        source = payload.get("file_path") or payload.get("source") or ""
        chunks.append({
            "id": str(doc_id),
            "preview": text[:200],
            "source": source,
        })

    return {
        "chunks": chunks,
        "next_offset": str(next_offset) if next_offset is not None else None,
        "count": len(chunks),
    }


# ── Evaluation endpoints ───────────────────────────────────────────────────────

# In-memory job store: {job_id: {"status": str, "report": dict | None, "error": str | None}}
_eval_jobs: Dict[str, Dict[str, Any]] = {}


async def _run_eval_job(job_id: str, cfg: AppConfig) -> None:
    from app.evaluation import run_evaluation_mode_a

    try:
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, run_evaluation_mode_a, cfg)
        _eval_jobs[job_id]["status"] = "done"
        _eval_jobs[job_id]["report"] = asdict(report)
    except (RuntimeError, OSError, ValueError, httpx.HTTPError) as exc:
        logger.exception("Evaluation job %s failed: %s", job_id, exc)
        _eval_jobs[job_id]["status"] = "error"
        _eval_jobs[job_id]["error"] = str(exc)


@app.post("/api/eval")
async def api_eval_start(request: Request):
    """Avvia una evaluation Mode A in background e ritorna il job_id."""
    cfg = _get_cfg(request)
    job_id = str(uuid.uuid4())
    _eval_jobs[job_id] = {"status": "running", "report": None, "error": None}
    asyncio.create_task(_run_eval_job(job_id, cfg))
    return {"job_id": job_id}


async def _run_eval_b_job(job_id: str, dataset: list, cfg: AppConfig) -> None:
    from app.evaluation import run_evaluation_mode_b

    try:
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, run_evaluation_mode_b, dataset, cfg)
        _eval_jobs[job_id]["status"] = "done"
        _eval_jobs[job_id]["report"] = asdict(report)
    except (RuntimeError, OSError, ValueError) as exc:
        logger.exception("Evaluation Mode B job %s failed: %s", job_id, exc)
        _eval_jobs[job_id]["status"] = "error"
        _eval_jobs[job_id]["error"] = str(exc)


@app.post("/api/eval/b")
async def api_eval_b_start(request: Request):
    """Avvia una evaluation Mode B con dataset JSON fornito dall'utente."""
    cfg = _get_cfg(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Body JSON non valido")

    dataset = body.get("dataset")
    if not isinstance(dataset, list) or not dataset:
        raise HTTPException(status_code=422, detail="Campo 'dataset' mancante o vuoto")

    job_id = str(uuid.uuid4())
    _eval_jobs[job_id] = {"status": "running", "report": None, "error": None}
    asyncio.create_task(_run_eval_b_job(job_id, dataset, cfg))
    return {"job_id": job_id}


@app.get("/api/eval/{job_id}")
async def api_eval_status(job_id: str):
    """Ritorna lo stato del job di evaluation e il report se completato."""
    job = _eval_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' non trovato")

    response: Dict[str, Any] = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "done":
        response["report"] = job["report"]
    elif job["status"] == "error":
        response["error"] = job["error"]
    return response
