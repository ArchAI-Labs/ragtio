# Pipeline RAG Q&A
import json
import logging
import time
from typing import AsyncGenerator, List, Optional

import httpx
from haystack import Document
from pydantic import BaseModel

from app.config import AppConfig
from app.query_enhancement import enhance_query
from app.retrieval import retrieve

logger = logging.getLogger(__name__)

_NO_DOCS_ANSWER = "Non ho trovato documenti rilevanti per rispondere alla domanda."


class SourceDocument(BaseModel):
    id: str
    content: str
    score: float
    source: str
    page: Optional[int] = None
    row_index: Optional[int] = None
    meta: dict = {}


class RAGResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str
    sub_queries: List[str] = []
    retrieval_mode: str = "hybrid"
    n_docs_retrieved: int = 0


# ── Context building ───────────────────────────────────────────────────────────

def _format_source_header(doc: Document, index: int) -> str:
    meta = doc.meta or {}
    source = meta.get("source", "sconosciuto")
    extras = []
    if page := meta.get("page"):
        extras.append(f"pagina {page}")
    if row := meta.get("row_index"):
        extras.append(f"riga {row}")
    extra_str = " — " + ", ".join(extras) if extras else ""
    return f"[Fonte {index}] {source}{extra_str}"


def build_context(
    documents: List[Document],
    max_context_length: int = 4000,
    separator: str = "\n\n---\n\n",
) -> tuple[str, int]:
    parts = []
    total_len = 0
    included = 0

    for i, doc in enumerate(documents):
        source_info = _format_source_header(doc, i + 1)
        chunk_text = f"{source_info}\n{doc.content}"
        chunk_len = len(chunk_text) + len(separator)

        if total_len + chunk_len > max_context_length:
            break

        parts.append(chunk_text)
        total_len += chunk_len
        included += 1

    return separator.join(parts), included


# ── Document helpers ───────────────────────────────────────────────────────────

def _doc_to_source(doc: Document) -> SourceDocument:
    meta = doc.meta or {}
    return SourceDocument(
        id=doc.id or "",
        content=doc.content or "",
        score=doc.score or 0.0,
        source=meta.get("source", "sconosciuto"),
        page=meta.get("page"),
        row_index=meta.get("row_index"),
        meta=meta,
    )


def _deduplicate(docs: List[Document]) -> List[Document]:
    seen: set = set()
    result = []
    for doc in docs:
        key = doc.id
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


def _collect_docs(query: str, enhanced_queries: List[str], cfg: AppConfig, filters: Optional[dict]) -> List[Document]:
    all_docs: List[Document] = []
    for q in enhanced_queries:
        docs = retrieve(query=q, cfg=cfg, filters=filters)
        all_docs.extend(docs)
    return _deduplicate(all_docs)


# ── LLM call helpers ───────────────────────────────────────────────────────────

def _build_messages(context: str, question: str, cfg: AppConfig) -> list:
    prompt = cfg.llm.rag_prompt_template.format(context=context, question=question)
    return [
        {"role": "system", "content": cfg.llm.system_prompt},
        {"role": "user", "content": prompt},
    ]


async def _call_ollama(context: str, question: str, cfg: AppConfig) -> str:
    messages = _build_messages(context, question, cfg)
    payload = {
        "model": cfg.llm.model,
        "messages": messages,
        "options": {
            "temperature": cfg.llm.temperature,
            "top_p": cfg.llm.top_p,
            "num_predict": cfg.llm.max_tokens,
        },
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=cfg.llm.timeout) as client:
        response = await client.post(f"{cfg.llm.host}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


async def _call_openai(context: str, question: str, cfg: AppConfig) -> str:
    from openai import AsyncOpenAI

    messages = _build_messages(context, question, cfg)
    client = AsyncOpenAI(
        api_key=cfg.llm.openai_api_key,
        base_url=cfg.llm.openai_base_url,
        timeout=cfg.llm.timeout,
    )
    response = await client.chat.completions.create(
        model=cfg.llm.model,
        messages=messages,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        max_tokens=cfg.llm.max_tokens,
    )
    return response.choices[0].message.content


async def _stream_ollama(context: str, question: str, cfg: AppConfig) -> AsyncGenerator[str, None]:
    messages = _build_messages(context, question, cfg)
    payload = {
        "model": cfg.llm.model,
        "messages": messages,
        "options": {
            "temperature": cfg.llm.temperature,
            "top_p": cfg.llm.top_p,
            "num_predict": cfg.llm.max_tokens,
        },
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=cfg.llm.timeout) as client:
        async with client.stream("POST", f"{cfg.llm.host}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if token := data.get("message", {}).get("content"):
                        yield token
                    if data.get("done"):
                        break


async def _stream_openai(context: str, question: str, cfg: AppConfig) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI

    messages = _build_messages(context, question, cfg)
    client = AsyncOpenAI(
        api_key=cfg.llm.openai_api_key,
        base_url=cfg.llm.openai_base_url,
        timeout=cfg.llm.timeout,
    )
    stream = await client.chat.completions.create(
        model=cfg.llm.model,
        messages=messages,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        max_tokens=cfg.llm.max_tokens,
        stream=True,
    )
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


async def _call_llm(context: str, question: str, cfg: AppConfig) -> str:
    if cfg.llm.provider == "openai":
        return await _call_openai(context, question, cfg)
    return await _call_ollama(context, question, cfg)


async def _stream_llm(context: str, question: str, cfg: AppConfig) -> AsyncGenerator[str, None]:
    if cfg.llm.provider == "openai":
        async for token in _stream_openai(context, question, cfg):
            yield token
    else:
        async for token in _stream_ollama(context, question, cfg):
            yield token


# ── Public API ─────────────────────────────────────────────────────────────────

async def ask(
    query: str,
    cfg: AppConfig,
    filters: Optional[dict] = None,
) -> RAGResponse:
    import time

    start = time.monotonic()
    logger.info("Avvio ask [provider=%s]: '%s'", cfg.llm.provider, query[:80])

    enhanced = enhance_query(query, cfg)
    docs = _collect_docs(query, enhanced.all_queries, cfg, filters)
    n_retrieved = len(docs)

    if not docs:
        logger.info("ask completato: nessun documento trovato in %.2f s", time.monotonic() - start)
        return RAGResponse(
            answer=_NO_DOCS_ANSWER,
            sources=[],
            query=query,
            sub_queries=enhanced.sub_queries + enhanced.expanded_queries,
            retrieval_mode=cfg.retrieval.mode,
            n_docs_retrieved=0,
        )

    context, n_included = build_context(docs, max_context_length=cfg.llm.max_context_length)
    answer = await _call_llm(context, query, cfg)

    logger.info(
        "ask completato in %.2f s: %d doc recuperati, %d nel contesto",
        time.monotonic() - start,
        n_retrieved,
        n_included,
    )
    return RAGResponse(
        answer=answer,
        sources=[_doc_to_source(d) for d in docs[:n_included]],
        query=query,
        sub_queries=enhanced.sub_queries + enhanced.expanded_queries,
        retrieval_mode=cfg.retrieval.mode,
        n_docs_retrieved=n_retrieved,
    )


async def ask_stream(
    query: str,
    cfg: AppConfig,
    filters: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    import time

    start = time.monotonic()
    logger.info("Avvio ask_stream [provider=%s]: '%s'", cfg.llm.provider, query[:80])

    enhanced = enhance_query(query, cfg)
    docs = _collect_docs(query, enhanced.all_queries, cfg, filters)

    if not docs:
        logger.info("ask_stream: nessun documento trovato in %.2f s", time.monotonic() - start)
        yield _NO_DOCS_ANSWER
        return

    context, n_included = build_context(docs, max_context_length=cfg.llm.max_context_length)
    logger.info(
        "ask_stream: %d doc recuperati, %d nel contesto, avvio streaming LLM",
        len(docs),
        n_included,
    )

    async for token in _stream_llm(context, query, cfg):
        yield token

    logger.info("ask_stream completato in %.2f s", time.monotonic() - start)
