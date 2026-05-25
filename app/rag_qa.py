# Pipeline RAG Q&A
import json
import logging
import re
import time
from typing import AsyncGenerator, List, Optional

import httpx
from haystack import Document
from pydantic import BaseModel

from app.config import AppConfig
from app.query_enhancement import enhance_query
from app.retrieval import retrieve
from app.utils import detect_language

logger = logging.getLogger(__name__)

_NO_DOCS_ANSWER = "Non ho trovato documenti rilevanti per rispondere alla domanda."

# ── Language detection (kept for backward compat, delegates to utils) ──────────

_LANG_MARKERS: dict[str, frozenset[str]] = {
    "Italian":    frozenset({"il","la","lo","gli","le","un","una","di","del","della","dei","degli","delle","che","è","non","con","per","si","come","dove","quando","questo","questa","questi","queste","quale","quali","sono","hai","siamo","avete","hanno","essere","fare","dire","vedere","sapere","volere","potere","andare","venire","cosa","perché","però","quindi","anche","già","ancora","sempre","mai","tutto","tutti","tutta","tutte"}),
    "French":     frozenset({"le","la","les","un","une","des","je","tu","il","elle","nous","vous","ils","elles","est","sont","que","pas","pour","avec","comme","où","quand","cette","quel","quelle","aussi","très","bien","mais","ou","donc","car","ni","or","tout","tous","toute","toutes","être","avoir","faire","dire","voir","savoir","vouloir","pouvoir","aller","venir","dans","sur","sous","entre","après","avant"}),
    "German":     frozenset({"der","die","das","des","dem","den","ein","eine","ist","sind","war","waren","nicht","mit","für","von","bei","nach","aus","an","auf","über","unter","zwischen","ich","du","er","sie","wir","ihr","auch","noch","schon","immer","alles","alle","sein","haben","werden","können","müssen","sollen","wollen","machen","sagen","gehen","kommen","wissen","sehen"}),
    "Spanish":    frozenset({"el","la","los","las","un","una","es","son","fue","fueron","que","no","con","para","por","como","donde","cuando","este","esta","estos","estas","cual","cuales","también","muy","bien","pero","sino","aunque","porque","todo","todos","toda","todas","ser","tener","hacer","decir","ver","saber","querer","poder","ir","venir","yo","tú","él","ella","nosotros","vosotros","ellos"}),
    "Portuguese": frozenset({"o","a","os","as","um","uma","é","são","foi","foram","que","não","com","para","por","como","onde","quando","este","esta","estes","estas","qual","quais","também","muito","bem","mas","porque","todo","todos","toda","todas","ser","ter","fazer","dizer","ver","saber","querer","poder","ir","vir","eu","tu","ele","ela","nós","vós","eles"}),
}


def _detect_language(text: str) -> str:
    return detect_language(text)


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


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).lstrip()


# ── LLM call helpers ───────────────────────────────────────────────────────────

def _build_messages(context: str, question: str, cfg: AppConfig) -> list:
    lang = _detect_language(question)
    system = cfg.llm.system_prompt + f"\n\nIMPORTANT: The user is writing in {lang}. You MUST reply in {lang} only. Never switch language."
    prompt = cfg.llm.rag_prompt_template.format(context=context, question=question)
    return [
        {"role": "system", "content": system},
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
        "think": cfg.llm.think,
    }
    async with httpx.AsyncClient(timeout=cfg.llm.timeout) as client:
        response = await client.post(f"{cfg.llm.host}/api/chat", json=payload)
        response.raise_for_status()
        return _strip_thinking(response.json()["message"]["content"])


async def _call_openai(context: str, question: str, cfg: AppConfig) -> str:
    from openai import AsyncOpenAI

    messages = _build_messages(context, question, cfg)
    client = AsyncOpenAI(
        api_key=cfg.llm.openai_api_key,
        base_url=cfg.llm.openai_base_url,
        timeout=cfg.llm.timeout,
    )
    kwargs: dict = dict(model=cfg.llm.model, messages=messages,
                        temperature=cfg.llm.temperature, top_p=cfg.llm.top_p)
    if cfg.llm.max_tokens != -1:
        kwargs["max_tokens"] = cfg.llm.max_tokens
    response = await client.chat.completions.create(**kwargs)
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
        "think": cfg.llm.think,
    }
    async with httpx.AsyncClient(timeout=cfg.llm.timeout) as client:
        async with client.stream("POST", f"{cfg.llm.host}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            pending = ""
            past_think = False
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    if past_think:
                        yield token
                    else:
                        pending += token
                        end_idx = pending.find("</think>")
                        if end_idx != -1:
                            after = pending[end_idx + len("</think>"):].lstrip("\n")
                            pending = ""
                            past_think = True
                            if after:
                                yield after
                if data.get("done"):
                    if pending:
                        yield pending
                    break


async def _stream_openai(context: str, question: str, cfg: AppConfig) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI

    messages = _build_messages(context, question, cfg)
    client = AsyncOpenAI(
        api_key=cfg.llm.openai_api_key,
        base_url=cfg.llm.openai_base_url,
        timeout=cfg.llm.timeout,
    )
    kwargs: dict = dict(model=cfg.llm.model, messages=messages,
                        temperature=cfg.llm.temperature, top_p=cfg.llm.top_p, stream=True)
    if cfg.llm.max_tokens != -1:
        kwargs["max_tokens"] = cfg.llm.max_tokens
    stream = await client.chat.completions.create(**kwargs)
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
