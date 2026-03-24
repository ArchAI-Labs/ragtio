"""Tests for app/rag_qa.py — mocking retrieval and Ollama."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document

from app.config import AppConfig
from app.query_enhancement import EnhancedQuery
from app.rag_qa import (
    RAGResponse,
    SourceDocument,
    _deduplicate,
    _doc_to_source,
    build_context,
    ask,
    ask_stream,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


def _make_doc(doc_id: str, content: str, score: float = 0.9, source: str = "test.pdf") -> Document:
    return Document(id=doc_id, content=content, score=score, meta={"source": source, "page": 1})


# ── build_context ──────────────────────────────────────────────────────────────

def test_build_context_basic():
    docs = [_make_doc("1", "chunk one"), _make_doc("2", "chunk two")]
    ctx, n = build_context(docs, max_context_length=10000)
    assert "chunk one" in ctx
    assert "chunk two" in ctx
    assert n == 2


def test_build_context_truncation():
    docs = [_make_doc("1", "A" * 3000), _make_doc("2", "B" * 3000)]
    ctx, n = build_context(docs, max_context_length=4000)
    assert n == 1
    assert "A" * 3000 in ctx
    assert "B" not in ctx


def test_build_context_empty():
    ctx, n = build_context([], max_context_length=4000)
    assert ctx == ""
    assert n == 0


def test_build_context_separator():
    docs = [_make_doc("1", "first"), _make_doc("2", "second")]
    ctx, _ = build_context(docs, separator="\n\n---\n\n")
    assert "\n\n---\n\n" in ctx


# ── _deduplicate ───────────────────────────────────────────────────────────────

def test_deduplicate_removes_same_id():
    docs = [_make_doc("1", "a"), _make_doc("1", "a"), _make_doc("2", "b")]
    result = _deduplicate(docs)
    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"


def test_deduplicate_preserves_order():
    docs = [_make_doc("3", "c"), _make_doc("1", "a"), _make_doc("2", "b")]
    result = _deduplicate(docs)
    assert [d.id for d in result] == ["3", "1", "2"]


# ── _doc_to_source ─────────────────────────────────────────────────────────────

def test_doc_to_source():
    doc = _make_doc("abc", "content text", score=0.85, source="file.pdf")
    src = _doc_to_source(doc)
    assert isinstance(src, SourceDocument)
    assert src.id == "abc"
    assert src.content == "content text"
    assert src.score == 0.85
    assert src.source == "file.pdf"
    assert src.page == 1


# ── ask() ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ask_no_docs(cfg):
    """When retrieval returns nothing, return the default answer without calling Ollama."""
    with patch("app.rag_qa.enhance_query") as mock_enhance, \
         patch("app.rag_qa.retrieve") as mock_retrieve:
        mock_enhance.return_value = EnhancedQuery(original_query="test")
        mock_retrieve.return_value = []

        result = await ask("test", cfg)

    assert isinstance(result, RAGResponse)
    assert "Non ho trovato" in result.answer
    assert result.sources == []
    assert result.n_docs_retrieved == 0


@pytest.mark.asyncio
async def test_ask_with_docs(cfg):
    docs = [_make_doc("1", "relevant content")]

    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "La risposta."}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.rag_qa.enhance_query") as mock_enhance, \
         patch("app.rag_qa.retrieve") as mock_retrieve, \
         patch("app.rag_qa.httpx.AsyncClient", return_value=mock_client):
        mock_enhance.return_value = EnhancedQuery(original_query="domanda")
        mock_retrieve.return_value = docs

        result = await ask("domanda", cfg)

    assert result.answer == "La risposta."
    assert len(result.sources) == 1
    assert result.n_docs_retrieved == 1
    assert result.query == "domanda"


@pytest.mark.asyncio
async def test_ask_deduplicates_docs(cfg):
    """When the same doc is returned for multiple queries, it should appear once."""
    doc = _make_doc("1", "content")

    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "answer"}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.rag_qa.enhance_query") as mock_enhance, \
         patch("app.rag_qa.retrieve") as mock_retrieve, \
         patch("app.rag_qa.httpx.AsyncClient", return_value=mock_client):
        # Two queries (expansion) both return the same doc
        mock_enhance.return_value = EnhancedQuery(
            original_query="q", expanded_queries=["q variant"]
        )
        mock_retrieve.return_value = [doc]

        result = await ask("q", cfg)

    assert result.n_docs_retrieved == 1


@pytest.mark.asyncio
async def test_ask_uses_filters(cfg):
    filters = {"source": "specific.pdf"}
    docs = [_make_doc("1", "content")]

    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "answer"}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.rag_qa.enhance_query") as mock_enhance, \
         patch("app.rag_qa.retrieve") as mock_retrieve, \
         patch("app.rag_qa.httpx.AsyncClient", return_value=mock_client):
        mock_enhance.return_value = EnhancedQuery(original_query="q")
        mock_retrieve.return_value = docs

        await ask("q", cfg, filters=filters)

    mock_retrieve.assert_called_once_with(query="q", cfg=cfg, filters=filters)


# ── ask_stream() ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ask_stream_no_docs(cfg):
    with patch("app.rag_qa.enhance_query") as mock_enhance, \
         patch("app.rag_qa.retrieve") as mock_retrieve:
        mock_enhance.return_value = EnhancedQuery(original_query="q")
        mock_retrieve.return_value = []

        tokens = [t async for t in ask_stream("q", cfg)]

    assert len(tokens) == 1
    assert "Non ho trovato" in tokens[0]


@pytest.mark.asyncio
async def test_ask_stream_yields_tokens(cfg):
    docs = [_make_doc("1", "relevant content")]

    async def _aiter_lines():
        yield json.dumps({"message": {"content": "Ciao"}, "done": False})
        yield json.dumps({"message": {"content": " mondo"}, "done": False})
        yield json.dumps({"message": {"content": ""}, "done": True})

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_lines = _aiter_lines
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.rag_qa.enhance_query") as mock_enhance, \
         patch("app.rag_qa.retrieve") as mock_retrieve, \
         patch("app.rag_qa.httpx.AsyncClient", return_value=mock_client):
        mock_enhance.return_value = EnhancedQuery(original_query="q")
        mock_retrieve.return_value = docs

        tokens = [t async for t in ask_stream("q", cfg)]

    assert tokens == ["Ciao", " mondo"]
