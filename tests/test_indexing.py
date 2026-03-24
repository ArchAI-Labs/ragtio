"""Test per il modulo app.indexing."""

import dataclasses
import logging
from typing import List

import pytest
from haystack import Document, component

from app.config import AppConfig
from app.indexing import IndexingResult, RecursiveDocumentSplitter, build_index


# ---------------------------------------------------------------------------
# Mock components per test senza dipendenze esterne
# ---------------------------------------------------------------------------

FAKE_DIM = 4


@component
class MockEmbedder:
    """Embedder che restituisce vettori fake senza caricare modelli."""

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        return {
            "documents": [
                dataclasses.replace(doc, embedding=[0.1] * FAKE_DIM) for doc in documents
            ]
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_store():
    """QdrantDocumentStore in-memory per test senza Qdrant esterno."""
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    return QdrantDocumentStore(
        location=":memory:",
        index="test_collection",
        embedding_dim=FAKE_DIM,
        use_sparse_embeddings=False,
        recreate_index=True,
    )


@pytest.fixture
def cfg():
    """AppConfig con valori di default."""
    return AppConfig()


def make_docs(n: int = 2, length: int = 300) -> List[Document]:
    """Genera n Document con contenuto di lunghezza sufficiente."""
    word = "Lorem ipsum dolor sit amet. "
    reps = max(1, length // len(word) + 1)
    return [
        Document(content=(word * reps)[:length], meta={"source": f"doc_{i}.txt"})
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test build_index — risultato e struttura
# ---------------------------------------------------------------------------


class TestBuildIndexResult:
    def test_returns_indexing_result(self, cfg, in_memory_store):
        docs = make_docs(2)
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert isinstance(result, IndexingResult)

    def test_n_documents_ingested(self, cfg, in_memory_store):
        docs = make_docs(3)
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.n_documents_ingested == 3

    def test_elapsed_time_positive(self, cfg, in_memory_store):
        docs = make_docs(1)
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.elapsed_time_seconds > 0

    def test_errors_is_list(self, cfg, in_memory_store):
        docs = make_docs(1)
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert isinstance(result.errors, list)

    def test_n_chunks_positive(self, cfg, in_memory_store):
        docs = make_docs(2, length=500)
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.n_chunks > 0

    def test_n_documents_written_positive(self, cfg, in_memory_store):
        docs = make_docs(2, length=500)
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.n_documents_written > 0


# ---------------------------------------------------------------------------
# Test build_index — lista vuota
# ---------------------------------------------------------------------------


class TestBuildIndexEmpty:
    def test_empty_input(self, cfg, in_memory_store):
        result = build_index([], cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.n_documents_ingested == 0
        assert result.n_chunks == 0
        assert result.n_documents_written == 0

    def test_empty_input_no_errors(self, cfg, in_memory_store):
        result = build_index([], cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.errors == []


# ---------------------------------------------------------------------------
# Test build_index — strategie di chunking
# ---------------------------------------------------------------------------


class TestChunkingStrategies:
    def _cfg_with(self, **kwargs) -> AppConfig:
        cfg = AppConfig()
        for key, val in kwargs.items():
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], val)
        return cfg

    def test_character_strategy_splits(self, in_memory_store):
        cfg = self._cfg_with(
            **{
                "indexing.chunking.strategy": "character",
                "indexing.chunking.chunk_size": 80,
                "indexing.chunking.chunk_overlap": 10,
                "indexing.min_chunk_length": 0,
            }
        )
        docs = [Document(content="A" * 500, meta={"source": "test.txt"})]
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.n_chunks > 1

    def test_paragraph_strategy(self, in_memory_store):
        cfg = self._cfg_with(
            **{
                "indexing.chunking.strategy": "paragraph",
                "indexing.min_chunk_length": 0,
            }
        )
        content = "Primo paragrafo con testo sufficiente.\n\nSecondo paragrafo con testo sufficiente."
        docs = [Document(content=content, meta={"source": "test.txt"})]
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.n_chunks == 2

    def test_recursive_strategy(self, in_memory_store):
        cfg = self._cfg_with(
            **{
                "indexing.chunking.strategy": "recursive",
                "indexing.chunking.chunk_size": 100,
                "indexing.chunking.chunk_overlap": 10,
                "indexing.min_chunk_length": 0,
            }
        )
        text = ("Paragrafo uno.\n\nParagrafo due.\n\nParagrafo tre con contenuto aggiuntivo " * 5)
        docs = [Document(content=text, meta={"source": "test.txt"})]
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.n_chunks > 1


# ---------------------------------------------------------------------------
# Test build_index — min_chunk_length
# ---------------------------------------------------------------------------


class TestMinChunkLength:
    def test_short_chunks_filtered(self, in_memory_store):
        cfg = AppConfig()
        cfg.indexing.chunking.strategy = "paragraph"
        cfg.indexing.min_chunk_length = 200

        short_content = "Breve.\n\nAnche questo è breve."
        docs = [Document(content=short_content, meta={"source": "test.txt"})]
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.n_chunks == 0
        assert len(result.errors) > 0

    def test_short_chunks_in_errors(self, in_memory_store):
        cfg = AppConfig()
        cfg.indexing.chunking.strategy = "paragraph"
        cfg.indexing.min_chunk_length = 200

        docs = [Document(content="Corto.\n\nAnche corto.", meta={"source": "test.txt"})]
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert all("scartato" in e for e in result.errors)

    def test_long_chunks_pass_filter(self, in_memory_store):
        cfg = AppConfig()
        cfg.indexing.chunking.strategy = "paragraph"
        cfg.indexing.min_chunk_length = 5

        docs = [Document(content="Abbastanza lungo per passare il filtro.", meta={"source": "x.txt"})]
        result = build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())
        assert result.n_chunks == 1
        assert result.errors == []


# ---------------------------------------------------------------------------
# Test build_index — WARNING per paragrafi lunghi
# ---------------------------------------------------------------------------


class TestParagraphWarning:
    def test_long_paragraph_logs_warning(self, in_memory_store, caplog):
        cfg = AppConfig()
        cfg.indexing.chunking.strategy = "paragraph"
        cfg.indexing.chunking.max_paragraph_length = 30
        cfg.indexing.min_chunk_length = 0

        long_para = "X" * 200
        docs = [Document(content=long_para, meta={"source": "test.txt"})]

        with caplog.at_level(logging.WARNING, logger="app.indexing"):
            build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())

        assert any("max_paragraph_length" in r.message for r in caplog.records)

    def test_short_paragraph_no_warning(self, in_memory_store, caplog):
        cfg = AppConfig()
        cfg.indexing.chunking.strategy = "paragraph"
        cfg.indexing.chunking.max_paragraph_length = 2000
        cfg.indexing.min_chunk_length = 0

        docs = [Document(content="Paragrafo breve.", meta={"source": "test.txt"})]

        with caplog.at_level(logging.WARNING, logger="app.indexing"):
            build_index(docs, cfg, document_store=in_memory_store, embedder=MockEmbedder())

        assert not any("max_paragraph_length" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test RecursiveDocumentSplitter (unit test)
# ---------------------------------------------------------------------------


class TestRecursiveDocumentSplitter:
    def test_short_text_not_split(self):
        splitter = RecursiveDocumentSplitter(chunk_size=1000)
        docs = [Document(content="Testo breve che non viene splittato.")]
        result = splitter.run(documents=docs)
        assert len(result["documents"]) == 1

    def test_long_text_split_on_double_newline(self):
        splitter = RecursiveDocumentSplitter(chunk_size=50, chunk_overlap=0)
        text = "Primo paragrafo abbastanza lungo.\n\nSecondo paragrafo abbastanza lungo."
        docs = [Document(content=text)]
        result = splitter.run(documents=docs)
        assert len(result["documents"]) > 1

    def test_chunk_size_respected(self):
        splitter = RecursiveDocumentSplitter(chunk_size=50, chunk_overlap=0)
        docs = [Document(content="A" * 300)]
        result = splitter.run(documents=docs)
        for doc in result["documents"]:
            assert len(doc.content) <= 50

    def test_empty_content_returns_nothing(self):
        splitter = RecursiveDocumentSplitter(chunk_size=100)
        docs = [Document(content="")]
        result = splitter.run(documents=docs)
        assert len(result["documents"]) == 0

    def test_whitespace_only_returns_nothing(self):
        splitter = RecursiveDocumentSplitter(chunk_size=100)
        docs = [Document(content="   \n\n  ")]
        result = splitter.run(documents=docs)
        assert len(result["documents"]) == 0

    def test_meta_propagated_to_chunks(self):
        splitter = RecursiveDocumentSplitter(chunk_size=30, chunk_overlap=0)
        meta = {"source": "test.txt", "page": 1}
        docs = [Document(content="A" * 200, meta=meta)]
        result = splitter.run(documents=docs)
        for doc in result["documents"]:
            assert doc.meta["source"] == "test.txt"
            assert "chunk_index" in doc.meta

    def test_chunk_index_sequential(self):
        splitter = RecursiveDocumentSplitter(chunk_size=30, chunk_overlap=0)
        docs = [Document(content="A" * 200)]
        result = splitter.run(documents=docs)
        indices = [d.meta["chunk_index"] for d in result["documents"]]
        assert indices == list(range(len(indices)))

    def test_parent_id_set(self):
        splitter = RecursiveDocumentSplitter(chunk_size=30, chunk_overlap=0)
        doc = Document(content="A" * 200)
        result = splitter.run(documents=[doc])
        for chunk in result["documents"]:
            assert chunk.meta["parent_id"] == doc.id

    def test_multiple_documents(self):
        splitter = RecursiveDocumentSplitter(chunk_size=50, chunk_overlap=0)
        docs = [Document(content="A" * 200), Document(content="B" * 200)]
        result = splitter.run(documents=docs)
        assert len(result["documents"]) > 2

    def test_overlap_increases_chunk_count(self):
        splitter_no_overlap = RecursiveDocumentSplitter(chunk_size=50, chunk_overlap=0)
        splitter_overlap = RecursiveDocumentSplitter(chunk_size=50, chunk_overlap=20)
        text = "A" * 300
        docs = [Document(content=text)]
        n_no_overlap = len(splitter_no_overlap.run(documents=docs)["documents"])
        n_overlap = len(splitter_overlap.run(documents=docs)["documents"])
        assert n_overlap >= n_no_overlap

    def test_custom_separators(self):
        splitter = RecursiveDocumentSplitter(
            chunk_size=30,
            chunk_overlap=0,
            separators=["||"],
        )
        text = "parte uno" + "||" + "parte due" + "||" + "parte tre molto più lunga del chunk"
        docs = [Document(content=text)]
        result = splitter.run(documents=docs)
        assert len(result["documents"]) >= 1
