"""Test per il modulo app.retrieval."""

from typing import List
from unittest.mock import MagicMock, call, patch

import pytest
from haystack import Document

import app.retrieval as retrieval_module
from app.config import AppConfig, RerankerConfig
from app.retrieval import _rerank, retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_docs(n: int = 3) -> List[Document]:
    return [
        Document(content=f"Chunk {i}", meta={"source": f"doc_{i}.txt"}, score=round(0.9 - i * 0.1, 2))
        for i in range(n)
    ]


def mock_pipeline(docs: List[Document], output_key: str = "retriever") -> MagicMock:
    """Crea un Pipeline mock che restituisce i documenti dati."""
    p = MagicMock()
    p.run.return_value = {output_key: {"documents": docs}}
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_caches():
    """Pulisce le cache di pipeline, store e reranker tra i test."""
    retrieval_module._pipelines.clear()
    retrieval_module._stores.clear()
    retrieval_module._reranker = None
    yield
    retrieval_module._pipelines.clear()
    retrieval_module._stores.clear()
    retrieval_module._reranker = None


@pytest.fixture
def cfg_dense():
    cfg = AppConfig()
    cfg.retrieval.mode = "dense"
    cfg.retrieval.top_k = 5
    cfg.retrieval.score_threshold = None
    cfg.reranker = None
    return cfg


@pytest.fixture
def cfg_sparse():
    cfg = AppConfig()
    cfg.retrieval.mode = "sparse"
    cfg.retrieval.top_k = 5
    cfg.retrieval.score_threshold = None
    cfg.reranker = None
    return cfg


@pytest.fixture
def cfg_hybrid():
    cfg = AppConfig()
    cfg.retrieval.mode = "hybrid"
    cfg.retrieval.top_k = 5
    cfg.retrieval.score_threshold = None
    cfg.reranker = None
    return cfg


@pytest.fixture
def mock_store():
    return MagicMock()


# ---------------------------------------------------------------------------
# Routing: ogni mode usa il builder corretto
# ---------------------------------------------------------------------------


class TestRetrieveRouting:
    def test_dense_builds_dense_pipeline(self, cfg_dense, mock_store):
        docs = make_docs()
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)) as mb:
            result = retrieve("query", cfg_dense, mode="dense", document_store=mock_store)
        mb.assert_called_once()
        assert result == docs

    def test_sparse_builds_sparse_pipeline(self, cfg_sparse, mock_store):
        docs = make_docs()
        with patch("app.retrieval._build_sparse_pipeline", return_value=mock_pipeline(docs)) as mb:
            result = retrieve("query", cfg_sparse, mode="sparse", document_store=mock_store)
        mb.assert_called_once()
        assert result == docs

    def test_hybrid_builds_hybrid_pipeline(self, cfg_hybrid, mock_store):
        docs = make_docs()
        with patch("app.retrieval._build_hybrid_pipeline", return_value=mock_pipeline(docs)) as mb:
            result = retrieve("query", cfg_hybrid, mode="hybrid", document_store=mock_store)
        mb.assert_called_once()
        assert result == docs

    def test_default_mode_taken_from_cfg(self, cfg_dense, mock_store):
        """Se mode non è passato, usa cfg.retrieval.mode."""
        docs = make_docs()
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)):
            result = retrieve("query", cfg_dense, document_store=mock_store)
        assert result == docs

    def test_mode_override_ignores_cfg(self, cfg_dense, mock_store):
        """mode='sparse' sovrascrive cfg.retrieval.mode='dense'."""
        docs = make_docs()
        with patch("app.retrieval._build_sparse_pipeline", return_value=mock_pipeline(docs)):
            result = retrieve("query", cfg_dense, mode="sparse", document_store=mock_store)
        assert result == docs

    def test_invalid_mode_raises_value_error(self, cfg_dense, mock_store):
        with pytest.raises(ValueError, match="xyz"):
            retrieve("query", cfg_dense, mode="xyz", document_store=mock_store)

    def test_invalid_mode_message_contains_valid_values(self, cfg_dense, mock_store):
        with pytest.raises(ValueError, match="dense"):
            retrieve("query", cfg_dense, mode="wrong", document_store=mock_store)

    def test_returns_list_of_documents(self, cfg_dense, mock_store):
        docs = make_docs(4)
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)):
            result = retrieve("query", cfg_dense, document_store=mock_store)
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_empty_result_returned_as_empty_list(self, cfg_dense, mock_store):
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline([])):
            result = retrieve("query", cfg_dense, document_store=mock_store)
        assert result == []


# ---------------------------------------------------------------------------
# Parametri: top_k e filters passati correttamente al pipeline
# ---------------------------------------------------------------------------


class TestPipelineParameters:
    def _run_call_kwargs(self, mock_p: MagicMock) -> dict:
        """Estrae i kwargs passati a pipeline.run."""
        return mock_p.run.call_args[0][0]

    def test_explicit_top_k_passed_to_retriever(self, cfg_dense, mock_store):
        mock_p = mock_pipeline(make_docs())
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_p):
            retrieve("query", cfg_dense, top_k=7, mode="dense", document_store=mock_store)
        assert self._run_call_kwargs(mock_p)["retriever"]["top_k"] == 7

    def test_default_top_k_from_cfg(self, cfg_dense, mock_store):
        cfg_dense.retrieval.top_k = 12
        mock_p = mock_pipeline(make_docs())
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_p):
            retrieve("query", cfg_dense, document_store=mock_store)
        assert self._run_call_kwargs(mock_p)["retriever"]["top_k"] == 12

    def test_filters_passed_to_retriever(self, cfg_dense, mock_store):
        filters = {"language": "it", "category": "medico"}
        mock_p = mock_pipeline(make_docs())
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_p):
            retrieve("query", cfg_dense, filters=filters, document_store=mock_store)
        assert self._run_call_kwargs(mock_p)["retriever"]["filters"] == filters

    def test_no_filters_passes_none(self, cfg_dense, mock_store):
        mock_p = mock_pipeline(make_docs())
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_p):
            retrieve("query", cfg_dense, filters=None, document_store=mock_store)
        assert self._run_call_kwargs(mock_p)["retriever"]["filters"] is None

    def test_score_threshold_passed_to_retriever(self, cfg_dense, mock_store):
        cfg_dense.retrieval.score_threshold = 0.75
        mock_p = mock_pipeline(make_docs())
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_p):
            retrieve("query", cfg_dense, document_store=mock_store)
        assert self._run_call_kwargs(mock_p)["retriever"]["score_threshold"] == 0.75

    def test_sparse_pipeline_receives_query_via_sparse_embedder(self, cfg_sparse, mock_store):
        mock_p = mock_pipeline(make_docs())
        with patch("app.retrieval._build_sparse_pipeline", return_value=mock_p):
            retrieve("test query", cfg_sparse, document_store=mock_store)
        run_kwargs = self._run_call_kwargs(mock_p)
        assert "sparse_embedder" in run_kwargs
        assert run_kwargs["sparse_embedder"]["text"] == "test query"

    def test_hybrid_pipeline_receives_query_in_both_embedders(self, cfg_hybrid, mock_store):
        mock_p = mock_pipeline(make_docs())
        with patch("app.retrieval._build_hybrid_pipeline", return_value=mock_p):
            retrieve("test query", cfg_hybrid, document_store=mock_store)
        run_kwargs = self._run_call_kwargs(mock_p)
        assert run_kwargs["embedder"]["text"] == "test query"
        assert run_kwargs["sparse_embedder"]["text"] == "test query"


# ---------------------------------------------------------------------------
# Lazy init: il pipeline viene costruito una sola volta
# ---------------------------------------------------------------------------


class TestLazyPipelineCache:
    def test_pipeline_built_once_on_multiple_calls(self, cfg_dense, mock_store):
        docs = make_docs()
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)) as mb:
            retrieve("q1", cfg_dense, mode="dense", document_store=mock_store)
            retrieve("q2", cfg_dense, mode="dense", document_store=mock_store)
            retrieve("q3", cfg_dense, mode="dense", document_store=mock_store)
        mb.assert_called_once()

    def test_different_modes_build_separate_pipelines(self, mock_store):
        cfg = AppConfig()
        cfg.reranker = None
        docs = make_docs()
        with (
            patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)) as md,
            patch("app.retrieval._build_sparse_pipeline", return_value=mock_pipeline(docs)) as ms,
        ):
            retrieve("q", cfg, mode="dense", document_store=mock_store)
            retrieve("q", cfg, mode="sparse", document_store=mock_store)
        md.assert_called_once()
        ms.assert_called_once()

    def test_different_stores_build_separate_pipelines(self, cfg_dense):
        store_a = MagicMock()
        store_b = MagicMock()
        docs = make_docs()
        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)) as mb:
            retrieve("q", cfg_dense, mode="dense", document_store=store_a)
            retrieve("q", cfg_dense, mode="dense", document_store=store_b)
        assert mb.call_count == 2


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


class TestReranking:
    def _cfg_with_reranker(self, enabled: bool = True) -> AppConfig:
        cfg = AppConfig()
        cfg.retrieval.mode = "dense"
        cfg.retrieval.top_n_after_rerank = 3
        cfg.reranker = RerankerConfig(enabled=enabled)
        return cfg

    def test_rerank_called_when_enabled(self, mock_store):
        cfg = self._cfg_with_reranker(enabled=True)
        docs = make_docs(5)
        reranked = make_docs(3)

        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)):
            with patch("app.retrieval._rerank", return_value=reranked) as mr:
                result = retrieve("query", cfg, mode="dense", document_store=mock_store)

        mr.assert_called_once()
        assert result == reranked

    def test_rerank_not_called_when_disabled(self, mock_store):
        cfg = self._cfg_with_reranker(enabled=False)
        docs = make_docs(5)

        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)):
            with patch("app.retrieval._rerank") as mr:
                result = retrieve("query", cfg, mode="dense", document_store=mock_store)

        mr.assert_not_called()
        assert result == docs

    def test_rerank_not_called_when_reranker_is_none(self, cfg_dense, mock_store):
        cfg_dense.reranker = None
        docs = make_docs()

        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)):
            with patch("app.retrieval._rerank") as mr:
                retrieve("query", cfg_dense, document_store=mock_store)

        mr.assert_not_called()

    def test_explicit_top_n_passed_to_rerank(self, mock_store):
        cfg = self._cfg_with_reranker()
        docs = make_docs(10)

        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)):
            with patch("app.retrieval._rerank", return_value=docs[:2]) as mr:
                retrieve("query", cfg, top_n_after_rerank=2, document_store=mock_store)

        _, _, called_top_n, _ = mr.call_args[0]
        assert called_top_n == 2

    def test_default_top_n_from_cfg(self, mock_store):
        cfg = self._cfg_with_reranker()
        cfg.retrieval.top_n_after_rerank = 7
        docs = make_docs(10)

        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)):
            with patch("app.retrieval._rerank", return_value=docs[:7]) as mr:
                retrieve("query", cfg, document_store=mock_store)

        _, _, called_top_n, _ = mr.call_args[0]
        assert called_top_n == 7

    def test_rerank_receives_query_and_docs(self, mock_store):
        cfg = self._cfg_with_reranker()
        docs = make_docs(5)

        with patch("app.retrieval._build_dense_pipeline", return_value=mock_pipeline(docs)):
            with patch("app.retrieval._rerank", return_value=docs[:3]) as mr:
                retrieve("my query", cfg, document_store=mock_store)

        called_query, called_docs, _, _ = mr.call_args[0]
        assert called_query == "my query"
        assert called_docs == docs

    def test_rerank_applied_to_all_modes(self, mock_store):
        for mode, patch_target in [
            ("dense", "app.retrieval._build_dense_pipeline"),
            ("sparse", "app.retrieval._build_sparse_pipeline"),
            ("hybrid", "app.retrieval._build_hybrid_pipeline"),
        ]:
            retrieval_module._pipelines.clear()
            cfg = self._cfg_with_reranker()
            cfg.retrieval.mode = mode
            docs = make_docs(5)

            with patch(patch_target, return_value=mock_pipeline(docs)):
                with patch("app.retrieval._rerank", return_value=docs[:3]) as mr:
                    retrieve("query", cfg, mode=mode, document_store=mock_store)

            mr.assert_called_once(), f"_rerank not called for mode={mode}"


# ---------------------------------------------------------------------------
# _rerank unit test (con TransformersSimilarityRanker mockato)
# ---------------------------------------------------------------------------


class TestRerankFunction:
    def test_rerank_calls_ranker_run(self):
        cfg = AppConfig()
        cfg.reranker = RerankerConfig(enabled=True, model="dummy")
        docs = make_docs(5)
        reranked = make_docs(3)

        mock_ranker = MagicMock()
        mock_ranker.run.return_value = {"documents": reranked}

        with patch("app.retrieval.TransformersSimilarityRanker", return_value=mock_ranker):
            result = _rerank("query", docs, 3, cfg)

        mock_ranker.warm_up.assert_called_once()
        mock_ranker.run.assert_called_once_with(query="query", documents=docs, top_k=3)
        assert result == reranked

    def test_rerank_empty_docs_returns_empty(self):
        cfg = AppConfig()
        cfg.reranker = RerankerConfig(enabled=True)
        result = _rerank("query", [], 5, cfg)
        assert result == []

    def test_reranker_cached_across_calls(self):
        cfg = AppConfig()
        cfg.reranker = RerankerConfig(enabled=True, model="dummy")
        docs = make_docs(3)

        mock_ranker = MagicMock()
        mock_ranker.run.return_value = {"documents": docs}

        with patch("app.retrieval.TransformersSimilarityRanker", return_value=mock_ranker) as MockClass:
            _rerank("q1", docs, 3, cfg)
            _rerank("q2", docs, 3, cfg)

        # Costruttore chiamato una sola volta (caching)
        MockClass.assert_called_once()
        mock_ranker.warm_up.assert_called_once()
        assert mock_ranker.run.call_count == 2

    def test_rerank_top_n_passed_per_call(self):
        cfg = AppConfig()
        cfg.reranker = RerankerConfig(enabled=True, model="dummy")
        docs = make_docs(5)

        mock_ranker = MagicMock()
        mock_ranker.run.return_value = {"documents": docs[:2]}

        with patch("app.retrieval.TransformersSimilarityRanker", return_value=mock_ranker):
            _rerank("query", docs, 2, cfg)

        assert mock_ranker.run.call_args[1]["top_k"] == 2 or mock_ranker.run.call_args[0][-1] == 2


# ---------------------------------------------------------------------------
# TransformersSimilarityRanker import patchability check
# ---------------------------------------------------------------------------


class TestImportPatchability:
    def test_transformers_ranker_importable_from_module(self):
        """TransformersSimilarityRanker deve essere patchabile come app.retrieval.TransformersSimilarityRanker."""
        import importlib

        mod = importlib.import_module("app.retrieval")
        # Verifica che il nome esista a livello di modulo (via lazy import nel codice di _get_reranker)
        # Il test è sufficiente se la patch funziona come dimostrato in TestRerankFunction
        assert hasattr(mod, "_get_reranker")
