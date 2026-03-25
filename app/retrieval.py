"""Modulo di retrieval ibrido su Qdrant con supporto dense, sparse e hybrid."""

import logging
import time
from typing import Any, Dict, List, Literal, Optional

from haystack import Document, Pipeline
from haystack.components.rankers import TransformersSimilarityRanker

from app.config import AppConfig

logger = logging.getLogger(__name__)

# ── Lazy cache ──────────────────────────────────────────────────────────────
_pipelines: Dict[str, Pipeline] = {}   # chiave: "{mode}:{id(store)}"
_stores: Dict[str, Any] = {}           # chiave: "{host}:{port}:{collection}"
_reranker: Optional[Any] = None


# ── Document store ────────────────────────────────────────────────────────────

def _get_document_store(cfg: AppConfig, document_store=None) -> Any:
    if document_store is not None:
        return document_store

    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    from app.indexing import _get_embedding_dim, _register_custom_embedder

    key = f"{cfg.qdrant.host}:{cfg.qdrant.port}:{cfg.qdrant.collection_name}"
    if key not in _stores:
        _register_custom_embedder(cfg.embedder)
        _stores[key] = QdrantDocumentStore(
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
            api_key=cfg.qdrant.api_key,
            index=cfg.qdrant.collection_name,
            embedding_dim=_get_embedding_dim(cfg.embedder),
            use_sparse_embeddings=True,
            recreate_index=False,
        )
    return _stores[key]


# ── Pipeline builders ─────────────────────────────────────────────────────────

def _build_dense_pipeline(cfg: AppConfig, store: Any) -> Pipeline:
    from haystack_integrations.components.embedders.fastembed import (
        FastembedTextEmbedder,
    )
    from haystack_integrations.components.retrievers.qdrant import (
        QdrantEmbeddingRetriever,
    )

    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        FastembedTextEmbedder(
            model=cfg.embedder.model,
            cache_dir=cfg.embedder.cache_dir,
            max_length=cfg.embedder.max_length,
        ),
    )
    pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=store))
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    return pipeline


def _build_sparse_pipeline(cfg: AppConfig, store: Any) -> Pipeline:
    from haystack_integrations.components.embedders.fastembed import (
        FastembedSparseTextEmbedder,
    )
    from haystack_integrations.components.retrievers.qdrant import (
        QdrantSparseEmbeddingRetriever,
    )

    pipeline = Pipeline()
    pipeline.add_component(
        "sparse_embedder",
        FastembedSparseTextEmbedder(cache_dir=cfg.embedder.cache_dir),
    )
    pipeline.add_component(
        "retriever", QdrantSparseEmbeddingRetriever(document_store=store)
    )
    pipeline.connect("sparse_embedder.sparse_embedding", "retriever.query_sparse_embedding")
    return pipeline


def _build_hybrid_pipeline(cfg: AppConfig, store: Any) -> Pipeline:
    from haystack_integrations.components.embedders.fastembed import (
        FastembedSparseTextEmbedder,
        FastembedTextEmbedder,
    )
    from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        FastembedTextEmbedder(
            model=cfg.embedder.model,
            cache_dir=cfg.embedder.cache_dir,
            max_length=cfg.embedder.max_length,
        ),
    )
    pipeline.add_component(
        "sparse_embedder",
        FastembedSparseTextEmbedder(cache_dir=cfg.embedder.cache_dir),
    )
    pipeline.add_component("retriever", QdrantHybridRetriever(document_store=store))
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("sparse_embedder.sparse_embedding", "retriever.query_sparse_embedding")
    return pipeline


# ── Pipeline cache ────────────────────────────────────────────────────────────

def _get_pipeline(mode: str, cfg: AppConfig, store: Any) -> Pipeline:
    key = f"{mode}:{id(store)}"
    if key not in _pipelines:
        if mode == "dense":
            _pipelines[key] = _build_dense_pipeline(cfg, store)
        elif mode == "sparse":
            _pipelines[key] = _build_sparse_pipeline(cfg, store)
        elif mode == "hybrid":
            _pipelines[key] = _build_hybrid_pipeline(cfg, store)
        else:
            raise ValueError(
                f"Modalità di retrieval non supportata: '{mode}'. "
                "Valori validi: dense, sparse, hybrid."
            )
    return _pipelines[key]


# ── Reranker ──────────────────────────────────────────────────────────────────

def _get_reranker(cfg: AppConfig) -> Any:
    global _reranker
    if _reranker is None:
        _reranker = TransformersSimilarityRanker(
            model=cfg.reranker.model,
            batch_size=cfg.reranker.batch_size,
        )
        _reranker.warm_up()
    return _reranker


def _rerank(
    query: str,
    docs: List[Document],
    top_n: int,
    cfg: AppConfig,
) -> List[Document]:
    if not docs:
        return docs
    ranker = _get_reranker(cfg)
    result = ranker.run(query=query, documents=docs, top_k=top_n)
    return result.get("documents", docs)


# ── Pipeline run helpers ──────────────────────────────────────────────────────

def _run_dense(
    pipeline: Pipeline,
    query: str,
    top_k: int,
    filters: Optional[dict],
    score_threshold: Optional[float],
) -> List[Document]:
    output = pipeline.run({
        "embedder": {"text": query},
        "retriever": {
            "top_k": top_k,
            "filters": filters,
            "score_threshold": score_threshold,
        },
    })
    return output.get("retriever", {}).get("documents", [])


def _run_sparse(
    pipeline: Pipeline,
    query: str,
    top_k: int,
    filters: Optional[dict],
    score_threshold: Optional[float],
) -> List[Document]:
    output = pipeline.run({
        "sparse_embedder": {"text": query},
        "retriever": {
            "top_k": top_k,
            "filters": filters,
            "score_threshold": score_threshold,
        },
    })
    return output.get("retriever", {}).get("documents", [])


def _run_hybrid(
    pipeline: Pipeline,
    query: str,
    top_k: int,
    filters: Optional[dict],
    score_threshold: Optional[float],
) -> List[Document]:
    output = pipeline.run({
        "embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "retriever": {
            "top_k": top_k,
            "filters": filters,
            "score_threshold": score_threshold,
        },
    })
    return output.get("retriever", {}).get("documents", [])


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    cfg: AppConfig,
    top_k: Optional[int] = None,
    filters: Optional[dict] = None,
    mode: Optional[Literal["dense", "sparse", "hybrid"]] = None,
    top_n_after_rerank: Optional[int] = None,
    document_store: Optional[Any] = None,
) -> List[Document]:
    """
    Recupera i documenti più rilevanti per la query.

    Args:
        query: Testo della query di ricerca.
        cfg: Configurazione applicativa completa.
        top_k: Numero di documenti da recuperare. Default: cfg.retrieval.top_k.
        filters: Filtri metadata (dict compatibile Qdrant). Default: None.
        mode: Strategia di retrieval. Default: cfg.retrieval.mode.
        top_n_after_rerank: Documenti finali dopo reranking. Default: cfg.retrieval.top_n_after_rerank.
        document_store: Store opzionale (per test in-memory).

    Returns:
        Lista di Document ordinata per rilevanza decrescente.

    Raises:
        ValueError: se mode non è uno dei valori supportati.
    """
    effective_mode = mode or cfg.retrieval.mode
    effective_top_k = top_k or cfg.retrieval.top_k
    effective_top_n = top_n_after_rerank or cfg.retrieval.top_n_after_rerank

    if effective_mode not in ("dense", "sparse", "hybrid"):
        raise ValueError(
            f"Modalità di retrieval non supportata: '{effective_mode}'. "
            "Valori validi: dense, sparse, hybrid."
        )

    logger.info("Avvio retrieve: mode=%s top_k=%d query='%s'", effective_mode, effective_top_k, query[:80])
    start = time.monotonic()

    store = _get_document_store(cfg, document_store)
    pipeline = _get_pipeline(effective_mode, cfg, store)
    score_threshold = cfg.retrieval.score_threshold

    _runners = {
        "dense": _run_dense,
        "sparse": _run_sparse,
        "hybrid": _run_hybrid,
    }
    docs = _runners[effective_mode](pipeline, query, effective_top_k, filters, score_threshold)

    if cfg.reranker is not None and cfg.reranker.enabled:
        docs = _rerank(query, docs, effective_top_n, cfg)

    logger.info(
        "retrieve completato: %d documenti restituiti in %.2f s",
        len(docs),
        time.monotonic() - start,
    )
    return docs
