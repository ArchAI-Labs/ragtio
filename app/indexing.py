"""Pipeline di indicizzazione documenti su Qdrant."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from haystack import Document, Pipeline, component
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

from app.config import AppConfig, EmbedderConfig

logger = logging.getLogger(__name__)

_EMBEDDING_DIMS = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-zh-v1.5": 768,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "intfloat/multilingual-e5-small": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


def _register_custom_embedder(cfg: EmbedderConfig) -> None:
    """Registra un custom embedder ONNX via TextEmbedding.add_custom_model().

    Deve essere chiamata prima di istanziare qualsiasi FastembedTextEmbedder
    quando cfg.custom è presente.
    """
    if cfg.custom is None:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    TextEmbedding.add_custom_model(
        model=cfg.model,
        pooling=PoolingType[cfg.custom.pooling],
        normalization=cfg.custom.normalization,
        sources=ModelSource(
            hf=cfg.custom.hf_repo,
            url=cfg.custom.url,
        ),
        dim=cfg.custom.dim,
        model_file=cfg.custom.model_file,
    )
    logger.debug(
        "Custom embedder registrato: model=%s dim=%d pooling=%s",
        cfg.model,
        cfg.custom.dim,
        cfg.custom.pooling,
    )


def _get_embedding_dim(cfg: EmbedderConfig) -> int:
    """Risolve la dimensione del vettore di embedding con priorità decrescente:
    1. cfg.embedding_dim (override esplicito)
    2. cfg.custom.dim (custom embedder)
    3. dizionario statico noto
    4. query a fastembed a runtime (fallback)
    """
    if cfg.embedding_dim is not None:
        return cfg.embedding_dim
    if cfg.custom is not None:
        return cfg.custom.dim
    if cfg.model in _EMBEDDING_DIMS:
        return _EMBEDDING_DIMS[cfg.model]
    # Fallback: interroga fastembed direttamente
    try:
        from fastembed import TextEmbedding
        _register_custom_embedder(cfg)
        model = TextEmbedding(model_name=cfg.model, cache_dir=cfg.cache_dir)
        dim = model.dim
        logger.info("Dimensione embedding rilevata da fastembed: model=%s dim=%d", cfg.model, dim)
        return dim
    except Exception as exc:
        logger.warning(
            "Impossibile rilevare la dimensione per '%s' (%s). Uso fallback 1024.",
            cfg.model,
            exc,
        )
        return 1024


@dataclass
class IndexingResult:
    """Risultato dell'operazione di indicizzazione."""

    n_documents_ingested: int
    n_chunks: int
    n_documents_written: int
    elapsed_time_seconds: float
    errors: List[str] = field(default_factory=list)


@component
class RecursiveDocumentSplitter:
    """Haystack component per splitting ricorsivo con separatori a priorità decrescente."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else ["\n\n", "\n", " ", ""]

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        result = []
        for doc in documents:
            chunks = self._split(doc.content or "", self.separators)
            for i, text in enumerate(chunks):
                meta = {**doc.meta, "parent_id": doc.id, "chunk_index": i}
                result.append(Document(content=text, meta=meta))
        return {"documents": result}

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text]

        for i, sep in enumerate(separators):
            if sep == "":
                return self._char_split(text)
            if sep not in text:
                continue
            parts = text.split(sep)
            merged = self._merge_parts(parts, sep, separators[i + 1:])
            return self._add_overlap(merged)

        return self._char_split(text)

    def _merge_parts(self, parts: List[str], sep: str, remaining: List[str]) -> List[str]:
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    sub = self._split(part, remaining)
                    if sub:
                        chunks.extend(sub[:-1])
                        current = sub[-1]
                    else:
                        current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    def _char_split(self, text: str) -> List[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i: i + self.chunk_size] for i in range(0, len(text), step)]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap:]
            result.append(tail + chunks[i])
        return result


def _get_splitter(cfg: AppConfig):
    strategy = cfg.indexing.chunking.strategy
    chunk_size = cfg.indexing.chunking.chunk_size
    chunk_overlap = cfg.indexing.chunking.chunk_overlap

    if strategy == "character":
        # DocumentSplitter non supporta split_by="character"; si usa il splitter ricorsivo
        # con separatori vuoti per ottenere un taglio a carattere puro
        return RecursiveDocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[""],
        )
    if strategy == "paragraph":
        return DocumentSplitter(
            split_by="passage",
            split_length=1,
            split_overlap=0,
        )
    if strategy == "recursive":
        return RecursiveDocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=cfg.indexing.chunking.separators or ["\n\n", "\n", " ", ""],
        )
    raise ValueError(f"Strategia di chunking non supportata: '{strategy}'")


def build_index(
    documents: List[Document],
    cfg: AppConfig,
    document_store: Optional[Any] = None,
    embedder: Optional[Any] = None,
) -> IndexingResult:
    """
    Esegue la pipeline di indicizzazione su una lista di Document Haystack.

    Args:
        documents: Lista di Document prodotti dal modulo di ingestione.
        cfg: Configurazione applicativa completa.
        document_store: DocumentStore opzionale (per test con in-memory Qdrant).
        embedder: Embedder opzionale (per test senza modelli reali).

    Returns:
        IndexingResult con statistiche sull'operazione.
    """
    from haystack.components.writers import DocumentWriter
    from haystack.document_stores.types import DuplicatePolicy
    from haystack_integrations.components.embedders.fastembed import (
        FastembedDocumentEmbedder as FastEmbedDocumentEmbedder,
    )
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    start = time.monotonic()
    errors: List[str] = []
    n_ingested = len(documents)

    logger.info("Avvio build_index: %d documenti da indicizzare", n_ingested)

    # Crea document store se non fornito
    if document_store is None:
        embedding_dim = _get_embedding_dim(cfg.embedder)
        document_store = QdrantDocumentStore(
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
            api_key=cfg.qdrant.api_key,
            index=cfg.qdrant.collection_name,
            embedding_dim=embedding_dim,
            use_sparse_embeddings=True,
            recreate_index=False,
        )

    # Step 1: Pulizia
    # remove_extra_whitespaces=False: evita di collassare \n\n (separatori di paragrafo)
    # remove_empty_lines=False: stessa ragione
    cleaner = DocumentCleaner(
        remove_empty_lines=False,
        remove_extra_whitespaces=False,
        remove_repeated_substrings=False,
    )
    cleaned = cleaner.run(documents=documents)["documents"]

    # Step 2: Splitting
    splitter = _get_splitter(cfg)
    chunks: List[Document] = splitter.run(documents=cleaned)["documents"]

    # Step 3: Filtra chunk corti e logga warning per paragrafi lunghi
    strategy = cfg.indexing.chunking.strategy
    min_len = cfg.indexing.min_chunk_length
    max_para_len = cfg.indexing.chunking.max_paragraph_length

    valid_chunks: List[Document] = []
    for chunk in chunks:
        length = len(chunk.content or "")
        if length < min_len:
            errors.append(f"Chunk scartato ({length} < {min_len} caratteri): '{(chunk.content or '')[:40]}...'")
            continue
        if strategy == "paragraph" and max_para_len and length > max_para_len:
            logger.warning(
                "Chunk paragrafo supera max_paragraph_length (%d > %d): '%s...'",
                length,
                max_para_len,
                (chunk.content or "")[:80],
            )
        valid_chunks.append(chunk)

    n_chunks = len(valid_chunks)

    if not valid_chunks:
        elapsed = time.monotonic() - start
        logger.info(
            "build_index completato: %d doc ingested, 0 chunk validi, 0 scritti in %.2f s",
            n_ingested,
            elapsed,
        )
        return IndexingResult(
            n_documents_ingested=n_ingested,
            n_chunks=0,
            n_documents_written=0,
            elapsed_time_seconds=elapsed,
            errors=errors,
        )

    # Step 4: Pipeline embedding + scrittura
    policy = (
        DuplicatePolicy.SKIP
        if cfg.indexing.duplicate_policy == "SKIP"
        else DuplicatePolicy.OVERWRITE
    )

    if embedder is None:
        _register_custom_embedder(cfg.embedder)
        embedder = FastEmbedDocumentEmbedder(
            model=cfg.embedder.model,
            batch_size=cfg.indexing.batch_size,
            cache_dir=cfg.embedder.cache_dir,
        )

    writer = DocumentWriter(document_store=document_store, policy=policy)

    pipeline = Pipeline()
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("writer", writer)
    pipeline.connect("embedder.documents", "writer.documents")

    output = pipeline.run({"embedder": {"documents": valid_chunks}})
    n_written = output.get("writer", {}).get("documents_written", 0)

    elapsed = time.monotonic() - start
    logger.info(
        "build_index completato: %d doc ingested, %d chunk, %d scritti in Qdrant in %.2f s",
        n_ingested,
        n_chunks,
        n_written,
        elapsed,
    )
    return IndexingResult(
        n_documents_ingested=n_ingested,
        n_chunks=n_chunks,
        n_documents_written=n_written,
        elapsed_time_seconds=elapsed,
        errors=errors,
    )
