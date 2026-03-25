"""Pipeline di evaluation del sistema RAG."""
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import httpx

from app.config import AppConfig
from app.metrics import (
    IRMetrics,
    aggregate_ir_metrics,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval import retrieve

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class GenMetrics:
    faithfulness: float       # 0.0 – 1.0
    answer_relevance: float   # 0.0 – 1.0
    context_precision: float  # 0.0 – 1.0


@dataclass
class EvaluationReport:
    mode: Literal["A", "B"]
    n_samples: int
    ir_metrics: IRMetrics
    gen_metrics: Optional[GenMetrics]  # None per Mode A
    k_values: List[int]
    retrieval_mode: str
    reranker_used: bool
    elapsed_time_seconds: float
    per_sample_results: List[dict]
    # Solo per Mode B:
    k_folds: Optional[int] = None
    ir_metrics_std: Optional[IRMetrics] = None


# ── Qdrant helpers ────────────────────────────────────────────────────────────


def _sample_chunks(cfg: AppConfig, n: int) -> list:
    """Campiona n chunk casuali dalla collection Qdrant."""
    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=cfg.qdrant.host,
        port=cfg.qdrant.port,
        api_key=cfg.qdrant.api_key,
        timeout=cfg.qdrant.timeout,
    )

    try:
        from qdrant_client.models import Sample, SampleQuery

        results = client.query_points(
            collection_name=cfg.qdrant.collection_name,
            query=SampleQuery(sample=Sample.Random),
            limit=n,
            with_payload=True,
        )
        return results.points
    except (ImportError, AttributeError):
        # Fallback per versioni precedenti di qdrant-client che non supportano SampleQuery
        records, _ = client.scroll(
            collection_name=cfg.qdrant.collection_name,
            limit=max(n * 5, 200),
            with_payload=True,
            with_vectors=False,
        )
        random.shuffle(records)
        return records[:n]


def _extract_chunk_id(record) -> Optional[str]:
    """Estrae il Haystack document ID dal payload del record Qdrant."""
    payload = getattr(record, "payload", {}) or {}
    doc_id = payload.get("id") or payload.get("_id")
    if doc_id:
        return str(doc_id)
    return str(record.id)


def _extract_chunk_text(record) -> Optional[str]:
    """Estrae il testo dal payload del record Qdrant."""
    payload = getattr(record, "payload", {}) or {}
    return payload.get("content") or payload.get("text")


# ── LLM helper ────────────────────────────────────────────────────────────────

QueryKind = Literal["question", "keywords"]


def _resolve_query_kind(cfg: AppConfig) -> QueryKind:
    """Determina il tipo di query da generare in base alla configurazione."""
    query_type = cfg.evaluation.mode_a.query_type
    if query_type == "auto":
        return "keywords" if cfg.retrieval.mode == "sparse" else "question"
    return query_type  # type: ignore[return-value]


def _generate_query(chunk_text: str, kind: QueryKind, cfg: AppConfig) -> Optional[str]:
    """Genera una domanda sintetica o una lista di keyword per un chunk tramite il LLM."""
    if kind == "keywords":
        prompt = cfg.evaluation.mode_a.keyword_gen_prompt.format(chunk=chunk_text)
    else:
        prompt = cfg.evaluation.mode_a.question_gen_prompt.format(chunk=chunk_text)

    try:
        if cfg.llm.provider == "openai":
            from openai import OpenAI

            client = OpenAI(
                api_key=cfg.llm.openai_api_key,
                base_url=cfg.llm.openai_base_url,
                timeout=cfg.llm.timeout,
            )
            response = client.chat.completions.create(
                model=cfg.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.llm.temperature,
                max_tokens=cfg.llm.max_tokens,
            )
            return response.choices[0].message.content.strip()
        else:
            payload = {
                "model": cfg.llm.model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    "temperature": cfg.llm.temperature,
                    "num_predict": cfg.llm.max_tokens,
                },
                "stream": False,
            }
            with httpx.Client(timeout=cfg.llm.timeout) as client:
                resp = client.post(f"{cfg.llm.host}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"].strip()
    except httpx.TimeoutException as exc:
        logger.warning("Timeout LLM nella generazione della query: %s", exc)
        return None
    except httpx.HTTPError as exc:
        logger.warning("Errore HTTP LLM nella generazione della query: %s", exc)
        return None
    except (KeyError, ValueError) as exc:
        logger.warning("Risposta LLM malformata nella generazione della query: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Errore LLM nella generazione della query: %s", exc)
        return None


# ── Report persistence ────────────────────────────────────────────────────────


def _save_report(report: EvaluationReport, output_dir: str) -> Path:
    """Salva il report di evaluation come JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    filename = out_path / f"eval_mode_{report.mode}_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    logger.info("Report salvato in %s", filename)
    return filename


# ── Mode A ────────────────────────────────────────────────────────────────────


def run_evaluation_mode_a(cfg: AppConfig) -> EvaluationReport:
    """
    Esegue la valutazione Mode A: genera domande sintetiche dai chunk
    e misura la qualità del retrieval.

    Args:
        cfg: Configurazione applicativa.

    Returns:
        EvaluationReport con metriche IR aggregate.
    """
    mode_cfg = cfg.evaluation.mode_a
    k_values = mode_cfg.k_values
    n_samples = mode_cfg.n_samples
    start = time.monotonic()

    query_kind = _resolve_query_kind(cfg)
    logger.info(
        "Avvio evaluation Mode A: %d campioni, k_values=%s, query_type=%s, retrieval_mode=%s",
        n_samples, k_values, query_kind, cfg.retrieval.mode,
    )

    # 1. Campionamento chunk casuali
    chunks = _sample_chunks(cfg, n_samples)
    if not chunks:
        raise RuntimeError("Nessun chunk disponibile nella collection Qdrant")

    per_sample: List[dict] = []

    for chunk in chunks:
        chunk_id = _extract_chunk_id(chunk)
        chunk_text = _extract_chunk_text(chunk)

        if not chunk_text:
            logger.warning("Chunk %s senza contenuto testuale, saltato", chunk_id)
            continue

        # 2. Generazione domanda sintetica o keyword
        query = _generate_query(chunk_text, query_kind, cfg)
        if not query:
            logger.warning("Impossibile generare query per chunk %s, saltato", chunk_id)
            continue

        # 3. Retrieval
        retrieved_docs = retrieve(query=query, cfg=cfg)
        retrieved_ids = [str(d.id) for d in retrieved_docs if d.id]
        relevant_ids = [chunk_id]

        # 4. Calcolo metriche per campione
        sample_result: dict = {
            "chunk_id": chunk_id,
            "query_type": query_kind,
            "query": query,
            "retrieved_ids": retrieved_ids,
            "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
        }
        for k in k_values:
            sample_result[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            sample_result[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, relevant_ids, k)
            sample_result[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

        per_sample.append(sample_result)

    if not per_sample:
        raise RuntimeError("Nessun campione valido processato durante l'evaluation")

    # 5. Aggregazione metriche
    ir_metrics = aggregate_ir_metrics(per_sample, k_values)
    elapsed = time.monotonic() - start
    reranker_used = cfg.reranker is not None and cfg.reranker.enabled

    report = EvaluationReport(
        mode="A",
        n_samples=len(per_sample),
        ir_metrics=ir_metrics,
        gen_metrics=None,
        k_values=k_values,
        retrieval_mode=cfg.retrieval.mode,
        reranker_used=reranker_used,
        elapsed_time_seconds=elapsed,
        per_sample_results=per_sample,
    )

    # 6. Salvataggio report
    _save_report(report, cfg.evaluation.output_dir)

    logger.info(
        "Evaluation Mode A completata in %.1f s. MRR=%.3f",
        elapsed,
        ir_metrics.mrr,
    )
    return report


def run_evaluation_mode_b(dataset_path: str, cfg: AppConfig) -> EvaluationReport:
    """
    Esegue la valutazione Mode B: K-Fold Cross-Validation su dataset annotato.
    Calcola metriche IR e metriche di generazione (LLM-as-judge).

    Args:
        dataset_path: Percorso al file JSONL del dataset di test.
        cfg: Configurazione applicativa.

    Returns:
        EvaluationReport con metriche IR, metriche di generazione e statistiche
        di varianza tra i fold.
    """
    raise NotImplementedError("Mode B non ancora implementato")
