"""Funzioni di metrica IR pure per l'evaluation del sistema RAG."""
import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class IRMetrics:
    recall_at_k: Dict[int, float]      # {1: 0.62, 3: 0.78, 5: 0.82, 10: 0.91}
    hit_rate_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant documents found in top-K."""
    if not relevant_ids:
        return 0.0
    top_k_ids = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k_ids & relevant_set) / len(relevant_set)


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """1 if at least one relevant document is in top-K, else 0."""
    top_k_ids = set(retrieved_ids[:k])
    return 1.0 if any(rid in top_k_ids for rid in relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Reciprocal of the rank of the first relevant document."""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (rilevanza binaria)."""
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(top_k)
        if doc_id in relevant_set
    )

    n_relevant_in_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_k))

    return dcg / idcg if idcg > 0 else 0.0


def aggregate_ir_metrics(per_sample: List[dict], k_values: List[int]) -> IRMetrics:
    """Aggrega le metriche IR su tutti i campioni."""
    n = len(per_sample)
    return IRMetrics(
        recall_at_k={k: sum(s[f"recall@{k}"] for s in per_sample) / n for k in k_values},
        hit_rate_at_k={k: sum(s[f"hit_rate@{k}"] for s in per_sample) / n for k in k_values},
        mrr=sum(s["mrr"] for s in per_sample) / n,
        ndcg_at_k={k: sum(s[f"ndcg@{k}"] for s in per_sample) / n for k in k_values},
    )
