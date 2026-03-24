"""Test per le funzioni di metrica IR pure in app/metrics.py."""
import math

import pytest

from app.metrics import (
    IRMetrics,
    aggregate_ir_metrics,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ── recall_at_k ───────────────────────────────────────────────────────────────


class TestRecallAtK:
    def test_empty_retrieved(self):
        assert recall_at_k([], ["doc1"], k=5) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["doc1", "doc2"], [], k=5) == 0.0

    def test_relevant_first_position(self):
        assert recall_at_k(["doc1", "doc2", "doc3"], ["doc1"], k=5) == 1.0

    def test_relevant_at_boundary_k(self):
        assert recall_at_k(["doc1", "doc2", "doc3"], ["doc3"], k=3) == 1.0

    def test_relevant_beyond_k(self):
        assert recall_at_k(["doc1", "doc2", "doc3"], ["doc3"], k=2) == 0.0

    def test_not_found(self):
        assert recall_at_k(["doc1", "doc2"], ["doc99"], k=5) == 0.0

    def test_partial_recall_multiple_relevant(self):
        result = recall_at_k(["doc1", "doc2", "doc3"], ["doc1", "doc4", "doc5"], k=3)
        assert result == pytest.approx(1 / 3)

    def test_full_recall_multiple_relevant(self):
        result = recall_at_k(["doc1", "doc2", "doc3"], ["doc1", "doc2"], k=3)
        assert result == pytest.approx(1.0)


# ── hit_rate_at_k ─────────────────────────────────────────────────────────────


class TestHitRateAtK:
    def test_empty_retrieved(self):
        assert hit_rate_at_k([], ["doc1"], k=5) == 0.0

    def test_empty_relevant(self):
        assert hit_rate_at_k(["doc1"], [], k=5) == 0.0

    def test_relevant_first_position(self):
        assert hit_rate_at_k(["doc1", "doc2"], ["doc1"], k=5) == 1.0

    def test_not_found(self):
        assert hit_rate_at_k(["doc1", "doc2"], ["doc99"], k=5) == 0.0

    def test_relevant_beyond_k(self):
        assert hit_rate_at_k(["doc1", "doc2", "doc3"], ["doc3"], k=2) == 0.0

    def test_one_of_many_relevant_found(self):
        # At least one found → 1.0
        assert hit_rate_at_k(["doc1", "doc2"], ["doc1", "doc99"], k=5) == 1.0


# ── reciprocal_rank / MRR ─────────────────────────────────────────────────────


class TestReciprocalRank:
    def test_empty_retrieved(self):
        assert reciprocal_rank([], ["doc1"]) == 0.0

    def test_empty_relevant(self):
        assert reciprocal_rank(["doc1"], []) == 0.0

    def test_first_position(self):
        assert reciprocal_rank(["doc1", "doc2"], ["doc1"]) == pytest.approx(1.0)

    def test_second_position(self):
        assert reciprocal_rank(["doc1", "doc2"], ["doc2"]) == pytest.approx(0.5)

    def test_third_position(self):
        assert reciprocal_rank(["doc1", "doc2", "doc3"], ["doc3"]) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert reciprocal_rank(["doc1", "doc2"], ["doc99"]) == 0.0

    def test_multiple_relevant_returns_first(self):
        # doc2 at rank 2, doc3 at rank 3 → reciprocal rank = 1/2
        assert reciprocal_rank(["doc1", "doc2", "doc3"], ["doc2", "doc3"]) == pytest.approx(0.5)


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────


class TestNdcgAtK:
    def test_empty_retrieved(self):
        assert ndcg_at_k([], ["doc1"], k=5) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["doc1"], [], k=5) == 0.0

    def test_relevant_first_position(self):
        # Ideal: doc1 at rank 1 → DCG = IDCG = 1/log2(2) = 1.0 → nDCG = 1.0
        assert ndcg_at_k(["doc1", "doc2"], ["doc1"], k=5) == pytest.approx(1.0)

    def test_not_found_within_k(self):
        assert ndcg_at_k(["doc1", "doc2", "doc3"], ["doc99"], k=3) == 0.0

    def test_relevant_beyond_k(self):
        assert ndcg_at_k(["doc1", "doc2", "doc3"], ["doc3"], k=2) == 0.0

    def test_relevant_at_second_position(self):
        # DCG = 1/log2(3), IDCG = 1/log2(2) → nDCG = log2(2)/log2(3)
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert ndcg_at_k(["doc1", "doc2"], ["doc2"], k=2) == pytest.approx(expected)

    def test_perfect_ranking_multiple_relevant(self):
        # Both relevant docs in positions 1 and 2 → nDCG = 1.0
        result = ndcg_at_k(["doc1", "doc2", "doc3"], ["doc1", "doc2"], k=3)
        assert result == pytest.approx(1.0)

    def test_reversed_ranking(self):
        # Relevant at positions 2 and 3, ideal is 1 and 2 → nDCG < 1.0
        result = ndcg_at_k(["doc3", "doc1", "doc2"], ["doc1", "doc2"], k=3)
        # DCG = 1/log2(3) + 1/log2(4), IDCG = 1/log2(2) + 1/log2(3)
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert result == pytest.approx(dcg / idcg)


# ── aggregate_ir_metrics ──────────────────────────────────────────────────────


class TestAggregateIRMetrics:
    def _make_sample(self, k_values, recall, hit_rate, ndcg, mrr):
        s = {"mrr": mrr}
        for k in k_values:
            s[f"recall@{k}"] = recall
            s[f"hit_rate@{k}"] = hit_rate
            s[f"ndcg@{k}"] = ndcg
        return s

    def test_single_perfect_sample(self):
        k_values = [1, 5]
        sample = self._make_sample(k_values, recall=1.0, hit_rate=1.0, ndcg=1.0, mrr=1.0)
        metrics = aggregate_ir_metrics([sample], k_values)
        assert isinstance(metrics, IRMetrics)
        assert metrics.mrr == pytest.approx(1.0)
        assert metrics.recall_at_k[1] == pytest.approx(1.0)
        assert metrics.hit_rate_at_k[5] == pytest.approx(1.0)
        assert metrics.ndcg_at_k[5] == pytest.approx(1.0)

    def test_average_over_two_samples(self):
        k_values = [5]
        s1 = self._make_sample(k_values, recall=1.0, hit_rate=1.0, ndcg=1.0, mrr=1.0)
        s2 = self._make_sample(k_values, recall=0.0, hit_rate=0.0, ndcg=0.0, mrr=0.0)
        metrics = aggregate_ir_metrics([s1, s2], k_values)
        assert metrics.mrr == pytest.approx(0.5)
        assert metrics.recall_at_k[5] == pytest.approx(0.5)
        assert metrics.ndcg_at_k[5] == pytest.approx(0.5)
