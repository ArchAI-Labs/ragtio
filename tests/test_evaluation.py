"""Test per app/evaluation.py."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import AppConfig
from app.evaluation import (
    EvaluationReport,
    _extract_chunk_id,
    _extract_chunk_text,
    _generate_query,
    _resolve_query_kind,
    _save_report,
    run_evaluation_mode_a,
)
from app.metrics import IRMetrics


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_record(payload: dict, record_id: str = "qdrant-uuid-1"):
    r = MagicMock()
    r.payload = payload
    r.id = record_id
    return r


def make_report(**overrides) -> EvaluationReport:
    defaults = dict(
        mode="A",
        n_samples=5,
        ir_metrics=IRMetrics(
            recall_at_k={5: 0.8},
            hit_rate_at_k={5: 0.9},
            mrr=0.75,
            ndcg_at_k={5: 0.82},
        ),
        gen_metrics=None,
        k_values=[5],
        retrieval_mode="hybrid",
        reranker_used=False,
        elapsed_time_seconds=1.5,
        per_sample_results=[],
    )
    defaults.update(overrides)
    return EvaluationReport(**defaults)


# ── _extract_chunk_id ────────────────────────────────────────────────────────


class TestExtractChunkId:
    def test_payload_has_id(self):
        record = make_record({"id": "haystack-doc-123"})
        assert _extract_chunk_id(record) == "haystack-doc-123"

    def test_payload_has__id(self):
        record = make_record({"_id": "alt-id-456"})
        assert _extract_chunk_id(record) == "alt-id-456"

    def test_fallback_to_record_id(self):
        record = make_record({}, record_id="fallback-id")
        assert _extract_chunk_id(record) == "fallback-id"

    def test_id_preferred_over__id(self):
        record = make_record({"id": "primary", "_id": "secondary"})
        assert _extract_chunk_id(record) == "primary"

    def test_id_converted_to_str(self):
        record = make_record({"id": 42})
        assert _extract_chunk_id(record) == "42"


# ── _extract_chunk_text ──────────────────────────────────────────────────────


class TestExtractChunkText:
    def test_extracts_content(self):
        record = make_record({"content": "some text"})
        assert _extract_chunk_text(record) == "some text"

    def test_extracts_text_field(self):
        record = make_record({"text": "alternative text"})
        assert _extract_chunk_text(record) == "alternative text"

    def test_returns_none_if_missing(self):
        record = make_record({})
        assert _extract_chunk_text(record) is None

    def test_content_preferred_over_text(self):
        record = make_record({"content": "primary", "text": "secondary"})
        assert _extract_chunk_text(record) == "primary"

    def test_no_payload_returns_none(self):
        record = MagicMock()
        record.payload = None
        assert _extract_chunk_text(record) is None


# ── _resolve_query_kind ──────────────────────────────────────────────────────


class TestResolveQueryKind:
    def test_auto_dense_returns_question(self):
        cfg = AppConfig()
        cfg.retrieval.mode = "dense"
        cfg.evaluation.mode_a.query_type = "auto"
        assert _resolve_query_kind(cfg) == "question"

    def test_auto_hybrid_returns_question(self):
        cfg = AppConfig()
        cfg.retrieval.mode = "hybrid"
        cfg.evaluation.mode_a.query_type = "auto"
        assert _resolve_query_kind(cfg) == "question"

    def test_auto_sparse_returns_keywords(self):
        cfg = AppConfig()
        cfg.retrieval.mode = "sparse"
        cfg.evaluation.mode_a.query_type = "auto"
        assert _resolve_query_kind(cfg) == "keywords"

    def test_explicit_question(self):
        cfg = AppConfig()
        cfg.evaluation.mode_a.query_type = "question"
        assert _resolve_query_kind(cfg) == "question"

    def test_explicit_keywords(self):
        cfg = AppConfig()
        cfg.evaluation.mode_a.query_type = "keywords"
        assert _resolve_query_kind(cfg) == "keywords"


# ── _generate_query ──────────────────────────────────────────────────────────


class TestGenerateQuery:
    def test_ollama_question_success(self, monkeypatch):
        import httpx

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "  Cosa descrive il testo?  "}}

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, **kwargs):
                return mock_resp

        monkeypatch.setattr(httpx, "Client", MockClient)

        cfg = AppConfig()
        result = _generate_query("testo di esempio", "question", cfg)
        assert result == "Cosa descrive il testo?"

    def test_ollama_keywords_uses_keyword_prompt(self, monkeypatch):
        import httpx

        captured = {}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "kw1, kw2"}}

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, json=None, **kwargs):
                captured["payload"] = json
                return mock_resp

        monkeypatch.setattr(httpx, "Client", MockClient)

        cfg = AppConfig()
        _generate_query("chunk testo", "keywords", cfg)

        # The keyword prompt should contain the chunk text
        messages = captured["payload"]["messages"]
        assert any("chunk testo" in m.get("content", "") for m in messages)

    def test_returns_none_on_timeout(self, monkeypatch):
        import httpx

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, **kwargs):
                raise httpx.TimeoutException("timeout")

        monkeypatch.setattr(httpx, "Client", MockClient)

        cfg = AppConfig()
        result = _generate_query("chunk", "question", cfg)
        assert result is None

    def test_returns_none_on_http_error(self, monkeypatch):
        import httpx

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, **kwargs):
                raise httpx.HTTPError("error")

        monkeypatch.setattr(httpx, "Client", MockClient)

        cfg = AppConfig()
        result = _generate_query("chunk", "question", cfg)
        assert result is None

    def test_returns_none_on_malformed_response(self, monkeypatch):
        import httpx

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"unexpected": "structure"}  # no "message" key

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, **kwargs):
                return mock_resp

        monkeypatch.setattr(httpx, "Client", MockClient)

        cfg = AppConfig()
        result = _generate_query("chunk", "question", cfg)
        assert result is None


# ── _save_report ──────────────────────────────────────────────────────────────


class TestSaveReport:
    def test_saves_json_file(self, tmp_path):
        report = make_report()
        path = _save_report(report, str(tmp_path))
        assert path.exists()
        assert path.suffix == ".json"

    def test_json_content_correct(self, tmp_path):
        report = make_report(n_samples=42)
        path = _save_report(report, str(tmp_path))
        data = json.loads(path.read_text())
        assert data["mode"] == "A"
        assert data["n_samples"] == 42

    def test_filename_contains_mode(self, tmp_path):
        report = make_report()
        path = _save_report(report, str(tmp_path))
        assert "mode_A" in path.name

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "nested" / "subdir"
        report = make_report()
        _save_report(report, str(nested))
        assert nested.exists()

    def test_multiple_reports_get_unique_names(self, tmp_path):
        import time

        report = make_report()
        path1 = _save_report(report, str(tmp_path))
        time.sleep(1.1)  # ensure different timestamp
        path2 = _save_report(report, str(tmp_path))
        assert path1 != path2


# ── run_evaluation_mode_a ────────────────────────────────────────────────────


class TestRunEvaluationModeA:
    def _make_chunk(self, chunk_id: str, text: str):
        return make_record({"id": chunk_id, "content": text}, record_id=chunk_id)

    def test_returns_evaluation_report(self, tmp_path):
        cfg = AppConfig()
        cfg.evaluation.mode_a.n_samples = 2
        cfg.evaluation.mode_a.k_values = [5]
        cfg.evaluation.output_dir = str(tmp_path)

        chunks = [
            self._make_chunk("c1", "Testo del primo documento."),
            self._make_chunk("c2", "Testo del secondo documento."),
        ]

        with patch("app.evaluation._sample_chunks", return_value=chunks), \
             patch("app.evaluation._generate_query", return_value="domanda test"), \
             patch("app.evaluation.retrieve", return_value=[]):
            report = run_evaluation_mode_a(cfg)

        assert isinstance(report, EvaluationReport)
        assert report.mode == "A"
        assert report.n_samples == 2
        assert report.gen_metrics is None

    def test_no_chunks_raises_runtime_error(self, tmp_path):
        cfg = AppConfig()
        cfg.evaluation.output_dir = str(tmp_path)

        with patch("app.evaluation._sample_chunks", return_value=[]):
            with pytest.raises(RuntimeError, match="Nessun chunk"):
                run_evaluation_mode_a(cfg)

    def test_skips_chunks_without_text(self, tmp_path):
        cfg = AppConfig()
        cfg.evaluation.mode_a.n_samples = 2
        cfg.evaluation.mode_a.k_values = [1]
        cfg.evaluation.output_dir = str(tmp_path)

        chunks = [
            make_record({}, "no-text-id"),  # no content
            self._make_chunk("valid-id", "Testo valido."),
        ]

        with patch("app.evaluation._sample_chunks", return_value=chunks), \
             patch("app.evaluation._generate_query", return_value="query"), \
             patch("app.evaluation.retrieve", return_value=[]):
            report = run_evaluation_mode_a(cfg)

        assert report.n_samples == 1

    def test_skips_chunks_with_failed_query_generation(self, tmp_path):
        cfg = AppConfig()
        cfg.evaluation.mode_a.n_samples = 2
        cfg.evaluation.mode_a.k_values = [1]
        cfg.evaluation.output_dir = str(tmp_path)

        chunks = [
            self._make_chunk("c1", "Testo uno."),
            self._make_chunk("c2", "Testo due."),
        ]

        call_count = {"n": 0}

        def mock_gen(*args, **kwargs):
            call_count["n"] += 1
            return "query" if call_count["n"] > 1 else None

        with patch("app.evaluation._sample_chunks", return_value=chunks), \
             patch("app.evaluation._generate_query", side_effect=mock_gen), \
             patch("app.evaluation.retrieve", return_value=[]):
            report = run_evaluation_mode_a(cfg)

        assert report.n_samples == 1

    def test_all_queries_fail_raises_runtime_error(self, tmp_path):
        cfg = AppConfig()
        cfg.evaluation.mode_a.n_samples = 1
        cfg.evaluation.output_dir = str(tmp_path)

        chunks = [self._make_chunk("c1", "testo")]

        with patch("app.evaluation._sample_chunks", return_value=chunks), \
             patch("app.evaluation._generate_query", return_value=None):
            with pytest.raises(RuntimeError, match="Nessun campione"):
                run_evaluation_mode_a(cfg)

    def test_saves_report_file(self, tmp_path):
        cfg = AppConfig()
        cfg.evaluation.mode_a.n_samples = 1
        cfg.evaluation.mode_a.k_values = [5]
        cfg.evaluation.output_dir = str(tmp_path)

        chunks = [self._make_chunk("c1", "testo")]

        with patch("app.evaluation._sample_chunks", return_value=chunks), \
             patch("app.evaluation._generate_query", return_value="query"), \
             patch("app.evaluation.retrieve", return_value=[]):
            run_evaluation_mode_a(cfg)

        reports = list(tmp_path.glob("eval_mode_A_*.json"))
        assert len(reports) == 1

    def test_ir_metrics_computed(self, tmp_path):
        from haystack import Document

        cfg = AppConfig()
        cfg.evaluation.mode_a.n_samples = 1
        cfg.evaluation.mode_a.k_values = [5]
        cfg.evaluation.output_dir = str(tmp_path)

        chunk_id = "c1"
        chunks = [self._make_chunk(chunk_id, "testo")]

        # Retrieval returns the chunk itself as top result
        retrieved_doc = Document(id=chunk_id, content="testo")

        with patch("app.evaluation._sample_chunks", return_value=chunks), \
             patch("app.evaluation._generate_query", return_value="query"), \
             patch("app.evaluation.retrieve", return_value=[retrieved_doc]):
            report = run_evaluation_mode_a(cfg)

        assert report.ir_metrics.mrr == pytest.approx(1.0)
        assert report.ir_metrics.hit_rate_at_k[5] == pytest.approx(1.0)

    def test_reranker_used_flag(self, tmp_path):
        from app.config import RerankerConfig

        cfg = AppConfig()
        cfg.evaluation.mode_a.n_samples = 1
        cfg.evaluation.mode_a.k_values = [5]
        cfg.evaluation.output_dir = str(tmp_path)
        cfg.reranker = RerankerConfig(enabled=True)

        chunks = [self._make_chunk("c1", "testo")]

        with patch("app.evaluation._sample_chunks", return_value=chunks), \
             patch("app.evaluation._generate_query", return_value="query"), \
             patch("app.evaluation.retrieve", return_value=[]):
            report = run_evaluation_mode_a(cfg)

        assert report.reranker_used is True

    def test_per_sample_results_populated(self, tmp_path):
        cfg = AppConfig()
        cfg.evaluation.mode_a.n_samples = 2
        cfg.evaluation.mode_a.k_values = [5]
        cfg.evaluation.output_dir = str(tmp_path)

        chunks = [
            self._make_chunk("c1", "testo uno"),
            self._make_chunk("c2", "testo due"),
        ]

        with patch("app.evaluation._sample_chunks", return_value=chunks), \
             patch("app.evaluation._generate_query", return_value="query"), \
             patch("app.evaluation.retrieve", return_value=[]):
            report = run_evaluation_mode_a(cfg)

        assert len(report.per_sample_results) == 2
        for sample in report.per_sample_results:
            assert "chunk_id" in sample
            assert "query" in sample
            assert "mrr" in sample
            assert "recall@5" in sample
            assert "hit_rate@5" in sample
            assert "ndcg@5" in sample
