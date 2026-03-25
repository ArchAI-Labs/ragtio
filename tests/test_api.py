"""Test per app/api.py — helper functions and API endpoints."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from haystack import Document

from app.api import QueryRequest, _apply_query_overrides, _build_ollama_payload, app
from app.config import (
    AppConfig,
    DecompositionConfig,
    ExpansionConfig,
    QueryEnhancementConfig,
)
from app.indexing import IndexingResult
from app.query_enhancement import EnhancedQuery


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


@pytest.fixture
def client(cfg):
    with TestClient(app) as c:
        app.state.cfg = cfg
        yield c


# ── _apply_query_overrides ────────────────────────────────────────────────────


class TestApplyQueryOverrides:
    def _req(self, **kwargs):
        return QueryRequest(query="test", **kwargs)

    def test_no_overrides_returns_same_object(self, cfg):
        req = self._req()
        result = _apply_query_overrides(cfg, req)
        assert result is cfg

    def test_retrieval_mode_override_deep_copies(self, cfg):
        req = self._req(retrieval_mode="sparse")
        result = _apply_query_overrides(cfg, req)
        assert result is not cfg
        assert result.retrieval.mode == "sparse"

    def test_original_cfg_not_mutated(self, cfg):
        original_mode = cfg.retrieval.mode
        req = self._req(retrieval_mode="dense")
        _apply_query_overrides(cfg, req)
        assert cfg.retrieval.mode == original_mode

    def test_top_k_override(self, cfg):
        req = self._req(top_k=7)
        result = _apply_query_overrides(cfg, req)
        assert result.retrieval.top_k == 7

    def test_top_n_after_rerank_override(self, cfg):
        req = self._req(top_n_after_rerank=3)
        result = _apply_query_overrides(cfg, req)
        assert result.retrieval.top_n_after_rerank == 3

    def test_expansion_override_when_configured(self, cfg):
        cfg.query_enhancement = QueryEnhancementConfig(
            expansion=ExpansionConfig(enabled=False)
        )
        req = self._req(use_expansion=True)
        result = _apply_query_overrides(cfg, req)
        assert result.query_enhancement.expansion.enabled is True

    def test_expansion_override_no_effect_when_not_configured(self, cfg):
        cfg.query_enhancement = None
        req = self._req(use_expansion=True)
        result = _apply_query_overrides(cfg, req)
        assert result.query_enhancement is None

    def test_decomposition_override_when_configured(self, cfg):
        cfg.query_enhancement = QueryEnhancementConfig(
            decomposition=DecompositionConfig(enabled=False)
        )
        req = self._req(use_decomposition=True)
        result = _apply_query_overrides(cfg, req)
        assert result.query_enhancement.decomposition.enabled is True


# ── _build_ollama_payload ─────────────────────────────────────────────────────


class TestBuildOllamaPayload:
    def test_structure(self, cfg):
        payload = _build_ollama_payload("ctx", "question", cfg, stream=False)
        assert payload["model"] == cfg.llm.model
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    def test_context_in_user_prompt(self, cfg):
        payload = _build_ollama_payload("my context", "q", cfg, stream=False)
        assert "my context" in payload["messages"][1]["content"]

    def test_question_in_user_prompt(self, cfg):
        payload = _build_ollama_payload("ctx", "my question", cfg, stream=False)
        assert "my question" in payload["messages"][1]["content"]

    def test_stream_true(self, cfg):
        payload = _build_ollama_payload("ctx", "q", cfg, stream=True)
        assert payload["stream"] is True

    def test_options_match_cfg(self, cfg):
        payload = _build_ollama_payload("ctx", "q", cfg, stream=False)
        assert payload["options"]["temperature"] == pytest.approx(cfg.llm.temperature)
        assert payload["options"]["top_p"] == pytest.approx(cfg.llm.top_p)
        assert payload["options"]["num_predict"] == cfg.llm.max_tokens


# ── /api/ingest ───────────────────────────────────────────────────────────────


class TestApiIngest:
    def test_success_txt(self, client, tmp_path):
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("Test document content")

        result_mock = IndexingResult(
            n_documents_ingested=1,
            n_chunks=1,
            n_documents_written=1,
            elapsed_time_seconds=0.1,
        )

        with patch("app.api.ingest", return_value=[Document(content="Test")]), \
             patch("app.api.build_index", return_value=result_mock):
            with open(txt_file, "rb") as f:
                response = client.post(
                    "/api/ingest", files={"file": ("doc.txt", f, "text/plain")}
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["n_documents_ingested"] == 1
        assert data["n_chunks"] == 1

    def test_file_too_large_returns_413(self, client, cfg, tmp_path):
        cfg.api.max_upload_size_mb = 1
        oversized = tmp_path / "large.txt"
        oversized.write_bytes(b"X" * (2 * 1024 * 1024))

        with open(oversized, "rb") as f:
            response = client.post(
                "/api/ingest", files={"file": ("large.txt", f, "text/plain")}
            )

        assert response.status_code == 413

    def test_unsupported_format_returns_400(self, client, tmp_path):
        from app.ingest import UnsupportedFormatError

        xyz_file = tmp_path / "doc.xyz"
        xyz_file.write_text("content")

        with patch("app.api.ingest", side_effect=UnsupportedFormatError(".xyz")):
            with open(xyz_file, "rb") as f:
                response = client.post(
                    "/api/ingest", files={"file": ("doc.xyz", f, "text/plain")}
                )

        assert response.status_code == 400

    def test_invalid_extra_metadata_returns_400(self, client, tmp_path):
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("content")

        with open(txt_file, "rb") as f:
            response = client.post(
                "/api/ingest",
                files={"file": ("doc.txt", f, "text/plain")},
                data={"extra_metadata": "not-valid-json"},
            )

        assert response.status_code == 400
        assert "extra_metadata" in response.json()["detail"]

    def test_invalid_duplicate_policy_returns_400(self, client, tmp_path):
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("content")

        with open(txt_file, "rb") as f:
            response = client.post(
                "/api/ingest",
                files={"file": ("doc.txt", f, "text/plain")},
                data={"duplicate_policy": "INVALID"},
            )

        assert response.status_code == 400

    def test_content_column_not_found_returns_400(self, client, tmp_path):
        csv_file = tmp_path / "doc.csv"
        csv_file.write_text("a,b\n1,2")

        with patch("app.api.ingest", side_effect=ValueError("Colonne disponibili: ['a', 'b']")):
            with open(csv_file, "rb") as f:
                response = client.post(
                    "/api/ingest",
                    files={"file": ("doc.csv", f, "text/csv")},
                    data={"content_column": "missing"},
                )

        assert response.status_code == 400

    def test_metadata_columns_parsed_as_list(self, client, tmp_path):
        """metadata_columns stringa 'a,b,c' viene splittatain lista."""
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("content")

        result_mock = IndexingResult(
            n_documents_ingested=1, n_chunks=1, n_documents_written=1,
            elapsed_time_seconds=0.1,
        )

        with patch("app.api.ingest", return_value=[Document(content="x")]) as mock_ingest, \
             patch("app.api.build_index", return_value=result_mock):
            with open(txt_file, "rb") as f:
                client.post(
                    "/api/ingest",
                    files={"file": ("doc.txt", f, "text/plain")},
                    data={"content_column": "col", "metadata_columns": "a, b, c"},
                )

        _, kwargs = mock_ingest.call_args
        assert kwargs["metadata_columns"] == ["a", "b", "c"]


# ── /api/query ────────────────────────────────────────────────────────────────


class TestApiQuery:
    def test_no_docs_returns_default_answer(self, client):
        enhanced = EnhancedQuery(original_query="test")

        with patch("app.api._collect_docs", return_value=([], enhanced)):
            response = client.post("/api/query", json={"query": "test"})

        assert response.status_code == 200
        data = response.json()
        assert "Non ho trovato" in data["answer"]
        assert data["sources"] == []
        assert data["n_sources_retrieved"] == 0

    def test_with_docs_returns_answer(self, client):
        docs = [Document(id="1", content="relevant", score=0.9, meta={"source": "t.pdf"})]
        enhanced = EnhancedQuery(original_query="domanda")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "La risposta."}}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("app.api._collect_docs", return_value=(docs, enhanced)), \
             patch("app.api.httpx.AsyncClient", return_value=mock_http):
            response = client.post("/api/query", json={"query": "domanda"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "La risposta."
        assert len(data["sources"]) == 1
        assert data["n_sources_retrieved"] == 1

    def test_mode_override_applied(self, client):
        enhanced = EnhancedQuery(original_query="q")

        with patch("app.api._collect_docs", return_value=([], enhanced)) as mock_collect:
            client.post("/api/query", json={"query": "q", "retrieval_mode": "sparse"})

        passed_cfg = mock_collect.call_args[0][1]
        assert passed_cfg.retrieval.mode == "sparse"

    def test_top_k_override_applied(self, client):
        enhanced = EnhancedQuery(original_query="q")

        with patch("app.api._collect_docs", return_value=([], enhanced)) as mock_collect:
            client.post("/api/query", json={"query": "q", "top_k": 3})

        passed_cfg = mock_collect.call_args[0][1]
        assert passed_cfg.retrieval.top_k == 3

    def test_ollama_timeout_returns_503(self, client):
        import httpx as real_httpx

        docs = [Document(id="1", content="c", score=0.9, meta={"source": "x.pdf"})]
        enhanced = EnhancedQuery(original_query="q")

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=real_httpx.TimeoutException("timeout")
        )
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("app.api._collect_docs", return_value=(docs, enhanced)), \
             patch("app.api.httpx.AsyncClient", return_value=mock_http):
            response = client.post("/api/query", json={"query": "q"})

        assert response.status_code == 503


# ── /api/status ───────────────────────────────────────────────────────────────


class TestApiStatus:
    def _make_http_mock(self, qdrant_ok: bool = True, ollama_ok: bool = True, n_docs: int = 5):
        def mock_get_side_effect(url, **kwargs):
            resp = MagicMock()
            if "healthz" in url:
                resp.status_code = 200 if qdrant_ok else 503
            elif "collections" in url:
                resp.status_code = 200
                resp.json.return_value = {"result": {"points_count": n_docs}}
            elif "api/tags" in url:
                resp.status_code = 200 if ollama_ok else 503
            else:
                resp.status_code = 404
            return resp

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=mock_get_side_effect)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        return mock_http

    def test_both_ok_returns_ok(self, client):
        mock_http = self._make_http_mock(qdrant_ok=True, ollama_ok=True)
        with patch("app.api.httpx.AsyncClient", return_value=mock_http):
            response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["qdrant"]["connected"] is True
        assert data["ollama"]["connected"] is True

    def test_qdrant_down_returns_degraded(self, client):
        import httpx as real_httpx

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=real_httpx.ConnectError("refused"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("app.api.httpx.AsyncClient", return_value=mock_http):
            response = client.get("/api/status")

        data = response.json()
        assert data["status"] in ("degraded", "down")
        assert data["qdrant"]["connected"] is False

    def test_response_contains_expected_fields(self, client):
        mock_http = self._make_http_mock()
        with patch("app.api.httpx.AsyncClient", return_value=mock_http):
            response = client.get("/api/status")

        data = response.json()
        assert "qdrant" in data
        assert "ollama" in data
        assert "collection_name" in data
        assert "embedder_model" in data
        assert "llm_model" in data


# ── /api/config ───────────────────────────────────────────────────────────────


class TestApiConfig:
    def test_get_config_returns_dict(self, client):
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "qdrant" in data
        assert "llm" in data
        assert "retrieval" in data

    def test_get_config_masks_api_key(self, client, cfg):
        cfg.qdrant.api_key = "secret-key"
        response = client.get("/api/config")
        data = response.json()
        assert data["qdrant"]["api_key"] == "***"

    def test_get_config_null_api_key_not_masked(self, client, cfg):
        cfg.qdrant.api_key = None
        response = client.get("/api/config")
        data = response.json()
        assert data["qdrant"]["api_key"] is None

    def test_post_config_updates_field(self, client):
        response = client.post("/api/config", json={"retrieval": {"mode": "sparse"}})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"
        assert "retrieval.mode" in data["changed_fields"]

    def test_post_config_locked_field_returns_400(self, client):
        response = client.post("/api/config", json={"qdrant": {"host": "new-host"}})
        assert response.status_code == 400

    def test_post_config_locked_port_returns_400(self, client):
        response = client.post("/api/config", json={"qdrant": {"port": 9999}})
        assert response.status_code == 400

    def test_post_config_unknown_section_returns_422(self, client):
        response = client.post("/api/config", json={"nonexistent": {"foo": "bar"}})
        assert response.status_code == 422

    def test_post_config_invalid_value_returns_422(self, client):
        response = client.post("/api/config", json={"retrieval": {"mode": "invalid_mode"}})
        assert response.status_code == 422

    def test_post_config_no_change_empty_changed_fields(self, client, cfg):
        current_mode = cfg.retrieval.mode
        response = client.post("/api/config", json={"retrieval": {"mode": current_mode}})
        assert response.status_code == 200
        assert "retrieval.mode" not in response.json()["changed_fields"]


# ── /api/index (DELETE) ───────────────────────────────────────────────────────


class TestApiDeleteIndex:
    def test_without_confirm_returns_400(self, client):
        response = client.delete("/api/index")
        assert response.status_code == 400
        assert "confirm" in response.json()["detail"].lower()

    def test_confirm_false_returns_400(self, client):
        response = client.delete("/api/index?confirm=false")
        assert response.status_code == 400

    def test_with_confirm_true_succeeds(self, client):
        import httpx as real_httpx

        # Let the initial count HTTP request fail gracefully (n_deleted → 0)
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=real_httpx.ConnectError("refused"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        mock_store = MagicMock()

        with patch("app.api.httpx.AsyncClient", return_value=mock_http), \
             patch(
                 "haystack_integrations.document_stores.qdrant.QdrantDocumentStore",
                 return_value=mock_store,
             ):
            response = client.delete("/api/index?confirm=true")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert "collection_name" in data
        assert "n_documents_deleted" in data


# ── /api/eval ─────────────────────────────────────────────────────────────────


class TestApiEval:
    def test_start_eval_returns_job_id(self, client):
        response = client.post("/api/eval")
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0

    def test_get_eval_not_found_returns_404(self, client):
        response = client.get("/api/eval/nonexistent-job-000")
        assert response.status_code == 404

    def test_get_eval_running_status(self, client):
        from app.api import _eval_jobs

        job_id = "test-running-job"
        _eval_jobs[job_id] = {"status": "running", "report": None, "error": None}
        try:
            response = client.get(f"/api/eval/{job_id}")
            assert response.status_code == 200
            assert response.json()["status"] == "running"
        finally:
            _eval_jobs.pop(job_id, None)

    def test_get_eval_done_includes_report(self, client):
        from app.api import _eval_jobs

        job_id = "test-done-job"
        _eval_jobs[job_id] = {
            "status": "done",
            "report": {"mode": "A", "n_samples": 5},
            "error": None,
        }
        try:
            response = client.get(f"/api/eval/{job_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "done"
            assert data["report"]["mode"] == "A"
        finally:
            _eval_jobs.pop(job_id, None)

    def test_get_eval_error_includes_error_message(self, client):
        from app.api import _eval_jobs

        job_id = "test-error-job"
        _eval_jobs[job_id] = {
            "status": "error",
            "report": None,
            "error": "Something went wrong",
        }
        try:
            response = client.get(f"/api/eval/{job_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "error"
            assert data["error"] == "Something went wrong"
        finally:
            _eval_jobs.pop(job_id, None)
