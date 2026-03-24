"""Test per il modulo app.query_enhancement."""

import json
from typing import Optional

import httpx
import pytest

from app.config import (
    AppConfig,
    DecompositionConfig,
    ExpansionConfig,
    LLMConfig,
    QueryEnhancementConfig,
)
from app.query_enhancement import (
    EnhancedQuery,
    _parse_llm_list_response,
    enhance_query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ollama_response(text: str) -> httpx.Response:
    """Crea una httpx.Response che simula la risposta JSON di Ollama /api/generate."""
    body = json.dumps({"response": text}).encode()
    return httpx.Response(200, content=body, headers={"content-type": "application/json"})


def make_transport(responses: list[httpx.Response]) -> httpx.MockTransport:
    """
    Restituisce un MockTransport che serve le risposte nell'ordine dato.
    Lancia AssertionError se vengono fatte più richieste del previsto.
    """
    calls = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return next(calls)

    return httpx.MockTransport(handler)


def make_cfg(
    expansion: Optional[ExpansionConfig] = None,
    decomposition: Optional[DecompositionConfig] = None,
    llm_timeout: int = 30,
) -> AppConfig:
    qe = QueryEnhancementConfig(expansion=expansion, decomposition=decomposition) if (
        expansion or decomposition
    ) else None
    return AppConfig(
        query_enhancement=qe,
        llm=LLMConfig(host="http://ollama:11434", model="mistral", timeout=llm_timeout),
    )


# ---------------------------------------------------------------------------
# _parse_llm_list_response
# ---------------------------------------------------------------------------


class TestParseLlmListResponse:
    def test_plain_lines(self):
        raw = "variante uno\nvariante due\nvariante tre"
        assert _parse_llm_list_response(raw, 3) == [
            "variante uno",
            "variante due",
            "variante tre",
        ]

    def test_removes_empty_lines(self):
        raw = "variante uno\n\nvariante due\n\n"
        assert _parse_llm_list_response(raw, 5) == ["variante uno", "variante due"]

    def test_strips_numbered_prefix_dot(self):
        raw = "1. prima\n2. seconda\n3. terza"
        assert _parse_llm_list_response(raw, 3) == ["prima", "seconda", "terza"]

    def test_strips_numbered_prefix_paren(self):
        raw = "1) prima\n2) seconda"
        assert _parse_llm_list_response(raw, 2) == ["prima", "seconda"]

    def test_strips_dash_bullet(self):
        raw = "- prima\n- seconda\n- terza"
        assert _parse_llm_list_response(raw, 3) == ["prima", "seconda", "terza"]

    def test_strips_bullet_symbols(self):
        raw = "• prima\n* seconda"
        assert _parse_llm_list_response(raw, 2) == ["prima", "seconda"]

    def test_truncates_to_expected_n(self):
        raw = "a\nb\nc\nd\ne"
        assert _parse_llm_list_response(raw, 3) == ["a", "b", "c"]

    def test_fewer_lines_than_expected(self):
        raw = "solo una"
        assert _parse_llm_list_response(raw, 5) == ["solo una"]


# ---------------------------------------------------------------------------
# EnhancedQuery.all_queries
# ---------------------------------------------------------------------------


class TestEnhancedQueryAllQueries:
    def test_no_enhancement(self):
        eq = EnhancedQuery(original_query="query")
        assert eq.all_queries == ["query"]

    def test_with_expanded(self):
        eq = EnhancedQuery(original_query="q", expanded_queries=["e1", "e2"])
        assert eq.all_queries == ["q", "e1", "e2"]

    def test_with_sub_queries(self):
        eq = EnhancedQuery(original_query="q", sub_queries=["s1", "s2"])
        assert eq.all_queries == ["q", "s1", "s2"]

    def test_deduplication_preserves_order(self):
        eq = EnhancedQuery(
            original_query="q",
            expanded_queries=["e1", "e1", "e2"],
            sub_queries=["e2", "s1"],
        )
        assert eq.all_queries == ["q", "e1", "e2", "s1"]

    def test_original_always_first(self):
        eq = EnhancedQuery(original_query="orig", expanded_queries=["x"])
        assert eq.all_queries[0] == "orig"


# ---------------------------------------------------------------------------
# enhance_query — nessuna feature abilitata
# ---------------------------------------------------------------------------


class TestEnhanceQueryDisabled:
    def test_no_query_enhancement_config(self):
        cfg = AppConfig(query_enhancement=None)
        result = enhance_query("test query", cfg)
        assert result.original_query == "test query"
        assert result.expanded_queries == []
        assert result.sub_queries == []
        assert result.all_queries == ["test query"]

    def test_both_disabled(self):
        cfg = make_cfg(
            expansion=ExpansionConfig(enabled=False),
            decomposition=DecompositionConfig(enabled=False),
        )
        result = enhance_query("test query", cfg)
        assert result.all_queries == ["test query"]

    def test_expansion_disabled_only(self):
        """Con solo expansion disabilitata e decomposition disabilitata → nessuna LLM call."""
        cfg = make_cfg(expansion=ExpansionConfig(enabled=False))
        result = enhance_query("test query", cfg)
        assert result.expanded_queries == []
        assert result.sub_queries == []


# ---------------------------------------------------------------------------
# enhance_query — expansion abilitata
# ---------------------------------------------------------------------------


class TestEnhanceQueryExpansion:
    def test_expansion_returns_variants(self, monkeypatch):
        variants_text = "variante A\nvariante B\nvariante C"

        def mock_call_ollama(prompt, llm_cfg):
            return variants_text

        monkeypatch.setattr("app.query_enhancement._call_ollama", mock_call_ollama)

        cfg = make_cfg(expansion=ExpansionConfig(enabled=True, n_variants=3))
        result = enhance_query("query originale", cfg)

        assert result.expanded_queries == ["variante A", "variante B", "variante C"]
        assert result.sub_queries == []
        assert result.all_queries == ["query originale", "variante A", "variante B", "variante C"]

    def test_expansion_deduplicates_original(self, monkeypatch):
        """Varianti identiche alla query originale (case-insensitive) vengono rimosse."""

        def mock_call_ollama(prompt, llm_cfg):
            return "Query Originale\nvariante diversa"

        monkeypatch.setattr("app.query_enhancement._call_ollama", mock_call_ollama)

        cfg = make_cfg(expansion=ExpansionConfig(enabled=True, n_variants=3))
        result = enhance_query("query originale", cfg)

        assert "query originale" not in [v.lower() for v in result.expanded_queries]
        assert "variante diversa" in result.expanded_queries

    def test_expansion_uses_prompt_template(self, monkeypatch):
        captured = {}

        def mock_call_ollama(prompt, llm_cfg):
            captured["prompt"] = prompt
            return "variante 1"

        monkeypatch.setattr("app.query_enhancement._call_ollama", mock_call_ollama)

        template = "Custom template {query} con {n} varianti"
        cfg = make_cfg(expansion=ExpansionConfig(enabled=True, n_variants=2, prompt_template=template))
        enhance_query("mia query", cfg)

        assert "mia query" in captured["prompt"]
        assert "2" in captured["prompt"]


# ---------------------------------------------------------------------------
# enhance_query — decomposition abilitata
# ---------------------------------------------------------------------------


class TestEnhanceQueryDecomposition:
    def test_decomposition_returns_subqueries(self, monkeypatch):
        sub_text = "sotto-query 1\nsotto-query 2\nsotto-query 3"

        def mock_call_ollama(prompt, llm_cfg):
            return sub_text

        monkeypatch.setattr("app.query_enhancement._call_ollama", mock_call_ollama)

        cfg = make_cfg(decomposition=DecompositionConfig(enabled=True, n_subqueries=3))
        result = enhance_query("query complessa", cfg)

        assert result.sub_queries == ["sotto-query 1", "sotto-query 2", "sotto-query 3"]
        assert result.expanded_queries == []

    def test_decomposition_uses_prompt_template(self, monkeypatch):
        captured = {}

        def mock_call_ollama(prompt, llm_cfg):
            captured["prompt"] = prompt
            return "sub 1"

        monkeypatch.setattr("app.query_enhancement._call_ollama", mock_call_ollama)

        template = "Scomponi {query} in {n} parti"
        cfg = make_cfg(decomposition=DecompositionConfig(enabled=True, n_subqueries=4, prompt_template=template))
        enhance_query("domanda complessa", cfg)

        assert "domanda complessa" in captured["prompt"]
        assert "4" in captured["prompt"]


# ---------------------------------------------------------------------------
# enhance_query — entrambe abilitate
# ---------------------------------------------------------------------------


class TestEnhanceQueryBoth:
    def test_both_enabled(self, monkeypatch):
        call_count = {"n": 0}

        def mock_call_ollama(prompt, llm_cfg):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Decomposition viene chiamata prima
                return "sub 1\nsub 2"
            return "exp 1\nexp 2"

        monkeypatch.setattr("app.query_enhancement._call_ollama", mock_call_ollama)

        cfg = make_cfg(
            expansion=ExpansionConfig(enabled=True, n_variants=2),
            decomposition=DecompositionConfig(enabled=True, n_subqueries=2),
        )
        result = enhance_query("q", cfg)

        assert result.sub_queries == ["sub 1", "sub 2"]
        assert result.expanded_queries == ["exp 1", "exp 2"]
        assert result.all_queries == ["q", "exp 1", "exp 2", "sub 1", "sub 2"]
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# enhance_query — gestione errori (fail-soft)
# ---------------------------------------------------------------------------


class TestEnhanceQueryErrorHandling:
    def test_timeout_returns_original(self, monkeypatch):
        def mock_call_ollama(prompt, llm_cfg):
            raise httpx.TimeoutException("timeout")

        monkeypatch.setattr("app.query_enhancement._call_ollama", mock_call_ollama)

        cfg = make_cfg(expansion=ExpansionConfig(enabled=True, n_variants=3))
        result = enhance_query("query", cfg)

        assert result.original_query == "query"
        assert result.expanded_queries == []
        assert result.sub_queries == []
        assert result.all_queries == ["query"]

    def test_connection_error_returns_original(self, monkeypatch):
        def mock_call_ollama(prompt, llm_cfg):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("app.query_enhancement._call_ollama", mock_call_ollama)

        cfg = make_cfg(decomposition=DecompositionConfig(enabled=True, n_subqueries=3))
        result = enhance_query("query", cfg)

        assert result.all_queries == ["query"]

    def test_timeout_logs_warning(self, monkeypatch, caplog):
        import logging

        def mock_call_ollama(prompt, llm_cfg):
            raise httpx.TimeoutException("timeout")

        monkeypatch.setattr("app.query_enhancement._call_ollama", mock_call_ollama)

        cfg = make_cfg(expansion=ExpansionConfig(enabled=True, n_variants=3))
        with caplog.at_level(logging.WARNING, logger="app.query_enhancement"):
            enhance_query("query", cfg)

        assert any("WARNING" in r.levelname or r.levelno >= logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# _call_ollama — test con httpx.MockTransport
# ---------------------------------------------------------------------------


class TestCallOllamaWithMockTransport:
    def test_successful_call(self, monkeypatch):
        """Verifica che _call_ollama usi correttamente httpx e parsi la risposta."""
        from app.query_enhancement import _call_ollama

        response_body = json.dumps({"response": "risposta dal LLM"}).encode()
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=response_body)
        )

        # Monkeypatcha httpx.Client per usare il MockTransport
        original_client = httpx.Client

        class MockClient(httpx.Client):
            def __init__(self, **kwargs):
                kwargs["transport"] = transport
                super().__init__(**kwargs)

        monkeypatch.setattr(httpx, "Client", MockClient)

        llm_cfg = LLMConfig(host="http://ollama:11434", model="mistral", timeout=10)
        result = _call_ollama("test prompt", llm_cfg)
        assert result == "risposta dal LLM"

    def test_timeout_raises(self, monkeypatch):
        """Verifica che un timeout propaghi httpx.TimeoutException."""
        from app.query_enhancement import _call_ollama

        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timeout", request=request)

        transport = httpx.MockTransport(timeout_handler)

        original_client = httpx.Client

        class MockClient(httpx.Client):
            def __init__(self, **kwargs):
                kwargs["transport"] = transport
                super().__init__(**kwargs)

        monkeypatch.setattr(httpx, "Client", MockClient)

        llm_cfg = LLMConfig(host="http://ollama:11434", model="mistral", timeout=1)
        with pytest.raises(httpx.TimeoutException):
            _call_ollama("prompt", llm_cfg)

    def test_request_payload(self, monkeypatch):
        """Verifica che il payload inviato a Ollama sia corretto."""
        from app.query_enhancement import _call_ollama

        captured_requests = []

        def capturing_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured_requests.append(body)
            return httpx.Response(
                200,
                content=json.dumps({"response": "ok"}).encode(),
            )

        transport = httpx.MockTransport(capturing_handler)

        class MockClient(httpx.Client):
            def __init__(self, **kwargs):
                kwargs["transport"] = transport
                super().__init__(**kwargs)

        monkeypatch.setattr(httpx, "Client", MockClient)

        llm_cfg = LLMConfig(
            host="http://ollama:11434",
            model="llama3",
            timeout=30,
            temperature=0.2,
            top_p=0.85,
            max_tokens=512,
        )
        _call_ollama("il mio prompt", llm_cfg)

        assert len(captured_requests) == 1
        payload = captured_requests[0]
        assert payload["model"] == "llama3"
        assert payload["prompt"] == "il mio prompt"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == pytest.approx(0.2)
        assert payload["options"]["top_p"] == pytest.approx(0.85)
        assert payload["options"]["num_predict"] == 512
