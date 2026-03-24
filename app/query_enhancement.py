"""Query enhancement: expansion e decomposition tramite Ollama."""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List

import httpx

from app.config import AppConfig, DecompositionConfig, ExpansionConfig, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQuery:
    """Risultato del query enhancement."""

    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)

    @property
    def all_queries(self) -> List[str]:
        """
        Unione di tutte le query da usare per il retrieval.
        La query originale è sempre la prima. Duplicati rimossi mantenendo l'ordine.
        """
        queries = [self.original_query] + self.expanded_queries + self.sub_queries
        seen: set = set()
        return [q for q in queries if not (q in seen or seen.add(q))]


def _parse_llm_list_response(response: str, expected_n: int) -> List[str]:
    """
    Parsa la risposta del LLM come lista di righe.

    - Rimuove righe vuote
    - Rimuove numerazione iniziale (es. "1.", "1)", "-", "•", "*")
    - Tronca al massimo a expected_n elementi
    """
    lines = response.strip().split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\d]+[.)]\s*|^[-•*]\s*", "", line).strip()
        if line:
            cleaned.append(line)
    return cleaned[:expected_n]


def _call_ollama(prompt: str, llm_cfg: LLMConfig) -> str:
    """
    Chiama Ollama /api/generate in modalità non-stream e restituisce la risposta.

    Raises:
        httpx.TimeoutException: se Ollama non risponde entro llm_cfg.timeout secondi.
        httpx.HTTPError: per altri errori di connessione.
    """
    url = f"{llm_cfg.host}/api/generate"
    payload = {
        "model": llm_cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": llm_cfg.temperature,
            "top_p": llm_cfg.top_p,
            "num_predict": llm_cfg.max_tokens,
        },
    }
    with httpx.Client(timeout=llm_cfg.timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]


def _deduplicate_against_original(variants: List[str], original_query: str) -> List[str]:
    """Rimuove varianti identiche alla query originale (case-insensitive)."""
    original_lower = original_query.lower()
    return [v for v in variants if v.lower() != original_lower]


def _expand_query(query: str, exp_cfg: ExpansionConfig, llm_cfg: LLMConfig) -> List[str]:
    prompt = exp_cfg.prompt_template.format(query=query, n=exp_cfg.n_variants)
    raw = _call_ollama(prompt, llm_cfg)
    variants = _parse_llm_list_response(raw, exp_cfg.n_variants)
    return _deduplicate_against_original(variants, query)


def _decompose_query(query: str, dec_cfg: DecompositionConfig, llm_cfg: LLMConfig) -> List[str]:
    prompt = dec_cfg.prompt_template.format(query=query, n=dec_cfg.n_subqueries)
    raw = _call_ollama(prompt, llm_cfg)
    sub_queries = _parse_llm_list_response(raw, dec_cfg.n_subqueries)
    return _deduplicate_against_original(sub_queries, query)


def enhance_query(query: str, cfg: AppConfig) -> EnhancedQuery:
    """
    Applica le tecniche di query enhancement abilitate nella configurazione.

    Se nessuna tecnica è abilitata, restituisce un EnhancedQuery con solo
    la query originale (nessuna chiamata al LLM).

    Se Ollama non risponde o va in timeout, logga WARNING e restituisce
    EnhancedQuery con solo la query originale (fail-soft).
    """
    start = time.monotonic()
    logger.info("Avvio enhance_query: '%s'", query[:80])

    result = EnhancedQuery(original_query=query)

    qe_cfg = cfg.query_enhancement
    if qe_cfg is None:
        logger.info("enhance_query completato: query enhancement non configurato, restituita query originale")
        return result

    expansion_enabled = qe_cfg.expansion is not None and qe_cfg.expansion.enabled
    decomposition_enabled = qe_cfg.decomposition is not None and qe_cfg.decomposition.enabled

    if not expansion_enabled and not decomposition_enabled:
        logger.info("enhance_query completato: nessuna tecnica abilitata, restituita query originale")
        return result

    try:
        # Decomposition prima (genera sotto-query dalla query originale)
        if decomposition_enabled:
            result.sub_queries = _decompose_query(query, qe_cfg.decomposition, cfg.llm)

        # Expansion sulla query originale
        if expansion_enabled:
            result.expanded_queries = _expand_query(query, qe_cfg.expansion, cfg.llm)

    except httpx.TimeoutException:
        logger.warning(
            "Ollama non ha risposto entro %s secondi durante query enhancement. "
            "Fallback alla query originale.",
            cfg.llm.timeout,
        )
        return EnhancedQuery(original_query=query)
    except httpx.HTTPError as exc:
        logger.warning(
            "Errore di connessione a Ollama durante query enhancement: %s. "
            "Fallback alla query originale.",
            exc,
        )
        return EnhancedQuery(original_query=query)

    logger.info(
        "enhance_query completato in %.2f s: %d expanded, %d sub-queries",
        time.monotonic() - start,
        len(result.expanded_queries),
        len(result.sub_queries),
    )
    return result
