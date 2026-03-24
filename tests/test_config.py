import os
import pytest
from pathlib import Path

from app.config import (
    AppConfig,
    ChunkingConfig,
    load_config,
    get_config,
    _config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
qdrant:
  host: "qdrant"
  port: 6333
  collection_name: "documents"

embedder:
  model: "BAAI/bge-m3"

llm:
  host: "http://ollama:11434"
  model: "mistral"

retrieval:
  mode: "hybrid"
  top_k: 5
"""


def write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Test 1: caricamento config minima valida
# ---------------------------------------------------------------------------


def test_load_minimal_config(tmp_path):
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))

    assert isinstance(cfg, AppConfig)
    assert cfg.qdrant.host == "qdrant"
    assert cfg.qdrant.port == 6333
    assert cfg.embedder.model == "BAAI/bge-m3"
    assert cfg.llm.model == "mistral"
    assert cfg.retrieval.mode == "hybrid"
    assert cfg.retrieval.top_k == 5
    # sezioni opzionali assenti → None
    assert cfg.query_enhancement is None
    assert cfg.reranker is None
    # sezioni con default
    assert cfg.indexing.chunking.strategy == "recursive"
    assert cfg.logging.level == "INFO"


# ---------------------------------------------------------------------------
# Test 2: validazione fallisce se chunk_overlap >= chunk_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["character", "recursive"])
def test_chunking_overlap_equal_to_size_raises(strategy):
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(strategy=strategy, chunk_size=500, chunk_overlap=500)


@pytest.mark.parametrize("strategy", ["character", "recursive"])
def test_chunking_overlap_greater_than_size_raises(strategy):
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(strategy=strategy, chunk_size=500, chunk_overlap=600)


@pytest.mark.parametrize("strategy", ["character", "recursive"])
def test_chunking_valid_overlap(strategy):
    cfg = ChunkingConfig(strategy=strategy, chunk_size=1000, chunk_overlap=150)
    assert cfg.chunk_overlap < cfg.chunk_size


def test_chunking_paragraph_ignores_overlap_validation():
    # paragraph non richiede chunk_overlap < chunk_size
    cfg = ChunkingConfig(strategy="paragraph", max_paragraph_length=3000)
    assert cfg.strategy == "paragraph"


def test_load_config_invalid_chunking(tmp_path):
    bad_yaml = MINIMAL_YAML + """\
indexing:
  chunking:
    strategy: "character"
    chunk_size: 200
    chunk_overlap: 200
"""
    cfg_path = write_yaml(tmp_path, bad_yaml)
    with pytest.raises(ValueError, match="chunk_overlap"):
        load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# Test 3: override da variabile d'ambiente
# ---------------------------------------------------------------------------


def test_env_override_qdrant_host(tmp_path, monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "my-qdrant-host")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.qdrant.host == "my-qdrant-host"


def test_env_override_qdrant_port(tmp_path, monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "6400")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.qdrant.port == 6400


def test_env_override_ollama_host(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://my-ollama:11434")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.llm.host == "http://my-ollama:11434"


def test_env_override_log_level(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.logging.level == "DEBUG"


# ---------------------------------------------------------------------------
# Test: get_config singleton
# ---------------------------------------------------------------------------


def test_get_config_singleton(tmp_path, monkeypatch):
    import app.config as config_module

    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    monkeypatch.setenv("CONFIG_PATH", str(cfg_path))
    monkeypatch.setattr(config_module, "_config", None)

    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2
