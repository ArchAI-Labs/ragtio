import os
import pytest
from pathlib import Path

from app.config import (
    AppConfig,
    ChunkingConfig,
    CustomEmbedderConfig,
    EmbedderConfig,
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


# ---------------------------------------------------------------------------
# Test: nuovi env override
# ---------------------------------------------------------------------------


def test_env_override_openai_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.llm.openai_api_key == "sk-test-key"


def test_env_override_openai_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.llm.model == "gpt-4o"


def test_env_override_openai_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-proxy.com/v1")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.llm.openai_base_url == "https://my-proxy.com/v1"


def test_env_override_embedder_model(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.embedder.model == "BAAI/bge-small-en-v1.5"


def test_env_override_reranker_model(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "cross-encoder/custom")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.reranker is not None
    assert cfg.reranker.model == "cross-encoder/custom"


def test_env_override_qdrant_collection_name(tmp_path, monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "my_collection")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.qdrant.collection_name == "my_collection"


# ---------------------------------------------------------------------------
# Test: EvaluationModeAConfig e EvaluationModeBConfig
# ---------------------------------------------------------------------------


def test_evaluation_mode_a_k_values_sorted():
    from app.config import EvaluationModeAConfig

    cfg = EvaluationModeAConfig(k_values=[10, 1, 5, 3])
    assert cfg.k_values == [1, 3, 5, 10]


def test_evaluation_mode_a_k_values_zero_raises():
    from app.config import EvaluationModeAConfig

    with pytest.raises(ValueError, match="K devono essere >= 1"):
        EvaluationModeAConfig(k_values=[0, 5])


def test_evaluation_mode_a_k_values_negative_raises():
    from app.config import EvaluationModeAConfig

    with pytest.raises(ValueError, match="K devono essere >= 1"):
        EvaluationModeAConfig(k_values=[-1, 5])


def test_evaluation_mode_b_k_values_sorted():
    from app.config import EvaluationModeBConfig

    cfg = EvaluationModeBConfig(k_values=[10, 1, 5])
    assert cfg.k_values == [1, 5, 10]


def test_evaluation_mode_b_k_values_negative_raises():
    from app.config import EvaluationModeBConfig

    with pytest.raises(ValueError, match="K devono essere >= 1"):
        EvaluationModeBConfig(k_values=[-1, 5])


# ---------------------------------------------------------------------------
# Test: EvaluationConfig defaults in AppConfig
# ---------------------------------------------------------------------------


def test_app_config_has_evaluation_defaults():
    cfg = AppConfig()
    assert cfg.evaluation.default_mode == "A"
    assert cfg.evaluation.mode_a.n_samples == 50
    assert cfg.evaluation.mode_a.k_values == [1, 3, 5, 10]
    assert cfg.evaluation.mode_a.query_type == "auto"
    assert cfg.evaluation.mode_b.k_folds == 5


def test_evaluation_config_loaded_from_yaml(tmp_path):
    yaml_with_eval = MINIMAL_YAML + """\
evaluation:
  default_mode: "B"
  mode_a:
    n_samples: 20
    k_values: [1, 5]
"""
    cfg_path = write_yaml(tmp_path, yaml_with_eval)
    cfg = load_config(str(cfg_path))
    assert cfg.evaluation.default_mode == "B"
    assert cfg.evaluation.mode_a.n_samples == 20
    assert cfg.evaluation.mode_a.k_values == [1, 5]


# ---------------------------------------------------------------------------
# Test: nuovi env override embedder
# ---------------------------------------------------------------------------


def test_env_override_embedder_batch_size(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDER_BATCH_SIZE", "16")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.embedder.batch_size == 16


def test_env_override_embedder_max_length(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDER_MAX_LENGTH", "256")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.embedder.max_length == 256


def test_env_override_embedder_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDER_CACHE_DIR", "/tmp/my-models")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.embedder.cache_dir == "/tmp/my-models"


def test_env_override_embedder_dim(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDER_DIM", "768")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.embedder.embedding_dim == 768


# ---------------------------------------------------------------------------
# Test: custom embedder via env vars
# ---------------------------------------------------------------------------


def test_env_override_custom_embedder_hf(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDER_CUSTOM_DIM", "384")
    monkeypatch.setenv("EMBEDDER_CUSTOM_HF_REPO", "org/my-model")
    monkeypatch.setenv("EMBEDDER_CUSTOM_POOLING", "CLS")
    monkeypatch.setenv("EMBEDDER_CUSTOM_NORMALIZATION", "false")
    monkeypatch.setenv("EMBEDDER_CUSTOM_MODEL_FILE", "onnx/model_quant.onnx")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))

    assert cfg.embedder.custom is not None
    assert cfg.embedder.custom.dim == 384
    assert cfg.embedder.custom.hf_repo == "org/my-model"
    assert cfg.embedder.custom.pooling == "CLS"
    assert cfg.embedder.custom.normalization is False
    assert cfg.embedder.custom.model_file == "onnx/model_quant.onnx"


def test_env_override_custom_embedder_normalization_true(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDER_CUSTOM_DIM", "512")
    monkeypatch.setenv("EMBEDDER_CUSTOM_HF_REPO", "org/model")
    monkeypatch.setenv("EMBEDDER_CUSTOM_NORMALIZATION", "true")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.embedder.custom.normalization is True


def test_env_override_custom_embedder_url(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDER_CUSTOM_DIM", "256")
    monkeypatch.setenv("EMBEDDER_CUSTOM_URL", "https://example.com/model.tar.gz")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.embedder.custom.url == "https://example.com/model.tar.gz"
    assert cfg.embedder.custom.hf_repo is None


def test_custom_embedder_not_activated_without_dim(tmp_path, monkeypatch):
    # senza EMBEDDER_CUSTOM_DIM il blocco custom non viene attivato
    monkeypatch.setenv("EMBEDDER_CUSTOM_HF_REPO", "org/model")
    cfg_path = write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(str(cfg_path))
    assert cfg.embedder.custom is None


# ---------------------------------------------------------------------------
# Test: CustomEmbedderConfig validazione
# ---------------------------------------------------------------------------


def test_custom_embedder_config_requires_source():
    with pytest.raises(ValueError, match="almeno uno tra"):
        CustomEmbedderConfig(dim=384)


def test_custom_embedder_config_hf_repo_ok():
    cfg = CustomEmbedderConfig(dim=384, hf_repo="org/model")
    assert cfg.hf_repo == "org/model"
    assert cfg.url is None


def test_custom_embedder_config_url_ok():
    cfg = CustomEmbedderConfig(dim=768, url="https://example.com/m.tar.gz")
    assert cfg.url == "https://example.com/m.tar.gz"


def test_custom_embedder_config_defaults():
    cfg = CustomEmbedderConfig(dim=384, hf_repo="org/model")
    assert cfg.pooling == "MEAN"
    assert cfg.normalization is True
    assert cfg.model_file == "onnx/model.onnx"


# ---------------------------------------------------------------------------
# Test: custom embedder via YAML
# ---------------------------------------------------------------------------


def test_custom_embedder_loaded_from_yaml(tmp_path):
    yaml_with_custom = MINIMAL_YAML + """\
embedder:
  model: "org/my-model"
  custom:
    dim: 384
    pooling: "MEAN"
    normalization: true
    hf_repo: "org/my-model"
    model_file: "onnx/model.onnx"
"""
    cfg_path = write_yaml(tmp_path, yaml_with_custom)
    cfg = load_config(str(cfg_path))
    assert cfg.embedder.custom is not None
    assert cfg.embedder.custom.dim == 384
    assert cfg.embedder.custom.hf_repo == "org/my-model"
