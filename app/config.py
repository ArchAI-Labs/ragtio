import os
import logging
from pathlib import Path
from typing import Optional, List, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)


class QdrantConfig(BaseModel):
    host: str = "qdrant"
    port: int = Field(default=6333, ge=1, le=65535)
    collection_name: str = "documents"
    api_key: Optional[str] = None
    grpc_port: int = Field(default=6334, ge=1, le=65535)
    prefer_grpc: bool = False
    timeout: int = Field(default=30, ge=1)


class CustomEmbedderConfig(BaseModel):
    """Parametri per registrare un modello ONNX custom via TextEmbedding.add_custom_model()."""

    pooling: Literal["MEAN", "CLS", "MAX"] = "MEAN"
    normalization: bool = True
    hf_repo: Optional[str] = None   # sources=ModelSource(hf=...)
    url: Optional[str] = None       # sources=ModelSource(url=...)
    model_file: str = "onnx/model.onnx"
    dim: int = Field(ge=1)

    @model_validator(mode="after")
    def require_at_least_one_source(self) -> "CustomEmbedderConfig":
        if not self.hf_repo and not self.url:
            raise ValueError("custom embedder richiede almeno uno tra 'hf_repo' e 'url'")
        return self


class EmbedderConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    batch_size: int = Field(default=32, ge=1)
    max_length: int = Field(default=512, ge=64, le=8192)
    cache_dir: str = "/app/models/fastembed"
    embedding_dim: Optional[int] = Field(default=None, ge=1)  # override manuale dimensione vettore
    custom: Optional[CustomEmbedderConfig] = None             # presente solo per modelli custom ONNX


class ChunkingConfig(BaseModel):
    strategy: Literal["character", "recursive", "paragraph"] = "recursive"
    chunk_size: Optional[int] = Field(default=1000, ge=1)
    chunk_overlap: Optional[int] = Field(default=150, ge=0)
    separators: Optional[List[str]] = None
    max_paragraph_length: Optional[int] = Field(default=2000, ge=1)

    @model_validator(mode="after")
    def validate_chunking_params(self) -> "ChunkingConfig":
        if self.strategy in ("character", "recursive"):
            if self.chunk_size is None:
                raise ValueError(f"chunk_size è obbligatorio per strategy '{self.strategy}'")
            if self.chunk_overlap is None:
                raise ValueError(f"chunk_overlap è obbligatorio per strategy '{self.strategy}'")
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError(
                    f"chunk_overlap ({self.chunk_overlap}) deve essere < chunk_size ({self.chunk_size})"
                )
        return self


class IndexingConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    duplicate_policy: Literal["SKIP", "OVERWRITE"] = "OVERWRITE"
    batch_size: int = Field(default=64, ge=1)
    min_chunk_length: int = Field(default=50, ge=0)


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai"] = "ollama"
    # Ollama
    host: str = "http://ollama:11434"
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None  # es. per Azure o proxy compatibili
    # Comune
    model: str = "mistral"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout: int = Field(default=120, ge=1)
    stream: bool = True
    system_prompt: str = (
        "Sei un assistente esperto e preciso. Rispondi alle domande basandoti "
        "esclusivamente sul contesto fornito. Se l'informazione non è presente "
        "nel contesto, dillo esplicitamente senza inventare. Rispondi nella "
        "stessa lingua della domanda."
    )
    rag_prompt_template: str = "Contesto:\n{context}\n\nDomanda: {question}\n\nRisposta:"
    max_context_length: int = Field(default=4000, ge=1)


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    top_n_after_rerank: int = Field(default=5, ge=1, le=100)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ExpansionConfig(BaseModel):
    enabled: bool = False
    n_variants: int = Field(default=3, ge=1, le=10)
    prompt_template: str = (
        "Genera {n} varianti semanticamente equivalenti della seguente query.\n"
        "Restituisci solo le varianti, una per riga, senza numerazione.\n"
        "Query: {query}"
    )


class DecompositionConfig(BaseModel):
    enabled: bool = False
    n_subqueries: int = Field(default=3, ge=1, le=10)
    prompt_template: str = (
        "Scomponi la seguente domanda complessa in {n} sotto-domande semplici.\n"
        "Restituisci solo le sotto-domande, una per riga, senza numerazione.\n"
        "Domanda: {query}"
    )


class QueryEnhancementConfig(BaseModel):
    expansion: Optional[ExpansionConfig] = None
    decomposition: Optional[DecompositionConfig] = None


class RerankerConfig(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = Field(default=16, ge=1)
    cache_dir: str = "/app/models/reranker"


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: List[str] = ["*"]
    max_upload_size_mb: int = Field(default=100, ge=1)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["text", "json"] = "text"
    file: Optional[str] = None


class EvaluationModeAConfig(BaseModel):
    n_samples: int = Field(default=50, ge=1)
    k_values: List[int] = Field(default=[1, 3, 5, 10])
    # "question" = sempre domanda sintetica (ottimale per Dense/Hybrid)
    # "keywords" = sempre lista di keyword (ottimale per Sparse/BM25)
    # "auto"     = domanda per dense/hybrid, keyword per sparse
    query_type: Literal["question", "keywords", "auto"] = "auto"
    question_gen_prompt: str = (
        "Leggi il seguente testo e formula una domanda a cui il testo risponde direttamente.\n"
        "Restituisci solo la domanda, senza altro testo.\n"
        "Testo: {chunk}"
    )
    keyword_gen_prompt: str = (
        "Leggi il seguente testo ed estrai da 3 a 6 keyword o brevi frasi chiave che lo rappresentano.\n"
        "Restituisci solo le keyword separate da virgola, senza altro testo.\n"
        "Testo: {chunk}"
    )

    @field_validator("k_values")
    @classmethod
    def k_values_must_be_positive(cls, v: List[int]) -> List[int]:
        if any(k < 1 for k in v):
            raise ValueError("Tutti i valori di K devono essere >= 1")
        return sorted(v)


class EvaluationModeBConfig(BaseModel):
    k_folds: int = Field(default=5, ge=2, le=20)
    k_values: List[int] = Field(default=[1, 3, 5, 10])
    judge_prompt_faithfulness: str = (
        "Valuta se la risposta è fedele al contesto fornito.\n"
        "Contesto: {context}\nRisposta: {answer}\n"
        'Rispondi in JSON: {"score": <0-1>, "reason": "<str>"}'
    )
    judge_prompt_relevance: str = (
        "Valuta se la risposta risponde alla domanda.\n"
        "Domanda: {question}\nRisposta: {answer}\n"
        'Rispondi in JSON: {"score": <0-1>, "reason": "<str>"}'
    )

    @field_validator("k_values")
    @classmethod
    def k_values_must_be_positive(cls, v: List[int]) -> List[int]:
        if any(k < 1 for k in v):
            raise ValueError("Tutti i valori di K devono essere >= 1")
        return sorted(v)


class EvaluationConfig(BaseModel):
    default_mode: Literal["A", "B"] = "A"
    mode_a: EvaluationModeAConfig = EvaluationModeAConfig()
    mode_b: EvaluationModeBConfig = EvaluationModeBConfig()
    output_dir: str = "/app/eval_reports"


class AppConfig(BaseModel):
    qdrant: QdrantConfig = QdrantConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    indexing: IndexingConfig = IndexingConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    query_enhancement: Optional[QueryEnhancementConfig] = None
    reranker: Optional[RerankerConfig] = None
    api: ApiConfig = ApiConfig()
    logging: LoggingConfig = LoggingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def apply_env_overrides(raw: dict) -> dict:
    """Sovrascrive i valori del dizionario YAML con le variabili d'ambiente."""
    if host := os.getenv("QDRANT_HOST"):
        raw.setdefault("qdrant", {})["host"] = host
    if port := os.getenv("QDRANT_PORT"):
        raw.setdefault("qdrant", {})["port"] = int(port)
    if llm_provider := os.getenv("LLM_PROVIDER"):
        raw.setdefault("llm", {})["provider"] = llm_provider
    if ollama_host := os.getenv("OLLAMA_HOST"):
        raw.setdefault("llm", {})["host"] = ollama_host
    if ollama_model := os.getenv("OLLAMA_MODEL"):
        raw.setdefault("llm", {})["model"] = ollama_model
    if openai_api_key := os.getenv("OPENAI_API_KEY"):
        raw.setdefault("llm", {})["openai_api_key"] = openai_api_key
    if openai_model := os.getenv("OPENAI_MODEL"):
        raw.setdefault("llm", {})["model"] = openai_model
    if openai_base_url := os.getenv("OPENAI_BASE_URL"):
        raw.setdefault("llm", {})["openai_base_url"] = openai_base_url
    if embedder_model := os.getenv("EMBEDDER_MODEL"):
        raw.setdefault("embedder", {})["model"] = embedder_model
    if embedder_batch_size := os.getenv("EMBEDDER_BATCH_SIZE"):
        raw.setdefault("embedder", {})["batch_size"] = int(embedder_batch_size)
    if embedder_max_length := os.getenv("EMBEDDER_MAX_LENGTH"):
        raw.setdefault("embedder", {})["max_length"] = int(embedder_max_length)
    if embedder_cache_dir := os.getenv("EMBEDDER_CACHE_DIR"):
        raw.setdefault("embedder", {})["cache_dir"] = embedder_cache_dir
    if embedder_dim := os.getenv("EMBEDDER_DIM"):
        raw.setdefault("embedder", {})["embedding_dim"] = int(embedder_dim)
    # Custom embedder via add_custom_model()
    if custom_dim := os.getenv("EMBEDDER_CUSTOM_DIM"):
        raw.setdefault("embedder", {}).setdefault("custom", {})["dim"] = int(custom_dim)
        if custom_pooling := os.getenv("EMBEDDER_CUSTOM_POOLING"):
            raw["embedder"]["custom"]["pooling"] = custom_pooling
        if custom_norm := os.getenv("EMBEDDER_CUSTOM_NORMALIZATION"):
            raw["embedder"]["custom"]["normalization"] = custom_norm.lower() not in ("false", "0", "no")
        if custom_hf := os.getenv("EMBEDDER_CUSTOM_HF_REPO"):
            raw["embedder"]["custom"]["hf_repo"] = custom_hf
        if custom_url := os.getenv("EMBEDDER_CUSTOM_URL"):
            raw["embedder"]["custom"]["url"] = custom_url
        if custom_file := os.getenv("EMBEDDER_CUSTOM_MODEL_FILE"):
            raw["embedder"]["custom"]["model_file"] = custom_file
    if reranker_model := os.getenv("RERANKER_MODEL"):
        raw.setdefault("reranker", {})["model"] = reranker_model
    if collection_name := os.getenv("QDRANT_COLLECTION_NAME"):
        raw.setdefault("qdrant", {})["collection_name"] = collection_name
    if log_level := os.getenv("LOG_LEVEL"):
        raw.setdefault("logging", {})["level"] = log_level
    return raw


def load_config(path: str = "config.yaml") -> AppConfig:
    """
    Carica e valida il file di configurazione YAML applicando gli override da env.

    Raises:
        FileNotFoundError: se il file non esiste
        ValueError: se il YAML non rispetta lo schema Pydantic
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = apply_env_overrides(raw or {})

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Configurazione non valida:\n{e}") from e


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Restituisce l'istanza singleton della configurazione, caricata una sola volta."""
    global _config
    if _config is None:
        config_path = os.getenv("CONFIG_PATH", "config.yaml")
        _config = load_config(config_path)
    return _config
