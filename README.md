# RAGtio

RAG (Retrieval-Augmented Generation) system built on Haystack, Qdrant, and Ollama (or OpenAI).

<br>

![ragtio_logo](https://github.com/ArchAI-Labs/ragtio/blob/main/ragtio_logo.png?raw=true)

<br>

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) >= 24
- [Docker Compose](https://docs.docker.com/compose/install/) >= 2.20 (included in Docker Desktop)

## Getting Started

```bash
docker compose up -d
```

On first startup, the `ollama-init` service automatically downloads the `mistral` model.
The download may take a few minutes depending on your connection speed.

## Web Interface

Once started, the interface is available at: http://localhost:8000

---

## User Manual

### How It Works

1. **Upload one or more documents** via the upload panel
2. **Ask a question** in the search bar
3. The system retrieves the most relevant passages from the indexed documents and generates a response

---

### Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | The entire file becomes a single document |
| Markdown | `.md` | The entire file becomes a single document (syntax is preserved) |
| PDF | `.pdf` | Each page becomes a separate document |
| Word | `.docx` | Paragraphs are merged into a single document |
| CSV | `.csv` | Each row becomes a document — requires specifying the text column |
| JSON / JSONL | `.json`, `.jsonl` | Each object becomes a document — requires specifying the text column |

---

### File Examples

#### TXT and Markdown

No special format required. The entire file content becomes a single document.

Example `.md`:

```markdown
# Photosynthesis

Photosynthesis is the process by which plants convert sunlight into chemical energy.

## Main Phases

- **Light phase**: occurs in thylakoids and produces ATP
- **Calvin cycle**: occurs in the stroma and fixes CO₂
```

#### TXT

```
Photosynthesis is the process by which plants convert sunlight into chemical energy.
This process occurs in chloroplasts and produces glucose and oxygen.
```

---

#### CSV

The CSV must have a header row (first row with column names). When uploading, you must specify
which column contains the main text to be indexed.

```csv
title,text,category
"Photosynthesis","Photosynthesis is the process by which plants convert sunlight into energy.","biology"
"Mitosis","Mitosis is the cell division process that produces two identical daughter cells.","biology"
"Linear Algebra","Linear algebra studies vectors, matrices, and linear transformations.","mathematics"
```

In this example, specify `text` as the content column when uploading.
The `title` and `category` columns will be saved as metadata and shown in the response sources.

> **Note:** the file must be UTF-8 encoded.

---

#### JSON

The JSON must be an array of objects, all with the same keys. Here too you must specify
which field contains the main text.

```json
[
  {
    "title": "Photosynthesis",
    "text": "Photosynthesis is the process by which plants convert sunlight into energy.",
    "category": "biology"
  },
  {
    "title": "Mitosis",
    "text": "Mitosis is the cell division process that produces two identical daughter cells.",
    "category": "biology"
  }
]
```

Specify `text` as the content column when uploading. The other fields become metadata.

---

#### JSONL

Same as JSON but one object per line (convenient format for large datasets):

```jsonl
{"title": "Photosynthesis", "text": "Photosynthesis converts sunlight into energy.", "category": "biology"}
{"title": "Mitosis", "text": "Mitosis produces two identical daughter cells.", "category": "biology"}
```

---

#### PDF and DOCX

No preparation needed: upload the file directly. The system extracts text automatically
(one page = one document for PDFs; all paragraphs merged for DOCX).

---

### Duplicate Handling

If you upload a file that is already in the index, you can choose between:

- **SKIP** — existing versions are kept, new ones are ignored
- **OVERWRITE** — existing documents are replaced with the new ones

---

### Practical File Tips

- For CSV/JSON documents with short texts (e.g. FAQs, product descriptions), the system works better with texts of at least 2–3 sentences per row.
- Avoid empty rows or rows with null text: they are automatically discarded.
- For scanned PDFs (images), text is not extracted — use PDFs with selectable text.
- The maximum file size is configurable (default: 100 MB).

---

## Interface Guide

The interface is divided into four tabs accessible from the top bar.
The header shows in real time the connection status to Qdrant and Ollama, latency, active models, document count, and collection name.

---

### Tab 1 — Ingestion

This tab is used to upload documents into the index.

**Steps:**

1. **Upload the file** — drag the file into the dashed area or click **Browse…**
2. **Tabular file options** *(CSV, JSON, JSONL only)* — fill in the fields:
   - **Text column**: name of the column containing the text to index (e.g. `text`)
   - **Metadata columns**: additional columns to keep as labels, comma-separated (e.g. `title, category`). If left empty, all columns except the text column are included.
3. **Additional metadata** *(optional)* — add key/value pairs to associate with all documents in the file, useful for filtering later (e.g. `source` = `manual_2024`)
4. **Chunking strategy** — choose how to split text into fragments:
   - **Character**: splits every N characters with a configurable overlap (default: 1000 chars, overlap 150)
   - **Recursive**: like Character but respects natural text separators (recommended, default)
   - **Sentence**: splits every N sentences with a sentence-level overlap
   - **Paragraph**: splits every N paragraphs with a paragraph-level overlap
5. **Duplicate policy** — if the file has already been uploaded:
   - **OVERWRITE**: replaces existing documents
   - **SKIP**: keeps existing documents and ignores new ones
6. Click **Index Documents** — the button activates only after selecting a file

At the end, a summary is shown with the number of documents and chunks created.

---

### Tab 2 — Query & Answer

This tab is used to query the indexed documents.

**Steps:**

1. **Write your question** in the text field (e.g. *"What are the phases of photosynthesis?"*)
2. **Retrieval options** *(optional)* — customize how documents are retrieved:
   - **Mode**: `Hybrid` (recommended), `Dense` (semantic embedding only), `Sparse` (BM25 keywords only)
   - **Top-K**: how many fragments to retrieve from the index (default: 10)
   - **Top-N after rerank**: how many fragments to pass to the model after reranking (default: 5)
3. **Metadata filters** *(optional)* — restrict the search to documents with specific metadata.
   Example: field `category`, operator `=`, value `biology` → search only in biology documents.
   Available operators: `=`, `>=`, `<=`, `in`, `not`.
4. **Query Enhancement** *(optional)* — automatically improve the query:
   - **Query Expansion**: generates N semantic variants of the question and uses them all for retrieval
   - **Query Decomposition**: breaks the question into N simpler sub-questions
5. **SSE Streaming** — if enabled, the response appears progressively; if disabled, it arrives all at once
6. Click **Send Question**

Below the response, the **sources** used (file name, page or row if applicable) and the **sub-queries** generated by enhancement are displayed.

---

### Tab 3 — Configuration

This tab allows you to modify system parameters without restarting.

> Changes are temporary: on container restart, values revert to those in `config.yaml`.

**LLM Model:**

| Field | Description | Default |
|-------|-------------|---------|
| Ollama Model | Model to use for generation (must already be downloaded) | `mistral` |
| Temperature | Response creativity: 0 = deterministic, 2 = very creative | `0.1` |
| Top-P | Nucleus sampling: lower values make responses more focused | `0.9` |
| Max tokens | Maximum response length in tokens | `1024` |
| System Prompt | System instructions sent to the model before each question | see UI |
| RAG Prompt Template | Template that assembles context and question. Use `{context}` and `{question}` as placeholders | see UI |

**Embedder Model:**

| Field | Description |
|-------|-------------|
| FastEmbed Model | Model used to transform text into vectors. Changing it requires re-indexing all documents |
| Batch size | Number of texts processed in parallel during embedding |
| Max length | Maximum number of tokens per text input to the embedder |

**Query Enhancement Prompts:** customizable prompts for query expansion and decomposition.

Click **Save Configuration** to apply changes or **Restore** to reload current values.

---

### Tab 4 — Evaluation

This tab measures retrieval quality automatically, without labeled datasets.

#### How It Works (Mode A)

1. The system samples **random chunks** from the index
2. For each chunk, it asks the LLM to generate a **synthetic query** whose type depends on the configured retrieval mode:
   - **Synthetic question** — used with **Dense** or **Hybrid** retrieval: captures the semantic meaning of the chunk
   - **Keywords** (3–6 key phrases) — used with **Sparse** retrieval (BM25): mirrors the exact vocabulary of the chunk
   - The choice is automatic with `query_type: auto` (default), or can be forced to `question` or `keywords` in `config.yaml`
3. It uses that query to perform retrieval and checks whether the original chunk is retrieved
4. It aggregates the results into quality metrics

**To start the evaluation:**

1. Make sure there are documents in the index (see Ingestion tab)
2. Click **Start Evaluation Mode A**
3. Wait for completion (may take a few minutes depending on the number of samples)

#### Returned Metrics

| Metric | What it measures | Ideal value |
|--------|-----------------|-------------|
| **MRR** (Mean Reciprocal Rank) | Average position of the first relevant result: 1.0 if always first, 0.5 if always second | closer to 1 |
| **Recall@K** | Fraction of relevant chunks found in the top K results | closer to 1 |
| **Hit Rate@K** | Percentage of questions for which at least one relevant result is in the top K | closer to 1 |
| **NDCG@K** | Like Recall@K but penalizes relevant results found at lower positions | closer to 1 |

Default K values are `[1, 3, 5, 10]`, configurable in `config.yaml`.

#### How to Interpret the Results

- **MRR > 0.7** and **Hit Rate@5 > 0.8** indicate good retrieval quality
- Low values suggest trying a different retrieval mode (Dense / Sparse / Hybrid in the Query tab) or chunking strategy
- After each configuration change, re-run the evaluation to compare values

Per-sample details (generated query, expected chunk, retrieved chunks) are visible in the table at the bottom of the page.

---

## Environment Variables

The following environment variables override values in `config.yaml`:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend: `ollama` or `openai` | `ollama` |
| `OLLAMA_HOST` | Ollama server URL | `http://ollama:11434` |
| `OLLAMA_MODEL` | Ollama model to use | `qwen3:4b` |
| `OPENAI_API_KEY` | OpenAI API key *(if `LLM_PROVIDER=openai`)* | — |
| `OPENAI_MODEL` | OpenAI model *(e.g. `gpt-4.1-mini`)* | — |
| `OPENAI_BASE_URL` | Custom OpenAI endpoint *(optional)* | — |
| `QDRANT_HOST` | Vector database host | `qdrant` |
| `QDRANT_PORT` | Qdrant REST port | `6333` |
| `QDRANT_COLLECTION_NAME` | Collection name | `documents` |
| `EMBEDDER_MODEL` | FastEmbed model for embeddings | `intfloat/multilingual-e5-base` |
| `EMBEDDER_BATCH_SIZE` | Number of texts processed per batch during embedding | `32` |
| `EMBEDDER_MAX_LENGTH` | Maximum token length per input to the embedder | `512` |
| `EMBEDDER_CACHE_DIR` | Local directory where FastEmbed caches downloaded models | `/app/models/fastembed` |
| `EMBEDDER_DIM` | Manual override of the embedding vector dimension | auto-detected |
| `EMBEDDER_CUSTOM_DIM` | Vector dimension for a custom ONNX model *(required to activate custom mode)* | — |
| `EMBEDDER_CUSTOM_POOLING` | Pooling strategy: `MEAN`, `CLS`, or `MAX` | `MEAN` |
| `EMBEDDER_CUSTOM_NORMALIZATION` | Whether to L2-normalize output vectors | `true` |
| `EMBEDDER_CUSTOM_HF_REPO` | Hugging Face repo ID of the custom model | — |
| `EMBEDDER_CUSTOM_URL` | Direct URL to the ONNX model archive *(alternative to `HF_REPO`)* | — |
| `EMBEDDER_CUSTOM_MODEL_FILE` | Path to the `.onnx` file inside the repo/archive | `onnx/model.onnx` |
| `RERANKER_MODEL` | Cross-encoder model for reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `LOG_LEVEL` | Log level (`DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `CONFIG_PATH` | Configuration file path | `config.yaml` |

Create a `.env` file in the project root to set variables:

```env
OLLAMA_MODEL=llama3

# To use OpenAI instead of Ollama:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
```

---

## Useful Commands

```bash
# Service status
docker compose ps

# Live logs
docker compose logs -f app

# Stop
docker compose down

# Stop and delete volumes (WARNING: removes all data)
docker compose down -v
```

## Changing the Ollama Model

Set the `OLLAMA_MODEL` environment variable before starting the stack:

```bash
OLLAMA_MODEL=llama3 docker compose up -d
```

Or create a `.env` file in the project root:

```env
OLLAMA_MODEL=llama3
```

## Using OpenAI as LLM Backend

To use GPT instead of Ollama, configure the environment variables in the `.env` file:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

The `ollama` and `ollama-init` services remain active in the stack but are not used for generation.
Embeddings continue to be computed locally via FastEmbed.

---

## Using a Custom Embedding Model

FastEmbed supports any model that has an ONNX export, even if it is not in the official supported list.
Use `EMBEDDER_CUSTOM_*` variables to register it before the embedder is instantiated.

**Example — load a model from Hugging Face:**

```env
EMBEDDER_MODEL=intfloat/multilingual-e5-small
EMBEDDER_CUSTOM_DIM=384
EMBEDDER_CUSTOM_POOLING=MEAN
EMBEDDER_CUSTOM_NORMALIZATION=true
EMBEDDER_CUSTOM_HF_REPO=intfloat/multilingual-e5-small
EMBEDDER_CUSTOM_MODEL_FILE=onnx/model.onnx
```

Or equivalently via `config.yaml`:

```yaml
embedder:
  model: "intfloat/multilingual-e5-small"
  custom:
    dim: 384
    pooling: "MEAN"
    normalization: true
    hf_repo: "intfloat/multilingual-e5-small"
    model_file: "onnx/model.onnx"
```

> **Note:** when switching to a different embedding model you must delete and re-index all documents,
> because the vector dimension stored in Qdrant must match the new model's output.

---

## API

The backend exposes the following REST APIs at `http://localhost:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | System status (Qdrant, Ollama, models, document count) |
| `POST` | `/api/ingest` | Upload and index a document |
| `POST` | `/api/query` | Run a RAG query (with SSE streaming support) |
| `GET` | `/api/config` | Read current configuration |
| `POST` | `/api/config` | Update configuration at runtime |
| `DELETE` | `/api/index` | Delete all documents from the index (`?confirm=true`) |
| `POST` | `/api/eval` | Start an evaluation job (asynchronous) |
| `GET` | `/api/eval/{job_id}` | Check the status and result of an evaluation job |
