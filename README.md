# RAGtio

Sistema RAG (Retrieval-Augmented Generation) basato su Haystack, Qdrant e Ollama.

<br>

![ragtio_logo](https://github.com/ArchAI-Labs/ragtio/blob/main/ragtio_logo.png?raw=true)

<br>

## Prerequisiti

- [Docker](https://docs.docker.com/get-docker/) >= 24
- [Docker Compose](https://docs.docker.com/compose/install/) >= 2.20 (incluso in Docker Desktop)

## Avvio

```bash
docker compose up -d
```

Al primo avvio, il servizio `ollama-init` scarica automaticamente il modello Mistral.
Il download può richiedere alcuni minuti a seconda della connessione.

## Cambiare il modello Ollama

Imposta la variabile d'ambiente `OLLAMA_MODEL` prima di avviare lo stack:

```bash
OLLAMA_MODEL=llama3 docker compose up -d
```

Oppure crea un file `.env` nella root del progetto:

```env
OLLAMA_MODEL=llama3
```

## Interfaccia web

Dopo l'avvio, l'interfaccia è disponibile su: http://localhost:8000

## Comandi utili

```bash
# Stato dei servizi
docker compose ps

# Log in tempo reale
docker compose logs -f app

# Arresto
docker compose down

# Arresto con cancellazione dei volumi (ATTENZIONE: rimuove tutti i dati)
docker compose down -v
```
