#!/bin/bash
# ollama-init.sh — scarica il modello LLM al primo avvio
# Uso: chmod +x ollama-init.sh && ./ollama-init.sh

MODEL=${OLLAMA_MODEL:-mistral}
OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}

echo "Attesa disponibilità Ollama su ${OLLAMA_URL}..."
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null; do
  echo "  Ollama non ancora pronto, riprovo tra 2s..."
  sleep 2
done

echo "Scaricamento modello: ${MODEL}"
docker exec rag_ollama ollama pull "${MODEL}"

echo "Modello ${MODEL} scaricato con successo."
