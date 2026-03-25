# RAGtio

Sistema RAG (Retrieval-Augmented Generation) basato su Haystack, Qdrant e Ollama (o OpenAI).

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

Al primo avvio, il servizio `ollama-init` scarica automaticamente il modello `mistral`.
Il download può richiedere alcuni minuti a seconda della connessione.

## Interfaccia web

Dopo l'avvio, l'interfaccia è disponibile su: http://localhost:8000

---

## Manuale utente

### Come funziona

1. **Carica uno o più documenti** tramite il pannello di caricamento
2. **Fai una domanda** nella barra di ricerca
3. Il sistema recupera i brani più rilevanti dai documenti caricati e genera una risposta

---

### Formati di file supportati

| Formato | Estensione | Note |
|---------|-----------|------|
| Testo semplice | `.txt` | Tutto il file diventa un unico documento |
| Markdown | `.md` | Tutto il file diventa un unico documento (la sintassi viene preservata) |
| PDF | `.pdf` | Ogni pagina diventa un documento separato |
| Word | `.docx` | I paragrafi vengono uniti in un unico documento |
| CSV | `.csv` | Ogni riga diventa un documento — richiede di specificare la colonna testo |
| JSON / JSONL | `.json`, `.jsonl` | Ogni oggetto diventa un documento — richiede di specificare la colonna testo |

---

### Esempi di file

#### TXT e Markdown

Nessun requisito di formato particolare. Il contenuto integrale del file diventa un unico documento.

Esempio `.md`:

```markdown
# Fotosintesi

La fotosintesi è il processo con cui le piante convertono la luce solare in energia chimica.

## Fasi principali

- **Fase luminosa**: avviene nei tilacoidi e produce ATP
- **Ciclo di Calvin**: avviene nello stroma e fissa la CO₂
```

#### TXT

```
La fotosintesi è il processo con cui le piante convertono la luce solare in energia chimica.
Questo processo avviene nei cloroplasti e produce glucosio e ossigeno.
```

---

#### CSV

Il CSV deve avere un'intestazione (prima riga con i nomi delle colonne). Al momento del caricamento
devi indicare quale colonna contiene il testo principale da indicizzare.

```csv
titolo,testo,categoria
"Fotosintesi","La fotosintesi è il processo con cui le piante convertono la luce solare in energia.","biologia"
"Mitosi","La mitosi è il processo di divisione cellulare che produce due cellule figlie identiche.","biologia"
"Algebra lineare","L'algebra lineare studia vettori, matrici e trasformazioni lineari.","matematica"
```

In questo esempio, al caricamento dovrai specificare `testo` come colonna del contenuto.
Le colonne `titolo` e `categoria` verranno salvate come metadati e compariranno nelle fonti della risposta.

> **Nota:** il file deve essere codificato in UTF-8.

---

#### JSON

Il JSON deve essere un array di oggetti, tutti con le stesse chiavi. Anche qui devi indicare
quale campo contiene il testo principale.

```json
[
  {
    "titolo": "Fotosintesi",
    "testo": "La fotosintesi è il processo con cui le piante convertono la luce solare in energia.",
    "categoria": "biologia"
  },
  {
    "titolo": "Mitosi",
    "testo": "La mitosi è il processo di divisione cellulare che produce due cellule figlie identiche.",
    "categoria": "biologia"
  }
]
```

Al caricamento specifica `testo` come colonna del contenuto. Gli altri campi diventano metadati.

---

#### JSONL

Come JSON ma un oggetto per riga (formato comodo per dataset grandi):

```jsonl
{"titolo": "Fotosintesi", "testo": "La fotosintesi converte la luce solare in energia.", "categoria": "biologia"}
{"titolo": "Mitosi", "testo": "La mitosi produce due cellule figlie identiche.", "categoria": "biologia"}
```

---

#### PDF e DOCX

Nessuna preparazione richiesta: carica il file direttamente. Il sistema estrae il testo
automaticamente (una pagina = un documento per i PDF; tutti i paragrafi uniti per i DOCX).

---

### Gestione dei duplicati

Se carichi un file già presente nell'indice puoi scegliere tra:

- **SKIP** — le versioni esistenti vengono mantenute, le nuove ignorate
- **OVERWRITE** — i documenti esistenti vengono sostituiti con quelli nuovi

---

### Consigli pratici sui file

- Per documenti CSV/JSON con testi brevi (es. FAQ, descrizioni prodotto), il sistema funziona meglio con testi di almeno 2-3 frasi per riga.
- Evita righe vuote o con testo nullo: vengono scartate automaticamente.
- Per PDF scansionati (immagini), il testo non viene estratto — usa PDF con testo selezionabile.
- La dimensione massima per file è configurabile (default: 100 MB).

---

## Guida all'interfaccia

L'interfaccia è divisa in quattro schede accessibili dalla barra in alto.
L'intestazione mostra in tempo reale lo stato della connessione a Qdrant e Ollama, la latenza, i modelli attivi, il numero di documenti e il nome della collection.

---

### Scheda 1 — Ingestione

Questa scheda serve a caricare documenti nell'indice.

**Passaggi:**

1. **Carica il file** — trascina il file nell'area tratteggiata oppure clicca **Sfoglia…**
2. **Opzioni file tabellare** *(solo per CSV, JSON, JSONL)* — compila i campi:
   - **Colonna testo**: nome della colonna che contiene il testo da indicizzare (es. `testo`)
   - **Colonne metadati**: colonne aggiuntive da conservare come etichette, separate da virgola (es. `titolo, categoria`). Se lasciato vuoto, vengono incluse tutte le colonne tranne quella del testo.
3. **Metadati aggiuntivi** *(opzionale)* — aggiungi coppie chiave/valore da associare a tutti i documenti del file, utili per filtrarli in seguito (es. `fonte` = `manuale_2024`)
4. **Strategia di chunking** — scegli come suddividere il testo in frammenti:
   - **Character**: taglia ogni N caratteri con un overlap configurabile (default: 1000 caratteri, overlap 150)
   - **Recursive**: come Character ma rispetta separatori naturali del testo (consigliato, default)
   - **Sentence**: taglia ogni N frasi con un overlap in frasi
   - **Paragraph**: taglia ogni N paragrafi con un overlap in paragrafi
5. **Policy duplicati** — se il file è già stato caricato in precedenza:
   - **OVERWRITE**: sostituisce i documenti esistenti
   - **SKIP**: mantiene i documenti esistenti e ignora i nuovi
6. Clicca **Indicizza Documenti** — il pulsante si attiva solo dopo aver selezionato un file

Al termine viene mostrato un riepilogo con il numero di documenti e chunk creati.

---

### Scheda 2 — Query & Risposta

Questa scheda serve a interrogare i documenti indicizzati.

**Passaggi:**

1. **Scrivi la domanda** nel campo di testo (es. *"Quali sono le fasi della fotosintesi?"*)
2. **Opzioni Retrieval** *(opzionale)* — personalizza come vengono recuperati i documenti:
   - **Modalità**: `Hybrid` (consigliata), `Dense` (solo embedding semantico), `Sparse` (solo keyword BM25)
   - **Top-K**: quanti frammenti recuperare dall'indice (default: 10)
   - **Top-N dopo rerank**: quanti frammenti passare al modello dopo il riordinamento (default: 5)
3. **Filtri Metadata** *(opzionale)* — limita la ricerca a documenti con specifici metadati.
   Esempio: campo `categoria`, operatore `=`, valore `biologia` → cerca solo nei documenti di biologia.
   Gli operatori disponibili sono: `=`, `>=`, `<=`, `in`, `not`.
4. **Query Enhancement** *(opzionale)* — migliora automaticamente la query:
   - **Query Expansion**: genera N varianti semantiche della domanda e le usa tutte per il retrieval
   - **Query Decomposition**: scompone la domanda in N sotto-domande più semplici
5. **Streaming SSE** — se attivo, la risposta appare progressivamente; se disattivo, arriva tutta in una volta
6. Clicca **Invia Domanda**

Sotto la risposta vengono mostrate le **fonti** usate (nome file, eventuale pagina o riga) e le **sotto-query** generate dall'enhancement.

---

### Scheda 3 — Configurazione

Questa scheda permette di modificare i parametri del sistema senza riavviare.

> Le modifiche sono temporanee: al riavvio del container i valori tornano a quelli del file `config.yaml`.

**Modello LLM:**

| Campo | Descrizione | Default |
|-------|-------------|---------|
| Modello Ollama | Modello da usare per la generazione (deve essere già scaricato) | `mistral` |
| Temperature | Creatività della risposta: 0 = deterministico, 2 = molto creativo | `0.1` |
| Top-P | Campionamento nucleare: valori bassi rendono la risposta più focalizzata | `0.9` |
| Max tokens | Lunghezza massima della risposta in token | `1024` |
| System Prompt | Istruzioni di sistema inviate al modello prima di ogni domanda | vedi UI |
| Prompt Template RAG | Template che assembla contesto e domanda. Usa `{context}` e `{question}` come segnaposto | vedi UI |

**Modello Embedder:**

| Campo | Descrizione |
|-------|-------------|
| Modello FastEmbed | Modello usato per trasformare il testo in vettori. Cambiarlo richiede la re-indicizzazione di tutti i documenti |
| Batch size | Numero di testi elaborati in parallelo durante l'embedding |
| Max length | Numero massimo di token per testo in ingresso all'embedder |

**Prompt Query Enhancement:** prompt personalizzabili per l'expansion e la decomposition delle query.

Clicca **Salva Configurazione** per applicare le modifiche o **Ripristina** per ricaricare i valori correnti.

---

### Scheda 4 — Evaluation

Questa scheda misura la qualità del retrieval in modo automatico, senza dataset etichettati.

#### Come funziona (Mode A)

1. Il sistema campiona **chunk casuali** dall'indice
2. Per ogni chunk chiede al LLM di generare una **query sintetica** il cui tipo dipende dalla modalità di retrieval configurata:
   - **Domanda sintetica** — usata con retrieval **Dense** o **Hybrid**: cattura il significato semantico del chunk
   - **Keyword** (3–6 frasi chiave) — usata con retrieval **Sparse** (BM25): rispecchia il vocabolario esatto del chunk
   - La scelta è automatica con `query_type: auto` (default), oppure forzabile a `question` o `keywords` nel file `config.yaml`
3. Usa quella query per fare retrieval e verifica se il chunk originale viene recuperato
4. Aggrega i risultati in metriche di qualità

**Per avviare l'evaluation:**

1. Assicurarsi che ci siano documenti nell'indice (vedi scheda Ingestione)
2. Cliccare **Avvia Evaluation Mode A**
3. Attendere il completamento (può richiedere alcuni minuti a seconda del numero di campioni)

#### Metriche restituite

| Metrica | Cosa misura | Valore ideale |
|---------|-------------|---------------|
| **MRR** (Mean Reciprocal Rank) | Posizione media del primo risultato rilevante: 1.0 se è sempre primo, 0.5 se è sempre secondo | più vicino a 1 |
| **Recall@K** | Frazione di chunk rilevanti trovati nei primi K risultati | più vicino a 1 |
| **Hit Rate@K** | Percentuale di domande per cui almeno un risultato rilevante è nei primi K | più vicino a 1 |
| **NDCG@K** | Come Recall@K ma penalizza i risultati rilevanti trovati in posizione bassa | più vicino a 1 |

I valori K di default sono `[1, 3, 5, 10]`, configurabili nel file `config.yaml`.

#### Come interpretare i risultati

- **MRR > 0.7** e **Hit Rate@5 > 0.8** indicano un retrieval di buona qualità
- Valori bassi suggeriscono di provare a cambiare la modalità di retrieval (Dense / Sparse / Hybrid nella scheda Query) o la strategia di chunking
- Dopo ogni modifica alla configurazione, rieseguire l'evaluation per confrontare i valori

Il dettaglio per campione (domanda generata, chunk atteso, chunk recuperati) è visibile nella tabella in fondo alla pagina.

---

## Variabili d'ambiente

Le seguenti variabili d'ambiente sovrascrivono i valori di `config.yaml`:

| Variabile | Descrizione | Default |
|-----------|-------------|---------|
| `LLM_PROVIDER` | Backend LLM: `ollama` o `openai` | `ollama` |
| `OLLAMA_HOST` | URL del server Ollama | `http://ollama:11434` |
| `OLLAMA_MODEL` | Modello Ollama da usare | `mistral` |
| `OPENAI_API_KEY` | API key OpenAI *(se `LLM_PROVIDER=openai`)* | — |
| `OPENAI_MODEL` | Modello OpenAI *(es. `gpt-4o-mini`)* | — |
| `OPENAI_BASE_URL` | Endpoint OpenAI custom *(opzionale)* | — |
| `QDRANT_HOST` | Host del database vettoriale | `qdrant` |
| `QDRANT_PORT` | Porta REST di Qdrant | `6333` |
| `QDRANT_COLLECTION_NAME` | Nome della collection | `documents` |
| `EMBEDDER_MODEL` | Modello FastEmbed per gli embedding | `intfloat/multilingual-e5-base` |
| `RERANKER_MODEL` | Modello cross-encoder per il reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `LOG_LEVEL` | Livello di log (`DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `CONFIG_PATH` | Percorso del file di configurazione | `config.yaml` |

Crea un file `.env` nella root del progetto per impostare le variabili:

```env
OLLAMA_MODEL=llama3

# Per usare OpenAI invece di Ollama:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
```

---

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

## Cambiare il modello Ollama

Imposta la variabile d'ambiente `OLLAMA_MODEL` prima di avviare lo stack:

```bash
OLLAMA_MODEL=llama3 docker compose up -d
```

Oppure crea un file `.env` nella root del progetto:

```env
OLLAMA_MODEL=llama3
```

## Usare OpenAI come backend LLM

Per usare GPT invece di Ollama, configura le variabili d'ambiente nel file `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Il servizio `ollama` e `ollama-init` restano attivi nello stack ma non vengono usati per la generazione.
Gli embedding continuano ad essere calcolati localmente tramite FastEmbed.

---

## API

Il backend espone le seguenti API REST su `http://localhost:8000`:

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| `GET` | `/api/status` | Stato del sistema (Qdrant, Ollama, modelli, conteggio documenti) |
| `POST` | `/api/ingest` | Carica e indicizza un documento |
| `POST` | `/api/query` | Esegui una query RAG (con supporto streaming SSE) |
| `GET` | `/api/config` | Leggi la configurazione corrente |
| `POST` | `/api/config` | Aggiorna la configurazione a runtime |
| `DELETE` | `/api/index` | Elimina tutti i documenti dall'indice (`?confirm=true`) |
| `POST` | `/api/eval` | Avvia un job di evaluation (asincrono) |
| `GET` | `/api/eval/{job_id}` | Controlla lo stato e il risultato di un job di evaluation |
