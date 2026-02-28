# Cognitor Assistant

Assistente virtuale con classificazione di intenti utilizzando PyTorch, FastText e NER.

## Quick Start

```bash
# Avvia il server API
uvicorn main:app --reload

# Avvia l'agent in modalità interattiva
python -m agent.agent
```

## Architettura dei Modelli

### 1. FastText (Word Embeddings)
Modello Skip-gram per generare rappresentazioni vettoriali delle parole.

- **Input**: File di testo (`data/fast-text.txt`)
- **Output**: `models/fasttext_model.bin`
- **Parametri**: dim=300, epoch=20, lr=0.1, minCount=3, wordNgrams=2

### 2. Intent Classifier (PyTorch)
Rete neurale BiGRU con Attention per la classificazione degli intenti.

- **Architettura**:
  - Embedding layer inizializzato con FastText
  - BiGRU bidirezionale (hidden_dim=256)
  - Attention layer
  - Dropout (0.3)
  - Linear layer per classificazione

## Struttura del Progetto

```
cognitor-assistant/
├── api/                    # API FastAPI
│   ├── auth.py            # Endpoint autenticazione
│   └── chatbot.py         # Endpoint chatbot
├── agent/                 # Agent conversazionale
│   └── agent.py
├── classes/               # Classi utility
├── intellective/          # Training modelli
│   ├── intent_classifier.py
│   ├── train_fast_text.py
│   └── train_intent_classifier.py
├── knowledge/             # Knowledge base
│   ├── intents/          # Intent definitions
│   ├── rules/            # Intent -> Response mapping
│   └── responses/        # Response templates
├── models/               # Modelli addestrati
├── pipeline/             # Pipeline di training
├── tests/                # Test
├── main.py               # FastAPI app
└── config.py
```

## API Endpoints

### POST /auth/token
Autenticazione utente, restituisce JWT token.

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=admin123"
```

### POST /chatbot/message
Invia un messaggio al chatbot (richiede token Bearer).

```bash
curl -X POST http://localhost:8000/chatbot/message \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Ciao!"}'
```

### GET /health
Health check.

## Knowledge Base

### Intents
Definiti in `knowledge/intents/`.

### Rules
Mappatura intent -> response in `knowledge/rules/`.

```json
{
  "rules": {
    "greeting": ["greeting_response"]
  }
}
```

### Responses
Risposte in `knowledge/responses/`.

```json
{
  "responses": {
    "greeting_response": [
      "Ciao! Come posso aiutarti?"
    ]
  }
}
```

## Training

### Pipeline Completa

```bash
python -m pipeline
```

### Step Manuali

1. **Preparazione dati**: `python -m pipeline.intent_builder`
2. **FastText**: `python -m intellective.train_fast_text`
3. **Intent Classifier**: `python -m intellective.train_intent_classifier`

## Dipendenze

- torch
- fasttext
- spacy
- fastapi
- uvicorn
- pandas
- scikit-learn
- nltk
- python-jose
