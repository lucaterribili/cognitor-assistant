# Introduzione

Cognitor Assistant è un assistente virtuale basato su intelligenza artificiale che utilizza modelli di deep learning per la classificazione degli intenti e il riconoscimento di entità nelle frasi degli utenti.

## Caratteristiche Principali

- **Classificazione degli intenti** con rete neurale BiGRU + Attention (PyTorch)
- **Word embeddings** con FastText (modello Skip-gram)
- **Named Entity Recognition** (NER) per l'estrazione di parametri
- **API REST** con FastAPI e autenticazione JWT
- **Knowledge base** configurabile in YAML

## Architettura

```
cognitor-assistant/
├── api/                    # API FastAPI
│   ├── auth.py            # Endpoint autenticazione
│   └── chatbot.py         # Endpoint chatbot
├── agent/                 # Agent conversazionale
├── classes/               # Classi utility
├── intellective/          # Training dei modelli
├── knowledge/             # Knowledge base
│   ├── intents/          # Definizione degli intent
│   ├── rules/            # Mappatura intent → response
│   └── responses/        # Template delle risposte
├── pipeline/              # Pipeline di training
├── tests/                 # Test automatici
└── main.py               # Entry point FastAPI
```

## Come Iniziare

Consulta la [guida Quick Start](/guide/quickstart) per avviare il progetto in pochi minuti.
