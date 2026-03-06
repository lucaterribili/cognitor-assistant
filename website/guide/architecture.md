# Architettura

## Overview

Cognitor Assistant è composto da due livelli principali:

1. **Layer di intelligenza** — Modelli ML per la comprensione del linguaggio naturale
2. **Layer applicativo** — API REST e agent conversazionale

## Modelli ML

### FastText (Word Embeddings)

Modello Skip-gram per generare rappresentazioni vettoriali delle parole.

- **Input**: File di testo (`knowledge/embeddings.txt`)
- **Output**: `models/fasttext_model.bin`
- **Parametri**: `dim=300`, `epoch=20`, `lr=0.1`, `minCount=3`, `wordNgrams=2`

### Intent Classifier (PyTorch)

Rete neurale BiGRU con Attention per la classificazione degli intenti.

```
Input Text
    │
    ▼
Tokenizzazione + Embedding (FastText)
    │
    ▼
BiGRU Bidirezionale (hidden_dim=256)
    │
    ▼
Attention Layer
    │
    ▼
Dropout (0.3)
    │
    ▼
Linear Layer → Softmax
    │
    ▼
Intent Label + Confidence Score
```

## Flusso di Elaborazione

```
Utente → API → Agent → Intent Classifier → Answer Manager → Risposta
                           │
                           └→ NER → Slot Filling
```

1. L'utente invia un messaggio all'API
2. L'agent preprocessa il testo
3. Il classificatore determina l'intento con un punteggio di confidenza
4. Il NER estrae le entità rilevanti (slot filling)
5. L'Answer Manager seleziona la risposta appropriata
6. La risposta viene restituita all'utente
