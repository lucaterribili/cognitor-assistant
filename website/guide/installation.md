# Installazione

## Requisiti di Sistema

| Componente | Versione minima |
|------------|----------------|
| Python | 3.10+ |
| pip | 22.0+ |
| RAM | 4 GB (8 GB consigliati) |
| Spazio disco | 2 GB |

## Installazione delle Dipendenze

```bash
pip install -r requirements.txt
```

Le principali dipendenze includono:

- `torch` — Framework deep learning
- `fasttext` — Word embeddings
- `spacy` — NER e NLP
- `fastapi` — Web framework per l'API
- `uvicorn` — ASGI server
- `python-jose` — JWT authentication

## Configurazione

Il file `config.py` contiene i parametri principali:

```python
# Soglia minima di confidenza per accettare un intent
MIN_INTENT_CONFIDENCE = 0.20

# Abilita modalità "doping" per migliorare con prompt wrapping
DOPING_ACTIVE = False
```

## Variabili d'Ambiente

Copia il file `.env.example` e personalizza i valori:

```bash
cp .env.example .env
```

## Training dei Modelli

Prima di avviare il server, è necessario addestrare i modelli:

```bash
python -m pipeline
```

Questo eseguirà la pipeline completa:
1. Preparazione dei dati di training
2. Training del modello FastText
3. Training del classificatore di intenti

Per il training manuale di ogni step:

```bash
# Step 1: Prepara i dati
python -m pipeline.intent_builder

# Step 2: Addestra FastText
python -m intellective.train_fast_text

# Step 3: Addestra il classificatore
python -m intellective.train_intent_classifier
```
