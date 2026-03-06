# Training

## Pipeline Completa

La pipeline automatica esegue tutti i passaggi in sequenza:

```bash
python -m pipeline
```

## Step Manuali

### 1. Preparazione dei Dati

Genera il file JSON unificato a partire dai file YAML degli intenti:

```bash
python -m pipeline.intent_builder
```

Output: `.cognitor/training_source.json`

### 2. Training FastText

Addestra il modello di word embeddings:

```bash
python -m intellective.train_fast_text
```

Output: `models/fasttext_model.bin`

**Parametri configurabili:**

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `dim` | 300 | Dimensione degli embedding |
| `epoch` | 20 | Numero di epoche |
| `lr` | 0.1 | Learning rate |
| `minCount` | 3 | Occorrenze minime di una parola |
| `wordNgrams` | 2 | N-gram di parole |

### 3. Training Intent Classifier

Addestra la rete neurale BiGRU:

```bash
python -m intellective.train_intent_classifier
```

Output: `models/intent_classifier.pt`

## Diagnostica

In caso di problemi durante il training, consulta i log in `diagnostics/`:

```bash
cat diagnostics/merge_errors.log
```

## Soglia di Confidenza

La soglia minima di confidenza per accettare un intento è configurabile in `config.py`:

```python
MIN_INTENT_CONFIDENCE = 0.20
```

Abbassare questo valore aumenta la sensibilità ma può causare falsi positivi.
