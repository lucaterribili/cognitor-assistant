# Arianna Assistant

Assistente virtuale con classificazione di intenti utilizzando PyTorch, FastText e SentencePiece.

## Architettura dei Modelli

### 1. FastText (Word Embeddings)
Modello Skip-gram per generare rappresentazioni vettoriali delle parole.

- **Input**: File di testo (`data/fast-text.txt`)
- **Output**: `models/fasttext_model.bin`
- **Parametri**: dim=300, epoch=20, lr=0.1, minCount=3, wordNgrams=2

### 2. SentencePiece (Tokenizer)
Tokenizer subword che converte testo in sequenze di interi.

- **Input**: Corpus di testo
- **Output**: `models/tokenizer/model.model`

### 3. Intent Classifier (PyTorch)
Rete neurale BiGRU con Attention per la classificazione degli intenti.

- **Architettura**:
  - Embedding layer inizializzato con FastText
  - BiGRU bidirezionale (hidden_dim=256)
  - Attention layer
  - Dropout (0.3)
  - Linear layer per classificazione

## Struttura del Progetto

```
arianna-assistant/
├── classes/              # Classi utility
│   ├── dataset_generator.py
│   └── intent_normalizer.py
├── intellective/        # Training modelli
│   ├── intent_classifier.py
│   ├── train_fast_text.py
│   └── train_intent_classifier.py
├── models/              # Modelli addestrati
├── pipeline/            # Pipeline di training
│   ├── __init__.py
│   └── intent_builder.py
├── training_data/       # Dati di training (versionati)
└── data/                # Dati generati automaticamente
```

## Dati di Training

### Formato Input

Il file `data/training_source.json` (o quelli in `training_data/`) deve seguire questa struttura:

```json
{
  "nlu": {
    "intents": [
      {
        "intent": "nome_intento",
        "examples": ["esempio 1", "esempio 2", ...]
      },
      ...
    ]
  }
}
```

### Gestione Dati Versionati

I dati di training sono memorizzati nella cartella `training_data/` e possono essere versionati con git. 

#### Merge dei dati

Per mergiare tutti i file JSON da `training_data/` in `data/training_source.json`:

```bash
python -m pipeline.merge_data
```

oppure specificare un file di output diverso:

```bash
python -m pipeline.merge_data -o custom_data.json
```

## Pipeline di Training

### Metodo 1: Pipeline Completa

Esegui tutto in un comando (incluso merge automatico dei dati):

```bash
python -m pipeline
```

Per saltare il merge:

```python
from pipeline import run_full_pipeline
run_full_pipeline(merge_data=False)
```

### Metodo 2: Step Manuali

#### 1. Preparazione dei Dati

```bash
python -m pipeline.intent_builder
```

oppure:

```bash
python test_intent_builder.py
```

Questo genera:
- `data/intent_dict.json` - Mappatura ID -> nome intent
- `data/nlu_data.csv` - Dati nel formato (INPUT, OUTPUT)
- `data/tokenized_data.npy` - Dati tokenizzati per il training

#### 2. Addestramento FastText

```bash
python -m intellective.train_fast_text
```

Genera `models/fasttext_model.bin`.

#### 3. Addestramento Intent Classifier

```bash
python -m intellective.train_intent_classifier
```

Genera `models/intent_model_fast.pth`.

## Dipendenze

- torch
- fasttext
- spacy
- numpy
- tqdm
- nltk
