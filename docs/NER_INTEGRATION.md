# Integrazione NER nel modello Intent Classifier

## Panoramica

Il modello `IntentClassifier` è stato esteso per supportare il **Named Entity Recognition (NER)** insieme alla classificazione degli intent. Ora il modello è in grado di:

1. **Classificare l'intent** della frase (es. restaurant_booking, send_email, ecc.)
2. **Estrarre entità** dalla frase (es. PERSON, LOCATION, DATE, TIME, NUMBER, ecc.)

## Architettura

### Multi-task Learning
Il modello utilizza un approccio di **joint training** con:
- **BiGRU condiviso**: encoder comune per entrambi i task
- **Branch Intent**: Attention + FC layer per classificazione
- **Branch NER**: FC + CRF layer per sequence labeling

```
Input Tokens
     ↓
 Embeddings (FastText)
     ↓
 BiGRU Bidirectional
     ↓
    / \
   /   \
  /     \
Attention  FC → CRF
  ↓           ↓
  FC      Viterbi Decode
  ↓           ↓
Intent    NER Tags
```

### Tipi di Entità Supportate

- **PERSON**: nomi di persone
- **LOCATION**: luoghi, città, indirizzi
- **DATE**: date
- **TIME**: orari
- **NUMBER**: numeri
- **PRODUCT**: prodotti
- **COMMAND**: comandi
- **TOPIC**: argomenti
- **EMAIL**: indirizzi email
- **TEAM**: team/gruppi

## Annotazione dei Dati

I dati in `/knowledge/intents/*.json` sono annotati con markup speciale:

```json
{
  "intent": "restaurant_booking",
  "examples": [
    "prenota per [due](NUMBER) persone",
    "tavolo a [Roma](LOCATION)",
    "prenota per [domani](DATE) alle [20:00](TIME)"
  ]
}
```

**Formato**: `[testo](ENTITY_TYPE)`

## Componenti Implementate

### 1. NERMarkupParser
Estrae entità dal markup e produce testo pulito:
```python
from classes.ner_markup_parser import NERMarkupParser

parser = NERMarkupParser()
clean_text, entities = parser.parse("prenota per [due](NUMBER) a [Roma](LOCATION)")
# Output:
# clean_text: "prenota per due a Roma"
# entities: [
#   {"start": 12, "end": 15, "entity": "NUMBER", "value": "due"},
#   {"start": 18, "end": 22, "entity": "LOCATION", "value": "Roma"}
# ]
```

### 2. NERTagBuilder
Converte entità in tag BIO (Begin-Inside-Outside) allineati ai token:
```python
from classes.ner_tag_builder import NERTagBuilder

builder = NERTagBuilder()
tokens = ['prenota', 'per', 'due', 'a', 'roma']
tag_ids = builder.align_tokens_to_bio(clean_text, tokens, entities)
# Output: [0, 0, 9, 0, 3]  # dove 0=O, 9=B-NUMBER, 3=B-LOCATION
```

### 3. IntentClassifier (Aggiornato)
Il modello ora supporta joint training e prediction:

```python
from intellective.intent_classifier import IntentClassifier

# Creazione modello
model = IntentClassifier(
    vocab_size=10000,
    embed_dim=300,
    hidden_dim=256,
    output_dim=8,
    dropout_prob=0.3,
    fasttext_model_path='models/fasttext_model.bin',
    freeze_embeddings=True
)

# Prediction
result = model.predict("prenota tavolo per due a Roma")
# Output:
# {
#   'intent_idx': 0,
#   'intent_confidence': 0.95,
#   'entities': [
#     {'start': 3, 'end': 4, 'entity': 'NUMBER', 'value': 'due'},
#     {'start': 5, 'end': 6, 'entity': 'LOCATION', 'value': 'Roma'}
#   ],
#   'tokens': ['prenota', 'tavolo', 'per', 'due', 'a', 'roma'],
#   'ner_tags': ['O', 'O', 'O', 'B-NUMBER', 'O', 'B-LOCATION']
# }
```

## Workflow Completo

### 1. Rigenerazione Dataset

```bash
python retrain_with_ner.py
```

Questo script:
- Carica tutti i file JSON da `knowledge/intents/`
- Parsa il markup NER con `NERMarkupParser`
- Genera tag BIO con `NERTagBuilder`
- Salva i dati processati in `data/tokenized_data.npy`
- Salva il tag builder in `data/ner_tag_builder.json`

### 2. Training del Modello

Il training ora gestisce **joint learning** con due loss:
- **Intent Loss**: CrossEntropyLoss per classificazione
- **NER Loss**: CRF negative log-likelihood per sequence labeling

```python
from intellective.train_intent_classifier import train_main_model

train_main_model()
```

I pesi delle loss possono essere configurati in `train_model()`:
- `intent_weight=1.0`
- `ner_weight=0.5`

### 3. Inference

```python
import torch
from intellective.intent_classifier import IntentClassifier

# Carica modello
model = IntentClassifier(...)
model.load_state_dict(torch.load('models/intent_model_fast.pth'))

# Predizione
result = model.predict("invia email a mario@test.it")
print(f"Intent: {result['intent_idx']}")
print(f"Entities: {result['entities']}")
```

## Modifiche ai File

### File Modificati:
1. **`intellective/intent_classifier.py`**
   - Aggiunto layer CRF per NER
   - Forward pass ritorna sia intent che NER
   - Metodo `predict()` estrae automaticamente le entità
   - Metodo `_extract_entities()` converte tag BIO in entità strutturate

2. **`classes/dataset_generator.py`**
   - Integrato `NERMarkupParser` per processare markup
   - CSV include colonne CLEAN_TEXT e ENTITIES
   - NPY include token_ids, intent_id, e ner_tag_ids
   - Salva NER tag builder per uso futuro

3. **`intellective/train_intent_classifier.py`**
   - Dataset carica anche ner_tags
   - `collate_fn` gestisce padding di ner_tags e maschere
   - `train_model()` implementa joint training con loss pesato

### File Nuovi:
1. **`test_ner_integration.py`** - Test di integrazione completo
2. **`retrain_with_ner.py`** - Script per rigenerazione e training

### Dipendenze Aggiunte:
- `pytorch-crf==0.7.2` - Layer CRF per sequence labeling

## Test

Per testare l'integrazione NER:

```bash
python test_ner_integration.py
```

Il test verifica:
1. ✓ Parsing del markup NER
2. ✓ Allineamento token → tag BIO
3. ✓ Forward pass del modello (training mode)
4. ✓ Forward pass del modello (inference mode)
5. ✓ Prediction end-to-end (richiede modello trainato)

## Prossimi Passi

1. **Rigenerare i dati** con il nuovo formato:
   ```bash
   python retrain_with_ner.py
   ```

2. **Trainare il modello** con supporto NER (lo script chiede conferma)

3. **Testare le predizioni** con frasi reali

## Note Tecniche

- **CRF Layer**: usa algoritmo di Viterbi per decodifica ottimale della sequenza
- **Maschere di Padding**: gestiscono correttamente sequenze di lunghezza variabile
- **Shared Encoder**: BiGRU condiviso migliora la generalizzazione
- **BIO Tagging**: schema standard per sequence labeling (Begin, Inside, Outside)

## Esempi di Output

```python
# Input: "prenota tavolo per due persone a Roma domani"
{
  'intent_idx': 0,  # restaurant_booking
  'intent_confidence': 0.89,
  'entities': [
    {'entity': 'NUMBER', 'value': 'due', 'start': 3, 'end': 4},
    {'entity': 'LOCATION', 'value': 'Roma', 'start': 6, 'end': 7},
    {'entity': 'DATE', 'value': 'domani', 'start': 7, 'end': 8}
  ]
}
```

## Troubleshooting

**Errore "Missing key(s) in state_dict"**: Il modello salvato ha una struttura vecchia. Rigenerare e ri-trainare.

**Errore "torchcrf not found"**: Installare `pip install pytorch-crf`

**Tag BIO non allineati**: Verificare che il tokenizer sia lo stesso usato per generare i dati

