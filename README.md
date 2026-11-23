# Cognitor Assistant

Assistente virtuale conversazionale con classificazione di intenti e riconoscimento di entità, basato su PyTorch, embedding di sotto-parola (FastText o skip-gram interno) e NER.

## Cos'è Cognitor

Cognitor è un assistente virtuale che utilizza tecniche di Natural Language Processing e Machine Learning per comprendere le intenzioni dell'utente e gestire conversazioni contestuali. Il sistema è composto da:

- **Intent Classifier**: Rete neurale BiGRU con Attention per la classificazione degli intenti
- **NER (Named Entity Recognition)**: Riconoscimento di entità nel testo usando tag BIO con CRF
- **Dialogue State Policy**: Gestione dello stato della conversazione con supporto per ML e fallback euristico
- **Sistema a Slot**: Raccolta di informazioni strutturate necessarie per completare azioni
- **Operations**: Azioni custom con auto-discovery

## Cosa è Basato

Il sistema si basa su tecnologie open source:

- **PyTorch**: Framework per il deep learning
- **SentencePiece** (default, `USE_FASTTEXT=false`): vocabolario di sotto-parole per l'embedding interno (skip-gram + negative sampling), senza dipendenza da FastText
- **FastText** (opzionale, `COGNITOR_USE_FASTTEXT=true`): word embeddings con n-grammi di carattere
- **PyTorch-CRF**: Conditional Random Field per sequence labeling
- **FastAPI**: API REST per l'interfaccia
- **YAML**: Formato per la configurazione dichiarativa

## Architettura

```
Cognitor Assistant
├── api/                    # API FastAPI
│   ├── auth.py            # Endpoint autenticazione JWT
│   └── chatbot.py         # Endpoint chatbot
├── agent/                 # Agent conversazionale
│   ├── agent.py           # Coordinator principale
│   ├── turn_processor.py  # Macchina a stati del turno (CLI + API)
│   ├── session_manager.py # Gestione e persistenza sessioni
│   ├── dialogue_state_policy.py  # Policy dialogo
│   ├── rule_interpreter.py       # Runtime DSL
│   ├── slot_manager.py    # Gestione slot (estrazione NER e validazione)
│   ├── keywords.py        # Comandi di annullamento/conferma e segnali di crisi
│   └── operations/        # Operazioni custom
├── intellective/         # Modelli ML
│   ├── intent_classifier.py    # BiGRU + Attention
│   ├── dialogue_policy.py       # GRU per dialogo
│   ├── train_fast_text.py       # Training embedding (FastText)
│   ├── train_own_embedder.py    # Training embedding alternativo (skip-gram + SentencePiece)
│   ├── train_intent_classifier.py
│   └── train_dialogue_policy.py
├── classes/              # Classi utility
├── pipeline/             # Pipeline training
├── knowledge/            # Knowledge base YAML
│   ├── intents/          # Definizioni intent
│   ├── rules/            # Mappatura intent→risposte
│   ├── responses/       # Template risposte
│   └── conversations/    # Storie conversazione
└── models/               # Modelli addestrati
```

### Componenti Principali

1. **Agent**: Coordina tutti i componenti, gestisce la pipeline di predizione
2. **Session Manager**: Mantiene stato e storico delle conversazioni
3. **Dialogue State Policy**: Predice la prossima azione del bot
4. **Rule Interpreter**: Runtime DSL per generare risposte
5. **Slot Manager**: Gestisce informazioni strutturate
6. **Operations Manager**: Esegue azioni custom

## Procedura di Installazione

### Requisiti

- Python 3.10+
- pip

### Installazione Dipendenze

```bash
pip install -r requirements.txt
```

Il file `requirements.txt` include:
- torch
- fasttext-wheel (opzionale, solo con `COGNITOR_USE_FASTTEXT=true`)
- sentencepiece
- fastapi
- uvicorn
- pandas
- scikit-learn
- nltk
- pyyaml
- python-jose
- torchcrf

### Download Pre-trained FastText (opzionale, solo con `USE_FASTTEXT=true`)

```bash
python scripts/download_pretrained_fasttext.py
```

## Procedura di Training

### Pipeline Completa

Esegui la pipeline completa che include:

1. Validazione del dataset
2. Generazione corpus per l'embedding
3. Training embedding (skip-gram interno di default, FastText se `USE_FASTTEXT`)
4. Generazione dataset NLU tokenizzato
5. Training Intent Classifier
6. Training Dialogue Policy

```bash
python -m pipeline
```

### Training Step-by-Step

```bash
# 1. Genera corpus per l'embedding
python -m pipeline.intent_builder

# 2. Allena l'embedding (uno dei due, a seconda di USE_FASTTEXT)
python -m intellective.train_own_embedder   # USE_FASTTEXT=false (default)
python -m intellective.train_fast_text      # USE_FASTTEXT=true

# 3. Allena Intent Classifier
python -m intellective.train_intent_classifier

# 4. Allena Dialogue Policy (opzionale)
python -m intellective.train_dialogue_policy
```

### Training Opzioni

`python -m pipeline` esegue sempre tutti gli step. Per saltare il training del classifier o della policy chiama la funzione da Python:

```python
from pipeline import run_full_pipeline

run_full_pipeline(train_classifier=False)  # salta l'Intent Classifier
run_full_pipeline(train_policy=False)      # salta la Dialogue Policy
```

## Esportare il modello NLU in ONNX

Per usare Intent Classifier e NER fuori da questo progetto (senza PyTorch) li si esporta in **un unico file ONNX**:

```bash
python -m intellective.export_onnx                              # scrive models/cognitor-intent.onnx
python -m intellective.export_onnx --output <percorso>/cognitor-intent.onnx
```

Il grafo ha input `input_ids` `[batch, seq]` (int64) e due output: `probs` `[batch, n_intents]` (softmax degli
intenti) e `ner_tags` `[batch, seq]` (int64, un tag per token: la decodifica CRF/Viterbi è dentro il grafo).
Nei metadati (`custom_metadata_map`) c'è tutto il resto, perché modello e tokenizer vanno sempre insieme:

| Chiave | Contenuto |
|---|---|
| `cognitor.manifest` | JSON: versione, nomi di input/output, tokenizer (`lowercase`), mappa indice → intento (`intents`), mappa indice → tag NER (`ner_tags`), `min_confidence` |
| `cognitor.tokenizer` | `sentencepiece.model` in base64 |

Le frasi vanno passate una alla volta (senza padding): il modello non usa maschere.

```python
session = onnxruntime.InferenceSession("cognitor-intent.onnx")
meta = session.get_modelmeta().custom_metadata_map
manifest = json.loads(meta["cognitor.manifest"])
sp = SentencePieceProcessor(model_proto=base64.b64decode(meta["cognitor.tokenizer"]))
ids = sp.encode(testo.lower(), out_type=int)
probs, tag_ids = session.run(["probs", "ner_tags"], {"input_ids": [ids]})
intent = manifest["intents"][str(probs[0].argmax())]
tags = [manifest["ner_tags"][str(t)] for t in tag_ids[0]]      # "O", "B-DATE", "I-DATE", ...
```

**Ricostruire le entità** (stessa logica di `IntentClassifier._extract_entities`): i token sono i pezzi
SentencePiece con il marcatore `▁` rimosso (`sp.id_to_piece(i).replace("▁", "")`). Un tag `B-X` apre un'entità di tipo
X, un `I-X` dello stesso tipo la prosegue, ogni altro tag (o un `I-` di tipo diverso) la chiude. Gli offset di
carattere si ottengono cercando i token in sequenza nel testo minuscolo, e il valore è la sottostringa del testo
minuscolo tra l'inizio del primo token e la fine dell'ultimo. Poi inizio e fine si estendono ai confini di parola
(finché il carattere accanto è alfanumerico): una parola spezzata in più pezzi, come "identità" → `i` + `dentità`, può avere
il tag solo sui primi pezzi e l'entità resterebbe a metà (`carta d'i`).

L'export confronta il file con il modello PyTorch su alcune frasi (leggendo il tokenizer dal file stesso) e
fallisce se le probabilità o i tag divergono. Richiede `onnx` e `onnxruntime` e il tokenizer SentencePiece
(`USE_FASTTEXT=false`). Test: `pytest tests/test_export_onnx.py`.

## Procedura di Avvio

### Avvio API Server

```bash
uvicorn main:app --reload
```

Il server sarà disponibile su `http://localhost:8000`

### Avvio Agent Interattivo

```bash
python -m agent.agent
```

Entra in modalità conversazione testuale.

## Utilizzo

### API Endpoints

#### Autenticazione

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=admin123"
```

Risposta:
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

#### Inviare un messaggio

```bash
curl -X POST http://localhost:8000/chatbot/message \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Ciao!"}'
```

#### Contratto chat v1 (token + messaggio)

Un client che parla questo contratto (es. un plugin CMS) funziona con o senza un gateway davanti: bastano Token URL e Chat URL.

```bash
# token (OAuth2 client_credentials, client da COGNITOR_CLIENTS="id:secret,...")
curl -X POST http://localhost:8000/api/oauth2/token \
  -u site-a:secret -d "grant_type=client_credentials&scope=CHAT"
# -> {"access_token": "...", "token_type": "Bearer", "expires_in": 1800}

# messaggio
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"session_id": "abc", "text": "Ciao!"}'
# -> [{"session_id": "abc", "text": "...", "link": null, "options": null, "waiting_for_slot": false}]
```

`POST /api/v1/chat/reset` con `{"session_id": "abc"}` azzera la conversazione. Le sessioni sono isolate per `client_id`.

#### Health Check

```bash
curl http://localhost:8000/health
```

### Modalità Interattiva

```bash
python -m agent.agent
```

Esempio di interazione:
```
COGNITOR AGENT - Interfaccia Testuale
Session ID: xxx
Sessioni attive: 1
Scrivi un messaggio (o 'esci' per terminare)

Tu: Ciao

Intent: greeting (95.2%)
Entita: nessuna

COGNITOR: Ciao! Come posso aiutarti?
```

### Configurazione

Modifica `config.py` per:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOPING_ACTIVE = False  # Data augmentation
MIN_INTENT_CONFIDENCE = 0.20  # Soglia minima intent
```

## Implementazione

### Aggiungere un Nuovo Intent

1. Definisci gli esempi in `knowledge/intents/<nome>.yaml`:
```yaml
nlu:
  intents:
  - intent: my_intent
    examples:
    - esempio uno
    - esempio due
```

2. Definisci la rule in `knowledge/rules/<nome>.yaml`:
```yaml
rules:
  my_intent:
    default: my_intent_response
```

3. Definisci la risposta in `knowledge/responses/<nome>.yaml`:
```yaml
responses:
  my_intent_response:
    - "Risposta per il nuovo intent"
```

4. Riaddestra: `python -m pipeline`

### Slot

Gli slot permettono di raccogliere informazioni:

```yaml
rules:
  prenota:
    slots:
      LOCATION:
        required: true
    cases:
      Roma: prenota_roma_response
      Milano: prenota_milano_response
    fallback: prenota_fallback
    wait: prenota_wait
```

### Operazioni Custom

Crea un'operazione in `agent/operations/<nome>.py`:

```python
from agent.operations.base import Operation

class MyOperation(Operation):
    @property
    def name(self) -> str:
        return "my_operation"

    def execute(self, intent_name: str, slots: dict = None) -> dict:
        return {
            "response": "Risposta",
            "slots": {},
            "metadata": {}
        }
```

Richiamala con `__<nome>` nella rule.

## Testing

```bash
# Test conversazione interattiva
python -m agent.agent

# Test unitari
pytest tests/

# Test specifico
pytest tests/test_dialogue_policy.py -v
```

## Struttura Knowledge Base

```
knowledge/            # Knowledge generica (chiacchiere di base, API esterne)
├── intents/          # Esempi NLU (YAML)
├── rules/            # Mappatura intent→risposte (YAML)
├── responses/        # Template risposte (YAML)
└── conversations/    # Storie dialogo (YAML)
training_data/        # Dati aggiuntivi (stessa struttura + operations/)
├── intents/ rules/ responses/ conversations/
└── operations/       # Operations aggiuntive (auto-discovery)
```

## Documentazione

La documentazione dettagliata è disponibile nella cartella `docs/`:
- `ARCHITETTURA.md`: Documentazione tecnica completa
- `IMPLEMENTAZIONE.md`: Guida all'implementazione
