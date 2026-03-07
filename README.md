# Cognitor Assistant

Assistente virtuale conversazionale con classificazione di intenti e riconoscimento di entità, basato su PyTorch, FastText e NER.

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
- **FastText**: Per word embeddings e tokenizzazione
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
│   ├── session_manager.py # Gestione sessioni
│   ├── dialogue_state_policy.py  # Policy dialogo
│   ├── rule_interpreter.py       # Runtime DSL
│   ├── slot_manager.py    # Gestione slot
│   ├── entity_manager.py # Gestione entità
│   └── operations/        # Operazioni custom
├── intellective/         # Modelli ML
│   ├── intent_classifier.py    # BiGRU + Attention
│   ├── dialogue_policy.py       # GRU per dialogo
│   ├── train_fast_text.py       # Training FastText
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
- fasttext
- fastapi
- uvicorn
- pandas
- scikit-learn
- nltk
- pyyaml
- python-jose
- torchcrf

### Download Pre-trained FastText (opzionale)

```bash
python scripts/download_pretrained_fasttext.py
```

## Procedura di Training

### Pipeline Completa

Esegui la pipeline completa che include:

1. Validazione del dataset
2. Generazione corpus FastText
3. Training FastText (word embeddings)
4. Generazione dataset NLU tokenizzato
5. Training Intent Classifier
6. Training Dialogue Policy

```bash
python -m pipeline
```

### Training Step-by-Step

```bash
# 1. Genera corpus FastText
python -m pipeline.intent_builder

# 2. Allena FastText (obbligatorio)
python -m intellective.train_fast_text

# 3. Allena Intent Classifier
python -m intellective.train_intent_classifier

# 4. Allena Dialogue Policy (opzionale)
python -m intellective.train_dialogue_policy
```

### Training Opzioni

Puoi saltare alcuni step:

```bash
# Skip training Intent Classifier
python -m pipeline --no-classifier

# Skip training Dialogue Policy
python -m pipeline --no-policy
```

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
knowledge/
├── intents/          # Esempi NLU (YAML)
├── rules/            # Mappatura intent→risposte (YAML)
├── responses/        # Template risposte (YAML)
└── conversations/    # Storie dialogo (YAML)
```

## Documentazione

La documentazione dettagliata è disponibile nella cartella `docs/`:
- `ARCHITETTURA.md`: Documentazione tecnica completa
- `IMPLEMENTAZIONE.md`: Guida all'implementazione
