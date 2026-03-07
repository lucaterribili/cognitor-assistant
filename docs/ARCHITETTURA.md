# Documentazione Tecnica - Cognitor Assistant

## Indice

1. [Panoramica del Sistema](#panoramica-del-sistema)
2. [Architettura Generale](#architettura-generale)
3. [Componenti Principali](#componenti-principali)
4. [Modelli ML](#modelli-ml)
5. [Pipeline di Training](#pipeline-di-training)
6. [Knowledge Base](#knowledge-base)
7. [Gestione del Dialogo](#gestione-del-dialogo)
8. [Flow di Esecuzione](#flow-di-esecuzione)

---

## Panoramica del Sistema

Cognitor Assistant è un assistente virtuale conversazionale basato su tecniche di Natural Language Understanding (NLU). Il sistema combina multiple tecnologie di Machine Learning per comprendere l'intenzione dell'utente e gestire conversazioni contestuali.

### Caratteristiche Principali

- **Classificazione Intent**: Identifica l'intenzione dell'utente tramite una rete neurale BiGRU con Attention
- **Riconoscimento Entità (NER)**: Estrae entità命名实体 dal testo usando tag BIO con CRF
- **Gestione Stato Dialogo**: Mantiene il contesto della conversazione e predice le azioni del bot
- **Sistema a Slot**: Gestisce informazioni strutturate necessarie per completare azioni
- **Operazioni Personalizzate**: Supporta azioni custom con auto-discovery

---

## Architettura Generale

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
│  ┌─────────────────┐           ┌─────────────────────────────┐  │
│  │   /auth/token   │           │      /chatbot/message      │  │
│  └─────────────────┘           └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent (agent/agent.py)                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     Prediction Pipeline                     │ │
│  │  1. Predict Intent + NER                                    │ │
│  │  2. Extract Entities                                       │ │
│  │  3. Update Session Slots                                   │ │
│  │  4. Dialogue State Policy → Next Action                  │ │
│  │  5. Generate Response                                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────┐      ┌─────────────────────┐      ┌─────────────────┐
│  Session      │      │  Dialogue State     │      │  Rule Interpreter│
│  Manager      │      │  Policy             │      │  (DSL Runtime)  │
└───────────────┘      └─────────────────────┘      └─────────────────┘
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐      ┌─────────────────────┐      ┌─────────────────┐
│  Entity       │      │  DialoguePolicy (ML) │      │  Responses      │
│  Manager      │      │  or Heuristic        │      │  Templates      │
└───────────────┘      └─────────────────────┘      └─────────────────┘
                                                       ┌─────────────────┐
                                                       │  Operations     │
                                                       │  Manager        │
                                                       └─────────────────┘
```

---

## Componenti Principali

### 1. Agent (`agent/agent.py`)

È il componente centrale che coordina tutti gli altri componenti.

**Responsabilità:**
- Caricamento dei modelli ML
- Predizione di intent ed entità
- Coordinamento della generazione delle risposte
- Gestione della pipeline di conversazione

**Metodi principali:**
- `load_models()`: Carica il modello Intent Classifier
- `load_knowledge()`: Carica rules, responses e conversations
- `predict(text)`: Predice intent ed entità per un testo
- `get_response(intent_name, slots, history)`: Genera la risposta

### 2. Session Manager (`agent/session_manager.py`)

Gestisce le sessioni di conversazione.

**Responsabilità:**
- Creazione e gestione di sessioni utente
- Mantenimento dello storico della conversazione
- Gestione del contesto (slot) per ogni sessione
- Timeout e cleanup delle sessioni inattive

**Struttura dati:**
```python
@dataclass
class ConversationSession:
    session_id: str
    created_at: datetime
    updated_at: datetime
    history: list[dict]  # [{role, content, intent, entities, timestamp}]
    context: dict        # Slot della conversazione
    metadata: dict
    waiting_for_slot: dict | None  # Se il bot aspetta uno slot
    agent_mode: str     # "predictable" o "inputable"
```

### 3. Entity Manager (`agent/entity_manager.py`)

Gestisce le entità estratte dalla conversazione.

**Responsabilità:**
- Storage delle entità riconosciute
- Lookup per nome o etichetta
- Gestione della confidence delle entità

### 4. Model Loader (`agent/model_loader.py`)

Carica i modelli ML e la knowledge base.

**Componenti:**
- `ModelLoader`: Carica il modello Intent Classifier e il vocabolario
- `KnowledgeLoader`: Carica rules, responses e conversations

### 5. Dialogue State Policy (`agent/dialogue_state_policy.py`)

Gestisce lo stato del dialogo e predice la prossima azione del bot.

**Due modalità:**
1. **ML Mode**: Usa un modello GRU addestrato (DialoguePolicy)
2. **Heuristic Mode**: Usa longest-suffix match sulle storie YAML (fallback)

**Ispirato a Rasa TED Policy:**
- Addestrato su "storie" di conversazione
- Rappresenta lo stato come sequenza di intent utente recenti
- Predice la prossima azione tramite classificazione

### 6. Rule Interpreter (`agent/rule_interpreter.py`)

Runtime DSL che interpreta le regole dichiarative definite in YAML.

**Responsabilità:**
- Interpretazione delle rules per generare risposte
- Gestione degli slot (required, optional)
- Matching dei casi (cases)
- Sostituzione dei placeholder `{SLOT_NAME}` nelle risposte
- Esecuzione delle operazioni (prefisso `__`)

**Logica di selezione risposta:**
1. Se slot richiesto mancante → rispondi con `wait`
2. Se slot fornito, cerca nei `cases` → risposta specifica
3. Se non trovato → `fallback`
4. Altrimenti → `default`

### 7. Operations Manager (`agent/operations/manager.py`)

Gestisce le operazioni personalizzate del bot.

**Caratteristiche:**
- Auto-discovery delle operations nella cartella `operations/`
- Supporta classi che ereditano da `Operation`
- Supporta funzioni con pattern `action_*` o `*_action`
- Espone i manager (Session, Entity) alle operations

**Operazioni disponibili:**
- `calculate`: Esegue calcoli matematici
- `location_query`: Query geografiche
- `geocoding`: Conversione indirizzi/coordinate

### 8. Slot Manager (`agent/slot_manager.py`)

Gestisce gli slot in modo data-driven basato sulle rules.

**Responsabilità:**
- Validazione dei valori degli slot
- Aggiornamento del contesto della sessione
- Gestione delle entità estratte dal NER

---

## Modelli ML

### 1. FastText (Word Embeddings)

**Scopo**: Generare rappresentazioni vettoriali delle parole

**Configurazione:**
- Modello: Skip-gram
- Dimensione: 300
- Epoch: 25
- Learning rate: 0.1
- minCount: 1
- wordNgrams: 2
- minn: 2, maxn: 5 (subwords)

**Output:**
- `models/fasttext_model.bin`: Modello completo
- `.cognitor/wordvectors.vec`: Matrice embeddings (word2vec format)
- `.cognitor/vocab.json`: Vocabolario

### 2. Intent Classifier (PyTorch BiGRU + Attention)

**Architettura:**
```
Input Text
    │
    ▼
Embedding Layer (FastText, freeze=True) ─── 300 dim
    │
    ▼
BiGRU (hidden_dim=256, bidirectional)
    │
    ├──────────────────┬───────────────────┐
    ▼                  ▼                   ▼
Attention Layer    NER Branch         Intent Branch
    │                  │                   │
    ▼                  ▼                   ▼
Dropout(0.3)      Linear+CRF         Dropout(0.3) → Linear
    │                  │                   │
    ▼                  ▼                   ▼
Intent logits     NER tags            Intent predictions
```

**Caratteristiche:**
- Embedding inizializzato con FastText (congelato)
- BiGRU bidirezionale per catturare contesto
- Attention per focalizzarsi su parti rilevanti
- CRF (Conditional Random Field) per NER sequence labeling
- Multi-task: Intent Classification + NER

### 3. Dialogue Policy (PyTorch GRU)

**Scopo**: Predire la prossima azione del bot dato lo storico

**Architettura:**
```
Context Intents [T] ──→ Embedding ──┐
Context Actions [T] ──→ Embedding ──┼──→ Sum ──→ GRU ──→ Dropout
                                    │         │
                                    └─────────┼───→ Concatenate ──→ Linear → Action logits
                                                │
Current Intent ──→ Embedding ──────────────────┘
```

**Iperparametri:**
- Embed dim: 64
- Hidden dim: 128
- Dropout: 0.3

---

## Pipeline di Training

### Pipeline Completa

```bash
python -m pipeline
```

### Step Manuali

```bash
# 1. Genera corpus FastText
python -m pipeline.intent_builder

# 2. Allena FastText
python -m intellective.train_fast_text

# 3. Allena Intent Classifier
python -m intellective.train_intent_classifier

# 4. Allena Dialogue Policy (opzionale)
python -m intellective.train_dialogue_policy
```

### Dettagli Pipeline

**Step 0: Validazione Dataset**
- Valida intents ed entità NER
- Controlla consistenza dei file YAML

**Step 1: Generazione Corpus FastText**
- Legge intents da `knowledge/intents/*.yaml`
- Genera `.cognitor/fast-text.txt` (raw text)

**Step 1.5: Merge Knowledge**
- Unisce rules da `knowledge/rules/` e `training_data/rules/`
- Unisce responses da `knowledge/responses/` e `training_data/responses/`
- Unisce conversations da `knowledge/conversations/` e `training_data/conversations/`
- Output in `.cognitor/`

**Step 2: Training FastText**
- Train unsupervised su corpus
- Estrae word vectors

**Step 3: Generazione Dataset NLU**
- Tokenizza usando FastText
- Genera tag NER BIO
- Salva `.cognitor/tokenized_data.npy`

**Step 4: Training Intent Classifier**
- BiGRU + Attention + CRF
- Early stopping con patience=10
- Salva `models/intent_model_fast.pth`

**Step 5: Training Dialogue Policy**
- GRU encoder per sequence di intent
- Addestrato su conversazioni YAML
- Salva `models/dialogue_policy.pth`

---

## Knowledge Base

### Struttura Directory

```
knowledge/
├── intents/          # Definizioni NLU (YAML)
│   ├── base.yaml
│   ├── command.yaml
│   └── ...
├── rules/            # Mappatura intent → risposte (YAML)
│   ├── base.yaml
│   ├── command.yaml
│   └── ...
├── responses/        # Template risposte (YAML)
│   ├── base.yaml
│   ├── command.yaml
│   └── ...
└── conversations/    # Storie di conversazione (YAML)
    └── base.yaml
```

### Formato Intents (`knowledge/intents/*.yaml`)

```yaml
nlu:
  intents:
  - intent: greeting
    examples:
    - ciao
    - salve
    - hey
    - buongiorno
  - intent: book_restaurant
    examples:
    - voglio prenotare un ristorante
    - prenota un tavolo
```

### Formato Rules (`knowledge/rules/*.yaml`)

```yaml
rules:
  # Intent semplice
  greeting:
    default: greeting_response

  # Intent con slot
  book_restaurant:
    slots:
      LOCATION:
        required: true
        type: string
      DATE:
        required: false
    cases:
      Roma: book_restaurant_roma_response
      Milano: book_restaurant_milano_response
    fallback: book_restaurant_fallback
    wait: book_restaurant_wait
```

### Formato Responses (`knowledge/responses/*.yaml`)

```yaml
responses:
  greeting_response:
    - "Ciao! Come posso aiutarti?"
    - "Hey! Sono qui per te."

  book_restaurant_roma_response:
    - "Ottima scelta! A Roma ci sono ristoranti eccellenti. Quante persone siete?"
```

Placeholder supportati:
- `{SLOT_NAME}`: Sostituito con il valore dello slot
- `$timestamp`: Data/ora corrente
- `{ALTRO_SLOT}`: Riferimento ad altro slot

### Formato Conversations (`knowledge/conversations/*.yaml`)

```yaml
conversations:
  restaurant_booking_flow:
    description: "Flusso per prenotazione ristorante"
    steps:
      - user: greeting
        bot: greeting_response
      - user: book_restaurant
        bot: restaurant_booking_response
      - user: thank_you
        bot: thank_you_response
      - user: farewell
        bot: farewell_response
```

---

## Gestione del Dialogo

### Flow di Predizione

```
1. Utente invia messaggio
         │
         ▼
2. Agent.predict() → Intent + Entities
         │
         ▼
3. SlotManager.update_session() → Estrae slot da entità
         │
         ▼
4. DialogueStatePolicy.predict_next_action()
   ├─ ML Mode: Usa DialoguePolicy (se disponibile)
   └─ Heuristic Mode: Longest-suffix match
         │
         ▼
5. Se azione trovata → Ritorna response dalla TED policy
   Altrimenti → RuleInterpreter.handle_intent()
         │
         ├─ Slot required mancante → wait response
         ├─ Slot fornito → cerca nei cases
         ├─ Valore non trovato → fallback
         └─ Default → default response
         │
         ▼
6. Applica bot_slots (set_slots dalla rule)
         │
         ▼
7. Risposta all'utente
```

### Modalità di Conversazione

**Predictable Mode (default):**
- Bot risponde normalmente agli intent rilevati

**Inputable Mode:**
- Bot aspetta input per uno slot specifico
- Utente può annullare con `#exit` o `#annulla`

### Slot Management

**Slot Types:**
- `required`: Obbligatorio, il bot chiede se mancante
- `optional`: Opzionale, il bot procede senza

**Bot Slots:**
- Impostati automaticamente dalla rule (`set_slots`)
- Estratti inline dalla response

---

## Flow di Esecuzione

### Avvio API Server

```bash
uvicorn main:app --reload
```

**Endpoints:**
- `POST /auth/token`: Autenticazione JWT
- `POST /chatbot/message`: Invia messaggio al chatbot
- `GET /health`: Health check

### Avvio Agent Interattivo

```bash
python -m agent.agent
```

Entra in un loop di conversazione testuale dove è possibile:
- Inviare messaggi
- Vedere intent predetti e confidenza
- Vedere entità estratte
- Interagire con slot richiesti

### Configurazione (`config.py`)

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOPING_ACTIVE = False  # Data augmentation
MIN_INTENT_CONFIDENCE = 0.20  # Soglia minima per accettare intent
```

---

## Dipendenze Principali

- **torch**: PyTorch per modelli neurali
- **fasttext**: Word embeddings e tokenizzazione
- **spacy**: NLP (opzionale)
- **fastapi**: API REST
- **uvicorn**: Server ASGI
- **pandas**: Manipolazione dati
- **scikit-learn**: Utilità ML
- **nltk**: NLP utilities
- **python-jose**: JWT authentication
- **pyyaml**: Parsing YAML
- **torchcrf**: CRF layer per NER
