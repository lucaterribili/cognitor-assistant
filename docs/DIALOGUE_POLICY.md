# Documentazione - Dialogue Policy (Policy Conversazione)

## Indice

1. [Panoramica](#panoramica)
2. [Architettura](#architettura)
3. [Come Viene Addestrata](#come-viene-addestrata)
4. [Come Interviene nella Conversazione](#come-interviene-nella-conversazione)
5. [File e Percorsi Rilevanti](#file-e-percorsi-rilevanti)

---

## Panoramica

La **Dialogue Policy** è il componente di Cognitor Assistant responsabile di predire la prossima azione che il bot deve eseguire dato lo stato corrente della conversazione.

Ispirata alla **TED (Transformer Embedding Dialogue) Policy** di Rasa, impara dai *flussi di conversazione* (file YAML) e sfrutta la storia degli intent utente recenti per contestualizzare la risposta più appropriata.

Il componente opera a livello di **Dialogue Management (DM)**, un livello distinto rispetto al NLU (che si occupa di classificare l'intent e riconoscere le entità). Il suo compito è rispondere alla domanda:

> *"Dato quello che l'utente ha detto (e quello che si è detto prima), qual è la prossima mossa del bot?"*

---

## Architettura

Il modello `DialoguePolicy` è implementato in `intellective/dialogue_policy.py` come classe PyTorch (`nn.Module`).

### Schema dell'Architettura

```
Context Intents [B, T] ──→ IntentEmbedding [E] ──┐
                                                   ├──→ Sum [B, T, E] ──→ GRU [B, H] ──→ Dropout
Context Actions [B, T] ──→ ActionEmbedding [E] ──┘                  │
                                                                      │
                                                   ┌──────────────────┘
                                                   │
Current Intent  [B]    ──→ IntentEmbedding [E] ───┴──→ Concat [B, H+E] ──→ Linear ──→ Action Logits [B, A]
```

**Legenda:**
- `B` = batch size
- `T` = lunghezza della sequenza di contesto (max `HISTORY_WINDOW = 5`)
- `E` = dimensione degli embedding (`EMBED_DIM = 64`)
- `H` = dimensione dello stato nascosto del GRU (`HIDDEN_DIM = 128`)
- `A` = numero di azioni distinte

### Componenti del Modello

| Componente              | Tipo                      | Descrizione                                                                  |
|-------------------------|---------------------------|------------------------------------------------------------------------------|
| `intent_embedding`      | `nn.Embedding(N_I+1, E)`  | Embedding condiviso per intent nel contesto e intent corrente (pad_idx=0)    |
| `action_embedding`      | `nn.Embedding(N_A+1, E)`  | Embedding separato per le azioni bot nel contesto (pad_idx=0)                |
| `gru`                   | `nn.GRU(E, H)`            | Encoder GRU unidirezionale che processa la sequenza di contesto storica      |
| `dropout`               | `nn.Dropout(0.3)`         | Dropout applicato all'output dello stato nascosto GRU                        |
| `fc`                    | `nn.Linear(H+E, N_A)`     | Classificatore finale che combina stato GRU e embedding dell'intent corrente |

### Iperparametri

Gli iperparametri condivisi tra training e inferenza sono definiti come costanti in `intellective/dialogue_policy.py`:

```python
DIALOGUE_POLICY_EMBED_DIM: int = 64     # Dimensione embedding
DIALOGUE_POLICY_HIDDEN_DIM: int = 128   # Dimensione hidden state GRU
DIALOGUE_POLICY_DROPOUT: float = 0.3    # Probabilità di dropout
```

### Forward Pass

```
1. context_emb = IntentEmbedding(context_intents) + ActionEmbedding(context_actions)
                 ─── somma element-wise degli embedding di intent e azione ad ogni passo temporale

2. _, hidden = GRU(context_emb)
               ─── hidden: [1, B, H] → squeeze → [B, H]

3. hidden = Dropout(hidden)

4. curr_emb = IntentEmbedding(current_intent)     ─── [B, E]

5. combined = Concat([hidden, curr_emb], dim=-1)   ─── [B, H+E]

6. logits = Linear(combined)                       ─── [B, A]
```

### Predizione (Inference)

Il metodo `predict()` calcola i logits, applica una `softmax` e restituisce l'azione con la confidenza più alta:

```python
probs = softmax(logits, dim=-1)
action_idx = argmax(probs)           # 0-indexed
confidence = probs[0, action_idx]    # float in [0.0, 1.0]
```

---

## Come Viene Addestrata

Il training della Dialogue Policy è implementato in `intellective/train_dialogue_policy.py` ed è eseguito come **Step 5** della pipeline di training.

### Flusso di Training

```
[1] Carica conversations da .cognitor/conversations.yaml
         │
         ▼
[2] Costruisce dizionari intent→id e action→id (1-indexed, 0 = padding)
         │
         ▼
[3] Genera campioni di training dalle storie (history window = 5)
         │
         ▼
[4] Crea DialoguePolicyDataset e DataLoader (batch_size ≤ 4, shuffle=True)
         │
         ▼
[5] Inizializza il modello DialoguePolicy con gli iperparametri condivisi
         │
         ▼
[6] Training loop con early stopping (epochs=150, lr=0.001, patience=20)
         │
         ▼
[7] Salva il modello e i dizionari
```

### Formato dei Dati di Training

Le storie di conversazione sono definite in file YAML nella cartella `knowledge/conversations/` (e `training_data/conversations/`):

```yaml
conversations:
  flusso_prenotazione:
    description: "Flusso di prenotazione ristorante"
    steps:
      - user: greeting
        bot: greeting_response
      - user: prenota_ristorante
        bot: prenota_ristorante_response
      - user: conferma_prenotazione
        bot: conferma_prenotazione_response
      - user: farewell
        bot: farewell_response
```

### Generazione dei Campioni

Per ogni passo `i` di ogni storia, viene generata una **quadrupla**:

```
(context_intent_ids, context_action_ids, current_intent_id, target_action_id)
```

- `context_intent_ids`: ID degli ultimi `HISTORY_WINDOW` intent utente precedenti al passo `i`
- `context_action_ids`: ID delle ultime `HISTORY_WINDOW` azioni bot precedenti al passo `i`
- `current_intent_id`: ID dell'intent utente al passo `i`
- `target_action_id`: ID dell'azione bot al passo `i` (1-indexed, da predire)

**Esempio** con `HISTORY_WINDOW = 5` sul flusso sopra:

| `i` | `context_intents`              | `context_actions`                  | `current_intent`        | `target_action`              |
|-----|--------------------------------|------------------------------------|-------------------------|------------------------------|
| 0   | `[]`                           | `[]`                               | `greeting`              | `greeting_response`          |
| 1   | `[greeting]`                   | `[greeting_response]`              | `prenota_ristorante`    | `prenota_ristorante_response`|
| 2   | `[greeting, prenota_ristorante]`| `[greeting_response, prenota_ristorante_response]` | `conferma_prenotazione` | `conferma_prenotazione_response` |
| 3   | `[greeting, prenota_ristorante, conferma_prenotazione]` | `[...]` | `farewell` | `farewell_response` |

### Padding e Batching

Le sequenze di contesto hanno lunghezze variabili. La funzione `collate_dialogue_fn` usa `pad_sequence` per allinearle a lunghezza uniforme all'interno del batch (pad_idx = 0).

Contesti vuoti (primo passo di ogni storia) vengono sostituiti con un tensore di zeri di lunghezza 1.

### Training Loop

```python
optimizer  = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion  = CrossEntropyLoss()
scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

for epoch in range(150):
    # Forward + backward + step
    # Calcola avg_loss sull'epoca
    # ReduceLROnPlateau aggiorna il lr se il loss non migliora per 3 epoche
    # Early stopping: se avg_loss non migliora per 20 epoche → stop
    # Ripristina il miglior stato del modello
```

| Parametro          | Valore |
|--------------------|--------|
| Ottimizzatore      | AdamW  |
| Learning rate      | 0.001  |
| Weight decay       | 1e-4   |
| Loss               | CrossEntropyLoss |
| Scheduler          | ReduceLROnPlateau (factor=0.5, patience=3) |
| Epoche massime     | 150    |
| Early stopping     | patience=20 |
| Batch size         | min(4, len(samples)) |

### Artefatti Prodotti

Dopo il training, vengono salvati tre file:

| File                                  | Contenuto                                  |
|---------------------------------------|--------------------------------------------|
| `models/dialogue_policy.pth`          | Pesi del modello PyTorch                   |
| `.cognitor/dialogue_intent_dict.json` | Dizionario `{ intent_name: id }` (1-indexed) |
| `.cognitor/dialogue_action_dict.json` | Dizionario `{ action_name: id }` (1-indexed) |

### Avviare il Training

```bash
# Intera pipeline (Step 0–5)
python -m pipeline

# Solo Dialogue Policy
python -m intellective.train_dialogue_policy
```

---

## Come Interviene nella Conversazione

Il componente `DialogueStatePolicy` (`agent/dialogue_state_policy.py`) è il punto di accesso alla policy durante l'inferenza. Viene richiamato dall'`Agent` a ogni turno di conversazione.

### Posizione nel Pipeline di Predizione

```
1. Utente invia messaggio
         │
         ▼
2. Agent.predict()       ─── Intent + Entities dal modello NLU (BiGRU + Attention + CRF)
         │
         ▼
3. SlotManager.update_session()   ─── Estrae slot dalle entità
         │
         ▼
4. DialogueStatePolicy.predict_next_action(current_intent, history)
   ├─── [Modalità ML]        → DialoguePolicy (GRU)
   └─── [Modalità Euristica] → Longest-suffix match sulle storie YAML
         │
         ▼
5. Se azione trovata → Risposta dalla Dialogue Policy
   Altrimenti        → RuleInterpreter.handle_intent()
         │
         ▼
6. Risposta finale all'utente
```

### Due Modalità Operative

#### Modalità ML (prioritaria)

Attivata se `models/dialogue_policy.pth`, `dialogue_intent_dict.json` e `dialogue_action_dict.json` sono presenti.

**Flusso:**

```
1. Estrai la sequenza degli intent utente precedenti dallo history (max 5)
2. Converti gli intent in ID tramite il dizionario
3. Costruisci i tensori di input (batch size = 1):
     context_intent_tensor: [1, T]  ─── sequenza di ID intent contesto
     context_action_tensor: [1, T]  ─── zeri (placeholder, TODO: tracciare azioni bot)
     current_tensor:        [1]     ─── ID intent corrente
4. Chiama model.predict(context_intent_tensor, context_action_tensor, current_tensor)
     ─── Restituisce (action_idx 0-indexed, confidence)
5. Converti action_idx in nome azione tramite dizionario inverso
     ─── action_dict_inv[action_idx + 1]  (riconversione a 1-indexed)
6. Restituisce {'action': action_name, 'confidence': confidence}
```

#### Modalità Euristica (fallback)

Usata quando il modello ML non è disponibile (es. prima esecuzione della pipeline).

**Algoritmo: Longest-Suffix Match**

```
1. Estrai la sequenza degli intent utente recenti dallo history (contesto corrente)
2. Per ogni transizione nelle storie YAML con user_intent == current_intent:
     - Calcola lo score di corrispondenza del suffisso tra contesto corrente e contesto della storia:
         score = max_suffix_match_length / len(story_context)
         score = 0.5 se story_context è vuoto (primo passo della storia)
3. Seleziona la transizione con lo score più alto
4. Restituisce {'action': best_action, 'confidence': best_score}
```

### Parametri di Runtime

| Parametro         | Valore | Descrizione                                                  |
|-------------------|--------|--------------------------------------------------------------|
| `HISTORY_WINDOW`  | 5      | Numero massimo di turni utente recenti considerati come contesto |
| `MIN_CONFIDENCE`  | 0.0    | Soglia minima di confidenza per applicare la predizione      |

### Storico della Conversazione

La policy utilizza il campo `history` della sessione, che è una lista di messaggi nel formato:

```python
[
    {
        'role': 'user',
        'content': 'Ciao, voglio prenotare un tavolo',
        'intent': 'prenota_ristorante',
        'entities': [...],
        'timestamp': '...'
    },
    {
        'role': 'bot',
        'content': 'Con piacere! Per quante persone?',
        ...
    },
    ...
]
```

La policy estrae solo i messaggi con `role == 'user'` e campo `intent` valorizzato per costruire la sequenza di contesto.

### Esempio di Predizione ML

Data una conversazione:

```
[utente] "ciao"          → intent: greeting
[bot]    "Ciao! ..."
[utente] "prenota tavolo" → intent: prenota_ristorante   ← turno corrente
```

La policy:
1. Estrae `context_intents = [greeting]`
2. Converte in ID, es. `context_intent_ids = [3]`
3. `current_intent = prenota_ristorante` → ID `7`
4. Costruisce `context_intent_tensor = [[3]]` (shape `[1, 1]`) e lo passa al GRU
5. Il GRU produce uno stato nascosto che cattura la transizione `greeting → prenota_ristorante`
6. Il classificatore predice l'azione più probabile, es. `prenota_ristorante_response` con confidenza `0.92`

### Gestione del Caso Senza Modello (Primo Avvio)

Se il modello non è ancora stato addestrato, la `DialogueStatePolicy` avverte e utilizza automaticamente la modalità euristica:

```
WARNING Dialogue Policy ML non disponibile: ...
[TED] Modalità: EURISTICA | intent='prenota_ristorante' | transizioni_disponibili=12
```

### Integrazione con il RuleInterpreter

L'output della policy (`{'action': '<response_key>', 'confidence': 0.92}`) viene usato dall'`Agent` per recuperare direttamente il template di risposta. Se la policy non produce un'azione valida (es. intent non in dizionario, confidenza insufficiente), il controllo passa al `RuleInterpreter` che elabora la risposta tramite le rules YAML.

---

## File e Percorsi Rilevanti

| File                                          | Ruolo                                                      |
|-----------------------------------------------|------------------------------------------------------------|
| `intellective/dialogue_policy.py`             | Definizione del modello `DialoguePolicy` (architettura ML) |
| `intellective/train_dialogue_policy.py`       | Logica di training: dati, dataset, loop, salvataggio       |
| `agent/dialogue_state_policy.py`              | Utilizzo del modello a runtime durante la conversazione    |
| `knowledge/conversations/*.yaml`              | Storie di conversazione usate per il training              |
| `training_data/conversations/*.yaml`          | Storie aggiuntive (mergiato nella pipeline)                |
| `.cognitor/conversations.yaml`                | File conversations mergiato (input del training)           |
| `models/dialogue_policy.pth`                  | Pesi del modello addestrato                                |
| `.cognitor/dialogue_intent_dict.json`         | Dizionario intent→id                                       |
| `.cognitor/dialogue_action_dict.json`         | Dizionario action→id                                       |
