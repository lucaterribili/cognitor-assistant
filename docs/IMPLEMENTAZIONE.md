# Guida all'Implementazione - Cognitor Assistant

## Indice

1. [Aggiungere un Nuovo Intent](#aggiungere-un-nuovo-intent)
2. [Aggiungere Slot agli Intent](#aggiungere-slot-agli-intent)
3. [Creare Operazioni Custom](#creare-operazioni-custom)
4. [Definire Nuove Entità NER](#definire-nuove-entità-ner)
5. [Creare Flussi di Conversazione](#creare-flussi-di-conversazione)
6. [Estendere il Sistema](#estendere-il-sistema)

---

## Aggiungere un Nuovo Intent

### Step 1: Definire gli esempi in `knowledge/intents/`

Crea o modifica un file in `knowledge/intents/`:

```yaml
# knowledge/intents/my_intents.yaml
nlu:
  intents:
  - intent: my_new_intent
    examples:
    - esempio uno
    - esempio due
    - altra frase di esempio
```

### Step 2: Definire la rule in `knowledge/rules/`

Crea o modifica un file in `knowledge/rules/`:

```yaml
# knowledge/rules/my_rules.yaml
rules:
  my_new_intent:
    default: my_new_intent_response
```

### Step 3: Definire la response in `knowledge/responses/`

Crea o modifica un file in `knowledge/responses/`:

```yaml
# knowledge/responses/my_responses.yaml
responses:
  my_new_intent_response:
    - "Hai attivato il nuovo intent!"
    - "Ecco la risposta per il tuo nuovo intent."
```

### Step 4: Riaddestrare i modelli

```bash
python -m pipeline
```

---

## Aggiungere Slot agli Intent

Gli slot permettono di raccogliere informazioni strutturate dall'utente.

### Slot Required

```yaml
# knowledge/rules/my_rules.yaml
rules:
  prenota_ristorante:
    slots:
      NUMERO_PERSONE:
        required: true
        type: integer
      DATA:
        required: true
        type: string
    default: prenota_ristorante_default_response
    wait: prenota_ristorante_wait_response
    fallback: prenota_ristorante_fallback_response
```

### Slot con Cases

```yaml
rules:
  cerca_meteo:
    slots:
      CITTA:
        required: true
        type: string
        entity: LOCATION
    cases:
      Roma: meteo_roma_response
      Milano: meteo_milano_response
      Napoli: meteo_napoli_response
    fallback: meteo_non_supportato_response
    wait: meteo_citta_wait_response
```

### Slot Opzionali

```yaml
rules:
  invia_messaggio:
    slots:
      DESTINATARIO:
        required: true
        type: string
        entity: PERSON
      MESSAGGIO:
        required: false
        type: string
    default: invia_messaggio_response
```

### Bot Slots (set_slots)

Per impostare slot automaticamente dalla risposta:

```yaml
rules:
  conferma_ordine:
    default: conferma_ordine_response
    set_slots:
      ORDINE_CONFERMATO: true
      DATA_CONFERMA: "$timestamp"
```

### Risposte con Slot Inline

Le risposte possono contenere slot estratti inline:

```yaml
responses:
  conferma_ordine_response:
    - "Ho confermato il tuo ordine per {NUMERO_PERSONE} persone."
```

---

## Creare Operazioni Custom

Le operazioni permettono di eseguire codice Python durante la generazione della risposta.

### Metodo 1: Classe Operation

Crea un file in `agent/operations/`:

```python
# agent/operations/calcola.py
from agent.operations.base import Operation


class Calculate(Operation):
    @property
    def name(self) -> str:
        return "calculate"

    def execute(self, intent_name: str, slots: dict = None) -> dict:
        expression = slots.get("expression", "0")
        
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {
                "response": f"Il risultato è: {result}",
                "slots": {"result": result},
                "metadata": {}
            }
        except Exception as e:
            return {
                "response": f"Errore nel calcolo: {e}",
                "slots": {},
                "metadata": {}
            }
```

### Metodo 2: Funzione

```python
# agent/operations/geocoding.py
def action_geocode(address: str, slots: dict, entity_manager) -> dict:
    # Logica di geocoding
    return {
        "response": f"Coordinate per {address}: 41.9028, 12.4964",
        "slots": {},
        "metadata": {}
    }
```

### Richiamare l'operazione nella Rule

```yaml
rules:
  calcola:
    default: __calculate
```

Il prefisso `__` indica che è un'operazione.

---

## Definire Nuove Entità NER

### Aggiungere entità nel codice

Modifica `classes/ner_tag_builder.py`:

```python
class NERTagBuilder:
    def __init__(self):
        self.tags = [
            'O',           # Outside
            'B-LOCATION',  # Begin Location
            'I-LOCATION', # Inside Location
            'B-DATE',      # Begin Date
            'I-DATE',      # Inside Date
            # Aggiungi qui nuove entità
        ]
```

### Usare entità negli Intent

```yaml
nlu:
  intents:
  - intent: cerca_evento
    examples:
    - cerca evento [musicale](TYPE) a [Roma](LOCATION)
    - eventi [sportivi](TYPE) a [Milano](LOCATION)
```

---

## Creare Flussi di Conversazione

### Definire conversation in YAML

```yaml
# knowledge/conversations/prenotazione.yaml
conversations:
  prenotazione_ristorante_flow:
    description: "Flusso completo prenotazione ristorante"
    steps:
      - user: greeting
        bot: greeting_response
      - user: cerca_ristorante
        bot: cerca_ristorante_response
      - user: prenota_ristorante
        bot: prenota_ristorante_response
      - user: conferma_prenotazione
        bot: conferma_prenotazione_response
      - user: farewell
        bot: farewell_response
```

### Training della Dialogue Policy

La Dialogue Policy impara dai flussi definiti:

1. Se `models/dialogue_policy.pth` esiste → usa ML
2. Altrimenti → usa longest-suffix match euristico

Per addestrare il modello ML:

```bash
python -m pipeline
```

---

## Estendere il Sistema

### Aggiungere un nuovo modulo Operations

1. Crea il file in `agent/operations/`
2. Implementa una classe `Operation` o una funzione `action_*`
3. Il sistema farà auto-discovery

### Modificare l'Intent Classifier

Per modificare l'architettura:

1. Modifica `intellective/intent_classifier.py`
2. Riaddestra con `python -m intellective.train_intent_classifier`

### Aggiungere nuovi tag NER

1. Modifica `classes/ner_tag_builder.py`
2. Aggiungi i tag nella lista
3. Riaddestra il modello

### Personalizzare la Dialogue Policy

Modifica iperparametri in `intellective/dialogue_policy.py`:

```python
DIALOGUE_POLICY_EMBED_DIM: int = 64   # default: 64
DIALOGUE_POLICY_HIDDEN_DIM: int = 128 # default: 128
DIALOGUE_POLICY_DROPOUT: float = 0.3  # default: 0.3
```

---

## Best Practices

### Nomenclatura

- Intent: `snake_case` (es. `prenota_ristorante`)
- Slot: `MAIUSCOLO` (es. `NUMERO_PERSONE`)
- Response key: `<intent>_<tipo>_response`

### Ordine dei file

- Mantieni i file base in `knowledge/base.yaml`
- Aggiungi file specifici per dominio (es. `culture.yaml`, `command.yaml`)

### Testing

```bash
# Test conversazione
python -m agent.agent

# Test modello
pytest tests/
```

### Debug

Il sistema stampa informazioni di debug durante l'esecuzione:
- Intent predetto e confidenza
- Entità estratte
- Slot attesi e ricevuti
- Risposta generata
