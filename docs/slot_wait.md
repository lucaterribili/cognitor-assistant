# Slot Waiting Mode — Traccia Modifiche

## Obiettivo

Implementare una modalità di attesa slot: quando il bot non ha le informazioni necessarie per risolvere un intent, entra in attesa di un input specifico dall'utente. Ricevuto l'input, imposta lo slot e ri-esegue il resolve sull'intent originale.

---

## Flusso

```
1. Utente: "informazioni turistiche"
2. predict() → intent: ask_city_touristic_information
3. resolve() → LOCATION not filled → risponde ask_location_wait_response
4. Sessione entra in waiting_for_slot: { intent: "ask_city_touristic_information", slot: "LOCATION" }

5. Utente: "Roma"
6. chat() vede waiting_for_slot attivo → salta predict()
7. Imposta LOCATION = "Roma" nel contesto di sessione
8. Re-esegue resolve("ask_city_touristic_information") → matcha eq Roma
9. Risponde ask_city_touristic_information_roma_response
10. waiting_for_slot viene resettato a None
```

---

## Modifiche

### 1. Rules JSON

Aggiungere il campo `wait_for_slot` nei branch che richiedono input utente.

```json
"ask_city_touristic_information": {
  "conditions": [
    {
      "if": [
        { "slot": "LOCATION", "operator": "eq", "value": "Roma" }
      ],
      "response": "ask_city_touristic_information_roma_response"
    },
    {
      "if": [
        { "slot": "LOCATION", "operator": "eq", "value": "Milano" }
      ],
      "response": "ask_city_touristic_information_milano_response"
    }
  ],
  "default": "ask_location_wait_response",
  "wait_for_slot": "LOCATION"
}
```

`wait_for_slot` è definito a livello di intent, non di branch: si attiva solo quando si cade nel `default`.

---

### 2. `ConversationSession` — session_manager.py

Aggiungere il campo `waiting_for_slot` al dataclass.

```python
@dataclass
class ConversationSession:
    session_id: str
    created_at: datetime
    updated_at: datetime
    history: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    waiting_for_slot: dict | None = None
    # esempio valore: {"intent": "ask_city_touristic_information", "slot": "LOCATION"}
```

---

### 3. `AnswerManager` — answer_manager.py

Il metodo `resolve()` deve restituire anche l'eventuale `wait_for_slot` da attivare.

Modificare `resolve()` per restituire un dizionario invece di una stringa:

```python
def resolve(self, intent: str, slots: dict) -> dict:
    rule = self.rules.get(intent)
    if not rule:
        return {"response": "default_fallback", "wait_for_slot": None}

    for branch in rule.get("conditions", []):
        if all(self._check_condition(condition, slots) for condition in branch["if"]):
            return {"response": branch["response"], "wait_for_slot": None}

    wait_for_slot = rule.get("wait_for_slot")

    return {
        "response": rule.get("default", "default_fallback"),
        "wait_for_slot": wait_for_slot
    }
```

Aggiornare `get_response()` di conseguenza:

```python
def get_response(self, intent: str, slots: dict, responses: dict) -> tuple[str, str | None]:
    resolved = self.resolve(intent, slots)
    response_key = resolved["response"]
    wait_for_slot = resolved["wait_for_slot"]

    response_list = responses.get(response_key, [])
    if not response_list:
        return f"Risposta non definita per {response_key}", None

    return random.choice(response_list), wait_for_slot
```

Restituisce una tupla `(testo_risposta, slot_da_attendere_o_None)`.

---

### 4. `Agent.chat()` — agent.py

Modificare il loop per gestire la modalità attesa.

```python
while True:
    user_input = input("Tu: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ['esci', 'exit', 'quit', 'q']:
        print("\n👋 Arrivederci!")
        break

    # Modalità attesa slot attiva
    if session.waiting_for_slot:
        slot_name = session.waiting_for_slot["slot"]
        pending_intent = session.waiting_for_slot["intent"]

        session.update_context(slot_name, user_input)
        session.waiting_for_slot = None

        response, wait_for_slot = self.get_response(pending_intent, session.context)
        print(f"\n🤖 Arianna: {response}\n")

        if wait_for_slot:
            session.waiting_for_slot = {"intent": pending_intent, "slot": wait_for_slot}

        session.add_message("user", user_input)
        session.add_message("assistant", response, pending_intent)
        continue

    # Flusso normale
    prediction = self.predict(user_input)
    print(f"\n🎯 Intent: {prediction['intent']} ({prediction['confidence']:.1%})")

    for entity in prediction.get('entities', []):
        session.update_context(entity['entity'], entity['value'])

    response, wait_for_slot = self.get_response(prediction['intent'], session.context)
    print(f"\n🤖 Arianna: {response}\n")

    if wait_for_slot:
        session.waiting_for_slot = {"intent": prediction['intent'], "slot": wait_for_slot}

    session.add_message("user", user_input, prediction['intent'], prediction.get('entities', []))
    session.add_message("assistant", response, prediction['intent'])
```

---

## Riepilogo file modificati

| File | Modifica |
|---|---|
| `knowledge/rules/*.json` | Aggiungere campo `wait_for_slot` agli intent che lo richiedono |
| `agent/session_manager.py` | Aggiungere campo `waiting_for_slot` a `ConversationSession` |
| `agent/answer_manager.py` | `resolve()` restituisce dict, `get_response()` restituisce tupla |
| `agent/agent.py` | Loop `chat()` gestisce modalità attesa prima del flusso normale |


# wait_for_slot — Modifica JSON

## Struttura aggiornata

`wait_for_slot` si sposta dentro la condition, non più a livello di intent.

```json
"ask_city_touristic_information": {
  "conditions": [
    {
      "if": [
        { "slot": "LOCATION", "operator": "eq", "value": "Roma" }
      ],
      "response": "ask_city_touristic_information_roma_response"
    },
    {
      "if": [
        { "slot": "LOCATION", "operator": "eq", "value": "Milano" }
      ],
      "response": "ask_city_touristic_information_milano_response"
    },
    {
      "if": [
        { "slot": "LOCATION", "operator": "not_filled" }
      ],
      "response": "ask_location_wait_response",
      "wait_for_slot": "LOCATION"
    }
  ],
  "default": "ask_location_wait_response"
}
```

## Modifica AnswerManager

`resolve()` legge `wait_for_slot` dal branch che matcha, non dalla rule.

```python
for branch in rule.get("conditions", []):
    if all(self._check_condition(condition, slots) for condition in branch["if"]):
        return {
            "response": branch["response"],
            "wait_for_slot": branch.get("wait_for_slot")
        }

return {"response": rule.get("default", "default_fallback"), "wait_for_slot": None}
```