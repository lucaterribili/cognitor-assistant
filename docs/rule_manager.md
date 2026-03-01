# Answer Manager — Traccia Modifiche

## Contesto

Sistema di gestione risposte per chatbot. L'answer manager seleziona la risposta corretta in base all'intent dell'utente e agli slot presenti in sessione.

---

## Versione 1 — Struttura base

Mapping semplice intent → risposta, senza condizioni.

```json
{
  "rules": {
    "restaurant_booking": ["restaurant_booking_response"],
    "order_food": ["order_food_response"],
    "taxi_ride": ["taxi_ride_response"],
    "book_flight": ["book_flight_response"],
    "book_hotel": ["book_hotel_response"],
    "alarm_set": ["alarm_set_response"],
    "play_music": ["play_music_response"],
    "read_news": ["read_news_response"],
    "schedule_meeting": ["schedule_meeting_response"],
    "send_email": ["send_email_response"]
  }
}
```

**Limiti:** nessuna gestione degli slot, risposta sempre uguale indipendentemente dal contesto.

---

## Versione 2 — Condizioni e slot in sessione

### Obiettivo

Gestire risposte diverse in base a:
- Presenza o assenza di uno slot (`filled` / `not_filled`)
- Valore specifico di uno slot (`eq`, `neq`, `gt`, `lt`, `contains`)

Gli slot vivono **in memoria per sessione**.

---

### Struttura JSON aggiornata

Ogni intent può avere:
- `conditions` — lista di branch valutati in ordine (vince il primo che matcha)
- `default` — risposta di fallback se nessuna condizione è soddisfatta

**Intent con condizioni:**

```json
{
  "rules": {
    "restaurant_booking": {
      "conditions": [
        {
          "if": [
            { "slot": "city", "operator": "filled" },
            { "slot": "date", "operator": "filled" },
            { "slot": "guests", "operator": "filled" }
          ],
          "response": "restaurant_booking_confirm"
        },
        {
          "if": [
            { "slot": "city", "operator": "filled" },
            { "slot": "guests", "operator": "gt", "value": 10 }
          ],
          "response": "restaurant_booking_large_group"
        },
        {
          "if": [
            { "slot": "city", "operator": "not_filled" }
          ],
          "response": "restaurant_booking_ask_city"
        }
      ],
      "default": "restaurant_booking_response"
    }
  }
}
```

**Intent senza condizioni (solo default):**

```json
{
  "rules": {
    "play_music": {
      "default": "play_music_response"
    },
    "read_news": {
      "default": "read_news_response"
    }
  }
}
```

---

### Operatori supportati

| Operatore    | Significato                        |
|--------------|------------------------------------|
| `filled`     | slot presente e non null           |
| `not_filled` | slot assente o null                |
| `eq`         | uguale a value                     |
| `neq`        | diverso da value                   |
| `gt`         | maggiore di value                  |
| `lt`         | minore di value                    |
| `contains`   | stringa contiene value             |

---

### Logica di valutazione

- Le condizioni dentro un `if` sono in **AND** — devono essere tutte vere
- I branch sono valutati **in ordine** — vince il primo che matcha
- Se nessun branch matcha, si usa il `default`
- Se l'intent non esiste nelle regole, si usa `"default_fallback"`

---

### Implementazione Python

```python
class AnswerManager:
    def __init__(self, rules: dict):
        self.rules = rules
        self.session_slots: dict = {}

    def set_slot(self, key: str, value):
        self.session_slots[key] = value

    def _check_condition(self, condition: dict) -> bool:
        slot_value = self.session_slots.get(condition["slot"])
        operator = condition["operator"]
        expected_value = condition.get("value")

        if operator == "filled":     return slot_value is not None
        if operator == "not_filled": return slot_value is None
        if operator == "eq":         return slot_value == expected_value
        if operator == "neq":        return slot_value != expected_value
        if operator == "gt":         return slot_value is not None and slot_value > expected_value
        if operator == "lt":         return slot_value is not None and slot_value < expected_value
        if operator == "contains":   return expected_value in (slot_value or "")
        return False

    def resolve(self, intent: str) -> str:
        rule = self.rules.get(intent)
        if not rule:
            return "default_fallback"

        for branch in rule.get("conditions", []):
            if all(self._check_condition(condition) for condition in branch["if"]):
                return branch["response"]

        return rule.get("default", "default_fallback")
```

---

### Esempio di utilizzo

```python
manager = AnswerManager(rules)

manager.set_slot("city", "Milano")
manager.set_slot("date", "2026-03-10")
manager.set_slot("guests", 12)

response_key = manager.resolve("restaurant_booking")
# → "restaurant_booking_large_group"  (guests > 10 ha priorità sul branch "tutti filled")
```

---

## Prossimi passi possibili

- Supporto condizioni in **OR** tra branch
- Reset selettivo degli slot dopo una risposta confermata
- Slot con scadenza (slot validi solo per N turni)