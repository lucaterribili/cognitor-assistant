# Refactoring Data-Driven: Da Hardcoded a Rules-Based

## 🎯 Obiettivo

Trasformare il sistema di gestione degli slot da **hardcoded** a **completamente data-driven**, dove tutto è guidato dalle **rules JSON** senza alcuna configurazione esterna.

## ❌ Problema Iniziale

### LocationManager Hardcoded
```python
# PRIMA - agent/location_manager.py
SUPPORTED_CITIES = {"Roma", "Milano"}  # ❌ Hardcoded
LOCATION_INTENTS = {"ask_city_touristic_information", "visit_city"}  # ❌ Hardcoded
CHANGE_LOCATION_PATTERNS = [  # ❌ Hardcoded
    r"visitare\s+\w+",
    r"andare\s+a\s+\w+",
    # ...
]

class LocationManager:
    # Logica specifica solo per LOCATION
    # Non estensibile ad altre entità
```

### Limitazioni
1. ❌ Solo entità `LOCATION` supportata
2. ❌ Solo città hardcoded `{"Roma", "Milano"}`
3. ❌ Pattern regex hardcoded nel codice
4. ❌ Impossibile aggiungere nuove entità senza modificare il codice
5. ❌ Non scalabile

## ✅ Soluzione Implementata

### SlotManager Data-Driven

Un sistema **completamente generico** che:
- ✅ Funziona con **qualsiasi entità** (LOCATION, PERSON, DATE, TIME, etc.)
- ✅ Estrae tutto dalle **rules JSON**
- ✅ Nessuna configurazione hardcoded
- ✅ Completamente estensibile

```python
# DOPO - agent/slot_manager.py
class SlotManager:
    """
    Sistema completamente data-driven basato sulle rules JSON.
    Non richiede configurazioni hardcoded: tutto viene dedotto dalle rules.
    """
    
    def __init__(self, rules: dict):
        # Analizza automaticamente le rules
        self.extractor = SlotExtractor(rules)
        self.context_manager = SlotContextManager(self.extractor)
```

## 🏗️ Architettura

### Componenti

```
SlotManager (Facade)
    ├── SlotExtractor
    │   ├── Analizza rules JSON
    │   ├── Estrae slot richiesti per intent
    │   ├── Estrae valori validi dalle conditions
    │   └── Valida valori automaticamente
    │
    └── SlotContextManager
        ├── Gestisce aggiornamento contesto sessione
        ├── Rileva intent consecutivi
        └── Gestisce validazione/invalidazione slot
```

### 1. SlotExtractor

**Responsabilità**: Estrazione e validazione degli slot

```python
class SlotExtractor:
    def __init__(self, rules: dict):
        self.rules = rules
        self._slot_entity_mapping = self._build_slot_entity_mapping()
        self._valid_values_cache = {}
    
    def get_valid_values_for_slot(self, intent: str, slot_name: str) -> list[str]:
        """Estrae valori validi dalle rules automaticamente"""
        # Analizza le conditions nelle rules
        # Trova tutti i valori in operator "eq"
```

**Come funziona**:
```json
// rules/base.json
{
  "ask_city_touristic_information": {
    "conditions": [
      {
        "if": [
          {"slot": "LOCATION", "operator": "eq", "value": "Roma"}
        ]
      },
      {
        "if": [
          {"slot": "LOCATION", "operator": "eq", "value": "Milano"}
        ]
      }
    ]
  }
}
```

Il sistema **estrae automaticamente**:
- Slot richiesto: `LOCATION`
- Valori validi: `["Roma", "Milano"]`
- Nessun hardcoding necessario!

### 2. SlotContextManager

**Responsabilità**: Aggiornamento intelligente del contesto

```python
class SlotContextManager:
    def update_session_context(
        self,
        session,
        intent: str,
        previous_intent: Optional[str],
        entities: list[dict],
        user_input: str
    ):
        """
        Aggiorna automaticamente il contesto basandosi su:
        - Intent corrente e precedente
        - Entità estratte dal NER
        - Rules definite nei JSON
        """
```

**Logica intelligente**:
1. **Analizza gli slot** richiesti dall'intent corrente
2. **Rileva slot comuni** tra intent consecutivi
3. **Estrae valori** dalle entità NER
4. **Valida automaticamente** usando le rules
5. **Gestisce casi speciali**:
   - Intent consecutivi → possibile cambio valore
   - Valore estratto ma non valido → slot invalidato
   - Nessun valore estratto su intent consecutivo → slot invalidato

## 📊 Confronto Before/After

### Prima: LocationManager (Hardcoded)

```python
# ❌ File: location_manager.py (95 righe)
SUPPORTED_CITIES = {"Roma", "Milano"}  # Hardcoded
LOCATION_INTENTS = {...}  # Hardcoded
CHANGE_LOCATION_PATTERNS = [...]  # Hardcoded regex

# Solo per LOCATION
# Non estensibile
# Modifiche richiedono cambio codice
```

### Dopo: SlotManager (Data-Driven)

```python
# ✅ File: slot_manager.py (280 righe)
# Nessuna configurazione hardcoded
# Funziona con qualsiasi entità
# Completamente estensibile

# Uso:
slot_manager = SlotManager(rules)  # ← solo le rules!
slot_manager.update_session_from_prediction(...)  # ← automatico!
```

## 🚀 Come Aggiungere Nuove Entità

### Prima (Hardcoded)
```python
# 1. Modificare LocationManager
# 2. Aggiungere costanti
# 3. Aggiungere pattern regex
# 4. Modificare logica di estrazione
# 5. Testing...
# = 5+ modifiche al codice!
```

### Dopo (Data-Driven)
```json
// 1. Aggiungere nelle rules JSON:
{
  "book_appointment": {
    "conditions": [
      {
        "if": [
          {"slot": "DATE", "operator": "filled"}
        ],
        "response": "appointment_confirmed"
      },
      {
        "if": [
          {"slot": "DATE", "operator": "not_filled"}
        ],
        "response": "ask_date",
        "wait_for_slot": "DATE"
      }
    ]
  }
}

// 2. Fatto! Il sistema riconosce automaticamente:
// - Slot richiesto: DATE
// - Entità da estrarre: DATE
// - Validazione automatica
// = 0 modifiche al codice!
```

## 🎯 Vantaggi del Sistema Data-Driven

### 1. **Zero Configurazione Hardcoded**
- ✅ Nessun pattern regex nel codice
- ✅ Nessuna lista di valori validi nel codice
- ✅ Nessun mapping slot→entità nel codice
- ✅ Tutto nelle rules JSON

### 2. **Completamente Generico**
```python
# Funziona per qualsiasi entità:
# - LOCATION (Roma, Milano)
# - PERSON (nomi)
# - DATE (date)
# - TIME (orari)
# - PRODUCT (prodotti)
# - ... qualsiasi cosa!
```

### 3. **Auto-Discovery**
```python
# Il sistema analizza automaticamente le rules e scopre:
slot_manager.extractor.get_valid_values_for_slot("ask_city_touristic_information", "LOCATION")
# → ["Roma", "Milano"]  (estratto dalle rules!)

slot_manager.context_manager.get_slots_for_intent("ask_city_touristic_information")
# → {"LOCATION", "LOCATION_UNSUPPORTED"}  (estratto dalle rules!)
```

### 4. **Validazione Automatica**
```python
# Non serve implementare validatori custom
slot_manager.validate_slot_value("ask_city_touristic_information", "LOCATION", "Napoli")
# → False (perché Napoli non è nelle rules)

slot_manager.validate_slot_value("ask_city_touristic_information", "LOCATION", "Roma")
# → True (perché Roma è nelle rules)
```

### 5. **Gestione Intent Consecutivi**
```python
# Scenario:
# 1. User: "Vorrei visitare Roma"
#    → Intent: visit_city, LOCATION: Roma
# 2. User: "Voglio vedere Milano"
#    → Intent: visit_city (stesso intent!)
#    → SlotManager rileva automaticamente: cambio di LOCATION
#    → LOCATION: Milano (aggiornato automaticamente)
```

## 📝 Esempi di Uso

### Esempio 1: Aggiungere Entità PERSON

```json
// knowledge/rules/personal.json
{
  "introduce_yourself": {
    "conditions": [
      {
        "if": [
          {"slot": "PERSON", "operator": "filled"}
        ],
        "response": "greeting_with_name"
      },
      {
        "if": [
          {"slot": "PERSON", "operator": "not_filled"}
        ],
        "response": "ask_name",
        "wait_for_slot": "PERSON"
      }
    ]
  }
}
```

**Nessuna modifica al codice necessaria!** Il sistema:
- Rileva automaticamente che serve lo slot `PERSON`
- Estrae automaticamente entità `PERSON` dal NER
- Valida e gestisce il contesto

### Esempio 2: Entità con Valori Validi

```json
// knowledge/rules/restaurant.json
{
  "book_table": {
    "conditions": [
      {
        "if": [
          {"slot": "TIME", "operator": "eq", "value": "12:00"}
        ],
        "response": "lunch_booking"
      },
      {
        "if": [
          {"slot": "TIME", "operator": "eq", "value": "20:00"}
        ],
        "response": "dinner_booking"
      },
      {
        "if": [
          {"slot": "TIME", "operator": "not_filled"}
        ],
        "response": "ask_time",
        "wait_for_slot": "TIME"
      }
    ]
  }
}
```

Il sistema automaticamente:
- Valori validi per `TIME`: `["12:00", "20:00"]`
- Altri orari → `TIME_UNSUPPORTED` = True
- Validazione case-insensitive

## 🔧 Integrazione con Agent

### Agent.py
```python
class Agent:
    def load_knowledge(self) -> None:
        self.rules, self.responses = self.knowledge_loader.load_all()
        
        # Inizializzazione SlotManager (completamente autonomo)
        self.slot_manager = SlotManager(self.rules)
        # ← Nessuna configurazione necessaria!
```

### ConversationHandler.py
```python
def _handle_location_update(self, user_input: str, session, prediction: dict):
    """Gestione slot completamente data-driven"""
    self.agent.slot_manager.update_session_from_prediction(
        session=session,
        current_intent=prediction['intent'],
        entities=prediction.get('entities', []),
        user_input=user_input
    )
    # ← Funziona per QUALSIASI entità!
```

## 📊 Metriche

| Aspetto | Prima (LocationManager) | Dopo (SlotManager) |
|---------|-------------------------|---------------------|
| Configurazioni hardcoded | 3+ (cities, intents, patterns) | 0 |
| Entità supportate | 1 (LOCATION) | ∞ (qualsiasi) |
| Righe di configurazione | ~40 | 0 |
| File di configurazione | 1 (slot_config.json) | 0 (eliminato) |
| Modifiche per nuova entità | 5+ file | 1 JSON (rules) |
| Scalabilità | Bassa | Altissima |
| Manutenibilità | Difficile | Facile |

## ✨ Best Practices Applicate

1. **Data-Driven Architecture**: tutto guidato dai dati, non dal codice
2. **Convention over Configuration**: convenzioni intelligenti (slot name = entity type)
3. **Zero Hardcoding**: nessun valore magico nel codice
4. **Auto-Discovery**: il sistema scopre automaticamente cosa serve
5. **Fail-Safe**: validazione robusta con fallback intelligenti
6. **Separation of Concerns**: logica business separata da configurazione

## 🎉 Risultato Finale

### File Eliminati
- ❌ `agent/location_manager.py` (95 righe) → sostituito da sistema generico
- ❌ `knowledge/slot_config.json` → non serve più

### File Creati
- ✅ `agent/slot_manager.py` (280 righe) → sistema generico data-driven

### Modifiche
- ✅ `agent/agent.py` → usa SlotManager invece di LocationManager
- ✅ `agent/conversation_handler.py` → chiamata semplificata
- ✅ `agent/model_loader.py` → rimossi riferimenti a slot_config

## 🚀 Prossimi Passi Possibili

### 1. Supporto Pattern Avanzati nelle Rules
```json
{
  "conditions": [
    {
      "if": [
        {"slot": "LOCATION", "operator": "in", "values": ["Roma", "Milano", "Napoli"]}
      ]
    }
  ]
}
```

### 2. Supporto Slot Composti
```json
{
  "conditions": [
    {
      "if": [
        {"slot": "DATETIME", "operator": "filled"},
        {"slot": "DATETIME.date", "operator": "gt", "value": "today"}
      ]
    }
  ]
}
```

### 3. Supporto Validazioni Custom
```json
{
  "conditions": [
    {
      "if": [
        {"slot": "EMAIL", "operator": "matches", "pattern": "^[a-z]+@[a-z]+\\.[a-z]+$"}
      ]
    }
  ]
}
```

---

**Conclusione**: Il sistema è ora completamente data-driven, scalabile ed estensibile. Aggiungere nuove entità richiede solo modifiche ai file JSON delle rules, senza toccare il codice Python! 🎯

