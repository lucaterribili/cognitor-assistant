# Sistema Operations - Custom Actions per Arianna

Il sistema di Operations è ispirato alle **Custom Actions di Rasa** e permette di eseguire codice personalizzato in risposta a specifici intent.

## Panoramica

Le Operations sono azioni che il bot può eseguire quando viene triggerato un intent specifico. A differenza delle semplici risposte testuali, le operations possono:

- Eseguire logica complessa
- Chiamare API esterne
- Manipolare lo stato della sessione
- Modificare gli slot
- Generare risposte dinamiche

## Come Funziona l'Auto-Discovery

Il sistema **scopre automaticamente** tutte le operations presenti nella cartella `agent/operations/`:

1. All'avvio, l'`OperationManager` scansiona tutti i file `.py` nella cartella
2. Per ogni file, cerca:
   - **Funzioni** che seguono il pattern `action_*` o `*_action`
   - **Classi** che ereditano da `Operation`
3. Le funzioni vengono automaticamente wrappate in una classe `Operation`
4. Tutte le operations vengono registrate e rese disponibili

**Non serve più registrare manualmente le operations!**

---

## Metodo 1: Funzioni (Consigliato) ⭐

Il modo più semplice e veloce per creare una operation è definire una funzione.

### Naming Convention

La funzione deve seguire uno di questi pattern:
- `action_<nome>` → registrata come `<nome>`
- `<nome>_action` → registrata come `<nome>`

### Esempio Base

```python
# agent/operations/my_action.py

def action_greet(intent_name: str) -> str:
    """Saluta l'utente."""
    return "Ciao! Come posso aiutarti?"
```

### Parametri Disponibili

La funzione può accettare questi parametri (tutti **opzionali**):

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `intent_name` o `intent` | `str` | Nome dell'intent che ha triggerato l'action |
| `slots` | `dict` | Dizionario degli slot disponibili |
| `session_manager` | `SessionManager` | Gestore delle sessioni utente |
| `entity_manager` | `EntityManager` | Gestore delle entità NER |

Il sistema **riconosce automaticamente** quali parametri accetta la tua funzione e li passa di conseguenza.

### Formato del Risultato

Puoi ritornare:

1. **Una stringa semplice** (viene automaticamente wrappata in un dict):
```python
def action_hello() -> str:
    return "Ciao!"
```

2. **Un dict completo** con tutte le informazioni:
```python
def action_calculate(intent_name: str, slots: dict) -> dict:
    result = slots.get("a", 0) + slots.get("b", 0)
    return {
        "response": f"Il risultato è {result}",
        "slots": {"result": result},  # Slot da impostare
        "metadata": {"operation": "calculate"}  # Metadati opzionali
    }
```

### Esempi Completi

#### Esempio 1: Action Semplice
```python
# agent/operations/joke.py

def action_joke() -> str:
    """Racconta una barzelletta."""
    return "Perché i programmatori preferiscono il buio? Perché la luce attira i bug!"
```

#### Esempio 2: Action con Slot
```python
# agent/operations/weather.py

def action_weather(slots: dict) -> dict:
    """Fornisce info sul meteo."""
    city = slots.get("city", "sconosciuta")
    
    # Qui potresti chiamare un'API meteo
    return {
        "response": f"Il meteo a {city} è soleggiato!",
        "slots": {"last_city": city},
        "metadata": {"source": "weather_api"}
    }
```

#### Esempio 3: Action con Session Manager
```python
# agent/operations/user_info.py

def action_save_name(slots: dict, session_manager) -> str:
    """Salva il nome dell'utente nella sessione."""
    name = slots.get("person_name")
    
    if name:
        session_manager.set("user_name", name)
        return f"Perfetto, ti chiamerò {name}!"
    
    return "Come ti chiami?"
```

#### Esempio 4: Action con API Esterna
```python
# agent/operations/news.py
import requests

def action_get_news(slots: dict) -> dict:
    """Recupera le ultime notizie."""
    category = slots.get("category", "general")
    
    try:
        response = requests.get(f"https://api.news.com/{category}")
        data = response.json()
        
        return {
            "response": f"Ultime notizie su {category}: {data['headline']}",
            "slots": {},
            "metadata": {"news_source": "api.news.com"}
        }
    except Exception as e:
        return {
            "response": "Non sono riuscito a recuperare le notizie.",
            "slots": {},
            "metadata": {"error": str(e)}
        }
```

---

## Metodo 2: Classi (Per Logica Complessa)

Per operations più complesse o che necessitano di stato, puoi usare le classi.

### Esempio

```python
# agent/operations/database_query.py
from agent.operations.base import Operation

class DatabaseQueryOperation(Operation):
    """Operation per query al database."""
    
    def __init__(self, session_manager=None, entity_manager=None):
        super().__init__(session_manager, entity_manager)
        self.db_connection = self._init_db()
    
    @property
    def name(self) -> str:
        return "database_query"
    
    def execute(self, intent_name: str, slots: dict = None) -> dict:
        query = slots.get("query")
        
        # Esegui la query
        results = self.db_connection.execute(query)
        
        return {
            "response": f"Trovati {len(results)} risultati",
            "slots": {"query_results": results},
            "metadata": {"rows": len(results)}
        }
    
    def _init_db(self):
        # Inizializza la connessione al database
        pass
```

### Quando Usare le Classi?

Usa le classi quando:
- Hai bisogno di **stato persistente** (connessioni, cache, ecc.)
- Vuoi **metodi helper** privati
- La logica è **molto complessa** e beneficia di una struttura OOP
- Devi fare **setup/teardown** di risorse

---

## Come Usare le Operations nelle Rules

Nelle tue rules YAML, puoi invocare le operations usando la direttiva `operation`:

```yaml
# knowledge/rules/calculate.yaml
rules:
  - intent: calculate
    operation: calculate  # Nome della operation
    priority: 10
```

Quando l'intent `calculate` viene triggerato:
1. Il `RuleInterpreter` trova la regola
2. Chiama `OperationManager.execute("calculate", intent_name, slots)`
3. L'`OperationManager` trova la function/class `action_calculate`
4. Esegue la function passando i parametri necessari
5. Ritorna il risultato

---

## Struttura del Progetto

```
agent/operations/
├── __init__.py           # Esporta Operation e OperationManager
├── base.py              # Classe base Operation (ABC)
├── manager.py           # OperationManager con auto-discovery
├── calculate.py         # Example: action_calculate()
├── location_query.py    # Example: action_location_query()
└── tools/               # Utility condivise
    └── geocoding.py
```

---

## Dettagli Tecnici dell'Auto-Discovery

### Processo di Caricamento

```python
# In agent.py all'avvio:
self.operation_manager = OperationManager(
    session_manager=self.session_manager,
    entity_manager=self.session_manager.entity_manager,
    auto_discover=True  # Attiva l'auto-discovery
)
```

### Cosa Viene Caricato?

L'`OperationManager` scansiona:
- ✅ Tutti i file `.py` in `agent/operations/`
- ❌ Esclude: `__init__.py`, `base.py`, `manager.py`
- ❌ Esclude: la cartella `tools/` (utility condivise)

### Logging

Durante l'avvio vedrai:
```
✓ Operation 'calculate' caricata da calculate.py (function)
✓ Operation 'location_query' caricata da location_query.py (function)
✓ Operation 'database_query' caricata da database_query.py (classe)
```

Se ci sono errori:
```
⚠ Errore nel caricare operations da broken.py: ModuleNotFoundError: No module named 'xyz'
```

---

## Best Practices

### ✅ DO

1. **Usa funzioni per logica semplice** - Più pulito e veloce
2. **Nomina chiaramente le actions** - `action_send_email` non `action_x`
3. **Gestisci gli errori** - Usa try/except e ritorna messaggi utili
4. **Ritorna sempre un dict completo** - Con response, slots e metadata
5. **Documenta i parametri** - Usa docstring chiare

### ❌ DON'T

1. **Non creare classi se non necessario** - Le funzioni sono più semplici
2. **Non fare blocking I/O lungo** - Usa async o thread se necessario
3. **Non salvare stato nelle funzioni** - Usa session_manager invece
4. **Non importare operations manualmente** - L'auto-discovery lo fa per te
5. **Non dimenticare di testare** - Scrivi test per le tue operations

---

## Testing

### Test Unitario di una Operation

```python
# tests/test_operations.py
from agent.operations.calculate import action_calculate

def test_calculate():
    result = action_calculate(
        intent_name="calculate",
        slots={"a": 5, "b": 3}
    )
    
    assert "response" in result
    assert result["slots"]["result"] == 8
```

### Test dell'OperationManager

```python
from agent.operations.manager import OperationManager

def test_auto_discovery():
    manager = OperationManager(auto_discover=True)
    
    # Verifica che le operations siano state caricate
    assert "calculate" in manager.list_operations()
    assert "location_query" in manager.list_operations()
    
    # Test esecuzione
    result = manager.execute("calculate", "calculate", {"a": 2, "b": 3})
    assert result["response"] is not None
```

---

## Troubleshooting

### La mia operation non viene caricata

**Checklist:**
- [ ] Il file è nella cartella `agent/operations/`?
- [ ] La funzione inizia con `action_` o finisce con `_action`?
- [ ] La classe eredita da `Operation`?
- [ ] Non ci sono errori di import nel file?
- [ ] Hai pulito la cache Python? (`find . -name "*.pyc" -delete`)

### Errori comuni

**1. `Operazione xyz non trovata`**
- L'operation non è stata caricata (vedi checklist sopra)
- Il nome nella rule non corrisponde al nome dell'operation

**2. `TypeError: missing required positional argument`**
- La tua funzione richiede parametri non supportati
- Usa solo i parametri documentati: `intent_name`, `slots`, `session_manager`, `entity_manager`

**3. `AttributeError: 'str' object has no attribute 'get'`**
- Stai assumendo che `slots` sia sempre un dict
- Aggiungi: `slots = slots or {}`

---

## Migrazione da Vecchio Sistema

Se hai vecchie operations basate su classi che vuoi mantenere:

### Prima (vecchio modo)
```python
# agent/agent.py
from agent.operations.calculate import CalculateOperation

self.operation_manager.register(CalculateOperation(
    session_manager=self.session_manager,
    entity_manager=self.session_manager.entity_manager
))
```

### Dopo (nuovo modo)
```python
# agent/agent.py
# Non serve più niente! L'auto-discovery fa tutto

# agent/operations/calculate.py
def action_calculate(intent_name: str, slots: dict = None) -> dict:
    return {"response": "...", "slots": {}, "metadata": {}}
```

---

## Esempi dal Progetto

### 1. Calculate Operation
```python
# agent/operations/calculate.py
def action_calculate(intent_name: str, slots: dict = None) -> dict:
    """Esegue calcoli matematici."""
    return {
        "response": f"Operation per {intent_name} non implementata.",
        "slots": {},
        "metadata": {"operation": "calculate"}
    }
```

### 2. Location Query Operation
```python
# agent/operations/location_query.py
import random
from agent.operations.tools.geocoding import Geocoding

def action_location_query(intent_name: str, slots: dict = None) -> dict:
    """Determina la posizione dell'utente."""
    slots = slots or {}
    lat = slots.get("lat") or random.uniform(41.0, 46.0)
    lon = slots.get("lon") or random.uniform(6.0, 19.0)
    
    geocoding = Geocoding()
    location = geocoding.get_location(browser_coords={"lat": lat, "lon": lon})
    
    if location and location.get("address"):
        address = location.get("address", {})
        city = address.get("city") or address.get("town")
        response = f"Ti trovi a {city}, {address.get('country', '')}"
    else:
        response = "Non sono riuscito a determinare la tua posizione."
    
    return {
        "response": response,
        "slots": {},
        "metadata": {"operation": "location_query", "location": location}
    }
```

---

## Conclusione

Il sistema di Operations offre:

✨ **Auto-discovery** - Zero configurazione, aggiungi file e funziona  
🚀 **Semplicità** - Funzioni invece di classi dove possibile  
🔧 **Flessibilità** - Supporta sia funzioni che classi  
📦 **Modularità** - Ogni action in un file separato  
🎯 **Type-safe** - Parametri opzionali con signature inspection  

È come Rasa, ma più semplice! 🎉

