# Operations - Quick Reference

Custom actions per il bot Arianna. Sistema con **auto-discovery** automatico.

## Come Creare una Nuova Operation

### Metodo 1: Funzione (Consigliato) ⭐

Crea un file e definisci una funzione con prefisso `action_`:

```python
# my_action.py

def action_my_action(intent_name: str, slots: dict = None) -> dict:
    """Descrizione dell'action."""
    return {
        "response": "Risposta generata",
        "slots": {},
        "metadata": {}
    }
```

**Parametri disponibili** (tutti opzionali):
- `intent_name` o `intent`: Nome dell'intent
- `slots`: Dizionario degli slot
- `session_manager`: SessionManager
- `entity_manager`: EntityManager

**Return format:**
- Stringa semplice: `return "Ciao!"`
- Dict completo: `return {"response": "...", "slots": {}, "metadata": {}}`

### Metodo 2: Classe (Per logica complessa)

```python
# my_operation.py
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

## Naming Convention

- **Funzioni**: `action_<nome>` → registrata come `<nome>`
- **Funzioni**: `<nome>_action` → registrata come `<nome>`
- **Classi**: Implementa la property `name()`

## Come Funziona

1. L'`OperationManager` scansiona automaticamente questa cartella all'avvio
2. Trova tutte le funzioni `action_*` e classi che ereditano da `Operation`
3. Le registra automaticamente
4. Quando una rule richiede un'operation, viene eseguita

**Non serve registrare manualmente!** 🎉

## Esempi dal Progetto

- `calculate.py` - Action di calcolo (funzione)
- `location_query.py` - Query posizione con API esterna (funzione)

## Documentazione Completa

Vedi: `/docs/OPERATIONS_SYSTEM.md`

## Testing

```python
from agent.operations.manager import OperationManager

manager = OperationManager(auto_discover=True)
print(manager.list_operations())  # Lista tutte le operations

result = manager.execute("my_action", "intent_name", {"key": "value"})
```

## Troubleshooting

**Operation non caricata?**
1. Controlla il nome: deve essere `action_<nome>` o ereditare da `Operation`
2. Verifica che il file sia in questa cartella (non in sottocartelle tipo `tools/`)
3. Controlla che non ci siano errori di import
4. Pulisci cache: `find . -name "*.pyc" -delete`

**Errori di parametri?**
- Usa solo i parametri supportati
- Tutti i parametri sono opzionali
- Aggiungi `slots = slots or {}` se usi slots

