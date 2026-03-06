"""
Esempio di operation completa che mostra tutte le funzionalità disponibili.

Questa è una operation di esempio che dimostra:
- Come usare tutti i parametri disponibili
- Come manipolare gli slot
- Come interagire con session_manager
- Come ritornare metadata
"""


def action_example(intent_name: str, slots: dict, session_manager, entity_manager) -> dict:
    """
    Operation di esempio completa.

    Questa action mostra come utilizzare tutti i parametri disponibili.

    Args:
        intent_name: Nome dell'intent che ha triggerato questa action
        slots: Dizionario degli slot estratti dall'input utente
        session_manager: Manager per accedere/modificare la sessione utente
        entity_manager: Manager per gestire le entità NER

    Returns:
        dict con:
            - response: Testo da mostrare all'utente
            - slots: Slot da impostare/aggiornare
            - metadata: Informazioni aggiuntive opzionali
    """
    # 1. Accedi agli slot
    user_name = slots.get("person_name")
    user_age = slots.get("age")

    # 2. Accedi alla sessione (dati persistenti dell'utente)
    visit_count = session_manager.get("visit_count", 0)
    session_manager.set("visit_count", visit_count + 1)

    last_intent = session_manager.get("last_intent")
    session_manager.set("last_intent", intent_name)

    # 3. Costruisci la risposta
    if user_name:
        greeting = f"Ciao {user_name}!"
    else:
        greeting = "Ciao!"

    if visit_count > 0:
        greeting += f" Questa è la tua visita numero {visit_count + 1}."

    # 4. Prepara gli slot da impostare
    new_slots = {}
    if user_name:
        new_slots["confirmed_name"] = user_name
    if user_age:
        new_slots["confirmed_age"] = user_age

    # 5. Ritorna il risultato completo
    return {
        "response": greeting,
        "slots": new_slots,
        "metadata": {
            "operation": "example",
            "intent": intent_name,
            "visit_count": visit_count + 1,
            "previous_intent": last_intent
        }
    }


# Puoi anche definire funzioni helper nella stessa file
def _calculate_something(value):
    """Helper privato per calcoli."""
    return value * 2


def action_simple_example() -> str:
    """
    Esempio più semplice: ritorna solo una stringa.

    Quando ritorni una stringa, viene automaticamente wrappata in:
    {
        "response": "la tua stringa",
        "slots": {},
        "metadata": {}
    }
    """
    return "Questa è una risposta semplice!"


def action_with_slots_only(slots: dict) -> dict:
    """
    Esempio che usa solo gli slot.

    Non tutti i parametri sono obbligatori! Usa solo quelli che ti servono.
    """
    # Sempre buona pratica fare il fallback a dict vuoto
    slots = slots or {}

    value = slots.get("number", 0)
    result = _calculate_something(value)

    return {
        "response": f"Il risultato è {result}",
        "slots": {"calculation_result": result},
        "metadata": {"operation": "simple_calc"}
    }


def action_with_error_handling(slots: dict) -> dict:
    """
    Esempio con gestione errori.

    È importante gestire gli errori e ritornare sempre un risultato valido.
    """
    try:
        # Simula un'operazione che potrebbe fallire
        value = slots.get("value")

        if value is None:
            raise ValueError("Valore mancante")

        result = 100 / value  # Potrebbe dividere per zero

        return {
            "response": f"Il risultato è {result}",
            "slots": {"result": result},
            "metadata": {"success": True}
        }

    except ZeroDivisionError:
        return {
            "response": "Errore: impossibile dividere per zero",
            "slots": {},
            "metadata": {"success": False, "error": "division_by_zero"}
        }

    except ValueError as e:
        return {
            "response": f"Errore: {str(e)}",
            "slots": {},
            "metadata": {"success": False, "error": "invalid_input"}
        }

    except Exception as e:
        return {
            "response": "Si è verificato un errore imprevisto",
            "slots": {},
            "metadata": {"success": False, "error": str(e)}
        }

