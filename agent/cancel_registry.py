"""Registro in-memory dei segnali di cancellazione per sessione.

Lo streaming endpoint registra un threading.Event per la sessione corrente
prima di eseguire il turno; le operation lente (es. web_search) lo ricevono
e possono interrompersi tra un tentativo e l'altro. Un secondo endpoint
(/chatbot/cancel) o la disconnessione del client impostano l'Event dall'esterno,
senza dover far passare la richiesta di stop per lo stesso stream HTTP che si
vuole interrompere.
"""
import threading

_events: dict[str, threading.Event] = {}
_lock = threading.Lock()


def register(session_id: str) -> threading.Event:
    event = threading.Event()
    with _lock:
        _events[session_id] = event
    return event


def cancel(session_id: str) -> bool:
    with _lock:
        event = _events.get(session_id)
    if event is None:
        return False
    event.set()
    return True


def clear(session_id: str) -> None:
    with _lock:
        _events.pop(session_id, None)
