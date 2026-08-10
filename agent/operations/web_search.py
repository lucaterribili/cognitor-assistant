"""Operation per l'intent web_search."""
import time
import threading

from ddgs import DDGS

_MAX_BODY_LENGTH = 200

# ddgs fa scraping delle pagine di ricerca DuckDuckGo (non è un'API ufficiale):
# occasionalmente un singolo tentativo fallisce per un rate-limit/blocco
# transitorio anche quando il servizio è raggiungibile. Un paio di retry con
# un piccolo backoff bastano a coprire questi casi senza introdurre una
# latenza percepibile in caso di fallimento reale/persistente.
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 1.5


def action_web_search(intent_name: str, slots: dict = None, cancel_event: threading.Event = None) -> dict:
    """
    Esegue una ricerca web tramite DuckDuckGo.

    Lo slot 'query' è required: se assente, il sistema entra in modalità
    inputable (gestita dalla rule YAML) e chiede all'utente cosa cercare.

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili, si aspetta la chiave 'query' con il termine di ricerca
        cancel_event: se impostato (utente ha premuto "Interrompi" mentre il
            bot stava cercando, via endpoint di streaming), interrompe il
            retry loop tra un tentativo e l'altro senza aspettare gli altri
            backoff

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    query = slots.get("query") or slots.get("QUERY") or slots.get("search_query")

    # Questo caso non dovrebbe più verificarsi grazie alla modalità inputable,
    # ma viene mantenuto come fallback di sicurezza.
    if not query:
        return {
            "response": "Cosa vuoi che cerchi? Dimmi l'argomento della ricerca.",
            "slots": {},
            "metadata": {"operation": "web_search", "query": None}
        }

    results = None
    last_error = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if cancel_event is not None and cancel_event.is_set():
            return {
                "response": "Ok, mi fermo! Dimmi pure se ti serve altro.",
                "slots": {},
                "metadata": {"operation": "web_search", "query": query, "cancelled": True}
            }
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, region="it-it"))
            break
        except Exception as e:
            last_error = e
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)

    if results is None:
        return {
            "response": "Mi dispiace, non riesco a effettuare la ricerca in questo momento. Riprova più tardi.",
            "slots": {},
            "metadata": {
                "operation": "web_search",
                "query": query,
                "error": str(last_error),
                "attempts": _MAX_ATTEMPTS,
            }
        }

    if not results:
        return {
            "response": f"Non ho trovato risultati per '{query}'. Prova con termini diversi.",
            "slots": {},
            "metadata": {"operation": "web_search", "query": query, "results": []}
        }

    lines = [f"Ecco cosa ho trovato per '{query}':\n"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "")
        body = result.get("body", "")
        href = result.get("href", "")
        lines.append(f"{i}. **{title}**")
        if body:
            lines.append(f"   {body[:_MAX_BODY_LENGTH]}{'...' if len(body) > _MAX_BODY_LENGTH else ''}")
        if href:
            lines.append(f"   🔗 {href}")
        lines.append("")

    response = "\n".join(lines).strip()

    return {
        "response": response,
        "slots": {},
        "metadata": {"operation": "web_search", "query": query, "results": results}
    }
