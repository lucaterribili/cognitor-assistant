"""Operation per l'intent search_tutorial — cerca tutorial nel backoffice Laravel (Programmato)."""
import requests

import config

_MAX_RESULTS = 5


def _title_of(item: dict) -> str:
    title = item.get("title") or "(senza titolo)"
    if isinstance(title, dict):
        title = title.get("it") or next(iter(title.values()), "(senza titolo)")
    return title


def action_search_tutorial(intent_name: str, slots: dict = None) -> dict:
    """
    Cerca tutorial (corsi capostipite) nel backoffice Laravel tramite l'endpoint
    scoped GET /api/chatbot/tutorials/search?title=... (richiede un token con
    ability "tutorials:search", vedi comando artisan `chatbot:cognitor-token`).

    L'endpoint filtra solo sui tutorial capostipite (corsi); come arricchimento
    lato bot, cerchiamo la query anche tra le lezioni figlie già incluse nella
    risposta, per segnalare all'utente in quale corso si trova la lezione.

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili, si aspetta la chiave 'query' con il termine di ricerca

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    query = slots.get("query") or slots.get("TUTORIAL_QUERY")

    if not query:
        return {
            "response": "Cosa devo cercare tra i tutorial?",
            "slots": {},
            "metadata": {"operation": "search_tutorial", "query": None},
        }

    if not config.BACKEND_API_TOKEN:
        return {
            "response": "La ricerca tutorial non è ancora configurata (manca il token di accesso al backoffice).",
            "slots": {},
            "metadata": {"operation": "search_tutorial", "query": query, "error": "missing_token"},
        }

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/tutorials/search"

    try:
        resp = requests.get(
            url,
            params={"title": query, "per_page": _MAX_RESULTS},
            headers={
                "Authorization": f"Bearer {config.BACKEND_API_TOKEN}",
                "Accept": "application/json",
            },
            timeout=config.BACKEND_API_TIMEOUT,
        )
    except requests.RequestException as e:
        return {
            "response": "Non riesco a raggiungere il backoffice in questo momento. Riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "search_tutorial", "query": query, "error": str(e)},
        }

    if resp.status_code == 401:
        return {
            "response": "Non sono autorizzato a cercare tutorial (token non valido o scaduto).",
            "slots": {},
            "metadata": {"operation": "search_tutorial", "query": query, "error": "unauthorized"},
        }

    if not resp.ok:
        return {
            "response": "Il backoffice ha risposto con un errore, non sono riuscito a completare la ricerca.",
            "slots": {},
            "metadata": {"operation": "search_tutorial", "query": query, "error": f"http_{resp.status_code}"},
        }

    try:
        payload = resp.json()
    except ValueError:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "search_tutorial", "query": query, "error": "invalid_json"},
        }

    tutorials = payload if isinstance(payload, list) else payload.get("data", [])

    if not tutorials:
        return {
            "response": f"Non ho trovato tutorial che corrispondono a '{query}'.",
            "slots": {},
            "metadata": {"operation": "search_tutorial", "query": query, "results": []},
        }

    lines = [f"Ho trovato questi corsi per '{query}':\n"]
    query_lower = query.lower()
    for i, tutorial in enumerate(tutorials[:_MAX_RESULTS], 1):
        root_title = _title_of(tutorial)
        lines.append(f"{i}. {root_title} (id {tutorial.get('id')})")

        matching_lessons = [
            _title_of(child)
            for child in tutorial.get("childs", []) or []
            if query_lower in _title_of(child).lower()
        ]
        for lesson_title in matching_lessons[:_MAX_RESULTS]:
            lines.append(f"   - lezione: {lesson_title}")

    return {
        "response": "\n".join(lines),
        "slots": {},
        "metadata": {"operation": "search_tutorial", "query": query, "results": tutorials},
    }
