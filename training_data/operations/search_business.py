"""Operation per l'intent search_business — cerca attività (Business) nel backoffice Laravel (Programmato)."""
import requests

import config

_MAX_RESULTS = 5


def action_search_business(intent_name: str, slots: dict = None) -> dict:
    """
    Cerca attività (Business) nel backoffice Laravel tramite l'endpoint scoped
    GET /api/chatbot/businesses?title=... (richiede un token con ability
    "businesses:read", vedi comando artisan `chatbot:cognitor-token`).

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili, si aspetta la chiave 'query' con il nome da cercare

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    query = slots.get("query") or slots.get("BUSINESS_QUERY")

    if not query:
        return {
            "response": "Come si chiama l'attività da cercare?",
            "slots": {},
            "metadata": {"operation": "search_business", "query": None},
        }

    if not config.BACKEND_API_TOKEN:
        return {
            "response": "La ricerca attività non è ancora configurata (manca il token di accesso al backoffice).",
            "slots": {},
            "metadata": {"operation": "search_business", "query": query, "error": "missing_token"},
        }

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/businesses"

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
            "metadata": {"operation": "search_business", "query": query, "error": str(e)},
        }

    if resp.status_code == 401:
        return {
            "response": "Non sono autorizzato a cercare attività (token non valido o scaduto).",
            "slots": {},
            "metadata": {"operation": "search_business", "query": query, "error": "unauthorized"},
        }

    if not resp.ok:
        return {
            "response": "Il backoffice ha risposto con un errore, non sono riuscito a completare la ricerca.",
            "slots": {},
            "metadata": {"operation": "search_business", "query": query, "error": f"http_{resp.status_code}"},
        }

    try:
        payload = resp.json()
    except ValueError:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "search_business", "query": query, "error": "invalid_json"},
        }

    businesses = payload if isinstance(payload, list) else payload.get("data", [])

    if not businesses:
        return {
            "response": f"Non ho trovato attività che corrispondono a '{query}'.",
            "slots": {},
            "metadata": {"operation": "search_business", "query": query, "results": []},
        }

    lines = [f"Ho trovato queste attività per '{query}':\n"]
    for i, business in enumerate(businesses[:_MAX_RESULTS], 1):
        title = business.get("title") or "(senza nome)"
        city = business.get("city")
        sectors = business.get("sectors") or []
        details = ", ".join(filter(None, [city, ", ".join(sectors) if sectors else None]))
        suffix = f" ({details})" if details else ""
        lines.append(f"{i}. {title}{suffix}")

    return {
        "response": "\n".join(lines),
        "slots": {},
        "metadata": {"operation": "search_business", "query": query, "results": businesses},
    }
