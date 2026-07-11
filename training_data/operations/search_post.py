"""Operation per l'intent search_post — cerca articoli nel backoffice Laravel (Programmato)."""
import requests

import config

_MAX_RESULTS = 5


def action_search_post(intent_name: str, slots: dict = None) -> dict:
    """
    Cerca articoli (Post) nel backoffice Laravel tramite l'endpoint scoped
    GET /api/chatbot/posts?title=... (richiede un token con ability "posts:read",
    vedi comando artisan `chatbot:cognitor-token`).

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili, si aspetta la chiave 'query' con il termine di ricerca

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    query = slots.get("query") or slots.get("POST_QUERY")

    if not query:
        return {
            "response": "Cosa devo cercare tra gli articoli?",
            "slots": {},
            "metadata": {"operation": "search_post", "query": None},
        }

    if not config.BACKEND_API_TOKEN:
        return {
            "response": "La ricerca articoli non è ancora configurata (manca il token di accesso al backoffice).",
            "slots": {},
            "metadata": {"operation": "search_post", "query": query, "error": "missing_token"},
        }

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/posts"

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
            "metadata": {"operation": "search_post", "query": query, "error": str(e)},
        }

    if resp.status_code == 401:
        return {
            "response": "Non sono autorizzato a cercare articoli (token non valido o scaduto).",
            "slots": {},
            "metadata": {"operation": "search_post", "query": query, "error": "unauthorized"},
        }

    if not resp.ok:
        return {
            "response": "Il backoffice ha risposto con un errore, non sono riuscito a completare la ricerca.",
            "slots": {},
            "metadata": {"operation": "search_post", "query": query, "error": f"http_{resp.status_code}"},
        }

    try:
        payload = resp.json()
    except ValueError:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "search_post", "query": query, "error": "invalid_json"},
        }

    # L'endpoint Laravel risponde con l'array dei post "grezzo" (non un envelope {"data": [...]}).
    posts = payload if isinstance(payload, list) else payload.get("data", [])

    if not posts:
        return {
            "response": f"Non ho trovato articoli che corrispondono a '{query}'.",
            "slots": {},
            "metadata": {"operation": "search_post", "query": query, "results": []},
        }

    lines = [f"Ho trovato questi articoli per '{query}':\n"]
    for i, post in enumerate(posts[:_MAX_RESULTS], 1):
        title = post.get("title") or "(senza titolo)"
        if isinstance(title, dict):
            title = title.get("it") or next(iter(title.values()), "(senza titolo)")
        lines.append(f"{i}. {title} (id {post.get('id')})")

    return {
        "response": "\n".join(lines),
        "slots": {},
        "metadata": {"operation": "search_post", "query": query, "results": posts},
    }
