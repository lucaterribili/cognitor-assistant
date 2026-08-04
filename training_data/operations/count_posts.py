"""Operation per l'intent count_posts — conta gli articoli pubblicati nel backoffice Laravel (Programmato)."""
import requests

import config


def action_count_posts(intent_name: str, slots: dict = None) -> dict:
    """
    Conta gli articoli pubblicati (Post con status "published") nel backoffice
    Laravel tramite l'endpoint scoped GET /api/chatbot/posts/count (richiede un
    token con ability "posts:read", la stessa già usata da search_post — vedi
    comando artisan `chatbot:cognitor-token`).

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili (non usati qui)

    Returns:
        dict con la risposta
    """
    if not config.BACKEND_API_TOKEN:
        return {
            "response": "Il conteggio degli articoli non è ancora configurato (manca il token di accesso al backoffice).",
            "slots": {},
            "metadata": {"operation": "count_posts", "error": "missing_token"},
        }

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/posts/count"

    try:
        resp = requests.get(
            url,
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
            "metadata": {"operation": "count_posts", "error": str(e)},
        }

    if resp.status_code == 401:
        return {
            "response": "Non sono autorizzato a contare gli articoli (token non valido o scaduto).",
            "slots": {},
            "metadata": {"operation": "count_posts", "error": "unauthorized"},
        }

    if not resp.ok:
        return {
            "response": "Il backoffice ha risposto con un errore, non sono riuscito a contare gli articoli.",
            "slots": {},
            "metadata": {"operation": "count_posts", "error": f"http_{resp.status_code}"},
        }

    try:
        payload = resp.json()
    except ValueError:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "count_posts", "error": "invalid_json"},
        }

    count = payload.get("count") if isinstance(payload, dict) else None
    if count is None:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "count_posts", "error": "missing_count"},
        }

    label = "articolo pubblicato" if count == 1 else "articoli pubblicati"
    return {
        "response": f"Ci sono {count} {label}.",
        "slots": {},
        "metadata": {"operation": "count_posts", "count": count},
    }
