"""Operation per l'intent publish_draft_content — pubblica tutti i Post in bozza di un
dominio nel backoffice Laravel (Programmato). Azione con effetto live: richiede una
conferma esplicita (slot 'confirm') prima di chiamare l'endpoint di pubblicazione."""
import requests

import config

_AFFIRMATIVE = {"si", "sì", "yes", "ok", "va bene", "d'accordo", "confermo", "procedi", "certo", "conferma"}


def _is_affirmative(value: str) -> bool:
    return str(value or "").strip().lower() in _AFFIRMATIVE


def action_publish_draft_content(intent_name: str, slots: dict = None) -> dict:
    """
    Pubblica i contenuti in bozza di un dominio tramite l'endpoint scoped
    POST /api/chatbot/posts/publish (richiede l'ability "posts:publish", vedi
    comando artisan `chatbot:cognitor-token`), solo se lo slot 'confirm' contiene
    una risposta affermativa.

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili — 'domain', 'confirm' (testo raw sì/no)

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    domain = slots.get("domain") or slots.get("DOMAIN_NAME")
    confirm = slots.get("confirm") or slots.get("CONFIRMATION")

    if not domain:
        return {
            "response": "Per quale dominio devo pubblicare i contenuti in bozza?",
            "slots": {},
            "metadata": {"operation": "publish_draft_content", "domain": domain},
        }

    if not _is_affirmative(confirm):
        return {
            "response": "Operazione annullata, non ho pubblicato nulla.",
            "slots": {},
            "metadata": {"operation": "publish_draft_content", "domain": domain, "confirmed": False},
        }

    if not config.BACKEND_API_TOKEN:
        return {
            "response": "La pubblicazione dei contenuti non è ancora configurata (manca il token di accesso al backoffice).",
            "slots": {},
            "metadata": {"operation": "publish_draft_content", "error": "missing_token"},
        }

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/posts/publish"

    try:
        resp = requests.post(
            url,
            json={"domain": domain},
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
            "metadata": {"operation": "publish_draft_content", "error": str(e)},
        }

    if resp.status_code == 401:
        return {
            "response": "Non sono autorizzato a pubblicare i contenuti (token non valido o scaduto).",
            "slots": {},
            "metadata": {"operation": "publish_draft_content", "error": "unauthorized"},
        }

    if resp.status_code == 422:
        try:
            message = resp.json().get("message", "Richiesta non valida.")
        except ValueError:
            message = "Richiesta non valida."
        return {
            "response": f"Non sono riuscito a pubblicare i contenuti: {message}",
            "slots": {},
            "metadata": {"operation": "publish_draft_content", "error": "validation", "message": message},
        }

    if not resp.ok:
        return {
            "response": "Il backoffice ha risposto con un errore, non sono riuscito a pubblicare i contenuti.",
            "slots": {},
            "metadata": {"operation": "publish_draft_content", "error": f"http_{resp.status_code}"},
        }

    try:
        data = resp.json()
    except ValueError:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "publish_draft_content", "error": "invalid_json"},
        }

    count = data.get("count", 0)
    domain_name = data.get("domain", domain)

    if count == 0:
        response_text = f"Non ho trovato contenuti in bozza da pubblicare per il dominio \"{domain_name}\"."
    else:
        response_text = f"Ho pubblicato {count} contenuti in bozza per il dominio \"{domain_name}\"."

    return {
        "response": response_text,
        "slots": {},
        "metadata": {"operation": "publish_draft_content", "domain": domain_name, "count": count, "confirmed": True},
    }
