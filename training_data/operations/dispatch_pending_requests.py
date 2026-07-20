"""Operation per l'intent dispatch_pending_requests — invia subito all'assistente
(Friday Agent, via RabbitMQ) tutte le Proposal in stato 'pending' nel backoffice
Laravel (Programmato), trigger on-demand dello stesso job schedulato rabbit:preprocess."""
import requests

import config


def action_dispatch_pending_requests(intent_name: str, slots: dict = None) -> dict:
    """
    Invia le richieste in sospeso tramite l'endpoint scoped
    POST /api/chatbot/proposals/dispatch (richiede l'ability "proposals:dispatch",
    vedi comando artisan `chatbot:cognitor-token`).

    Returns:
        dict con la risposta
    """
    if not config.BACKEND_API_TOKEN:
        return {
            "response": "L'invio delle richieste in sospeso non è ancora configurato (manca il token di accesso al backoffice).",
            "slots": {},
            "metadata": {"operation": "dispatch_pending_requests", "error": "missing_token"},
        }

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/proposals/dispatch"

    try:
        resp = requests.post(
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
            "metadata": {"operation": "dispatch_pending_requests", "error": str(e)},
        }

    if resp.status_code == 401:
        return {
            "response": "Non sono autorizzato a inviare le richieste in sospeso (token non valido o scaduto).",
            "slots": {},
            "metadata": {"operation": "dispatch_pending_requests", "error": "unauthorized"},
        }

    if not resp.ok:
        return {
            "response": "Il backoffice ha risposto con un errore, non sono riuscito a inviare le richieste.",
            "slots": {},
            "metadata": {"operation": "dispatch_pending_requests", "error": f"http_{resp.status_code}"},
        }

    try:
        data = resp.json()
    except ValueError:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "dispatch_pending_requests", "error": "invalid_json"},
        }

    count = data.get("count", 0)

    if count == 0:
        response_text = "Non c'erano richieste in sospeso da inviare."
    else:
        response_text = f"Ho inviato {count} richieste in sospeso all'assistente."

    return {
        "response": response_text,
        "slots": {},
        "metadata": {"operation": "dispatch_pending_requests", "count": count},
    }
