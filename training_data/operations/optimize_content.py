"""Operation per l'intent optimize_content — invia in ottimizzazione AI (task 'optimize'
verso Friday Agent) tutti i Post in bozza di un dominio, nel backoffice Laravel (Programmato)."""
import requests

import config


def action_optimize_content(intent_name: str, slots: dict = None) -> dict:
    """
    Ottimizza i contenuti in bozza di un dominio tramite l'endpoint scoped
    POST /api/chatbot/content/optimize (richiede l'ability "content:optimize", vedi
    comando artisan `chatbot:cognitor-token`).

    Il risultato dell'ottimizzazione viene salvato lato Laravel come Revision da
    rivedere manualmente (non pubblica né sovrascrive il contenuto esistente).

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili — 'domain'

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    domain = slots.get("domain") or slots.get("DOMAIN_NAME")

    if not domain:
        return {
            "response": "Per quale dominio devo ottimizzare i contenuti in bozza?",
            "slots": {},
            "metadata": {"operation": "optimize_content", "domain": domain},
        }

    if not config.BACKEND_API_TOKEN:
        return {
            "response": "L'ottimizzazione dei contenuti non è ancora configurata (manca il token di accesso al backoffice).",
            "slots": {},
            "metadata": {"operation": "optimize_content", "error": "missing_token"},
        }

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/content/optimize"

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
            "metadata": {"operation": "optimize_content", "error": str(e)},
        }

    if resp.status_code == 401:
        return {
            "response": "Non sono autorizzato a ottimizzare i contenuti (token non valido o scaduto).",
            "slots": {},
            "metadata": {"operation": "optimize_content", "error": "unauthorized"},
        }

    if resp.status_code == 422:
        try:
            message = resp.json().get("message", "Richiesta non valida.")
        except ValueError:
            message = "Richiesta non valida."
        return {
            "response": f"Non sono riuscito a ottimizzare i contenuti: {message}",
            "slots": {},
            "metadata": {"operation": "optimize_content", "error": "validation", "message": message},
        }

    if not resp.ok:
        return {
            "response": "Il backoffice ha risposto con un errore, non sono riuscito ad avviare l'ottimizzazione.",
            "slots": {},
            "metadata": {"operation": "optimize_content", "error": f"http_{resp.status_code}"},
        }

    try:
        data = resp.json()
    except ValueError:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "optimize_content", "error": "invalid_json"},
        }

    count = data.get("count", 0)
    domain_name = data.get("domain", domain)

    if count == 0:
        response_text = f"Non ho trovato contenuti in bozza da ottimizzare per il dominio \"{domain_name}\"."
    else:
        response_text = (
            f"Ho avviato l'ottimizzazione di {count} contenuti in bozza per il dominio \"{domain_name}\". "
            "I risultati saranno pronti in revisione a breve."
        )

    return {
        "response": response_text,
        "slots": {},
        "metadata": {"operation": "optimize_content", "domain": domain_name, "count": count},
    }
