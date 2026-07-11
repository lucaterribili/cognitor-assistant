"""Operation per l'intent create_content — genera articoli o tutorial nel backoffice
Laravel (Programmato), a seconda dello slot content_type."""
import requests

import config


def action_create_content(intent_name: str, slots: dict = None) -> dict:
    """
    Crea una Proposal nel backoffice Laravel tramite l'endpoint scoped
    POST /api/chatbot/articles/generate (task 'generate-article', richiede l'ability
    "articles:generate") o POST /api/chatbot/tutorials/generate (task
    'generate-tutorial', richiede l'ability "tutorials:generate"), a seconda del valore
    dello slot 'content_type'. Vedi il comando artisan `chatbot:cognitor-token` per le
    ability del token di servizio.

    La generazione vera e propria è asincrona: Laravel prende in carico la richiesta
    (rabbit:preprocess ogni 5 minuti) e un worker esterno genera il contenuto.

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili — 'content_type' ('article'|'tutorial'), 'domain',
               'category' (solo per article), 'tutorial' (solo per tutorial), 'titles'

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    content_type = (slots.get("content_type") or slots.get("CONTENT_TYPE") or "article").lower()
    domain = slots.get("domain") or slots.get("DOMAIN_NAME")
    titles = slots.get("titles") or slots.get("ARTICLE_TITLES") or []

    if isinstance(titles, str):
        titles = [t.strip() for t in titles.split(",") if t.strip()]

    if content_type == "tutorial":
        endpoint = "tutorials/generate"
        content_slot = slots.get("tutorial") or slots.get("TUTORIAL_NAME")
        content_key = "tutorial"
        label = "tutorial"
    else:
        endpoint = "articles/generate"
        content_slot = slots.get("category") or slots.get("CATEGORY_NAME")
        content_key = "category"
        label = "articoli"

    if not domain or not content_slot or not titles:
        return {
            "response": f"Mi mancano ancora delle informazioni per creare {label}.",
            "slots": {},
            "metadata": {
                "operation": "create_content",
                "content_type": content_type,
                "domain": domain,
                content_key: content_slot,
                "titles": titles,
            },
        }

    if not config.BACKEND_API_TOKEN:
        return {
            "response": f"La creazione {label} non è ancora configurata (manca il token di accesso al backoffice).",
            "slots": {},
            "metadata": {"operation": "create_content", "error": "missing_token"},
        }

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/{endpoint}"

    try:
        resp = requests.post(
            url,
            json={"domain": domain, content_key: content_slot, "titles": titles},
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
            "metadata": {"operation": "create_content", "error": str(e)},
        }

    if resp.status_code == 401:
        return {
            "response": f"Non sono autorizzato a creare {label} (token non valido o scaduto).",
            "slots": {},
            "metadata": {"operation": "create_content", "error": "unauthorized"},
        }

    if resp.status_code == 422:
        try:
            message = resp.json().get("message", "Richiesta non valida.")
        except ValueError:
            message = "Richiesta non valida."
        return {
            "response": f"Non sono riuscito a creare {label}: {message}",
            "slots": {},
            "metadata": {"operation": "create_content", "error": "validation", "message": message},
        }

    if not resp.ok:
        return {
            "response": f"Il backoffice ha risposto con un errore, non sono riuscito a creare {label}.",
            "slots": {},
            "metadata": {"operation": "create_content", "error": f"http_{resp.status_code}"},
        }

    try:
        data = resp.json()
    except ValueError:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "create_content", "error": "invalid_json"},
        }

    count = data.get("count", len(titles))
    domain_name = data.get("domain", domain)
    content_name = data.get(content_key, content_slot)

    if content_type == "tutorial":
        response_text = (
            f"Ho avviato la generazione di {count} tutorial per il dominio \"{domain_name}\", "
            f"capostipite \"{content_name}\". Saranno pronti in qualche minuto."
        )
    else:
        response_text = (
            f"Ho avviato la generazione di {count} articoli per il dominio \"{domain_name}\", "
            f"categoria \"{content_name}\". Saranno pronti in qualche minuto."
        )

    return {
        "response": response_text,
        "slots": {},
        "metadata": {"operation": "create_content", "proposal_id": data.get("proposal_id"), "count": count},
    }
