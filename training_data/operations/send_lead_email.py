"""Operation per l'intent ask_service.

Nome/recapito/messaggio non attivano una logica nuova da mantenere: vengono
inoltrati a POST {BACKEND_API_BASE_URL}/chatbot/lead, che riusa l'infrastruttura
email già esistente del form di contatto del sito (stesso modello Contact,
stesse mail di notifica) invece di duplicarla lato Cognitor.
"""
import requests

import config

_TIMEOUT = 5.0


def action_send_lead_email(intent_name: str = None, slots: dict = None) -> dict:
    slots = slots or {}
    name = slots.get("NOME")
    contact = slots.get("RECAPITO")
    message = slots.get("MESSAGGIO_RICHIESTA")

    base_url = (config.BACKEND_API_BASE_URL or "").rstrip("/")
    if not base_url:
        return {
            "response": "Ho preso nota, ma al momento non riesco a inoltrare la richiesta. Riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "send_lead_email", "error": "missing_base_url"},
        }

    try:
        resp = requests.post(
            f"{base_url}/chatbot/lead",
            json={"name": name, "contact": contact, "message": message},
            headers={"Accept": "application/json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return {
            "response": (
                "Ho preso nota della tua richiesta, ma non sono riuscito a "
                "inoltrarla subito: riprova più tardi o scrivici direttamente."
            ),
            "slots": {},
            "metadata": {"operation": "send_lead_email", "error": str(e)},
        }

    return {
        "response": f"Grazie {name}! Ho inoltrato la tua richiesta al nostro team, ti contatteranno a breve al recapito che mi hai dato.",
        "slots": {},
        "metadata": {"operation": "send_lead_email"},
    }
