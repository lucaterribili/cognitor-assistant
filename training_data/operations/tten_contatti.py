"""Operation per l'intent ask_tten_contatti.

Indirizzo/email/PEC non sono nel training dataset: cambiano raramente ma,
quando succede, sono dati "ufficiali" che vivono lato Laravel (pannello
impostazioni del package geco/chatbot) - qui li recuperiamo dal vivo via
GET /chatbot/company-info invece di duplicarli in una response statica
che andrebbe disallineata a ogni modifica lato sito.
"""
import requests

import config

_TIMEOUT = 5.0


def action_tten_contatti(intent_name: str = None, slots: dict = None) -> dict:
    """
    Recupera indirizzo/email/PEC da GET {BACKEND_API_BASE_URL}/chatbot/company-info
    (endpoint pubblico del sito TTen) e compone la risposta.

    Args:
        intent_name: Nome dell'intent che ha attivato l'operation
        slots: Slot disponibili (non usati qui)

    Returns:
        dict con la risposta e metadati
    """
    base_url = (config.BACKEND_API_BASE_URL or "").rstrip("/")
    if not base_url:
        return {
            "response": "Non riesco a recuperare i contatti in questo momento. Riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "tten_contatti", "error": "missing_base_url"},
        }

    url = f"{base_url}/chatbot/company-info"

    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {
            "response": "Non riesco a recuperare i contatti in questo momento. Riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "tten_contatti", "error": str(e)},
        }

    address = data.get("address")
    email = data.get("email")
    pec = data.get("pec")
    phone = data.get("phone")

    if not (address or email or pec or phone):
        return {
            "response": "Non riesco a recuperare i contatti in questo momento. Riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "tten_contatti", "error": "empty_response"},
        }

    parts = []
    if address:
        parts.append(f"Ci trovi in {address}.")
    if phone:
        parts.append(f"Ci puoi chiamare al {phone}.")
    if email:
        email_sentence = f"Puoi scriverci a {email}"
        email_sentence += f", oppure via PEC a {pec}." if pec else "."
        parts.append(email_sentence)
    elif pec:
        parts.append(f"Puoi scriverci via PEC a {pec}.")

    response = " ".join(parts)

    return {
        "response": response,
        "slots": {},
        "metadata": {"operation": "tten_contatti", "source": url},
    }
