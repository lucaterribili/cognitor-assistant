"""Provider di opzioni dinamiche (bottoni) per gli slot domain/category/tutorial
dell'intent create_content — dati letti dal backoffice Laravel (Programmato).

Un fallimento di rete/autenticazione qui non deve mai bloccare la conversazione: in
quel caso si ritorna una lista vuota e il bot ricade sulla normale domanda in testo
libero (vedi OperationManager.get_options)."""
import requests

import config


def _get(endpoint: str, params: dict = None) -> list:
    if not config.BACKEND_API_TOKEN:
        return []

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/{endpoint}"
    resp = requests.get(
        url,
        params=params or {},
        headers={
            "Authorization": f"Bearer {config.BACKEND_API_TOKEN}",
            "Accept": "application/json",
        },
        timeout=config.BACKEND_API_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, list) else payload.get("data", [])


def options_domains(slots: dict = None) -> list:
    """Domini disponibili, GET /chatbot/domains."""
    items = _get("domains")
    return [{"value": item["name"], "label": item["name"]} for item in items if item.get("name")]


def options_categories(slots: dict = None) -> list:
    """Categorie del dominio già raccolto, GET /chatbot/categories?domain=<nome>."""
    slots = slots or {}
    domain = slots.get("domain") or slots.get("DOMAIN_NAME")
    if not domain:
        return []
    items = _get("categories", {"domain": domain})
    return [{"value": item["title"], "label": item["title"]} for item in items if item.get("title")]


def options_tutorials(slots: dict = None) -> list:
    """Tutorial capostipiti del dominio già raccolto, GET /chatbot/tutorials?domain=<nome>."""
    slots = slots or {}
    domain = slots.get("domain") or slots.get("DOMAIN_NAME")
    if not domain:
        return []
    items = _get("tutorials", {"domain": domain})
    return [{"value": item["title"], "label": item["title"]} for item in items if item.get("title")]
