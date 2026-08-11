"""Cache del set di intent disabilitati da admin.

Il pannello impostazioni del package geco/chatbot (lato Laravel) espone quali
intent l'amministratore ha deciso di disattivare (GET /chatbot/disabled-intents,
stesso pattern già usato per company-info). Qui li recuperiamo e mettiamo in
cache con un TTL breve: interrogare Laravel ad ogni turno sarebbe inutile
latenza aggiuntiva per un dato che cambia raramente.
"""
import time

import requests

import config

_CACHE_TTL_SECONDS = 30
_cache = {"set": set(), "fetched_at": 0.0}


def get_disabled_intents() -> set:
    now = time.time()
    if now - _cache["fetched_at"] < _CACHE_TTL_SECONDS:
        return _cache["set"]

    base_url = (config.BACKEND_API_BASE_URL or "").rstrip("/")
    if not base_url:
        return _cache["set"]

    try:
        resp = requests.get(f"{base_url}/chatbot/disabled-intents", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        _cache["set"] = set(data.get("disabled") or [])
        _cache["fetched_at"] = now
    except requests.RequestException:
        # Laravel irraggiungibile: manteniamo l'ultimo valore noto (anche
        # vuoto) invece di bloccare la conversazione o disabilitare tutto.
        pass

    return _cache["set"]
