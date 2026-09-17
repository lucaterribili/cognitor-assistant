"""
Operation per l'intent ask_capital.

Niente API esterna: le capitali non cambiano quasi mai e l'unica fonte
gratuita "senza registrazione" con questi dati (REST Countries) ha ritirato
le versioni pubbliche v1-v4 e richiede ora un account/API key per la v5
(verificato: https://restcountries.com/docs/countries/legacy-api-deprecation).
Per i ~20 paesi coperti dagli esempi di training una mappa statica è più
semplice, più veloce e non dipende dalla disponibilità di un servizio terzo.
"""
import unicodedata

# variante normalizzata -> (nome canonico da mostrare in risposta, capitale)
_CAPITALS_IT = {
    "francia": ("Francia", "Parigi"),
    "italia": ("Italia", "Roma"),
    "germania": ("Germania", "Berlino"),
    "giappone": ("Giappone", "Tokyo"),
    "brasile": ("Brasile", "Brasilia"),
    "ucraina": ("Ucraina", "Kiev"),
    "cina": ("Cina", "Pechino"),
    "cinese": ("Cina", "Pechino"),
    "repubblica popolare cinese": ("Cina", "Pechino"),
    "spagna": ("Spagna", "Madrid"),
    "regno unito": ("Regno Unito", "Londra"),
    "inghilterra": ("Regno Unito", "Londra"),
    "portogallo": ("Portogallo", "Lisbona"),
    "stati uniti": ("Stati Uniti", "Washington D.C."),
    "usa": ("Stati Uniti", "Washington D.C."),
    "australia": ("Australia", "Canberra"),
    "canada": ("Canada", "Ottawa"),
    "russia": ("Russia", "Mosca"),
    "india": ("India", "New Delhi"),
    "messico": ("Messico", "Città del Messico"),
    "egitto": ("Egitto", "Il Cairo"),
    "grecia": ("Grecia", "Atene"),
    "argentina": ("Argentina", "Buenos Aires"),
    "svizzera": ("Svizzera", "Berna"),
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def action_ask_capital(intent_name: str = None, slots: dict = None) -> dict:
    slots = slots or {}
    country = slots.get("LOCATION")

    if not country:
        return {
            "response": "Di quale paese vuoi sapere la capitale?",
            "slots": {},
            "metadata": {"operation": "ask_capital", "error": "missing_country"},
        }

    match = _CAPITALS_IT.get(_normalize(country))
    if not match:
        return {
            "response": f"Non ho la capitale di '{country}' a disposizione, mi dispiace.",
            "slots": {},
            "metadata": {"operation": "ask_capital", "error": "country_not_found", "country": country},
        }

    canonical_name, capital = match
    return {
        "response": f"La capitale di {canonical_name} è {capital}.",
        "slots": {},
        "metadata": {"operation": "ask_capital", "country": canonical_name, "capital": capital},
    }
