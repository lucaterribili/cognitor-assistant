"""Operation per l'intent ask_food_nutrition (Open Food Facts, nessuna API key).

Usa search.openfoodfacts.org (il nuovo motore di ricerca "search-a-licious"),
non il vecchio endpoint cgi/search.pl: quello vecchio ha dato risultati
irrilevanti in test diretti (es. "nutella" -> un formaggio fresco) e in
alcuni casi un 503 "not available to anonymous users" anche a parità di
parametri - il nuovo endpoint è più affidabile e ha rilevanza testuale
molto migliore.
"""
import time

import requests

_TIMEOUT = 5
_SEARCH_URL = "https://search.openfoodfacts.org/search"
# Open Food Facts chiede un User-Agent descrittivo per ogni richiesta
# (policy pubblica): senza, alcune risposte arrivano vuote o rifiutate.
_HEADERS = {"User-Agent": "CognitorAssistant/1.0 (chatbot NLU demo)"}

# Anche il nuovo endpoint può rispondere con un errore transitorio sotto
# carico: un paio di retry bastano a coprire i casi passeggeri senza
# introdurre una latenza percepibile in caso di fallimento reale.
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 1.0


def _format_value(value, unit: str) -> str:
    if value is None:
        return "n/d"
    try:
        return f"{float(value):.1f} {unit}"
    except (TypeError, ValueError):
        return "n/d"


def action_ask_food_nutrition(intent_name: str = None, slots: dict = None) -> dict:
    slots = slots or {}
    food_name = slots.get("query")

    if not food_name:
        return {
            "response": "Di quale alimento vuoi conoscere i valori nutrizionali?",
            "slots": {},
            "metadata": {"operation": "ask_food_nutrition", "error": "missing_food"},
        }

    hits = None
    last_error = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                _SEARCH_URL,
                params={
                    "q": food_name,
                    "page_size": 1,
                    "fields": "product_name,nutriments,nutriscore_grade",
                },
                headers=_HEADERS,
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            hits = response.json().get("hits", [])
            break
        except requests.RequestException as e:
            last_error = e
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)

    if hits is None:
        return {
            "response": "Il database alimentare non è raggiungibile in questo momento, riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "ask_food_nutrition", "error": str(last_error)},
        }

    if not hits:
        return {
            "response": f"Non ho trovato informazioni nutrizionali per '{food_name}'.",
            "slots": {},
            "metadata": {"operation": "ask_food_nutrition", "error": "not_found", "food": food_name},
        }

    product = hits[0]
    nutriments = product.get("nutriments") or {}
    product_name = product.get("product_name") or food_name.strip().title()
    nutriscore = (product.get("nutriscore_grade") or "").upper()
    nutriscore = nutriscore if nutriscore in {"A", "B", "C", "D", "E"} else None

    text = (
        f"Per 100g di {product_name}: "
        f"{_format_value(nutriments.get('energy-kcal_100g'), 'kcal')}, "
        f"proteine {_format_value(nutriments.get('proteins_100g'), 'g')}, "
        f"carboidrati {_format_value(nutriments.get('carbohydrates_100g'), 'g')}, "
        f"grassi {_format_value(nutriments.get('fat_100g'), 'g')}."
    )
    if nutriscore:
        text += f" Nutri-Score: {nutriscore}."

    return {
        "response": text,
        "slots": {},
        "metadata": {
            "operation": "ask_food_nutrition",
            "product_name": product_name,
            "nutriments": nutriments,
            "nutriscore": nutriscore,
        },
    }
