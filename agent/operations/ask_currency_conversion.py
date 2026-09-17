"""Operation per l'intent ask_currency_conversion (Frankfurter, nessuna API key)."""
import unicodedata

import requests

_TIMEOUT = 5
_API_URL = "https://api.frankfurter.dev/v1/latest"

# Nomi comuni in italiano (normalizzati: minuscolo, senza accenti) -> codice
# ISO 4217. Frankfurter copre le valute dei paesi UE + le principali valute
# mondiali (non le criptovalute): https://frankfurter.dev
_CURRENCY_CODES = {
    "euro": "EUR",
    "euri": "EUR",
    "dollaro": "USD",
    "dollari": "USD",
    "dollaro americano": "USD",
    "dollari americani": "USD",
    "sterlina": "GBP",
    "sterline": "GBP",
    "sterlina inglese": "GBP",
    "yen": "JPY",
    "yen giapponese": "JPY",
    "franco svizzero": "CHF",
    "franchi svizzeri": "CHF",
    "rublo": "RUB",
    "rubli": "RUB",
    "yuan": "CNY",
    "renminbi": "CNY",
    "dollaro canadese": "CAD",
    "dollari canadesi": "CAD",
    "dollaro australiano": "AUD",
    "dollari australiani": "AUD",
    "corona svedese": "SEK",
    "corona norvegese": "NOK",
    "corona danese": "DKK",
    "zloty": "PLN",
    "real brasiliano": "BRL",
    "rupia indiana": "INR",
    "won": "KRW",
    "won sudcoreano": "KRW",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _to_code(currency_name: str) -> str | None:
    return _CURRENCY_CODES.get(_normalize(currency_name))


def _parse_amount(raw) -> float:
    if not raw:
        return 1.0
    try:
        return float(str(raw).replace(",", "."))
    except ValueError:
        return 1.0


def action_ask_currency_conversion(intent_name: str = None, slots: dict = None) -> dict:
    slots = slots or {}
    from_name = slots.get("CURRENCY_FROM")
    to_name = slots.get("CURRENCY_TO")
    amount = _parse_amount(slots.get("NUMBER"))

    if not from_name or not to_name:
        return {
            "response": "Quali due valute vuoi convertire? Es. \"100 euro in dollari\".",
            "slots": {},
            "metadata": {"operation": "ask_currency_conversion", "error": "missing_currency"},
        }

    from_code = _to_code(from_name)
    to_code = _to_code(to_name)
    if not from_code or not to_code:
        unknown = from_name if not from_code else to_name
        return {
            "response": f"Non conosco la valuta '{unknown}', mi dispiace.",
            "slots": {},
            "metadata": {"operation": "ask_currency_conversion", "error": "unknown_currency"},
        }

    try:
        response = requests.get(
            _API_URL,
            params={"amount": amount, "base": from_code, "symbols": to_code},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        converted = data["rates"][to_code]
    except (requests.RequestException, KeyError) as e:
        return {
            "response": "Il servizio di cambio valuta non è raggiungibile in questo momento, riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "ask_currency_conversion", "error": str(e)},
        }

    amount_label = int(amount) if amount == int(amount) else amount
    converted_label = round(converted, 2)
    text = f"{amount_label} {from_code} equivalgono a {converted_label} {to_code}."

    return {
        "response": text,
        "slots": {},
        "metadata": {
            "operation": "ask_currency_conversion",
            "amount": amount,
            "from": from_code,
            "to": to_code,
            "converted": converted,
        },
    }
