"""Operation per l'intent ask_public_holiday (Nager.Date, nessuna API key)."""
import unicodedata

import requests

_TIMEOUT = 5
_MAX_HOLIDAYS_SHOWN = 5

# Nomi paese in italiano (normalizzati) -> codice ISO 3166-1 alpha-2 richiesto
# da Nager.Date. Copre i paesi usati negli esempi di training; se il paese
# richiesto non è qui, si ricade sull'Italia (il dominio più probabile per
# un assistente in italiano) invece di rifiutare la richiesta.
_COUNTRY_CODES = {
    "italia": "IT",
    "francia": "FR",
    "germania": "DE",
    "spagna": "ES",
    "stati uniti": "US",
    "usa": "US",
    "regno unito": "GB",
    "inghilterra": "GB",
    "portogallo": "PT",
    "svizzera": "CH",
    "austria": "AT",
    "belgio": "BE",
    "olanda": "NL",
    "paesi bassi": "NL",
    "grecia": "GR",
    "polonia": "PL",
    "irlanda": "IE",
    "canada": "CA",
    "australia": "AU",
    "brasile": "BR",
}

_DEFAULT_COUNTRY_CODE = "IT"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _country_code(location: str | None) -> str:
    if not location:
        return _DEFAULT_COUNTRY_CODE
    return _COUNTRY_CODES.get(_normalize(location), _DEFAULT_COUNTRY_CODE)


def _format_holiday_list(holidays: list[dict]) -> str:
    lines = [f"- {h['date']}: {h['localName']}" for h in holidays[:_MAX_HOLIDAYS_SHOWN]]
    return "\n".join(lines)


def action_ask_public_holiday(intent_name: str = None, slots: dict = None) -> dict:
    slots = slots or {}
    country_code = _country_code(slots.get("LOCATION"))
    year_raw = slots.get("NUMBER")

    try:
        if year_raw:
            year = int(str(year_raw).strip())
            url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
        else:
            year = None
            url = f"https://date.nager.at/api/v3/NextPublicHolidays/{country_code}"

        response = requests.get(url, timeout=_TIMEOUT)
        response.raise_for_status()
        holidays = response.json()
    except (requests.RequestException, ValueError) as e:
        return {
            "response": "Il servizio delle festività non è raggiungibile in questo momento, riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "ask_public_holiday", "error": str(e)},
        }

    if not holidays:
        return {
            "response": "Non ho trovato festività per questa richiesta.",
            "slots": {},
            "metadata": {"operation": "ask_public_holiday", "country_code": country_code, "year": year},
        }

    if year is None:
        text = "Le prossime festività sono:\n" + _format_holiday_list(holidays)
    else:
        text = f"Le festività del {year} sono:\n" + _format_holiday_list(holidays)
        if len(holidays) > _MAX_HOLIDAYS_SHOWN:
            text += f"\n... e altre {len(holidays) - _MAX_HOLIDAYS_SHOWN}."

    return {
        "response": text,
        "slots": {},
        "metadata": {
            "operation": "ask_public_holiday",
            "country_code": country_code,
            "year": year,
            "count": len(holidays),
        },
    }
