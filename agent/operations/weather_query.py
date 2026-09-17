"""Operation per l'intent weather_query (Open-Meteo, nessuna API key)."""
import requests

from agent.operations.tools.geocoding import Geocoding

_TIMEOUT = 5

# Codici WMO (weather_code di Open-Meteo) -> descrizione italiana. Raggruppati
# per fasce, non uno a uno: la scelta della frase esatta non è mai critica
# quanto la temperatura.
_WEATHER_CODE_IT = {
    0: "cielo sereno",
    1: "poco nuvoloso",
    2: "parzialmente nuvoloso",
    3: "nuvoloso",
    45: "nebbia",
    48: "nebbia con brina",
    51: "pioviggine leggera",
    53: "pioviggine moderata",
    55: "pioviggine intensa",
    56: "pioviggine gelata",
    57: "pioviggine gelata intensa",
    61: "pioggia leggera",
    63: "pioggia moderata",
    65: "pioggia forte",
    66: "pioggia gelata",
    67: "pioggia gelata intensa",
    71: "neve leggera",
    73: "neve moderata",
    75: "neve forte",
    77: "granelli di neve",
    80: "rovesci di pioggia leggeri",
    81: "rovesci di pioggia moderati",
    82: "rovesci di pioggia violenti",
    85: "rovesci di neve leggeri",
    86: "rovesci di neve forti",
    95: "temporale",
    96: "temporale con grandine leggera",
    99: "temporale con grandine forte",
}

# Espressioni relative usate negli esempi di training (DATE): offset in
# giorni da oggi rispetto a quanto restituito da Open-Meteo (daily.time[0] è
# sempre oggi). Qualsiasi altra espressione non riconosciuta resta su "oggi"
# (offset 0) piuttosto che rifiutare la richiesta.
_DAY_OFFSETS = {
    "oggi": 0,
    "domani": 1,
    "dopodomani": 2,
}


def _weather_description(code) -> str:
    try:
        return _WEATHER_CODE_IT.get(int(code), "condizioni non determinate")
    except (TypeError, ValueError):
        return "condizioni non determinate"


def action_weather_query(intent_name: str = None, slots: dict = None) -> dict:
    """
    Recupera meteo attuale o previsione (in base allo slot DATE) per la
    località richiesta, geocodificata con lo stesso tool già usato da
    location_query (Nominatim).
    """
    slots = slots or {}
    location_name = slots.get("LOCATION")
    date_expr = str(slots.get("DATE") or "").strip().lower()

    if not location_name:
        return {
            "response": "Per quale città vuoi le previsioni?",
            "slots": {},
            "metadata": {"operation": "weather_query", "error": "missing_location"},
        }

    geocoding = Geocoding()
    place = geocoding.get_location(place=location_name)
    if not place or place.get("lat") is None:
        return {
            "response": f"Non sono riuscito a trovare '{location_name}'. Puoi controllare il nome della città?",
            "slots": {},
            "metadata": {"operation": "weather_query", "error": "location_not_found"},
        }

    day_offset = _DAY_OFFSETS.get(date_expr, 0)

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["lat"],
                "longitude": place["lon"],
                "current": "temperature_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "forecast_days": max(day_offset + 1, 1),
                "timezone": "auto",
            },
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return {
            "response": "Il servizio meteo non è raggiungibile in questo momento, riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "weather_query", "error": str(e)},
        }

    city_label = location_name.strip().title()

    if day_offset == 0 and "current" in data:
        temp = data["current"]["temperature_2m"]
        desc = _weather_description(data["current"]["weather_code"])
        text = f"A {city_label} adesso ci sono {temp}°C, {desc}."
    else:
        daily = data.get("daily", {})
        try:
            temp_max = daily["temperature_2m_max"][day_offset]
            temp_min = daily["temperature_2m_min"][day_offset]
            desc = _weather_description(daily["weather_code"][day_offset])
            when = {1: "domani", 2: "dopodomani"}.get(day_offset, "nei prossimi giorni")
            text = f"A {city_label} {when} sono previsti {desc}, con temperature tra {temp_min}°C e {temp_max}°C."
        except (KeyError, IndexError):
            text = f"Non ho previsioni disponibili per {city_label} in questo momento."

    return {
        "response": text,
        "slots": {},
        "metadata": {"operation": "weather_query", "location": place, "raw": data},
    }
