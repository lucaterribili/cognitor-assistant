"""Operation per l'intent location_query."""
import random

from agent.operations.tools.geocoding import Geocoding


def action_location_query(intent_name: str, slots: dict = None) -> dict:
    """
    Esegue l'operazione di query sulla posizione.

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    lat = slots.get("lat") or random.uniform(41.0, 46.0)
    lon = slots.get("lon") or random.uniform(6.0, 19.0)

    geocoding = Geocoding()
    location = geocoding.get_location(browser_coords={"lat": lat, "lon": lon})

    if location and location.get("address"):
        address = location.get("address", {})
        city = address.get("city") or address.get("town") or address.get("village")
        if city:
            response = f"Ti trovi a {city}, {address.get('country', '')}"
        elif address.get("display_name"):
            response = f"Ti trovi in: {address['display_name']}"
        else:
            response = f"La tua posizione approssimativa è: lat {location['lat']}, lon {location['lon']}"
    else:
        response = "Non sono riuscito a determinare la tua posizione."

    return {
        "response": response,
        "slots": {},
        "metadata": {"operation": "location_query", "location": location}
    }
