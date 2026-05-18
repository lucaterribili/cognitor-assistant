"""Operation per l'intent book_flight."""

def action_book_flight(intent_name: str, slots: dict = None) -> dict:
    """
    Esegue l'operazione di ricerca voli in base alle località fornite.

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili

    Returns:
        dict con la risposta e eventuali metadati
    """
    slots = slots or {}
    location = slots.get("LOCATION")
    departure = slots.get("DEPARTURE")

    if not location or not departure:
        missing = "LOCATION" if not location else "DEPARTURE"
        return {
            "response": f"Mi serve sia la città di partenza che quella di arrivo. Mancante: {missing}.",
            "slots": {},
            "metadata": {"operation": "book_flight", "missing_slot": missing}
        }

    response = f"Cerco voli da {departure} a {location}. Ti faccio sapere i risultati appena possibile."
    return {
        "response": response,
        "slots": {},
        "metadata": {
            "operation": "book_flight",
            "departure": departure,
            "location": location
        }
    }
