"""Operation per l'intent time_query."""

from datetime import datetime


def action_time_query_response(intent_name: str = None, slots: dict = None) -> dict:
    """
    Restituisce l'orario corrente in formato umano.

    Args:
        intent_name: Nome dell'intent che ha attivato l'operation
        slots: Slot disponibili (non usati qui)

    Returns:
        dict con la risposta e eventuali metadati
    """
    now = datetime.now()
    formatted_time = now.strftime("%H:%M")
    response = f"Sono le {formatted_time}." if formatted_time else "Non riesco a leggere l'ora al momento."

    return {
        "response": response,
        "slots": {},
        "metadata": {
            "operation": "time_query_response",
            "timestamp": now.isoformat()
        }
    }
