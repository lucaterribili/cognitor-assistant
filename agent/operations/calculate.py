"""Operation per l'intent calculate."""


def action_calculate(intent_name: str, slots: dict = None) -> dict:
    """
    Esegue l'operazione di calcolo.

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili

    Returns:
        dict con la risposta
    """
    return {
        "response": f"Operation per {intent_name} non implementata.",
        "slots": {},
        "metadata": {"operation": "calculate"}
    }
