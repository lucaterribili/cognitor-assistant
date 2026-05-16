"""Operation placeholder per traduzioni verso il russo."""


def action_russian_translate(intent_name: str, slots: dict = None) -> dict:
    """
    Prepara i dati per una futura integrazione con un traduttore esterno.

    Args:
        intent_name: Nome dell'intent che attiva la traduzione
        slots: Slot disponibili, con TRANSLATION_TEXT valorizzato dal NER

    Returns:
        dict con risposta, slot e metadati
    """
    slots = slots or {}
    text_to_translate = (
        slots.get("TRANSLATION_TEXT")
        or slots.get("translation_text")
        or slots.get("QUERY")
    )

    if not text_to_translate:
        return {
            "response": "Non ho trovato la parola o frase da tradurre. Scrivimela e riproviamo.",
            "slots": {},
            "metadata": {
                "operation": "russian_translate",
                "target_language": "ru",
                "source_text": None,
                "status": "missing_input",
            },
        }

    return {
        "response": (
            f"Perfetto, ho ricevuto '{text_to_translate}'. "
            "Il traduttore automatico non è ancora integrato, ma il testo è pronto per essere inviato al modulo di traduzione russo."
        ),
        "slots": {"LAST_TRANSLATION_TEXT": text_to_translate},
        "metadata": {
            "operation": "russian_translate",
            "intent": intent_name,
            "target_language": "ru",
            "source_text": text_to_translate,
            "status": "queued_for_translator",
        },
    }
