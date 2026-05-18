"""Operation per la traduzione verso lingue target (default: russo)."""

_LANGUAGE_ALIASES = {
    "russo": "ru",
    "inglese": "en",
    "francese": "fr",
    "spagnolo": "es",
    "tedesco": "de",
    "italiano": "it",
    "cinese": "zh",
    "arabo": "ar",
    "giapponese": "ja",
    "portoghese": "pt",
}


def action_cognitor_translate(intent_name: str, slots: dict = None) -> dict:
    """
    Gestisce le richieste di traduzione estratte dall'intent `translate`.

    Legge i slot TRANSLATION_TEXT (obbligatorio) e LANGUAGE (opzionale,
    default "russo") e prepara i dati per il modulo di traduzione esterno.

    Args:
        intent_name: Nome dell'intent che attiva la traduzione
        slots: Slot disponibili con TRANSLATION_TEXT e LANGUAGE valorizzati dal NER

    Returns:
        dict con risposta, slot e metadati
    """
    slots = slots or {}
    text_to_translate = slots.get("TRANSLATION_TEXT")
    language_raw = (slots.get("LANGUAGE") or "russo").strip().lower()
    language_code = _LANGUAGE_ALIASES.get(language_raw, language_raw)

    if not text_to_translate:
        return {
            "response": "Non ho trovato la parola o frase da tradurre. Scrivimela e riproviamo.",
            "slots": {},
            "metadata": {
                "operation": "cognitor_translate",
                "target_language": language_code,
                "target_language_name": language_raw,
                "source_text": None,
                "status": "missing_input",
            },
        }

    return {
        "response": (
            f"Perfetto, ho ricevuto '{text_to_translate}' da tradurre in {language_raw}. "
            "Il traduttore automatico non è ancora integrato, ma il testo è pronto per essere inviato al modulo di traduzione."
        ),
        "slots": {
            "LAST_TRANSLATION_TEXT": text_to_translate,
            "LAST_TRANSLATION_LANGUAGE": language_raw,
        },
        "metadata": {
            "operation": "cognitor_translate",
            "intent": intent_name,
            "target_language": language_code,
            "target_language_name": language_raw,
            "source_text": text_to_translate,
            "status": "queued_for_translator",
        },
    }
