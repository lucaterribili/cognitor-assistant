"""Operation per la traduzione verso lingue target (default: russo)."""

from agent.operations.tools.translator import (
    load_translator,
    normalize_language,
    resolve_translation_paths,
    translate_text,
)

_SOURCE_LANGUAGE = "it"


def action_cognitor_translate(intent_name: str, slots: dict = None) -> dict:
    """
    Gestisce le richieste di traduzione estratte dall'intent `translate`.

    Legge i slot TRANSLATION_TEXT (obbligatorio) e LANGUAGE (opzionale,
    default "russo"), carica il modello e restituisce la traduzione.

    Args:
        intent_name: Nome dell'intent che attiva la traduzione
        slots: Slot disponibili con TRANSLATION_TEXT e LANGUAGE valorizzati dal NER

    Returns:
        dict con risposta, slot e metadati
    """
    slots = slots or {}
    text_to_translate = slots.get("TRANSLATION_TEXT")
    language_raw = (slots.get("LANGUAGE") or "russo").strip().lower()
    target_language = normalize_language(language_raw)

    if not text_to_translate:
        return {
            "response": "Non ho trovato la parola o frase da tradurre. Scrivimela e riproviamo.",
            "slots": {},
            "metadata": {
                "operation": "cognitor_translate",
                "target_language": target_language,
                "target_language_name": language_raw,
                "source_text": None,
                "status": "missing_input",
            },
        }

    if target_language is None:
        return {
            "response": (
                "Non ho riconosciuto la lingua di destinazione. Prova con 'russo' o 'ru'."
            ),
            "slots": {
                "LAST_TRANSLATION_TEXT": text_to_translate,
                "LAST_TRANSLATION_LANGUAGE": language_raw,
            },
            "metadata": {
                "operation": "cognitor_translate",
                "intent": intent_name,
                "target_language": None,
                "target_language_name": language_raw,
                "source_text": text_to_translate,
                "status": "unsupported_language",
            },
        }

    if resolve_translation_paths(target_language, _SOURCE_LANGUAGE) is None:
        return {
            "response": (
                f"Al momento posso tradurre solo dall'italiano al russo. Prova a chiedere una traduzione in russo."
            ),
            "slots": {
                "LAST_TRANSLATION_TEXT": text_to_translate,
                "LAST_TRANSLATION_LANGUAGE": language_raw,
            },
            "metadata": {
                "operation": "cognitor_translate",
                "intent": intent_name,
                "target_language": target_language,
                "target_language_name": language_raw,
                "source_text": text_to_translate,
                "status": "unsupported_language_pair",
            },
        }

    try:
        model, tokenizer = load_translator(target_language, _SOURCE_LANGUAGE)
        translated_text = translate_text(model, tokenizer, text_to_translate)
    except Exception as exc:
        return {
            "response": (
                "Si è verificato un errore durante la traduzione. Riprova tra un attimo."
            ),
            "slots": {
                "LAST_TRANSLATION_TEXT": text_to_translate,
                "LAST_TRANSLATION_LANGUAGE": language_raw,
            },
            "metadata": {
                "operation": "cognitor_translate",
                "intent": intent_name,
                "target_language": target_language,
                "target_language_name": language_raw,
                "source_text": text_to_translate,
                "status": "translation_error",
                "error": str(exc),
            },
        }

    return {
        "response": translated_text,
        "slots": {
            "LAST_TRANSLATION_TEXT": text_to_translate,
            "LAST_TRANSLATION_LANGUAGE": language_raw,
            "TRANSLATED_TEXT": translated_text,
        },
        "metadata": {
            "operation": "cognitor_translate",
            "intent": intent_name,
            "source_language": _SOURCE_LANGUAGE,
            "target_language": target_language,
            "target_language_name": language_raw,
            "source_text": text_to_translate,
            "translated_text": translated_text,
            "status": "translated",
        },
    }
