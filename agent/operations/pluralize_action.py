"""Operation per pluralizzare un termine italiano e tradurlo (default: russo)."""

from classes.pluralize import pluralize
from agent.operations.cognitor_translate import action_cognitor_translate

def action_pluralize(intent_name: str, slots: dict = None) -> dict:
    """
    Pluralizza il testo in TRANSLATION_TEXT e poi lo traduce.
    
    Args:
        intent_name: Nome dell'intent (pluralize)
        slots: Slot con TRANSLATION_TEXT e opzionalmente LANGUAGE
        
    Returns:
        Risposta con il plurale italiano e la traduzione
    """
    slots = slots or {}
    original_text = slots.get("TRANSLATION_TEXT")
    language = slots.get("LANGUAGE") or "russo"
    
    if not original_text:
        return {
            "response": "Non hai specificato la parola di cui vuoi il plurale.",
            "slots": {},
            "metadata": {"status": "missing_input"}
        }

    # 1. Pluralizzazione in italiano
    plural_result = pluralize(original_text)
    plural_text = plural_result.plural
    
    # 2. Traduzione del plurale
    # Prepariamo gli slot per l'azione di traduzione
    translation_slots = {
        "TRANSLATION_TEXT": plural_text,
        "LANGUAGE": language
    }
    
    translation_result = action_cognitor_translate("translate", translation_slots)
    
    # 3. Costruzione della risposta finale
    if translation_result.get("metadata", {}).get("status") == "translated":
        translated_plural = translation_result["response"]
        response = (
            f"Il plurale di **{original_text}** in italiano è **{plural_text}**.\n"
            f"In {language} si dice: **{translated_plural}**"
        )
        
        # Metadati dell'operazione
        metadata = {
            "operation": "pluralize",
            "status": "success",
            "original_singular": original_text,
            "italian_plural": plural_text,
            "target_language": language,
            "translated_plural": translated_plural,
            "rule_applied": plural_result.rule
        }
        
        return {
            "response": response,
            "slots": {
                "LAST_PLURAL": plural_text,
                "LAST_TRANSLATION": translated_plural
            },
            "metadata": metadata
        }
    else:
        # Se la traduzione fallisce (es. lingua non supportata), diamo comunque il plurale
        return {
            "response": f"Il plurale di **{original_text}** è **{plural_text}**, ma non sono riuscito a tradurlo in '{language}'.",
            "slots": {"LAST_PLURAL": plural_text},
            "metadata": {
                "operation": "pluralize",
                "status": "partial_success",
                "italian_plural": plural_text,
                "translation_error": translation_result.get("response")
            }
        }
