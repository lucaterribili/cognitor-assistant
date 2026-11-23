"""
Gestione data-driven degli slot: estrazione dalle entità NER, validazione e
aggiornamento del contesto di sessione, tutto dedotto dalle rules DSL.
"""
import re
from typing import Optional, Any

from agent.rule_interpreter import RuleInterpreter


class SlotManager:

    # Tipi di entità "a soggetto libero": usati da intent generici tipo
    # "parlami di X"/"cerca X" (ask_culture, ask_definition, web_search, ...).
    # Il NER può etichettare lo stesso testo con un tipo più specifico invece
    # che con quello dichiarato per lo slot (es. "Russia" → LOCATION invece di
    # KEYWORD). Per questi slot si accetta comunque la prima entità disponibile
    # di un altro tipo, dato che l'intent ha un solo slot "soggetto".
    #
    # location/departure sono nello stesso set perché un nome di città isolato
    # ("Roma", senza "da"/"a" davanti) non dà al NER nessun segnale per scegliere
    # tra i due tag e risolve quasi sempre a LOCATION: senza questo fallback,
    # rispondere "Roma" allo slot DEPARTURE di book_flight farebbe scattare la
    # "via di fuga" per cambio di argomento in TurnProcessor._handle_slot_input.
    _OPEN_TOPIC_ENTITY_TYPES = {"keyword", "query", "topic", "location", "departure"}

    _TRANSLATE_LANGUAGES = [
        'russo', 'inglese', 'francese', 'spagnolo', 'tedesco', 'arabo',
        'cinese', 'italiano', 'portoghese', 'giapponese', 'coreano', 'olandese',
    ]
    _TRANSLATE_TRIGGERS = [
        'come si dice in', 'come si traduce in', 'come si dice',
        'come si traduce', 'traduzione di', 'traduzione', 'traduci in', 'traduci',
    ]

    def __init__(self, rules: dict, rule_interpreter: RuleInterpreter | None = None):
        self.rules = rules
        self.rule_interpreter = rule_interpreter or RuleInterpreter(rules, {})
        # intent_name -> {slot_name: entity_type}. Scoped per intent: due intent
        # possono dichiarare lo stesso slot (es. "query") con entità NER diverse.
        self._slot_entity_mapping: dict[str, dict[str, str]] = {
            intent_name: {
                slot_name: slot_config.get('entity', slot_name)
                for slot_name, slot_config in (rule.get('slots') or {}).items()
                if not slot_name.endswith('_UNSUPPORTED')
            }
            for intent_name, rule in rules.items()
        }

    # ------------------------------------------------------------------ estrazione

    def get_slot_entity_type(self, intent_name: str, slot_name: str) -> Optional[str]:
        return self._slot_entity_mapping.get(intent_name, {}).get(slot_name)

    def extract_from_entities(self, intent_name: str, slot_name: str, entities: list[dict]) -> Optional[str]:
        """Valore dello slot estratto dalle entità NER (o None)."""
        value, _ = self._extract_from_entities_and_index(intent_name, slot_name, entities)
        return value

    def _extract_from_entities_and_index(
        self, intent_name: str, slot_name: str, entities: list[dict], exclude_indexes: set[int] | None = None
    ) -> tuple[Optional[str], int | None]:
        """Come extract_from_entities, ritornando anche l'indice dell'entità usata
        (per non riutilizzare la stessa entità su più slot)."""
        entity_type = self.get_slot_entity_type(intent_name, slot_name)
        if not entity_type:
            return None, None

        exclude_indexes = exclude_indexes or set()
        candidates = [(idx, e) for idx, e in enumerate(entities) if idx not in exclude_indexes]

        entity_type_lower = entity_type.lower()
        for idx, entity in candidates:
            if entity.get('entity', '').lower() == entity_type_lower:
                return entity.get('value'), idx

        # Slot "a soggetto libero": accetta la prima entità disponibile di altro tipo
        if entity_type_lower in self._OPEN_TOPIC_ENTITY_TYPES and candidates:
            idx, entity = candidates[0]
            return entity.get('value'), idx

        return None, None

    # ------------------------------------------------------------------ validazione

    def get_valid_values(self, intent: str, slot_name: str) -> list[str]:
        return self.rule_interpreter.get_valid_values_for_slot(intent, slot_name)

    def validate_slot_value(self, intent: str, slot_name: str, value: Any) -> bool:
        return self.rule_interpreter.is_valid_value(intent, slot_name, value)

    def get_slots_for_intent(self, intent: str) -> set[str]:
        """Slot usati dall'intent, più il flag `<SLOT>_UNSUPPORTED` per quelli required."""
        rule = self.rules.get(intent)
        if not rule:
            return set()
        slots = set()
        for slot_name, slot_config in (rule.get('slots') or {}).items():
            slots.add(slot_name)
            if slot_config.get('required', False):
                slots.add(f"{slot_name}_UNSUPPORTED")
        return slots

    # ------------------------------------------------------------------ contesto

    def update_session_from_prediction(self, session, current_intent: str, entities: list[dict], user_input: str) -> None:
        """
        Aggiorna il contesto della sessione dopo una predizione, in base a intent
        corrente/precedente, entità NER e rules. Da chiamare dopo ogni predizione.
        """
        intent = current_intent
        previous_intent = self._get_previous_intent(session)

        current_slots = {s for s in self.get_slots_for_intent(intent) if not s.endswith('_UNSUPPORTED')}
        previous_slots = {
            s for s in (self.get_slots_for_intent(previous_intent) if previous_intent else set())
            if not s.endswith('_UNSUPPORTED')
        }

        print(f"[SlotManager] intent='{intent}' | slot attesi={current_slots} | entità NER={[(e.get('entity'), e.get('value')) for e in entities]}")
        print(f"[SlotManager] mapping slot→entity: { {s: self.get_slot_entity_type(intent, s) for s in current_slots} }")

        consecutive_slots = current_slots & previous_slots
        used_entity_indexes: set[int] = set()

        ner_language = next(
            (e.get('value') for e in entities if e.get('entity', '').upper() == 'LANGUAGE'),
            None
        )

        for slot_name in current_slots:
            extracted_value, extracted_index = self._extract_from_entities_and_index(
                intent, slot_name, entities, exclude_indexes=used_entity_indexes
            )
            if extracted_index is not None:
                used_entity_indexes.add(extracted_index)

            # Fallback regex per TRANSLATION_TEXT quando il NER è insufficiente
            if intent == 'translate' and slot_name == 'TRANSLATION_TEXT':
                regex_value = self._extract_translate_fallback(user_input, ner_language or session.context.get('LANGUAGE'))
                if regex_value and (not extracted_value or len(regex_value) > len(extracted_value)):
                    extracted_value = regex_value
                    print(f"[SlotManager] TRANSLATION_TEXT: regex fallback → '{extracted_value}'")

            is_consecutive = slot_name in consecutive_slots
            print(f"[SlotManager] slot='{slot_name}' → extracted='{extracted_value}' | consecutivo={is_consecutive}")

            if extracted_value:
                # Stesso slot su intent consecutivi: un valore non valido azzera lo
                # slot; su un nuovo slot invece il valore grezzo viene conservato.
                self._store_slot_value(session, intent, slot_name, extracted_value,
                                       keep_invalid_value=not is_consecutive)
            elif (not is_consecutive and intent == 'translate'
                  and slot_name == 'TRANSLATION_TEXT' and previous_intent != 'translate'):
                # Nuova richiesta di traduzione senza estrazione: azzera il valore
                # precedente per forzare il bot a chiedere cosa tradurre
                session.update_context(slot_name, None)
                print("[SlotManager] TRANSLATION_TEXT azzerato (nuova richiesta translate senza estrazione)")

    def _store_slot_value(self, session, intent: str, slot_name: str, value: str, keep_invalid_value: bool) -> None:
        if self.validate_slot_value(intent, slot_name, value):
            casted_value = self.rule_interpreter.cast_slot_value(intent, slot_name, value)
            session.update_context(slot_name, casted_value)
            session.update_context(f"{slot_name}_UNSUPPORTED", False)
            print(f"[SlotManager] Slot '{slot_name}' impostato: {casted_value} (type: {type(casted_value).__name__})")
        else:
            session.update_context(slot_name, value if keep_invalid_value else None)
            session.update_context(f"{slot_name}_UNSUPPORTED", True)
            print(f"[SlotManager] Slot '{slot_name}' non supportato: {value}")

    @staticmethod
    def _get_previous_intent(session) -> Optional[str]:
        """Intent dell'ultimo messaggio utente precedente a quello corrente."""
        if len(session.history) >= 2:
            for msg in reversed(session.history[:-1]):
                if msg.get('role') == 'user' and msg.get('intent'):
                    return msg.get('intent')
        return None

    def _extract_translate_fallback(self, user_input: str, language: str = None) -> str | None:
        """
        Estrae il testo da tradurre tramite regex quando il NER è insufficiente:
        1. testo tra virgolette/apici → traduci in russo 'Sei brutto'
        2. testo dopo il nome della lingua → come si dice in russo non capisci niente
        3. testo rimanente dopo aver rimosso trigger words, lingua e preposizioni
        """
        text = user_input.strip()

        match = re.search(r'["""\'](.+?)["""\']', text)
        if match:
            return match.group(1).strip()

        langs_to_check = [language.lower()] if language else self._TRANSLATE_LANGUAGES
        for lang in langs_to_check:
            m = re.search(rf'\b{re.escape(lang)}\b\s+(.+?)(?:\s*[?!.]?\s*$)', text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip(' ?!.,')
                if candidate:
                    return candidate

        cleaned = text
        for trigger in sorted(self._TRANSLATE_TRIGGERS, key=len, reverse=True):
            cleaned = re.sub(rf'(?i)\b{re.escape(trigger)}\b', '', cleaned)
        for lang in langs_to_check:
            cleaned = re.sub(rf'(?i)\bin\s+{re.escape(lang)}\b', '', cleaned)
            cleaned = re.sub(rf'(?i)\b{re.escape(lang)}\b', '', cleaned)

        cleaned = cleaned.strip(' ?!.,')
        return cleaned if cleaned else None
