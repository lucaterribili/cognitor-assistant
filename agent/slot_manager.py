"""
Sistema generico per la gestione degli slot basato sulle rules.
Questo modulo sostituisce la logica hardcoded con un approccio data-driven.

Completamente autonomo: deduce tutto dalle rules JSON senza configurazioni esterne.
"""
import re
from typing import Optional, Any


class SlotExtractor:
    """
    Estrae automaticamente i valori degli slot dalle entità NER
    basandosi sulle rules definite nei JSON/YAML.
    """

    # Tipi di entità "a soggetto libero": usati da intent generici tipo
    # "parlami di X"/"cerca X" (ask_culture, ask_definition, web_search, ...).
    # Il NER può etichettare lo stesso testo con un tipo più specifico invece
    # che con quello dichiarato per lo slot (es. "Russia" → LOCATION invece di
    # KEYWORD, perché il modello lo riconosce anche come nome di luogo). Per
    # questi slot si accetta comunque la prima entità disponibile di un altro
    # tipo, dato che l'intent ha un solo slot "soggetto" e non c'è ambiguità
    # su quale entità dovesse riempirlo.
    #
    # location/departure sono nello stesso set per un motivo diverso ma con lo
    # stesso sintomo: un nome di città isolato ("Roma", senza "da"/"a" davanti)
    # non dà al NER nessun segnale testuale per scegliere tra i due tag, e di
    # fatto risolve quasi sempre a LOCATION. Senza questo fallback, rispondere
    # con un nome di città nudo allo slot DEPARTURE di book_flight non trova
    # match esatto (ner_value vuoto) e fa scattare la "via di fuga" per cambio
    # di argomento in TurnProcessor._handle_slot_input, abbandonando il flow a
    # metà con una risposta di un altro intent invece di accettare il valore.
    _OPEN_TOPIC_ENTITY_TYPES = {"keyword", "query", "topic", "location", "departure"}

    def __init__(self, rules: dict, rule_interpreter=None):
        """
        Args:
            rules: Dizionario delle rules caricato dai JSON/YAML
            rule_interpreter: Opzionale RuleInterpreter per delegare validazione
        """
        self.rules = rules
        self.rule_interpreter = rule_interpreter
        self._slot_entity_mapping = self._build_slot_entity_mapping()
        self._valid_values_cache = {}

    def _build_slot_entity_mapping(self) -> dict[str, dict[str, str]]:
        """
        Costruisce automaticamente il mapping intent_name -> slot_name -> entity_type
        analizzando le rules (formato DSL YAML).

        Scoped per intent: due intent diversi possono dichiarare uno slot con lo
        stesso nome (es. "query") ma associato a un'entità NER diversa (es.
        web_search.query -> QUERY, ask_culture.query -> KEYWORD) senza collidere.

        Estrae dall'attributo 'entity' nella configurazione degli slot.

        Returns:
            dict: mapping intent_name -> {slot_name: entity_type}
        """
        mapping: dict[str, dict[str, str]] = {}

        # Analizza tutte le rules per trovare gli slot
        for intent_name, rule in self.rules.items():
            # Nuovo formato DSL YAML
            slots_config = rule.get('slots', {})
            for slot_name, slot_config in slots_config.items():
                if not slot_name.endswith('_UNSUPPORTED'):
                    # Usa l'entity specificata o fallback al nome dello slot
                    entity_type = slot_config.get('entity', slot_name)
                    mapping.setdefault(intent_name, {})[slot_name] = entity_type

        return mapping

    def get_slot_entity_type(self, intent_name: str, slot_name: str) -> Optional[str]:
        """
        Ottiene il tipo di entità NER corrispondente allo slot per un dato intent.

        Args:
            intent_name: Nome dell'intent a cui appartiene lo slot
            slot_name: Nome dello slot (es. "LOCATION")

        Returns:
            Tipo di entità NER o None
        """
        return self._slot_entity_mapping.get(intent_name, {}).get(slot_name)

    def extract_from_entities(self, intent_name: str, slot_name: str, entities: list[dict]) -> Optional[str]:
        """
        Estrae il valore di uno slot dalle entità NER.

        Args:
            intent_name: Nome dell'intent a cui appartiene lo slot
            slot_name: Nome dello slot da estrarre
            entities: Lista di entità dal NER

        Returns:
            Valore estratto o None
        """
        value, _ = self.extract_from_entities_and_index(intent_name, slot_name, entities)
        return value

    def extract_from_entities_and_index(
        self,
        intent_name: str,
        slot_name: str,
        entities: list[dict],
        exclude_indexes: set[int] | None = None
    ) -> tuple[Optional[str], int | None]:
        """
        Estrae il valore di uno slot dalle entità NER e ritorna anche l'indice dell'entità usata.

        Args:
            intent_name: Nome dell'intent a cui appartiene lo slot
            slot_name: Nome dello slot da estrarre
            entities: Lista di entità dal NER
            exclude_indexes: Indici delle entità già utilizzate

        Returns:
            Tuple[value, index] dove index è None se non trovato
        """
        entity_type = self.get_slot_entity_type(intent_name, slot_name)
        if not entity_type:
            return None, None

        entity_type_lower = entity_type.lower()
        for idx, entity in enumerate(entities):
            if exclude_indexes and idx in exclude_indexes:
                continue
            if entity.get('entity', '').lower() == entity_type_lower:
                return entity.get('value'), idx

        # Fallback per slot "a soggetto libero" (vedi _OPEN_TOPIC_ENTITY_TYPES):
        # nessuna entità del tipo dichiarato, ma se lo slot atteso è generico
        # accetta comunque la prima entità disponibile di un altro tipo.
        if entity_type_lower in self._OPEN_TOPIC_ENTITY_TYPES:
            for idx, entity in enumerate(entities):
                if exclude_indexes and idx in exclude_indexes:
                    continue
                return entity.get('value'), idx

        return None, None

    def get_valid_values_for_slot(self, intent: str, slot_name: str) -> list[str]:
        """
        Estrae tutti i valori validi per uno slot da un intent,
        analizzando le conditions nelle rules o delegando al RuleInterpreter.

        Args:
            intent: Nome dell'intent
            slot_name: Nome dello slot

        Returns:
            Lista di valori validi
        """
        # Se abbiamo il RuleInterpreter, delega a lui
        if self.rule_interpreter:
            return self.rule_interpreter.get_valid_values_for_slot(intent, slot_name)

        # Altrimenti usa la logica legacy
        cache_key = f"{intent}:{slot_name}"
        if cache_key in self._valid_values_cache:
            return self._valid_values_cache[cache_key]

        valid_values = []
        rule = self.rules.get(intent)

        if not rule:
            self._valid_values_cache[cache_key] = valid_values
            return valid_values

        # Nuovo formato DSL (YAML)
        if "cases" in rule:
            valid_values = list(rule["cases"].keys())
        # Vecchio formato (JSON con conditions)
        else:
            for branch in rule.get('conditions', []):
                for condition in branch.get('if', []):
                    if condition.get('slot') == slot_name and condition.get('operator') == 'eq':
                        value = condition.get('value')
                        if value and isinstance(value, str):
                            valid_values.append(value)

        self._valid_values_cache[cache_key] = valid_values
        return valid_values

    def is_valid_value(self, intent: str, slot_name: str, value: Any) -> bool:
        """
        Valida se un valore è accettabile per uno slot in un determinato intent.

        Args:
            intent: Nome dell'intent
            slot_name: Nome dello slot
            value: Valore da validare

        Returns:
            True se il valore è valido
        """
        # Se abbiamo il RuleInterpreter, delega a lui
        if self.rule_interpreter:
            return self.rule_interpreter.is_valid_value(intent, slot_name, value)

        # Altrimenti usa la logica legacy
        if not value:
            return False

        valid_values = self.get_valid_values_for_slot(intent, slot_name)

        # Se non ci sono vincoli espliciti, accetta qualsiasi valore
        if not valid_values:
            return True

        # Confronto case-insensitive
        value_lower = str(value).lower() if value else ""
        return any(value_lower == valid.lower() for valid in valid_values)


class SlotContextManager:
    """
    Gestisce l'aggiornamento del contesto della sessione basandosi
    sulle entità estratte e le rules.

    Completamente data-driven: non usa pattern hardcoded.
    """

    def __init__(self, slot_extractor: SlotExtractor):
        self.slot_extractor = slot_extractor

    def get_slots_for_intent(self, intent: str) -> set[str]:
        """
        Ottiene tutti gli slot utilizzati da un intent analizzando le rules.
        Supporta sia il nuovo formato DSL YAML che il vecchio formato JSON.

        Args:
            intent: Nome dell'intent

        Returns:
            Set di nomi di slot
        """
        rule = self.slot_extractor.rules.get(intent)
        if not rule:
            return set()

        slots = set()

        # Nuovo formato DSL (YAML) - ha attributo 'slots'
        if 'slots' in rule:
            for slot_name in rule['slots'].keys():
                slots.add(slot_name)
                # Aggiungi anche il flag _UNSUPPORTED se lo slot è required
                slot_config = rule['slots'][slot_name]
                if slot_config.get('required', False):
                    slots.add(f"{slot_name}_UNSUPPORTED")

        # Vecchio formato JSON (legacy) - ha 'conditions'
        elif 'conditions' in rule:
            for branch in rule.get('conditions', []):
                for condition in branch.get('if', []):
                    slot_name = condition.get('slot')
                    if slot_name:
                        slots.add(slot_name)

        return slots

    def update_session_context(
        self,
        session,
        intent: str,
        previous_intent: Optional[str],
        entities: list[dict],
        user_input: str
    ) -> None:
        """
        Aggiorna automaticamente il contesto della sessione basandosi su:
        - Intent corrente e precedente
        - Entità estratte dal NER
        - Rules definite nei JSON
        - Pattern di cambio rilevati

        Args:
            session: Sessione corrente
            intent: Intent corrente
            previous_intent: Intent precedente (può essere None)
            entities: Lista di entità estratte
            user_input: Input testuale dell'utente
        """
        # Ottieni tutti gli slot rilevanti per questo intent
        current_slots = self.get_slots_for_intent(intent)
        previous_slots = self.get_slots_for_intent(previous_intent) if previous_intent else set()

        # Filtra solo gli slot veri (non i flag _UNSUPPORTED)
        current_slots_real = {s for s in current_slots if not s.endswith('_UNSUPPORTED')}
        previous_slots_real = {s for s in previous_slots if not s.endswith('_UNSUPPORTED')}

        print(f"[SlotManager] intent='{intent}' | slot attesi={current_slots_real} | entità NER={[(e.get('entity'), e.get('value')) for e in entities]}")
        print(f"[SlotManager] mapping slot→entity: { {s: self.slot_extractor.get_slot_entity_type(intent, s) for s in current_slots_real} }")

        # Se ci sono slot in comune tra intent consecutivi
        consecutive_slots = current_slots_real & previous_slots_real
        used_entity_indexes: set[int] = set()

        # Pre-estrai lingua per il fallback translate (usata sotto)
        ner_language = next(
            (e.get('value') for e in entities if e.get('entity', '').upper() == 'LANGUAGE'),
            None
        )

        for slot_name in current_slots_real:
            # Estrai valore dalle entità, evitando di riutilizzare la stessa entità per più slot
            extracted_value, extracted_index = self.slot_extractor.extract_from_entities_and_index(
                intent, slot_name, entities, exclude_indexes=used_entity_indexes
            )
            if extracted_index is not None:
                used_entity_indexes.add(extracted_index)

            # Fallback intelligente per TRANSLATION_TEXT quando il NER è insufficiente
            if intent == 'translate' and slot_name == 'TRANSLATION_TEXT':
                regex_value = self._extract_translate_fallback(user_input, ner_language or session.context.get('LANGUAGE'))
                if regex_value and (not extracted_value or len(regex_value) > len(extracted_value)):
                    extracted_value = regex_value
                    print(f"[SlotManager] TRANSLATION_TEXT: regex fallback → '{extracted_value}'")

            print(f"[SlotManager] slot='{slot_name}' → extracted='{extracted_value}' | consecutivo={slot_name in consecutive_slots}")

            # Caso 1: Intent consecutivi con stesso slot
            if slot_name in consecutive_slots:
                self._handle_consecutive_intent_slot(
                    session, intent, slot_name, extracted_value, user_input
                )

            # Caso 2: Nuovo slot o primo intent
            elif extracted_value:
                self._handle_new_slot_value(
                    session, intent, slot_name, extracted_value
                )
            elif intent == 'translate' and slot_name == 'TRANSLATION_TEXT' and previous_intent != 'translate':
                # Nuova richiesta di traduzione da un intent diverso e NER non ha trovato nulla:
                # azzera il valore precedente per forzare il bot a chiedere cosa tradurre
                session.update_context(slot_name, None)
                print(f"[SlotManager] TRANSLATION_TEXT azzerato (nuova richiesta translate senza estrazione)")

    def _extract_translate_fallback(self, user_input: str, language: str = None) -> str | None:
        """
        Estrae il testo da tradurre tramite regex quando il NER è insufficiente.

        Strategie (in ordine di priorità):
        1. Testo tra virgolette/apici → es. traduci in russo 'Sei brutto'
        2. Testo dopo il nome della lingua → es. come si dice in russo non capisci niente
        3. Testo dopo trigger words, rimosse lingua e preposizioni
        """
        text = user_input.strip()

        # Priorità 1: testo tra virgolette o apici
        match = re.search(r'["""\'](.+?)["""\']', text)
        if match:
            return match.group(1).strip()

        # Priorità 2: testo dopo la lingua → "in LANGUAGE <testo>"
        known_langs = [
            'russo', 'inglese', 'francese', 'spagnolo', 'tedesco', 'arabo',
            'cinese', 'italiano', 'portoghese', 'giapponese', 'coreano', 'olandese',
        ]
        langs_to_check = [language.lower()] if language else known_langs
        for lang in langs_to_check:
            pattern = rf'\b{re.escape(lang)}\b\s+(.+?)(?:\s*[?!.]?\s*$)'
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip(' ?!.,')
                if candidate:
                    return candidate

        # Priorità 3: rimuovi trigger words e lingua, il resto è il testo
        cleaned = text
        triggers = [
            'come si dice in', 'come si traduce in', 'come si dice',
            'come si traduce', 'traduzione di', 'traduzione', 'traduci in', 'traduci',
        ]
        for trigger in sorted(triggers, key=len, reverse=True):
            cleaned = re.sub(rf'(?i)\b{re.escape(trigger)}\b', '', cleaned)

        for lang in langs_to_check:
            cleaned = re.sub(rf'(?i)\bin\s+{re.escape(lang)}\b', '', cleaned)
            cleaned = re.sub(rf'(?i)\b{re.escape(lang)}\b', '', cleaned)

        cleaned = cleaned.strip(' ?!.,')
        return cleaned if cleaned else None

    def _handle_consecutive_intent_slot(
        self,
        session,
        intent: str,
        slot_name: str,
        extracted_value: Optional[str],
        user_input: str
    ) -> None:
        """
        Gestisce l'aggiornamento di uno slot quando due intent consecutivi
        utilizzano lo stesso slot.
        """
        if extracted_value:
            # Valida il nuovo valore
            if self.slot_extractor.is_valid_value(intent, slot_name, extracted_value):
                # Esegui il casting prima di salvare nel contesto
                casted_value = extracted_value
                if self.slot_extractor.rule_interpreter:
                    casted_value = self.slot_extractor.rule_interpreter.cast_slot_value(intent, slot_name, extracted_value)
                
                session.update_context(slot_name, casted_value)
                session.update_context(f"{slot_name}_UNSUPPORTED", False)
                print(f"[SlotManager] Slot '{slot_name}' aggiornato: {casted_value} (type: {type(casted_value).__name__})")
            else:
                # Valore estratto ma non valido -> invalida
                session.update_context(slot_name, None)
                session.update_context(f"{slot_name}_UNSUPPORTED", True)
                print(f"[SlotManager] Slot '{slot_name}' non supportato: {extracted_value}")

    def _handle_new_slot_value(
        self,
        session,
        intent: str,
        slot_name: str,
        extracted_value: str
    ) -> None:
        """
        Gestisce l'aggiornamento di uno slot con un nuovo valore estratto.
        """
        if self.slot_extractor.is_valid_value(intent, slot_name, extracted_value):
            # Esegui il casting prima di salvare nel contesto
            casted_value = extracted_value
            if self.slot_extractor.rule_interpreter:
                casted_value = self.slot_extractor.rule_interpreter.cast_slot_value(intent, slot_name, extracted_value)
            
            session.update_context(slot_name, casted_value)
            session.update_context(f"{slot_name}_UNSUPPORTED", False)
            print(f"[SlotManager] Slot '{slot_name}' impostato: {casted_value} (type: {type(casted_value).__name__})")
        else:
            session.update_context(slot_name, extracted_value)
            session.update_context(f"{slot_name}_UNSUPPORTED", True)
            print(f"[SlotManager] Slot '{slot_name}' non supportato: {extracted_value}")


class SlotManager:
    """
    Facade principale per la gestione degli slot.
    Sistema completamente data-driven basato sulle rules JSON/YAML.

    Non richiede configurazioni hardcoded: tutto viene dedotto dalle rules.
    """

    def __init__(self, rules: dict, rule_interpreter=None):
        """
        Args:
            rules: Dizionario delle rules caricato dai JSON/YAML
            rule_interpreter: Opzionale RuleInterpreter per validazione avanzata
        """
        self.extractor = SlotExtractor(rules, rule_interpreter)
        self.context_manager = SlotContextManager(self.extractor)

    def update_session_from_prediction(
        self,
        session,
        current_intent: str,
        entities: list[dict],
        user_input: str
    ) -> None:
        """
        Aggiorna la sessione analizzando prediction e history.
        Questo è il metodo principale da chiamare dopo ogni predizione.

        Args:
            session: Sessione corrente
            current_intent: Intent predetto
            entities: Entità estratte dal NER
            user_input: Input testuale dell'utente
        """
        # Trova l'intent precedente nella history
        previous_intent = self._get_previous_intent(session)

        # Aggiorna il contesto
        self.context_manager.update_session_context(
            session=session,
            intent=current_intent,
            previous_intent=previous_intent,
            entities=entities,
            user_input=user_input
        )

    def _get_previous_intent(self, session) -> Optional[str]:
        """Estrae l'intent precedente dalla history della sessione."""
        if len(session.history) >= 2:
            for msg in reversed(session.history[:-1]):
                if msg.get('role') == 'user' and msg.get('intent'):
                    return msg.get('intent')
        return None

    def get_valid_values(self, intent: str, slot_name: str) -> list[str]:
        """
        Ottiene i valori validi per uno slot in un dato intent.
        Utile per validazione e suggerimenti all'utente.

        Args:
            intent: Nome dell'intent
            slot_name: Nome dello slot

        Returns:
            Lista di valori validi
        """
        return self.extractor.get_valid_values_for_slot(intent, slot_name)

    def validate_slot_value(self, intent: str, slot_name: str, value: Any) -> bool:
        """
        Valida un valore per uno slot.

        Args:
            intent: Nome dell'intent
            slot_name: Nome dello slot
            value: Valore da validare

        Returns:
            True se valido
        """
        return self.extractor.is_valid_value(intent, slot_name, value)





