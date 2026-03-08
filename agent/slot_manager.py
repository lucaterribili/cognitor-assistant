"""
Sistema generico per la gestione degli slot basato sulle rules.
Questo modulo sostituisce la logica hardcoded con un approccio data-driven.

Completamente autonomo: deduce tutto dalle rules JSON senza configurazioni esterne.
"""
from typing import Optional, Any


class SlotExtractor:
    """
    Estrae automaticamente i valori degli slot dalle entità NER
    basandosi sulle rules definite nei JSON/YAML.
    """

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

    def _build_slot_entity_mapping(self) -> dict[str, str]:
        """
        Costruisce automaticamente il mapping slot_name -> entity_type
        analizzando le rules (formato DSL YAML).

        Estrae dall'attributo 'entity' nella configurazione degli slot.

        Returns:
            dict: mapping slot_name -> entity_type
        """
        mapping = {}

        # Analizza tutte le rules per trovare gli slot
        for intent_name, rule in self.rules.items():
            # Nuovo formato DSL YAML
            slots_config = rule.get('slots', {})
            for slot_name, slot_config in slots_config.items():
                if not slot_name.endswith('_UNSUPPORTED'):
                    # Usa l'entity specificata o fallback al nome dello slot
                    entity_type = slot_config.get('entity', slot_name)
                    mapping[slot_name] = entity_type

        return mapping

    def get_slot_entity_type(self, slot_name: str) -> Optional[str]:
        """
        Ottiene il tipo di entità NER corrispondente allo slot.

        Args:
            slot_name: Nome dello slot (es. "LOCATION")

        Returns:
            Tipo di entità NER o None
        """
        return self._slot_entity_mapping.get(slot_name)

    def extract_from_entities(self, slot_name: str, entities: list[dict]) -> Optional[str]:
        """
        Estrae il valore di uno slot dalle entità NER.

        Args:
            slot_name: Nome dello slot da estrarre
            entities: Lista di entità dal NER

        Returns:
            Valore estratto o None
        """
        entity_type = self.get_slot_entity_type(slot_name)
        if not entity_type:
            return None

        # Cerca la prima entità del tipo corretto (case-insensitive)
        entity_type_lower = entity_type.lower()
        for entity in entities:
            if entity.get('entity', '').lower() == entity_type_lower:
                return entity.get('value')

        return None

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
        print(f"[SlotManager] mapping slot→entity: { {s: self.slot_extractor.get_slot_entity_type(s) for s in current_slots_real} }")

        # Se ci sono slot in comune tra intent consecutivi
        consecutive_slots = current_slots_real & previous_slots_real

        for slot_name in current_slots_real:
            # Estrai valore dalle entità
            extracted_value = self.slot_extractor.extract_from_entities(slot_name, entities)
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

        Logica completamente data-driven:
        - Se NER estrae un valore valido -> lo aggiorna
        - Se NER estrae un valore non valido -> invalida e chiede di nuovo
        - Se NER non estrae nulla -> MANTIENE il valore precedente (l'utente sta solo facendo domande)
        """
        current_value = session.get_context(slot_name)

        if extracted_value:
            # Valida il nuovo valore
            if self.slot_extractor.is_valid_value(intent, slot_name, extracted_value):
                session.update_context(slot_name, extracted_value)
                session.update_context(f"{slot_name}_UNSUPPORTED", False)
                print(f"[SlotManager] Slot '{slot_name}' aggiornato: {extracted_value}")
            else:
                # Valore estratto ma non valido -> invalida
                session.update_context(slot_name, None)
                session.update_context(f"{slot_name}_UNSUPPORTED", True)
                print(f"[SlotManager] Slot '{slot_name}' non supportato: {extracted_value}")

        # Se NER non estrae nulla, MANTIENE il valore precedente
        # (l'utente probabilmente sta solo facendo altre domande sulla stessa location)

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
            session.update_context(slot_name, extracted_value)
            session.update_context(f"{slot_name}_UNSUPPORTED", False)
            print(f"[SlotManager] Slot '{slot_name}' impostato: {extracted_value}")
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





