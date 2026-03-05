"""
Runtime per interpretare il DSL dichiarativo delle rules (YAML).

Questo è l'INTERPRETE - il motore che esegue le regole definite nel DSL.
Il DSL è stabile, le regole cambiano fuori dal codice.

Separazione pulita:
- DSL (YAML) = COSA fare
- Runtime (questo file) = COME farlo
"""
from typing import Optional, Any
import random


class RuleInterpreter:
    """
    Interprete generico per le rules in formato DSL.

    Legge la configurazione YAML e decide quale risposta dare
    basandosi sugli slot disponibili.
    """

    def __init__(self, rules: dict, responses: dict):
        """
        Args:
            rules: Dizionario delle rules caricato da YAML
            responses: Dizionario delle responses caricato da YAML
        """
        self.rules = rules
        self.responses = responses

    def handle_intent(self, intent_name: str, slots: dict = None) -> tuple[str, Optional[str]]:
        """
        Interpreta una rule e restituisce la risposta appropriata.

        Questo è il CUORE dell'interprete.

        Args:
            intent_name: Nome dell'intent
            slots: Dizionario degli slot disponibili

        Returns:
            tuple: (risposta, slot_da_attendere)
        """
        if slots is None:
            slots = {}

        # Ottieni la rule per questo intent
        rule = self.rules.get(intent_name)
        if not rule:
            return "Intent non trovato nel DSL", None

        # Caso 1: Intent semplice (solo default)
        if "default" in rule and "slots" not in rule:
            response_key = rule["default"]
            return self._get_random_response(response_key, slots), None

        # Caso 2: Intent con slot
        if "slots" in rule:
            return self._handle_slot_based_intent(rule, slots)

        # Fallback
        return "Configurazione intent non valida", None

    def _handle_slot_based_intent(self, rule: dict, slots: dict) -> tuple[str, Optional[str]]:
        """
        Gestisce intent che richiedono slot.

        Logica dell'interprete:
        1. Controlla se ci sono slot required non forniti → wait
        2. Controlla se lo slot ha un valore con unsupported → fallback
        3. Cerca il valore dello slot nei cases → risposta specifica
        4. Altrimenti → fallback o default
        """
        rule_slots = rule.get("slots", {})

        # Step 1: Controlla slot required
        for slot_name, slot_config in rule_slots.items():
            if slot_config.get("required", False):
                slot_value = slots.get(slot_name)
                unsupported_flag = slots.get(f"{slot_name}_UNSUPPORTED")

                # Slot non fornito o unsupported → wait
                if not slot_value or unsupported_flag:
                    wait_key = rule.get("wait")
                    if wait_key:
                        return self._get_random_response(wait_key, slots), slot_name
                    # Fallback se non c'è wait definito
                    fallback_key = rule.get("fallback", rule.get("default"))
                    return self._get_random_response(fallback_key, slots), None

        # Step 2: Slot fornito, cerca nei cases
        for slot_name, slot_config in rule_slots.items():
            slot_value = slots.get(slot_name)
            if slot_value:
                cases = rule.get("cases", {})

                # Case-insensitive match
                for case_key, response_key in cases.items():
                    if str(slot_value).lower() == str(case_key).lower():
                        return self._get_random_response(response_key, slots), None

                # Valore non trovato nei cases → fallback
                fallback_key = rule.get("fallback")
                if fallback_key:
                    return self._get_random_response(fallback_key, slots), None

        # Step 3: Default se disponibile
        default_key = rule.get("default")
        if default_key:
            return self._get_random_response(default_key, slots), None

        return "Nessuna risposta configurata", None

    def _get_random_response(self, response_key: str, slots: dict) -> str:
        """
        Ottiene una risposta random dalla lista e sostituisce i placeholder.

        Args:
            response_key: Chiave della response
            slots: Dizionario degli slot per sostituire i placeholder

        Returns:
            Risposta con placeholder sostituiti
        """
        response_list = self.responses.get(response_key, [])

        if not response_list:
            return f"Risposta non definita per {response_key}"

        # Scegli una risposta random
        response = random.choice(response_list)

        # Sostituisci placeholder {SLOT_NAME} con valori
        for slot_name, slot_value in slots.items():
            if slot_value and not slot_name.endswith("_UNSUPPORTED"):
                placeholder = f"{{{slot_name}}}"
                response = response.replace(placeholder, str(slot_value))

        return response

    def get_valid_values_for_slot(self, intent_name: str, slot_name: str) -> list[str]:
        """
        Estrae i valori validi per uno slot dai cases del DSL.

        Args:
            intent_name: Nome dell'intent
            slot_name: Nome dello slot

        Returns:
            Lista di valori validi
        """
        rule = self.rules.get(intent_name)
        if not rule:
            return []

        # Controlla che lo slot sia definito
        rule_slots = rule.get("slots", {})
        if slot_name not in rule_slots:
            return []

        # Estrai i case keys
        cases = rule.get("cases", {})
        return list(cases.keys())

    def is_valid_value(self, intent_name: str, slot_name: str, value: Any) -> bool:
        """
        Valida un valore per uno slot.

        Args:
            intent_name: Nome dell'intent
            slot_name: Nome dello slot
            value: Valore da validare

        Returns:
            True se valido
        """
        if not value:
            return False

        valid_values = self.get_valid_values_for_slot(intent_name, slot_name)

        # Se non ci sono vincoli espliciti, accetta qualsiasi valore
        if not valid_values:
            return True

        # Case-insensitive match
        value_lower = str(value).lower()
        return any(value_lower == str(valid).lower() for valid in valid_values)

    def get_slots_for_intent(self, intent_name: str) -> dict:
        """
        Ottiene la configurazione degli slot per un intent.

        Args:
            intent_name: Nome dell'intent

        Returns:
            Dizionario della configurazione slot
        """
        rule = self.rules.get(intent_name)
        if not rule:
            return {}

        return rule.get("slots", {})

    def is_slot_required(self, intent_name: str, slot_name: str) -> bool:
        """
        Verifica se uno slot è required per un intent.

        Args:
            intent_name: Nome dell'intent
            slot_name: Nome dello slot

        Returns:
            True se required
        """
        slots_config = self.get_slots_for_intent(intent_name)
        slot_config = slots_config.get(slot_name, {})
        return slot_config.get("required", False)

