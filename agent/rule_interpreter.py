"""
Runtime per interpretare il DSL dichiarativo delle rules (YAML).

- DSL (YAML) = COSA fare
- Runtime (questo file) = COME farlo

Convenzioni del DSL usate qui:
- `default` / `fallback` / `wait` / valori dei `cases`: response key, oppure
  `__nome` per eseguire l'operation `nome` (vedi agent/operations/).
- Nelle response: `{SLOT}` viene sostituito col valore dello slot, `[TIME]`/`[DATE]`
  con ora/data correnti, e `{SLOT=valore}` imposta uno slot lato bot (slot inline,
  rimosso dal testo visibile).
"""
from typing import Optional, Any
import random
import re
from datetime import datetime

from agent.operations.manager import OperationManager

# Slot inline nelle response: "Imposto Roma {LOCATION=Roma}." -> {"LOCATION": "Roma"}
_INLINE_SLOT_PATTERN = re.compile(r'\{([^=]+)=([^}]+)}')


def parse_inline_slots(response: str) -> tuple[str, dict]:
    """Estrae gli slot inline `{SLOT=valore}` e li rimuove dal testo visibile."""
    slots = {m.group(1).strip(): m.group(2).strip() for m in _INLINE_SLOT_PATTERN.finditer(response)}
    return _INLINE_SLOT_PATTERN.sub('', response).strip(), slots


class RuleInterpreter:
    """Interprete delle rules DSL: decide quale risposta dare in base agli slot."""

    def __init__(self, rules: dict, responses: dict, operation_manager: OperationManager | None = None):
        self.rules = rules
        self.responses = responses
        self.operation_manager = operation_manager

    # ------------------------------------------------------------------ entry point

    def handle_intent_with_bot_slots(
        self, intent_name: str, slots: dict = None, raw_text: str = None, cancel_event=None
    ) -> tuple[str, Optional[str], dict]:
        """
        Interpreta la rule dell'intent.

        Args:
            intent_name: Nome dell'intent
            slots: Slot disponibili
            raw_text: Testo grezzo del turno, inoltrato alle operation che lo dichiarano
            cancel_event: threading.Event opzionale, inoltrato alle operation che lo
                dichiarano (vedi Operation.execute)

        Returns:
            tuple: (risposta, slot_da_attendere, slot_da_impostare_dal_bot).
            Le eventuali opzioni (bottoni) dello slot in attesa viaggiano in
            bot_slots sotto la chiave riservata `__options__`.
        """
        if slots is None:
            slots = {}

        rule = self.rules.get(intent_name)
        if not rule:
            return "Intent non trovato nel DSL", None, {}

        bot_slots = self.extract_set_slots(intent_name, slots)

        if "slots" in rule:
            response, wait_slot, options, inline_slots = self._handle_slot_based_intent(
                rule, slots, raw_text, cancel_event
            )
            bot_slots.update(inline_slots)
            if options:
                bot_slots["__options__"] = options
            return response, wait_slot, bot_slots

        # Intent semplice senza slot
        default_key = rule.get("default", "")
        op_result = self._run_operation(default_key, slots, raw_text, cancel_event)
        if op_result is not None:
            return op_result["response"], None, {**bot_slots, **op_result.get("slots", {})}

        if "default" in rule:
            response, inline_slots = self._render_response(rule["default"], slots)
            return response, None, {**bot_slots, **inline_slots}

        return "Configurazione intent non valida", None, bot_slots

    def _handle_slot_based_intent(
        self, rule: dict, slots: dict, raw_text: str = None, cancel_event=None
    ) -> tuple[str, Optional[str], Optional[list], dict]:
        """
        Gestisce intent con slot:
        1. primo slot required (e pertinente, vedi `when:`) mancante → wait
        2. chiave composita SLOT1|SLOT2 nei cases
        3. matching sul primo slot con valore → case, altrimenti fallback
        4. operation di default (`__nome`) o response di default

        I `cases` vengono controllati PRIMA dell'operation di `default`: un dominio
        può così intercettare valori specifici su un intent generico a operation fissa
        (es. `web_search` → `__web_search`) senza toccare gli altri valori.

        Returns:
            tuple: (risposta, slot_da_attendere, opzioni, slot_inline_trovati)
        """
        rule_slots = rule.get("slots", {})
        wait_config = rule.get("wait")
        default_key = rule.get("default", "")
        cases = rule.get("cases", {})

        # Step 1: slot required mancanti, nell'ordine di definizione
        for slot_name, slot_config in rule_slots.items():
            if not self._slot_applies(slot_config, slots) or not slot_config.get("required", False):
                continue
            if slots.get(slot_name) and not slots.get(f"{slot_name}_UNSUPPORTED"):
                continue

            # `wait` può essere un dict per-slot o una stringa globale
            wait_key = wait_config.get(slot_name) if isinstance(wait_config, dict) else wait_config
            if wait_key:
                response, inline_slots = self._render_response(wait_key, slots)
                return response, slot_name, self._get_slot_options(slot_config, slots), inline_slots
            fallback_key = rule.get("fallback", rule.get("default"))
            if fallback_key:
                response, inline_slots = self._render_response(fallback_key, slots)
                return response, None, None, inline_slots
            return "Slot richiesto non fornito", None, None, {}

        # Step 2: chiave composita (tutti i required pertinenti con valore, nell'ordine)
        required_values = [
            slots.get(s)
            for s, c in rule_slots.items()
            if self._slot_applies(c, slots) and c.get("required", False) and slots.get(s)
        ]
        if len(required_values) > 1:
            composite_key = "|".join(str(v) for v in required_values).lower()
            for case_key, target in cases.items():
                if composite_key == str(case_key).lower():
                    return self._resolve_case_target(target, slots, raw_text, cancel_event)

        # Step 3: matching sul primo slot con valore. Il fallback ha senso solo come
        # esito di un case-lookup fallito: senza `cases` si prosegue all'operation.
        if cases:
            for slot_name in rule_slots:
                slot_value = slots.get(slot_name)
                if not slot_value:
                    continue
                for case_key, target in cases.items():
                    if str(slot_value).lower() == str(case_key).lower():
                        return self._resolve_case_target(target, slots, raw_text, cancel_event)
                fallback_key = rule.get("fallback")
                if fallback_key:
                    response, inline_slots = self._render_response(fallback_key, slots)
                    return response, None, None, inline_slots

        # Step 4: operation o response di default
        op_result = self._run_operation(default_key, slots, raw_text, cancel_event)
        if op_result is not None:
            return op_result["response"], None, None, {}

        if default_key and not default_key.startswith("__"):
            response, inline_slots = self._render_response(default_key, slots)
            return response, None, None, inline_slots

        return "Nessuna risposta configurata", None, None, {}

    def _resolve_case_target(
        self, target: str, slots: dict, raw_text: str = None, cancel_event=None
    ) -> tuple[str, None, None, dict]:
        """Target di un case matchato: `__nome` esegue l'operation, altrimenti è una response key."""
        op_result = self._run_operation(target, slots, raw_text, cancel_event)
        if op_result is not None:
            return op_result["response"], None, None, {}
        response, inline_slots = self._render_response(target, slots)
        return response, None, None, inline_slots

    # ------------------------------------------------------------------ helpers

    def _run_operation(self, key: str, slots: dict, raw_text: str = None, cancel_event=None) -> Optional[dict]:
        """Esegue l'operation se `key` è nella forma `__nome` ed è registrata, altrimenti None."""
        if not key.startswith("__") or not self.operation_manager:
            return None
        operation_name = key[2:]
        if not self.operation_manager.has_operation(operation_name):
            return None
        return self.operation_manager.execute(operation_name, operation_name, slots, raw_text, cancel_event)

    def _render_response(self, response_key: str, slots: dict) -> tuple[str, dict]:
        """Sceglie una response random, sostituisce i placeholder ed estrae gli slot inline."""
        response_list = self.responses.get(response_key, [])
        if not response_list:
            return f"Risposta non definita per {response_key}", {}

        response = random.choice(response_list)
        for slot_name, slot_value in slots.items():
            if slot_value and not slot_name.endswith("_UNSUPPORTED"):
                response = response.replace(f"{{{slot_name}}}", str(slot_value))

        now = datetime.now()
        response = response.replace("[TIME]", now.strftime("%H:%M"))
        response = response.replace("[DATE]", now.strftime("%d/%m/%Y"))

        return parse_inline_slots(response)

    @staticmethod
    def _slot_applies(slot_config: dict, slots: dict) -> bool:
        """Uno slot con `when: {altro_slot: valore}` è pertinente solo quando l'altro
        slot già raccolto ha quel valore; senza `when` si applica sempre."""
        when = slot_config.get("when")
        if not when:
            return True
        return all(str(slots.get(k)) == str(v) for k, v in when.items())

    def _get_slot_options(self, slot_config: dict, slots: dict) -> Optional[list]:
        """Opzioni (bottoni) per lo slot in attesa: lista statica `options` oppure
        provider dinamico `options_source` (funzione `options_<nome>` nelle operations)."""
        if "options" in slot_config:
            return slot_config["options"]

        options_source = slot_config.get("options_source")
        if options_source and self.operation_manager:
            return self.operation_manager.get_options(options_source, slots)

        return None

    # ------------------------------------------------------------------ slot config

    def get_slots_for_intent(self, intent_name: str) -> dict:
        """Configurazione degli slot dichiarati dalla rule dell'intent."""
        rule = self.rules.get(intent_name)
        return rule.get("slots", {}) if rule else {}

    def get_valid_values_for_slot(self, intent_name: str, slot_name: str) -> list[str]:
        """
        Valori validi per uno slot, ricavati dai `cases` (anche compositi "Roma|musei",
        prendendo la parte nella posizione dello slot tra i required).

        Se il `default` è un'operation (`__nome`) o la rule ha `open_cases: true`, i
        cases sono solo eccezioni puntuali sopra un catch-all generico (es. web_search,
        ask_programming_concept), non un enum chiuso come in `open_app`: lo slot resta
        libero, altrimenti qualunque valore non elencato verrebbe rifiutato.
        """
        rule = self.rules.get(intent_name)
        if not rule:
            return []

        rule_slots = rule.get("slots", {})
        if slot_name not in rule_slots:
            return []

        if rule.get("open_cases") or str(rule.get("default", "")).startswith("__"):
            return []

        cases = rule.get("cases", {})
        if not cases:
            return []

        if any("|" in str(k) for k in cases.keys()):
            required_slots = [s for s, c in rule_slots.items() if c.get("required", False)]
            if slot_name in required_slots:
                slot_position = required_slots.index(slot_name)
                values = set()
                for case_key in cases.keys():
                    parts = str(case_key).split("|")
                    if slot_position < len(parts):
                        values.add(parts[slot_position])
                return list(values)

        return list(cases.keys())

    def cast_slot_value(self, intent_name: str, slot_name: str, value: Any) -> Any:
        """Converte il valore nel `type` dichiarato (string, integer, float, boolean, list).
        Se la conversione fallisce ritorna il valore originale."""
        if value is None:
            return None

        slot_config = self.get_slots_for_intent(intent_name).get(slot_name, {})
        target_type = slot_config.get("type", "string").lower()

        try:
            if target_type == "integer":
                return int(value)
            elif target_type == "float":
                return float(str(value).replace(',', '.'))
            elif target_type == "boolean":
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ("true", "1", "si", "sì", "yes", "ok")
            elif target_type == "list":
                if isinstance(value, list):
                    return value
                return [s.strip() for s in str(value).split(',')]
            else:
                return str(value)
        except (ValueError, TypeError):
            return value

    def is_valid_value(self, intent_name: str, slot_name: str, value: Any) -> bool:
        """Valida un valore per uno slot: tipo, `pattern` opzionale e appartenenza ai cases."""
        if value is None:
            return False

        casted_value = self.cast_slot_value(intent_name, slot_name, value)
        slot_config = self.get_slots_for_intent(intent_name).get(slot_name, {})
        target_type = slot_config.get("type", "string").lower()

        if target_type == "integer" and not isinstance(casted_value, int):
            return False
        if target_type == "float" and not isinstance(casted_value, (int, float)):
            return False

        # Gli slot free_text (nome/recapito/messaggio...) non hanno cases: un
        # `pattern` opzionale permette di validarne il FORMATO (es. email/telefono).
        pattern = slot_config.get("pattern")
        if pattern and not re.search(pattern, str(value).strip(), re.IGNORECASE):
            return False

        valid_values = self.get_valid_values_for_slot(intent_name, slot_name)
        if not valid_values:
            return True

        value_lower = str(value).lower()
        return any(value_lower == str(valid).lower() for valid in valid_values)

    def extract_set_slots(self, intent_name: str, current_slots: dict = None) -> dict:
        """
        Slot da impostare lato bot dichiarati in `set_slots` della rule. Valori
        supportati: statici ("Roma", true, 123), riferimenti a slot ("{ALTRO_SLOT}")
        e "$timestamp".
        """
        current_slots = current_slots or {}
        rule = self.rules.get(intent_name)
        if not rule:
            return {}
        return {
            slot_name: self._resolve_slot_value(value, current_slots)
            for slot_name, value in (rule.get("set_slots") or {}).items()
        }

    @staticmethod
    def _resolve_slot_value(value: Any, current_slots: dict) -> Any:
        if not isinstance(value, str):
            return value
        if value.startswith("{") and value.endswith("}"):
            return current_slots.get(value[1:-1])
        if value == "$timestamp":
            return datetime.now().isoformat()
        return value
