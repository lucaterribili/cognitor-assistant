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
from datetime import datetime

from agent.response_slot_parser import ResponseSlotParser
from agent.operations.manager import OperationManager


class RuleInterpreter:
    """
    Interprete generico per le rules in formato DSL.

    Legge la configurazione YAML e decide quale risposta dare
    basandosi sugli slot disponibili.
    """

    def __init__(self, rules: dict, responses: dict, operation_manager: OperationManager | None = None):
        """
        Args:
            rules: Dizionario delle rules caricato da YAML
            responses: Dizionario delle responses caricato da YAML
            operation_manager: Gestore delle operations (opzionale)
        """
        self.rules = rules
        self.responses = responses
        self.operation_manager = operation_manager

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
        1. Controlla se ci sono slot required non forniti → wait (per-slot o globale)
        2. Prova chiave composita SLOT1|SLOT2 nei cases (multi-slot)
        3. Matching singolo sul primo slot con valore
        4. Altrimenti → fallback o default
        """
        rule_slots = rule.get("slots", {})
        wait_config = rule.get("wait")

        # Step 1: Controlla slot required nell'ordine di definizione
        for slot_name, slot_config in rule_slots.items():
            if slot_config.get("required", False):
                slot_value = slots.get(slot_name)
                unsupported_flag = slots.get(f"{slot_name}_UNSUPPORTED")

                # Slot non fornito o unsupported → wait
                if not slot_value or unsupported_flag:
                    # Supporta wait come dict per-slot o stringa globale
                    if isinstance(wait_config, dict):
                        wait_key = wait_config.get(slot_name)
                    else:
                        wait_key = wait_config
                    if wait_key:
                        return self._get_random_response(wait_key, slots), slot_name
                    # Fallback se non c'è wait definito
                    fallback_key = rule.get("fallback", rule.get("default"))
                    return self._get_random_response(fallback_key, slots), None

        cases = rule.get("cases", {})

        # Step 2: Prova chiave composita (tutti i required slot con valore, nell'ordine)
        required_values = [
            slots.get(s)
            for s, c in rule_slots.items()
            if c.get("required", False) and slots.get(s)
        ]
        if len(required_values) > 1:
            composite_key = "|".join(str(v) for v in required_values)
            for case_key, response_key in cases.items():
                if composite_key.lower() == str(case_key).lower():
                    return self._get_random_response(response_key, slots), None

        # Step 3: Matching singolo (primo slot con valore)
        for slot_name, slot_config in rule_slots.items():
            slot_value = slots.get(slot_name)
            if slot_value:
                for case_key, response_key in cases.items():
                    if str(slot_value).lower() == str(case_key).lower():
                        return self._get_random_response(response_key, slots), None

                # Valore non trovato nei cases → fallback
                fallback_key = rule.get("fallback")
                if fallback_key:
                    return self._get_random_response(fallback_key, slots), None

        # Step 4: Default se disponibile
        default_key = rule.get("default")
        if default_key:
            return self._get_random_response(default_key, slots), None

        return "Nessuna risposta configurata", None

    def _slot_applies(self, slot_config: dict, slots: dict) -> bool:
        """
        Verifica se uno slot è pertinente nel turno corrente.

        Uno slot può dichiarare `when: {altro_slot: valore}` per essere richiesto solo
        quando un altro slot già raccolto ha un certo valore (es. lo slot "category" si
        applica solo se "content_type" == "article"). Senza `when`, lo slot si applica
        sempre. Usato per rendere condizionale la partecipazione di uno slot al flusso di
        raccolta, non solo la sua obbligatorietà.

        Args:
            slot_config: Configurazione dello slot
            slots: Slot correnti

        Returns:
            True se lo slot è pertinente nel turno corrente
        """
        when = slot_config.get("when")
        if not when:
            return True
        return all(str(slots.get(k)) == str(v) for k, v in when.items())

    def _get_slot_options(self, slot_config: dict, slots: dict) -> Optional[list]:
        """
        Calcola le opzioni (bottoni) disponibili per uno slot in attesa.

        Supporta due sorgenti:
        - `options`: lista statica dichiarata direttamente nella rule.
        - `options_source`: nome di un provider dinamico registrato nell'OperationManager
          (funzione `options_<nome>` in agent/operations/*.py), interrogato con gli slot
          correnti (utile per opzioni dipendenti da uno slot già raccolto).

        Args:
            slot_config: Configurazione dello slot in attesa
            slots: Slot correnti

        Returns:
            Lista di opzioni [{"value": ..., "label": ...}, ...] oppure None se lo slot
            non ha opzioni configurate
        """
        if "options" in slot_config:
            return slot_config["options"]

        options_source = slot_config.get("options_source")
        if options_source and self.operation_manager:
            return self.operation_manager.get_options(options_source, slots)

        return None

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

        response = random.choice(response_list)

        for slot_name, slot_value in slots.items():
            if slot_value and not slot_name.endswith("_UNSUPPORTED"):
                placeholder = f"{{{slot_name}}}"
                response = response.replace(placeholder, str(slot_value))

        # Sostituisce i placeholder temporali con i valori reali
        now = datetime.now()
        response = response.replace("[TIME]", now.strftime("%H:%M"))
        response = response.replace("[DATE]", now.strftime("%d/%m/%Y"))

        return response

    def _get_response_with_slots(
        self, response_key: str, slots: dict
    ) -> tuple[str, dict]:
        """
        Ottiene una risposta random, sostituisce placeholder ed estrae slot inline.

        Args:
            response_key: Chiave della response
            slots: Dizionario degli slot

        Returns:
            tuple: (risposta_processata, slot_inline_trovati)
        """
        response_list = self.responses.get(response_key, [])

        if not response_list:
            return f"Risposta non definita per {response_key}", {}

        response = random.choice(response_list)

        for slot_name, slot_value in slots.items():
            if slot_value and not slot_name.endswith("_UNSUPPORTED"):
                placeholder = f"{{{slot_name}}}"
                response = response.replace(placeholder, str(slot_value))

        # Sostituisce i placeholder temporali con i valori reali
        now = datetime.now()
        response = response.replace("[TIME]", now.strftime("%H:%M"))
        response = response.replace("[DATE]", now.strftime("%d/%m/%Y"))

        cleaned_response, inline_slots = ResponseSlotParser.parse(response)

        return cleaned_response, inline_slots

    def get_valid_values_for_slot(self, intent_name: str, slot_name: str) -> list[str]:
        """
        Estrae i valori validi per uno slot dai cases del DSL.
        Supporta chiavi composite (es. "Roma|musei") estraendo il valore
        nella posizione corretta per il dato slot.

        Se il `default` della rule esegue un'operation (`__nome`), i cases sono
        eccezioni puntuali sovrapposte a un catch-all generico (es. `ask_culture`/
        `web_search` → `__web_search`), non un enum chiuso di valori ammessi come
        in `open_app` (dove i cases SONO l'insieme esaustivo supportato, senza
        operation di default). In quel caso lo slot resta libero: validarlo contro
        i cases rifiuterebbe qualunque ricerca libera che non coincida con
        un'eccezione elencata.

        Args:
            intent_name: Nome dell'intent
            slot_name: Nome dello slot

        Returns:
            Lista di valori validi
        """
        rule = self.rules.get(intent_name)
        if not rule:
            return []

        rule_slots = rule.get("slots", {})
        if slot_name not in rule_slots:
            return []

        if str(rule.get("default", "")).startswith("__"):
            return []

        cases = rule.get("cases", {})
        if not cases:
            return []

        # Se ci sono chiavi composite (contengono |), estrai i valori per la posizione del slot
        has_composite = any("|" in str(k) for k in cases.keys())
        if has_composite:
            required_slots = [s for s, c in rule_slots.items() if c.get("required", False)]
            if slot_name in required_slots:
                slot_position = required_slots.index(slot_name)
                values = set()
                for case_key in cases.keys():
                    parts = str(case_key).split("|")
                    if slot_position < len(parts):
                        values.add(parts[slot_position])
                return list(values)

        # Formato semplice: restituisce tutti i case keys
        return list(cases.keys())

    def cast_slot_value(self, intent_name: str, slot_name: str, value: Any) -> Any:
        """
        Converte il valore dello slot nel tipo configurato nel DSL.

        Supportati: string, integer, float, boolean, list.

        Args:
            intent_name: Nome dell'intent
            slot_name: Nome dello slot
            value: Valore grezzo

        Returns:
            Valore convertito o valore originale se conversione fallisce
        """
        if value is None:
            return None

        slots_config = self.get_slots_for_intent(intent_name)
        slot_config = slots_config.get(slot_name, {})
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
        """
        Valida un valore per uno slot, considerando anche il tipo.

        Args:
            intent_name: Nome dell'intent
            slot_name: Nome dello slot
            value: Valore da validare

        Returns:
            True se valido
        """
        if value is None:
            return False

        # Tenta il casting per validare il tipo
        casted_value = self.cast_slot_value(intent_name, slot_name, value)
        
        slots_config = self.get_slots_for_intent(intent_name)
        slot_config = slots_config.get(slot_name, {})
        target_type = slot_config.get("type", "string").lower()

        # Se il casting ha fallito il cambio di tipo quando richiesto
        if target_type == "integer" and not isinstance(casted_value, int):
            return False
        if target_type == "float" and not isinstance(casted_value, (int, float)):
            return False

        valid_values = self.get_valid_values_for_slot(intent_name, slot_name)

        # Se non ci sono vincoli espliciti sui valori (cases), accetta qualsiasi valore del tipo corretto
        if not valid_values:
            return True

        # Case-insensitive match per stringhe
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

    def extract_set_slots(self, intent_name: str, current_slots: dict = None) -> dict:
        """
        Estrae le azioni set_slots definite nella rule.

        Args:
            intent_name: Nome dell'intent
            current_slots: Slot attuali per risolvere riferimenti dinamici

        Returns:
            Dizionario {slot_name: valore} da impostare
        """
        if current_slots is None:
            current_slots = {}

        rule = self.rules.get(intent_name)
        if not rule:
            return {}

        set_slots_config = rule.get("set_slots", {})
        if not set_slots_config:
            return {}

        resolved_slots = {}
        for slot_name, value in set_slots_config.items():
            resolved_value = self._resolve_slot_value(value, current_slots)
            resolved_slots[slot_name] = resolved_value

        return resolved_slots

    def _resolve_slot_value(self, value: Any, current_slots: dict) -> Any:
        """
        Risolve un valore che può essere statico o dinamico.

        Valori supportati:
        - Valori statici: "Roma", true, 123
        - Riferimenti a slot: "{ALTRO_SLOT}"
        - Variabili speciali: $timestamp, $session_id

        Args:
            value: Valore da risolvere
            current_slots: Slot attuali

        Returns:
            Valore risolto
        """
        if not isinstance(value, str):
            return value

        # Riferimento a slot esistente {SLOT_NAME}
        if value.startswith("{") and value.endswith("}"):
            ref_slot = value[1:-1]
            return current_slots.get(ref_slot)

        # Variabile speciale: $timestamp
        if value == "$timestamp":
            return datetime.now().isoformat()

        # Ritorna il valore così com'è
        return value

    def handle_intent_with_bot_slots(
        self, intent_name: str, slots: dict = None
    ) -> tuple[str, Optional[str], dict]:
        """
        Interpreta una rule e restituisce la risposta + slot da impostare dal bot.

        Args:
            intent_name: Nome dell'intent
            slots: Dizionario degli slot disponibili

        Returns:
            tuple: (risposta, slot_da_attendere, slot_da_impostare)
        """
        if slots is None:
            slots = {}

        rule = self.rules.get(intent_name)
        if not rule:
            return "Intent non trovato nel DSL", None, {}

        bot_slots = self.extract_set_slots(intent_name, slots)

        # Intent con slot: la gestione dell'operation (dopo aver raccolto gli slot)
        # è delegata a _handle_slot_based_intent_with_slots
        if "slots" in rule:
            response, wait_slot, options, inline_slots = self._handle_slot_based_intent_with_slots(rule, slots)
            # Gli slot inline (sintassi {SLOT=value} nel template di risposta) vengono
            # fusi in bot_slots con la stessa strategia usata nel percorso "default" più
            # sotto: inline_slots ha priorità sulle proprie chiavi, senza cancellare le
            # altre eventualmente già impostate da extract_set_slots.
            bot_slots.update(inline_slots)
            # Le opzioni (bottoni) viaggiano nel canale bot_slots con una chiave riservata,
            # per non allargare la tupla pubblica (response, wait_slot, bot_slots) — i
            # chiamanti la estraggono e la rimuovono prima di applicare bot_slots al contesto.
            if options:
                bot_slots["__options__"] = options
            return response, wait_slot, bot_slots

        # Intent semplice senza slot: esegui l'operation se present
        default_key = rule.get("default", "")
        if default_key.startswith("__") and self.operation_manager:
            operation_name = default_key[2:]
            if self.operation_manager.has_operation(operation_name):
                operation_result = self.operation_manager.execute(
                    operation_name, operation_name, slots
                )
                all_bot_slots = {**bot_slots, **operation_result.get("slots", {})}
                return operation_result["response"], None, all_bot_slots

        if "default" in rule:
            response_key = rule["default"]
            response, inline_slots = self._get_response_with_slots(response_key, slots)
            all_bot_slots = {**bot_slots, **inline_slots}
            return response, None, all_bot_slots

        return "Configurazione intent non valida", None, bot_slots

    def _handle_slot_based_intent_with_slots(
        self, rule: dict, slots: dict
    ) -> tuple[str, Optional[str], Optional[list], dict]:
        """
        Gestisce intent che richiedono slot, restituendo anche slot inline.
        Supporta wait per-slot (dict), case compositi multi-slot (SLOT1|SLOT2) e slot
        condizionali (`when:`, vedi `_slot_applies`).

        I `cases` vengono controllati PRIMA dell'operation di `default`: questo
        permette a un dominio di intercettare valori di slot specifici su un
        intent generico a operation fissa (es. `ask_culture`/`web_search` →
        `__web_search`) senza toccare il comportamento per tutti gli altri
        valori. Un case può puntare a una response key statica oppure, con la
        stessa convenzione `__nome` del default, a un'operation (vedi
        `_resolve_case_target`).

        Returns:
            tuple: (risposta, slot_da_attendere, opzioni, slot_inline_trovati)
        """
        rule_slots = rule.get("slots", {})
        wait_config = rule.get("wait")
        default_key = rule.get("default", "")
        cases = rule.get("cases", {})

        # Step 1: Controlla slot required nell'ordine di definizione (saltando quelli
        # non pertinenti nel turno corrente per via di un eventuale `when:`)
        for slot_name, slot_config in rule_slots.items():
            if not self._slot_applies(slot_config, slots):
                continue
            if slot_config.get("required", False):
                slot_value = slots.get(slot_name)
                unsupported_flag = slots.get(f"{slot_name}_UNSUPPORTED")

                if not slot_value or unsupported_flag:
                    # Supporta wait come dict per-slot o stringa globale
                    if isinstance(wait_config, dict):
                        wait_key = wait_config.get(slot_name)
                    else:
                        wait_key = wait_config
                    if wait_key:
                        response, inline_slots = self._get_response_with_slots(wait_key, slots)
                        options = self._get_slot_options(slot_config, slots)
                        return response, slot_name, options, inline_slots
                    fallback_key = rule.get("fallback", rule.get("default"))
                    if fallback_key:
                        response, inline_slots = self._get_response_with_slots(fallback_key, slots)
                        return response, None, None, inline_slots
                    return "Slot richiesto non fornito", None, None, {}

        # Step 2: Prova chiave composita (tutti i required slot pertinenti con valore, nell'ordine)
        required_values = [
            slots.get(s)
            for s, c in rule_slots.items()
            if self._slot_applies(c, slots) and c.get("required", False) and slots.get(s)
        ]
        if len(required_values) > 1:
            composite_key = "|".join(str(v) for v in required_values)
            for case_key, response_key in cases.items():
                if composite_key.lower() == str(case_key).lower():
                    return self._resolve_case_target(response_key, slots)

        # Step 3: Matching singolo (primo slot con valore)
        for slot_name, slot_config in rule_slots.items():
            slot_value = slots.get(slot_name)
            if slot_value:
                for case_key, response_key in cases.items():
                    if str(slot_value).lower() == str(case_key).lower():
                        return self._resolve_case_target(response_key, slots)

                fallback_key = rule.get("fallback")
                if fallback_key:
                    response, inline_slots = self._get_response_with_slots(fallback_key, slots)
                    return response, None, None, inline_slots

        # Nessun case ha fatto match: tutti gli slot required (pertinenti) sono
        # presenti. Se è definita un'operation (default: __<name>), eseguila ora.
        if default_key.startswith("__") and self.operation_manager:
            operation_name = default_key[2:]
            if self.operation_manager.has_operation(operation_name):
                op_result = self.operation_manager.execute(operation_name, operation_name, slots)
                return op_result["response"], None, None, {}

        if default_key and not default_key.startswith("__"):
            response, inline_slots = self._get_response_with_slots(default_key, slots)
            return response, None, None, inline_slots

        return "Nessuna risposta configurata", None, None, {}

    def _resolve_case_target(self, response_key: str, slots: dict) -> tuple[str, None, None, dict]:
        """
        Risolve il target di un `case` matchato: se usa la stessa convenzione
        `__nome` del `default` esegue l'operation corrispondente, altrimenti
        tratta `response_key` come una normale response key statica.

        Args:
            response_key: Valore del case matchato (response key o `__operation`)
            slots: Slot correnti da passare all'operation o da interpolare nella response

        Returns:
            tuple: (risposta, None, None, slot_inline_trovati)
        """
        if response_key.startswith("__") and self.operation_manager:
            operation_name = response_key[2:]
            if self.operation_manager.has_operation(operation_name):
                op_result = self.operation_manager.execute(operation_name, operation_name, slots)
                return op_result["response"], None, None, {}

        response, inline_slots = self._get_response_with_slots(response_key, slots)
        return response, None, None, inline_slots

