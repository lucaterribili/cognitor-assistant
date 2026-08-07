"""
Motore condiviso della macchina a stati conversazionale.

Sia l'interfaccia testuale (agent/conversation_handler.py) sia l'endpoint HTTP
(api/chatbot.py) devono gestire lo stesso ciclo per ogni turno: comando di
annullamento -> raccolta slot in corso (modalità "inputable") -> classificazione
intent. Prima di questo modulo le due interfacce reimplementavano la stessa
logica in modo indipendente, ed erano finite fuori sincrono più volte (es. un
fix a un'invariante di stato applicato a una sola delle due copie). Qui vive
un'unica implementazione; CLI e API restano responsabili solo dell'I/O
(stampa a schermo vs risposta JSON).
"""
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.cancel_commands import is_cancel_command
from config import INPUTABLE_SWITCH_CONFIDENCE


@dataclass
class TurnResult:
    """Esito di un turno, con tutti i dati che un adapter (CLI o HTTP) può volere."""
    kind: str  # "cancel" | "slot_invalid" | "slot_filled" | "prediction"
    response: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    entities: list = field(default_factory=list)
    options: Optional[list] = None
    wait_for_slot: Optional[str] = None
    prediction: Optional[dict] = None
    slot_name: Optional[str] = None
    ner_slot_value: Optional[str] = None
    casted_value: Any = None
    casted_type: Optional[str] = None


class TurnProcessor:
    """Esegue un turno di conversazione per una sessione data."""

    def __init__(self, agent):
        self.agent = agent

    def process(self, user_input: str, session, on_predict=None) -> TurnResult:
        """Elabora un turno. `on_predict`, se fornito, viene invocato con il dict di
        predizione subito dopo la classificazione (solo nel percorso non-inputable) e
        prima di eseguire get_response — permette all'adapter CLI di stampare il debug
        di intent/entità nello stesso punto cronologico in cui veniva stampato prima
        di questo refactor, senza spargere formattazione da terminale in questa classe."""
        awaiting_slot = session.agent_mode == "inputable" and session.waiting_for_slot

        if awaiting_slot and is_cancel_command(user_input):
            return self._handle_cancel(user_input, session)

        if awaiting_slot:
            return self._handle_slot_input(user_input, session, on_predict)

        return self._handle_prediction(user_input, session, on_predict)

    def _handle_cancel(self, user_input: str, session) -> TurnResult:
        session.waiting_for_slot = None
        session.agent_mode = "predictable"
        response_text = "Input annullato. Puoi fornire un nuovo comando."
        session.add_message("user", user_input)
        session.add_message("assistant", response_text, None)
        return TurnResult(kind="cancel", response=response_text)

    def _handle_slot_input(self, user_input: str, session, on_predict=None) -> TurnResult:
        slot_name = session.waiting_for_slot["slot"]
        pending_intent = session.waiting_for_slot["intent"]

        # Prova a estrarre il valore dello slot tramite NER, altrimenti usa il testo grezzo
        prediction = self.agent.predict(user_input)
        entities = prediction.get('entities', [])
        ner_value = self.agent.slot_manager.extractor.extract_from_entities(pending_intent, slot_name, entities)

        # Via di fuga dalla modalità inputable: se il NER non trova nulla di
        # pertinente per lo slot atteso E il messaggio classifica con alta
        # confidenza come un intent diverso, l'utente ha quasi certamente
        # cambiato argomento — non ha senso forzarlo come valore di slot
        # (rischio di restare bloccati in loop "Selezione non valida").
        # Controllo fatto DOPO il tentativo NER: se il NER trova comunque
        # un'entità del tipo giusto (es. una città per lo slot LOCATION),
        # quella resta prioritaria anche se l'intent complessivo della frase
        # è un altro (es. "Roma" da solo classifica come
        # choose_flight_destination ma è comunque una risposta valida allo
        # slot LOCATION di book_flight).
        if (
            not ner_value
            and prediction['intent'] != pending_intent
            and prediction['intent'] != 'low_confidence_fallback'
            and prediction['confidence'] >= INPUTABLE_SWITCH_CONFIDENCE
        ):
            print(f"[INPUTABLE] Cambio di contesto rilevato → '{prediction['intent']}' "
                  f"(confidenza={prediction['confidence']:.2f}) abbandona lo slot '{slot_name}' "
                  f"di '{pending_intent}'")
            session.waiting_for_slot = None
            session.agent_mode = "predictable"
            return self._build_prediction_result(user_input, session, prediction, on_predict)

        slot_value = ner_value if ner_value else user_input
        if ner_value:
            print(f"[INPUTABLE] NER → slot '{slot_name}' estratto: '{ner_value}'")
        else:
            print(f"[INPUTABLE] NER non ha trovato '{slot_name}', uso testo grezzo")

        if not self.agent.slot_manager.validate_slot_value(pending_intent, slot_name, slot_value):
            response_text = "Selezione non valida. Riprova."
            session.add_message("user", user_input)
            session.add_message("assistant", response_text, pending_intent)
            return TurnResult(
                kind="slot_invalid",
                response=response_text,
                intent=pending_intent,
                prediction=prediction,
                slot_name=slot_name,
                ner_slot_value=ner_value,
            )

        # Esegui il casting prima di salvare nel contesto
        casted_value = self.agent.rule_interpreter.cast_slot_value(pending_intent, slot_name, slot_value)

        session.update_context(slot_name, casted_value)
        session.update_context(f"{slot_name}_UNSUPPORTED", False)
        session.waiting_for_slot = None
        session.agent_mode = "predictable"
        print(f"[INPUTABLE] Slot '{slot_name}' impostato = '{casted_value}' "
              f"(type: {type(casted_value).__name__})")

        response_text, wait_for_slot, bot_slots = self.agent.get_response(
            pending_intent, session.context, session.history, raw_text=user_input
        )
        options = self._apply_bot_slots(session, bot_slots)

        if wait_for_slot:
            session.waiting_for_slot = {"intent": pending_intent, "slot": wait_for_slot}
            session.agent_mode = "inputable"

        session.add_message("user", user_input)
        session.add_message("assistant", response_text, pending_intent)

        return TurnResult(
            kind="slot_filled",
            response=response_text,
            intent=pending_intent,
            options=options,
            wait_for_slot=wait_for_slot,
            prediction=prediction,
            slot_name=slot_name,
            ner_slot_value=ner_value,
            casted_value=casted_value,
            casted_type=type(casted_value).__name__,
        )

    def _handle_prediction(self, user_input: str, session, on_predict=None) -> TurnResult:
        prediction = self.agent.predict(user_input)
        return self._build_prediction_result(user_input, session, prediction, on_predict)

    def _build_prediction_result(self, user_input: str, session, prediction: dict, on_predict=None) -> TurnResult:
        if on_predict:
            on_predict(prediction)

        self.agent.slot_manager.update_session_from_prediction(
            session=session,
            current_intent=prediction['intent'],
            entities=prediction.get('entities', []),
            user_input=user_input
        )

        response_text, wait_for_slot, bot_slots = self.agent.get_response(
            prediction['intent'], session.context, session.history, raw_text=user_input
        )
        options = self._apply_bot_slots(session, bot_slots)

        if wait_for_slot:
            session.waiting_for_slot = {"intent": prediction['intent'], "slot": wait_for_slot}
            session.agent_mode = "inputable"

        session.add_message("user", user_input, prediction['intent'], prediction.get('entities', []))
        session.add_message("assistant", response_text, prediction['intent'])

        return TurnResult(
            kind="prediction",
            response=response_text,
            intent=prediction['intent'],
            confidence=prediction['confidence'],
            entities=prediction.get('entities', []),
            options=options,
            wait_for_slot=wait_for_slot,
            prediction=prediction,
        )

    @staticmethod
    def _apply_bot_slots(session, bot_slots: Optional[dict[str, Any]]) -> Optional[list]:
        """Applica gli slot impostati dal bot al contesto sessione e ne estrae le
        opzioni (bottoni) eventualmente allegate dal canale riservato `__options__`."""
        if not bot_slots:
            return None
        options = bot_slots.pop("__options__", None)
        for slot_name, slot_value in bot_slots.items():
            if slot_value:
                session.update_context(slot_name, slot_value)
                session.update_context(f"{slot_name}_UNSUPPORTED", False)
                print(f"[BotSlot] Impostato {slot_name} = {slot_value}")
        return options
