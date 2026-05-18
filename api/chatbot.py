from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.auth import User, get_current_user
from api.dependencies import get_agent
from agent.agent import Agent

router = APIRouter()


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    entities: Optional[list] = None


@router.post("/message", response_model=ChatResponse)
def send_message(
    chat_message: ChatMessage,
    current_user: User = Depends(get_current_user),
    agent: Agent = Depends(get_agent),
):
    user_message = chat_message.message.strip()
    session_id = chat_message.session_id

    if not session_id:
        session_id = agent.session_manager.create_session(user_id=current_user.username)
    
    session = agent.session_manager.get_session(session_id)
    if not session:
        # Se la sessione è scaduta o non trovata, ne creiamo una nuova
        session_id = agent.session_manager.create_session(user_id=current_user.username)
        session = agent.session_manager.get_session(session_id)

    # Gestione comandi di cancellazione
    cancel_commands = {'#exit', '#annulla', '#cancel', '#abort'}
    if user_message.lower() in cancel_commands and session.agent_mode == "inputable":
        session.waiting_for_slot = None
        session.agent_mode = "predictable"
        response_text = "Input annullato. Come posso aiutarti?"
        session.add_message("user", user_message)
        session.add_message("assistant", response_text, None)
        return ChatResponse(response=response_text, session_id=session_id)

    # Gestione modalità inputable (attesa slot)
    if session.agent_mode == "inputable" and session.waiting_for_slot:
        slot_name = session.waiting_for_slot["slot"]
        pending_intent = session.waiting_for_slot["intent"]

        # Usa il modello anche in inputable per estrarre entità dal testo utente
        prediction = agent.predict(user_message)
        entities = prediction.get('entities', [])
        extracted_slot_value = agent.slot_manager.extractor.extract_from_entities(
            slot_name, entities
        )

        slot_value = extracted_slot_value if extracted_slot_value else user_message
        if extracted_slot_value:
            print(f"[INPUTABLE] Estratto valore slot '{slot_name}' da NER: {extracted_slot_value}")

        if not agent.slot_manager.validate_slot_value(pending_intent, slot_name, slot_value):
            response_text = "Selezione non valida. Riprova."
            session.add_message("user", user_message)
            return ChatResponse(response=response_text, session_id=session_id)

        # Esegui il casting
        casted_value = agent.rule_interpreter.cast_slot_value(pending_intent, slot_name, slot_value)

        session.update_context(slot_name, casted_value)
        session.update_context(f"{slot_name}_UNSUPPORTED", False)
        session.waiting_for_slot = None
        session.agent_mode = "predictable"

        response_text, wait_for_slot, bot_slots = agent.get_response(
            pending_intent, session.context, session.history
        )

        if bot_slots:
            for s_name, s_val in bot_slots.items():
                if s_val:
                    session.update_context(s_name, s_val)
                    session.update_context(f"{s_name}_UNSUPPORTED", False)

        if wait_for_slot:
            session.waiting_for_slot = {"intent": pending_intent, "slot": wait_for_slot}
            session.agent_mode = "inputable"

        session.add_message("user", user_message)
        session.add_message("assistant", response_text, pending_intent)

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            intent=pending_intent
        )

    # Modalità normale
    prediction = agent.predict(user_message)
    
    agent.slot_manager.update_session_from_prediction(
        session=session,
        current_intent=prediction['intent'],
        entities=prediction.get('entities', []),
        user_input=user_message
    )

    response_text, wait_for_slot, bot_slots = agent.get_response(
        prediction['intent'], session.context, session.history
    )

    if bot_slots:
        for s_name, s_val in bot_slots.items():
            if s_val:
                session.update_context(s_name, s_val)
                session.update_context(f"{s_name}_UNSUPPORTED", False)

    if wait_for_slot:
        session.waiting_for_slot = {"intent": prediction['intent'], "slot": wait_for_slot}
        session.agent_mode = "inputable"

    session.add_message("user", user_message, prediction['intent'], prediction.get('entities', []))
    session.add_message("assistant", response_text, prediction['intent'])

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        intent=prediction['intent'],
        confidence=prediction['confidence'],
        entities=prediction.get('entities', [])
    )
