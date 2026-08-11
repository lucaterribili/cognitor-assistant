from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth import User, get_current_user
from api.dependencies import get_agent
from agent.agent import Agent
from agent.turn_processor import TurnProcessor

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
    options: Optional[list] = None
    # True quando il turno lascia la sessione in modalità "inputable" (in attesa
    # del valore di uno slot, session.waiting_for_slot popolato da TurnProcessor):
    # il client può usarlo per mostrare un modo per interrompere il flusso invece
    # di dover indovinare lo stato dall'assenza di altri segnali.
    waiting_for_slot: bool = False


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

    result = TurnProcessor(agent).process(user_message, session)

    return ChatResponse(
        response=result.response,
        session_id=session_id,
        intent=result.intent,
        confidence=result.confidence,
        entities=result.entities or None,
        options=result.options,
        waiting_for_slot=bool(result.wait_for_slot),
    )


@router.get("/intents")
def list_intents(
    current_user: User = Depends(get_current_user),
    agent: Agent = Depends(get_agent),
):
    """Elenco degli intent noti al classificatore, usato dall'admin di
    geco/chatbot per popolare la selezione degli intent da disabilitare
    (vedi agent/disabled_intents.py)."""
    return {"intents": sorted(set(agent.intent_dict.values()))}
