"""
Endpoint di streaming (SSE) per il chatbot — spento di default, vedi
config.STREAMING_ENABLED (mount condizionale in main.py).

L'engine non genera testo token-per-token: seleziona una risposta già pronta
(regola/ML) o esegue un'operation, quasi sempre in pochi millisecondi. Lo
streaming qui non genera nulla di nuovo: rivela la risposta già calcolata
a parola per parola, per un effetto di "digitazione" percepibile e coerente
con l'interfaccia. Il valore reale è l'interruzione: un'operation lenta
(es. web_search) può essere fermata a metà, e la connessione può essere
chiusa dal client in qualsiasi momento — anche mentre il bot sta "generando",
non solo durante lo slot-filling.
"""
import asyncio
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import cancel_registry
from agent.agent import Agent
from agent.turn_processor import TurnProcessor
from api.auth import User, get_current_user
from api.chatbot import ChatMessage
from api.dependencies import get_agent

router = APIRouter()

_WORD_DELAY_SECONDS = 0.05
_DISCONNECT_POLL_SECONDS = 0.2


def _sse(event_type: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    if event_type == "chunk":
        return f"data: {payload}\n\n"
    return f"event: {event_type}\ndata: {payload}\n\n"


@router.post("/message/stream")
async def send_message_stream(
    chat_message: ChatMessage,
    request: Request,
    current_user: User = Depends(get_current_user),
    agent: Agent = Depends(get_agent),
):
    user_message = chat_message.message.strip()
    session_id = chat_message.session_id

    if not session_id:
        session_id = agent.session_manager.create_session(user_id=current_user.username)

    session = agent.session_manager.get_session(session_id)
    if not session:
        session_id = agent.session_manager.create_session(user_id=current_user.username)
        session = agent.session_manager.get_session(session_id)

    cancel_event = cancel_registry.register(session_id)

    async def generate():
        try:
            task = asyncio.create_task(
                asyncio.to_thread(TurnProcessor(agent).process, user_message, session, None, cancel_event)
            )

            # Il turno vero e proprio gira in un thread separato (è codice
            # sincrono/bloccante); qui lo aspettiamo controllando in parallelo
            # se il client ha chiuso la connessione (es. AbortController lato
            # widget), nel qual caso non c'è più nessuno a cui inviare dati.
            while not task.done():
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                await asyncio.sleep(_DISCONNECT_POLL_SECONDS)

            result = task.result()

            words = result.response.split(" ")
            for i, word in enumerate(words):
                if cancel_event.is_set():
                    break
                separator = "" if i == 0 else " "
                yield _sse("chunk", {"chunk": f"{separator}{word}"})
                await asyncio.sleep(_WORD_DELAY_SECONDS)

            # Se nel frattempo è arrivato un cancel esplicito (POST /chatbot/cancel),
            # quello ha già resettato lo stato "in attesa di slot" della sessione:
            # rileggiamo lo stato live invece di fidarci del risultato del turno,
            # che potrebbe essere stato calcolato prima della cancellazione.
            yield _sse("done", {
                "session_id": session_id,
                "intent": result.intent,
                "confidence": result.confidence,
                "entities": result.entities or None,
                "options": result.options,
                "waiting_for_slot": bool(session.waiting_for_slot),
                "cancelled": cancel_event.is_set(),
            })
        finally:
            cancel_registry.clear(session_id)

    return StreamingResponse(generate(), media_type="text/event-stream")


class CancelRequest(BaseModel):
    session_id: str


@router.post("/cancel")
def cancel_message(
    payload: CancelRequest,
    current_user: User = Depends(get_current_user),
    agent: Agent = Depends(get_agent),
):
    """Interrompe il turno in corso per la sessione indicata (bottone "Interrompi"
    del widget, cliccabile anche mentre il bot sta "generando"). Resetta anche lo
    stato di attesa slot della sessione, con la stessa semantica del comando di
    annullamento testuale (TurnProcessor._handle_cancel): dopo uno stop il turno
    successivo parte da zero, non resta legato allo slot del turno interrotto."""
    cancelled = cancel_registry.cancel(payload.session_id)

    session = agent.session_manager.get_session(payload.session_id)
    if session:
        session.waiting_for_slot = None
        session.agent_mode = "predictable"

    return {"cancelled": cancelled}
