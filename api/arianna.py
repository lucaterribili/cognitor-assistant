"""
Arianna / Rasa Webhook Compatibility Router for Cognitor Assistant.
Allows seamless communication with chatbot-chat-api (Spring Boot) and legacy webhook clients.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    User,
    _ADMIN_USERNAME,
)
from api.dependencies import get_agent
from agent.agent import Agent
from agent.turn_processor import TurnProcessor

router = APIRouter()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class WebhookMessageRequest(BaseModel):
    message: Optional[str] = None
    text: Optional[str] = None
    sender: Optional[str] = None
    sessionId: Optional[str] = None
    session_id: Optional[str] = None


class WebhookMessageResponse(BaseModel):
    recipient_id: str
    sessionId: str
    text: str


@router.post("/token", response_model=TokenResponse)
async def webhook_token(request: Request):
    """
    Rasa / Arianna webhook token endpoint.
    Accepts application/json {"username": "admin", "password": "..."}
    or application/x-www-form-urlencoded form.
    """
    content_type = request.headers.get("content-type", "")
    username = _ADMIN_USERNAME
    password = ""

    if "application/json" in content_type:
        try:
            body = await request.json()
            username = body.get("username", _ADMIN_USERNAME)
            password = body.get("password", "")
        except Exception:
            pass
    elif "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
            username = form.get("username", _ADMIN_USERNAME)
            password = form.get("password", "")
        except Exception:
            pass
    else:
        try:
            body = await request.json()
            username = body.get("username", _ADMIN_USERNAME)
            password = body.get("password", "")
        except Exception:
            pass

    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user["username"]})
    return TokenResponse(access_token=token, token_type="bearer")


@router.post("/message", response_model=List[WebhookMessageResponse])
def webhook_message(
    chat_message: WebhookMessageRequest,
    current_user: User = Depends(get_current_user),
    agent: Agent = Depends(get_agent),
):
    """
    Rasa / Arianna webhook message endpoint.
    Receives {"sender": session_id, "message": text} or {"sessionId": ..., "text": ...}
    Processes via Cognitor TurnProcessor and returns [{recipient_id: session_id, text: reply}].
    """
    user_text = (chat_message.message or chat_message.text or "").strip()
    session_id = chat_message.sender or chat_message.sessionId or chat_message.session_id or "user"

    session = agent.session_manager.get_session(session_id)
    if not session:
        session_id = agent.session_manager.create_session(user_id=current_user.username, session_id=session_id)
        session = agent.session_manager.get_session(session_id)

    result = TurnProcessor(agent).process(user_text, session)

    return [
        WebhookMessageResponse(
            recipient_id=session_id,
            sessionId=session_id,
            text=result.response or ""
        )
    ]
