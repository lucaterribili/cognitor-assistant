from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.auth import User, get_current_user

router = APIRouter()


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None


@router.post("/message", response_model=ChatResponse)
def send_message(
    chat_message: ChatMessage,
    current_user: User = Depends(get_current_user),
):
    user_message = chat_message.message
    
    response_text = f"Echo: {user_message}"
    
    return ChatResponse(
        response=response_text,
        session_id=chat_message.session_id
    )
