"""
Contratto chat v1: l'unico punto di ingresso per token e messaggio.

    POST /api/oauth2/token   Basic client_id:secret + form grant_type=client_credentials&scope=CHAT
                             -> {access_token, token_type, expires_in}
    POST /api/v1/chat        Bearer, JSON {session_id, text}
                             -> [{session_id, text, link, options, waiting_for_slot}]
    POST /api/v1/chat/reset  Bearer, JSON {session_id}

Un client (es. un plugin CMS) configurato con Token URL + Chat URL funziona allo stesso
modo contro qualunque servizio che esponga questo contratto, con o senza un gateway davanti.

Il client_id presente nel JWT (claim `sub`) fa da tenant: le sessioni sono isolate per
client_id, quindi due siti (o due visitatori) non condividono mai stato di conversazione.

Client ammessi: COGNITOR_CLIENTS="id1:secret1,id2:secret2" (il segreto può contenere ':').
Se non impostata, l'unico client è l'utente di servizio AUTH_ADMIN_USERNAME/AUTH_ADMIN_PASSWORD.
"""
import base64
import binascii
import hashlib
import hmac
import os
import re
from datetime import timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Form, Header, HTTPException, status
from pydantic import AliasChoices, BaseModel, Field

import config  # noqa: F401  (assicura che .env sia caricato prima di leggere le env var)
from agent.agent import Agent
from agent.turn_processor import TurnProcessor
from api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    _ADMIN_PASSWORD,
    _ADMIN_USERNAME,
    User,
    create_access_token,
    get_current_user,
)
from api.dependencies import get_agent

oauth_router = APIRouter()
chat_router = APIRouter()

ALLOWED_SCOPES = {"CHAT"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ---------------------------------------------------------------------------
# Client credentials
# ---------------------------------------------------------------------------

def _load_clients() -> dict:
    raw = os.getenv("COGNITOR_CLIENTS", "").strip()
    clients = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        client_id, sep, secret = entry.partition(":")
        if sep and client_id and secret:
            clients[client_id] = secret
    if not clients:
        clients[_ADMIN_USERNAME] = _ADMIN_PASSWORD
    return clients


def _valid_client(client_id: str, secret: str) -> bool:
    expected = _load_clients().get(client_id)
    if expected is None:
        # confronto comunque a tempo costante per non rivelare se il client esiste
        hmac.compare_digest(secret.encode(), b"x")
        return False
    return hmac.compare_digest(secret.encode(), expected.encode())


def _invalid_client() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "invalid_client"},
        headers={"WWW-Authenticate": 'Basic realm="cognitor"'},
    )


def _parse_basic(authorization: Optional[str]) -> Optional[tuple]:
    if not authorization or not authorization.lower().startswith("basic "):
        return None
    try:
        decoded = base64.b64decode(authorization[6:].strip(), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    client_id, sep, secret = decoded.partition(":")
    if not sep:
        return None
    return client_id, secret


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int


@oauth_router.post("/token", response_model=TokenResponse)
def token(
    authorization: Optional[str] = Header(default=None),
    grant_type: str = Form(...),
    scope: str = Form(default="CHAT"),
    client_id: Optional[str] = Form(default=None),
    client_secret: Optional[str] = Form(default=None),
):
    if grant_type != "client_credentials":
        raise HTTPException(status_code=400, detail={"error": "unsupported_grant_type"})
    if scope.upper() not in ALLOWED_SCOPES:
        raise HTTPException(status_code=400, detail={"error": "invalid_scope"})

    # RFC 6749 §2.3.1: Basic preferito, credenziali nel body tollerate
    credentials = _parse_basic(authorization)
    if credentials is None and client_id and client_secret:
        credentials = (client_id, client_secret)
    if credentials is None or not _valid_client(*credentials):
        raise _invalid_client()

    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": credentials[0], "scope": scope.upper()}, expires_delta=expires
    )
    return TokenResponse(access_token=access_token, expires_in=int(expires.total_seconds()))


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    # `session_id` è il nome del contratto; gli alias coprono i widget che mandano
    # `sessionId` (plugin WordPress storico) o `sender` (stile Rasa).
    session_id: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("session_id", "sessionId", "sender")
    )
    text: str = Field(validation_alias=AliasChoices("text", "message"))


class ChatResetRequest(BaseModel):
    session_id: str = Field(validation_alias=AliasChoices("session_id", "sessionId", "sender"))


class ChatResponse(BaseModel):
    session_id: str
    text: str
    link: Optional[str] = None
    options: Optional[list] = None
    waiting_for_slot: bool = False


def _slug(value: str) -> str:
    return value if _SAFE_ID.match(value) else hashlib.sha256(value.encode()).hexdigest()[:32]


def internal_session_id(client_id: str, session_id: str) -> str:
    """Id di sessione interno: namespace per client + valore sicuro.

    L'id finisce nel nome del file di log della sessione, quindi un valore scelto dal
    client non può contenere separatori di percorso: se non è [A-Za-z0-9_-]{1,64}
    viene sostituito da un hash.
    """
    return f"{_slug(client_id)}__{_slug(session_id)}"


@chat_router.post("/chat", response_model=List[ChatResponse])
def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    agent: Agent = Depends(get_agent),
):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail={"error": "text_required"})

    client_session_id = request.session_id or os.urandom(16).hex()
    internal_id = internal_session_id(current_user.username, client_session_id)

    session = agent.session_manager.get_session(internal_id)
    if not session:
        agent.session_manager.create_session(user_id=current_user.username, session_id=internal_id)
        session = agent.session_manager.get_session(internal_id)

    result = TurnProcessor(agent).process(text, session)

    return [
        ChatResponse(
            session_id=client_session_id,
            text=result.response or "",
            options=result.options or None,
            waiting_for_slot=bool(result.wait_for_slot),
        )
    ]


@chat_router.post("/chat/reset", status_code=status.HTTP_200_OK)
def reset(
    request: ChatResetRequest,
    current_user: User = Depends(get_current_user),
    agent: Agent = Depends(get_agent),
):
    agent.session_manager.delete_session(internal_session_id(current_user.username, request.session_id))
    return {}
