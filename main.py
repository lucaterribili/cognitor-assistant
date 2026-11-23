import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config  # assicura che .env sia caricato prima di leggere le env var sotto

app = FastAPI(title="Cognitor Assistant API")

# L'unico chiamante previsto è il gateway Spring Boot (server-to-server, non un browser),
# che autorizza le richieste in base al dominio prima di inoltrarle qui: CORS non è la
# barriera di sicurezza reale (quella è il JWT), ma non c'è motivo di lasciare "*" con
# allow_credentials=True. Nessuna origine autorizzata di default: va impostata esplicitamente
# CORS_ALLOWED_ORIGINS (comma-separated) solo se serve davvero chiamare l'API da browser (es. docs).
_cors_allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api import auth, chatbot, chat_v1

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chatbot.router, prefix="/chatbot", tags=["chatbot"])
# Contratto chat v1: unico punto di ingresso per token (/api/oauth2/token) e messaggio (/api/v1/chat)
app.include_router(chat_v1.oauth_router, prefix="/api/oauth2", tags=["chat-v1"])
app.include_router(chat_v1.chat_router, prefix="/api/v1", tags=["chat-v1"])

# Endpoint di streaming, spento di default (config.STREAMING_ENABLED, abilitato
# con COGNITOR_STREAMING_ENABLED=true all'avvio): un client che non lo sa deve
# trovare un 404 pulito, non un endpoint montato ma inerte.
if config.STREAMING_ENABLED:
    from api import chatbot_stream

    app.include_router(chatbot_stream.router, prefix="/chatbot", tags=["chatbot-stream"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
