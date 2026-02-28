from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Arianna Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api import auth, chatbot

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chatbot.router, prefix="/chatbot", tags=["chatbot"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
