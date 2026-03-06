# Quick Start

Segui questi passaggi per avviare Cognitor Assistant in pochi minuti.

## Prerequisiti

- Python 3.10+
- pip

## Installazione

### 1. Clona il repository

```bash
git clone https://github.com/lucaterribili/cognitor-assistant.git
cd cognitor-assistant
```

### 2. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 3. Avvia il server API

```bash
uvicorn main:app --reload
```

Il server sarà disponibile su `http://localhost:8000`.

## Primo Utilizzo

### Ottieni un token di accesso

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=admin123"
```

La risposta sarà:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Invia un messaggio al chatbot

```bash
curl -X POST http://localhost:8000/chatbot/message \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Ciao!"}'
```

## Modalità Interattiva

Puoi anche avviare l'agent in modalità interattiva da terminale:

```bash
python -m agent.agent
```

## Prossimi Passi

- Consulta la [guida all'installazione](/guide/installation) per la configurazione avanzata
- Esplora la [knowledge base](/guide/knowledge-base) per personalizzare gli intenti
- Leggi la [documentazione API](/api/) per l'integrazione
