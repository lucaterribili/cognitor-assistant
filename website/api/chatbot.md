# Chatbot

## Invia un Messaggio

```http
POST /chatbot/message
Authorization: Bearer <token>
Content-Type: application/json
```

**Body:**

```json
{
  "message": "Ciao, come stai?"
}
```

**Esempio:**

```bash
curl -X POST http://localhost:8000/chatbot/message \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Ciao!"}'
```

**Risposta:**

```json
{
  "response": "Ciao! Come posso aiutarti?",
  "intent": "greeting",
  "confidence": 0.97
}
```

## Parametri della Risposta

| Campo | Tipo | Descrizione |
|-------|------|-------------|
| `response` | string | Risposta dell'assistente |
| `intent` | string | Intento classificato |
| `confidence` | float | Punteggio di confidenza (0–1) |

## Errori

| Codice | Descrizione |
|--------|-------------|
| `401` | Token mancante o non valido |
| `422` | Dati della richiesta non validi |
| `500` | Errore interno del server |
