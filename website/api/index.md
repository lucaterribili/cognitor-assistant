# API Reference

## Base URL

```
http://localhost:8000
```

## Autenticazione

L'API utilizza autenticazione Bearer JWT. Includi il token in ogni richiesta:

```http
Authorization: Bearer <token>
```

## Endpoint

### Health Check

```http
GET /health
```

Verifica lo stato del server.

**Risposta:**
```json
{
  "status": "ok"
}
```

---

Consulta le sezioni dedicate per i dettagli su:
- [Autenticazione](/api/auth) — Login e gestione token
- [Chatbot](/api/chatbot) — Invio messaggi e risposte
