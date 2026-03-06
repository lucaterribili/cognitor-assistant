# Autenticazione

## Ottieni un Token

```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded
```

**Parametri:**

| Campo | Tipo | Descrizione |
|-------|------|-------------|
| `username` | string | Nome utente |
| `password` | string | Password |

**Esempio:**

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=admin123"
```

**Risposta:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

## Ottieni l'Utente Corrente

```http
GET /auth/me
Authorization: Bearer <token>
```

**Risposta:**

```json
{
  "username": "admin",
  "email": "admin@example.com",
  "disabled": false
}
```

## Errori

| Codice | Descrizione |
|--------|-------------|
| `401` | Credenziali non valide o token scaduto |
| `422` | Dati della richiesta non validi |
