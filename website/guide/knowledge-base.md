# Knowledge Base

La knowledge base è il cuore configurabile di Cognitor Assistant. È organizzata in tre sezioni principali.

## Intenti (`knowledge/intents/`)

Gli intenti definiscono le categorie di richieste che l'utente può fare, insieme agli esempi di frasi.

### Formato YAML

```yaml
nlu:
  intents:
    - intent: greeting
      examples:
        - "Ciao"
        - "Buongiorno"
        - "Salve, come stai?"
    
    - intent: open_app
      examples:
        - "apri [WhatsApp](PRODUCT)"
        - "apri [Telegram](PRODUCT)"
```

## Regole (`knowledge/rules/`)

Le regole mappano ogni intento alle risposte appropriate, con supporto per slot condizionali.

```yaml
rules:
  open_app:
    slots:
      PRODUCT:
        required: true
        type: string
        entity: PRODUCT
    
    cases:
      WhatsApp: open_app_whatsapp_response
      Telegram: open_app_telegram_response
    
    fallback: open_app_unsupported_response
    wait: open_app_wait_response
```

## Risposte (`knowledge/responses/`)

I template delle risposte, con supporto per varianti multiple (scelta casuale).

```yaml
responses:
  greeting_response:
    - "Ciao! Come posso aiutarti?"
    - "Salve! In cosa posso esserti utile?"
  
  open_app_whatsapp_response:
    - "Apro WhatsApp per te! 📱"
    - "Sto aprendo WhatsApp..."
```

## Aggiungere un Nuovo Intento

1. Aggiungi il file YAML in `knowledge/intents/`
2. Aggiungi la regola in `knowledge/rules/base.yaml`
3. Aggiungi le risposte in `knowledge/responses/base.yaml`
4. Riesegui la pipeline: `python -m pipeline`
