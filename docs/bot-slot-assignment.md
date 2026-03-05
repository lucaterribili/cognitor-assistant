# Strategia: Assegnazione Slot da Parte del Bot

## Problema
Attualmente gli slot vengono assegnati solo attraverso l'estrazione delle entità dagli intenti dell'utente. Il bot non può assegnare slot autonomamente.

## Soluzione Proposta

Il bot può assegnare slot attraverso due meccanismi:

### 1. Assegnazione via Rules (bot-side slot setting)

Le rules possono definire un'azione `set_slot` che imposta uno slot automaticamente quando la rule viene eseguita.

**Sintassi nelle rules:**
```yaml
rules:
  intent_example:
    default: response_key
    set_slots:
      SLOT_NAME: "valore_fisso"
      ALTRO_SLOT: "{dynamic_value}"
```

**Esempio concreto:**
```yaml
rules:
  confirm_location:
    default: confirm_location_response
    set_slots:
      LOCATION: "Roma"
      CONFIRMATION: true
```

#### Valori dinamici supportati:
- **Valori fissi**: `"Roma"`, `true`, `123`
- **Riferimenti a slot esistenti**: `{ALTRO_SLOT}` - copia il valore da un altro slot
- **Valori speciali**:
  - `$context.VARIABILE` - usa variabili del contesto
  - `$timestamp` - data/ora corrente
  - `$session_id` - ID della sessione

### 2. Assegnazione via Risposte (embedded slot setting)

Le risposte possono contenere istruzioni inline per impostare slot. Questo è utile per conferme o confermazioni implicite.

**Sintassi nelle risposte:**
```
Risposta normale {SLOT_NAME=value} e altro {ALTRO_SLOT=altro_valore}
```

**Esempio:**
```yaml
responses:
  confirm_location_response:
    - "Perfetto, imposterò la località come {LOCATION=Roma}. Vuoi procedere?"
```

#### Parser delle istruzioni inline:
- Trova tutti i pattern `{SLOT_NAME=value}` nella risposta
- Estrae il nome dello slot e il valore
- Rimuove l'istruzione dalla risposta visualizzata all'utente
- Imposta lo slot nel contesto

### 3. Ordine di Valutazione

1. **Rules `set_slots`** - Ha priorità più alta, eseguito per primo
2. **Risposte inline** - Ha priorità più bassa, eseguito dopo

## Implementazione

### Modifiche necessarie:

1. **RuleInterpreter**: 
   - Aggiungere metodo per estrarre `set_slots` dalla rule
   - Eseguire il setting degli slot prima di generare la risposta

2. **AnswerManager** (se usato):
   - Aggiungere parser per istruzioni inline `{SLOT=value}`
   - Restituire anche i slot da impostare oltre alla risposta

3. **ConversationHandler**:
   - Dopo aver ottenuto la risposta, processare le istruzioni inline
   - Aggiornare il contesto con i nuovi slot

4. **SlotManager**:
   - Aggiungere metodo per impostare slot manualmente (da rule o risposta)
   - Mantenere la validazione dei valori

## Esempi d'Uso

### Esempio 1: Conferma automatica via Rule
```yaml
rules:
  user_says_yes_to_roma:
    default: confirmed_roma_response
    set_slots:
      LOCATION: "Roma"
      CONFIRMATION: true
```

Quando l'utente conferma "sì" per Roma, il bot imposta automaticamente lo slot LOCATION a "Roma".

### Esempio 2: Copia valore via Rule
```yaml
rules:
  copy_location_to_temp:
    default: temp_location_response
    set_slots:
      TEMP_LOCATION: "{LOCATION}"
```

### Esempio 3: Istruzioni inline nella risposta
```yaml
responses:
  ask_confirmation_response:
    - "Imposto Roma come località {LOCATION=Roma}. Confermi?"
```

L'utente vedrà: "Imposto Roma come località. Confermi?"
Il bot avrà impostato LOCATION="Roma".

## Considerazioni

1. **Validazione**: Gli slot impostati dal bot devono comunque passare la validazione (se presente)
2. **Ciclo infinito**: Evitare situazioni dove il bot continua a impostare slot in loop
3. **Debug**: Loggare quando il bot imposta slot automaticamente
4. **Precedenza**: Gli slot impostati dall'utente (da entità) hanno precedenza su quelli impostati dal bot?

## Roadmap

- [x] Implementare parsing `set_slots` nel RuleInterpreter
- [x] Implementare parsing istruzioni inline nelle risposte
- [x] Aggiornare ConversationHandler per processare i nuovi slot
- [ ] Aggiungere test per i nuovi comportamenti
- [ ] Documentare la sintassi nel DSL
