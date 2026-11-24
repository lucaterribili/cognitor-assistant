from enum import Enum


class Action(str, Enum):
    """Cosa fa il bot a un turno. Sono generiche (valgono per ogni sito): le risposte vere vengono da regole/RAG/QA."""
    ANSWER = "answer"              # risponde alla domanda (regola, casi, RAG o modello QA)
    ASK_SLOT = "ask_slot"          # chiede l'entità che manca ("di quale dipartimento parli?")
    CLARIFY = "clarify"            # chiede di riformulare / di precisare
    FALLBACK = "fallback"          # ammette di non sapere (fuori argomento o nessun contenuto trovato)
    GREET = "greet"
    GOODBYE = "goodbye"
    THANKS_REPLY = "thanks_reply"  # "prego, serve altro?"
    HANDOFF = "handoff"            # passa a un operatore
    DEESCALATE = "deescalate"      # risponde a un insulto
    OFFER_MORE = "offer_more"      # "posso aiutarti con altro?"
    PROMPT = "prompt"              # "dimmi pure" (invito ad andare avanti)
    REPEAT = "repeat"              # ripete l'ultima risposta


ACTIONS = [a.value for a in Action]


class Move(str, Enum):
    """Cosa fa l'utente a un turno (nascosto al sistema: serve a generare i dati e a dare l'etichetta giusta)."""
    QUESTION = "question"                  # domanda con argomento (con o senza entità)
    SLOT_REPLY = "slot_reply"              # risponde a "di quale...?": solo l'entità
    FOLLOW_UP_ENTITY = "follow_up_entity"  # "e il distretto 2?": solo una nuova entità, argomento implicito
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    INSULT = "insult"
    HANDOFF = "handoff"                    # vuole un operatore
    CONFIRM = "confirm"
    DENY = "deny"
    REPEAT_REQUEST = "repeat_request"      # "non ho capito, puoi ripetere?"
    UNCLEAR = "unclear"                    # frase incomprensibile
    OUT_OF_SCOPE = "out_of_scope"          # fuori argomento
