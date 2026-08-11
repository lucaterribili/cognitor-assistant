"""
Parole riconosciute come conferma affermativa breve ("sì", "ok", "va bene"...).

Il classificatore ML fatica a distinguere con sicurezza parole cortissime come
"si"/"ok" (pochi n-grammi di caratteri su cui basarsi): in pratica finiscono
per classificare come un intent qualunque con confidenza bassa/media (es.
"schedule_meeting", per la sola somiglianza con l'esempio di training
"schedule"). Non ha senso combattere questa fragilità del modello quando il
problema è in realtà un pattern-match deterministico su un vocabolario chiuso
- stesso approccio già usato per i comandi di annullamento (cancel_commands.py).
Usato solo per risolvere il caso "il bot ha appena proposto qualcosa
(CHATBOT_PROPOSAL) e l'utente conferma", non come sostituto generale
dell'intent 'confirm'.
"""
import re

CONFIRM_COMMANDS = {
    'si', 'sì', 'ok', 'okay', 'va bene', 'vabbene', 'certo', 'certamente',
    'confermo', 'esatto', 'perfetto', 'd\'accordo', 'daccordo', 'ovviamente',
    'sicuro', 'procedi', 'yes',
}

_TRAILING_PUNCTUATION = re.compile(r'[!?.,;:]+$')


def is_confirm_command(user_input: str) -> bool:
    """True se `user_input` è una conferma affermativa breve (case-insensitive,
    tollerante alla punteggiatura finale tipo 'sì!' o 'ok.'), anche seguita da
    una cortesia aggiunta che non ne cambia il significato ('sì, grazie',
    'ok grazie mille', 'va bene allora')."""
    normalized = _TRAILING_PUNCTUATION.sub('', user_input.strip().lower())
    if normalized in CONFIRM_COMMANDS:
        return True

    for phrase in CONFIRM_COMMANDS:
        rest = normalized[len(phrase):]
        if normalized.startswith(phrase) and rest[:1] in (' ', ','):
            return True

    return False
