"""
Comandi/parole riconosciuti come richiesta di annullare l'input dello slot corrente
durante la raccolta slot (session.agent_mode == "inputable").

Fonte unica condivisa tra CLI (agent/conversation_handler.py) e API
(api/chatbot.py): prima erano due set duplicati e disallineati — l'API
riconosceva solo le forme con "#" (#exit, #annulla, ...), quindi una parola
naturale come "stop" veniva presa come valore letterale dello slot in attesa
invece che come richiesta di uscita dal flusso.
"""
import re

CANCEL_COMMANDS = {
    # forme storiche con prefisso "#" (compatibilità con la documentazione CLI)
    '#exit', '#annulla', '#cancel', '#abort', '#stop', '#basta', '#annullare', '#esci', '#cancella',
    # forme naturali senza prefisso, quelle che un utente scrive per davvero
    'stop', 'basta', 'annulla', 'annullare', 'esci', 'cancella', 'cancel', 'exit', 'abort',
}

_TRAILING_PUNCTUATION = re.compile(r'[!?.,;:]+$')


def is_cancel_command(user_input: str) -> bool:
    """True se `user_input` è una richiesta di annullamento (case-insensitive,
    tollerante alla punteggiatura finale tipo 'stop!' o 'basta.')."""
    normalized = _TRAILING_PUNCTUATION.sub('', user_input.strip().lower())
    return normalized in CANCEL_COMMANDS
