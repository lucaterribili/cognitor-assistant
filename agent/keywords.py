"""
Riconoscimento deterministico (pattern-match su vocabolario chiuso) di input
che non devono dipendere dalla stabilità del classificatore ML:

- comandi di annullamento durante la raccolta slot ("stop", "#annulla", ...)
- conferme brevi ("sì", "ok", "va bene", ...), che il modello fatica a
  classificare per i pochi n-grammi su cui basarsi
- segnali di rischio autolesionismo/suicidio: un retrain che tocca solo altri
  intent può spostare lo spazio degli embedding e far perdere la
  classificazione corretta di crisis_support (osservato in pratica). Un
  pattern-match non regredisce mai per effetto di modifiche altrove nel
  dataset. I pattern vanno tenuti volutamente stringenti: i falsi positivi
  (risposta di crisi per una frustrazione qualunque) sono un problema reale.

Fonte unica condivisa tra CLI e API (via TurnProcessor / Agent.predict).
"""
import re

_TRAILING_PUNCTUATION = re.compile(r'[!?.,;:]+$')


def _normalize(user_input: str) -> str:
    """Minuscolo, senza spazi ai bordi né punteggiatura finale ('Stop!' -> 'stop')."""
    return _TRAILING_PUNCTUATION.sub('', user_input.strip().lower())


# --- Annullamento -------------------------------------------------------------

CANCEL_COMMANDS = {
    # forme storiche con prefisso "#" (compatibilità con la documentazione CLI)
    '#exit', '#annulla', '#cancel', '#abort', '#stop', '#basta', '#annullare', '#esci', '#cancella',
    # forme naturali senza prefisso, quelle che un utente scrive per davvero
    'stop', 'basta', 'annulla', 'annullare', 'esci', 'cancella', 'cancel', 'exit', 'abort',
}


def is_cancel_command(user_input: str) -> bool:
    """True se `user_input` è una richiesta di annullamento dello slot in attesa."""
    return _normalize(user_input) in CANCEL_COMMANDS


# --- Conferma -----------------------------------------------------------------

CONFIRM_COMMANDS = {
    'si', 'sì', 'ok', 'okay', 'va bene', 'vabbene', 'certo', 'certamente',
    'confermo', 'esatto', 'perfetto', 'd\'accordo', 'daccordo', 'ovviamente',
    'sicuro', 'procedi', 'yes',
}


def is_confirm_command(user_input: str) -> bool:
    """True se `user_input` è una conferma affermativa breve, anche seguita da
    una cortesia che non ne cambia il significato ('sì, grazie', 'ok grazie mille').
    Usato solo per sbloccare una proposta in sospeso (CHATBOT_PROPOSAL)."""
    normalized = _normalize(user_input)
    if normalized in CONFIRM_COMMANDS:
        return True

    for phrase in CONFIRM_COMMANDS:
        rest = normalized[len(phrase):]
        if normalized.startswith(phrase) and rest[:1] in (' ', ','):
            return True

    return False


# --- Segnali di crisi ---------------------------------------------------------

_CRISIS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    # "farla finita" da solo è un segnale forte, ma "farla finita con X"
    # ("finire con questo progetto/esame") è l'uso comune, non ambiguo,
    # quindi va escluso esplicitamente (negative lookahead su "con").
    r'\b(voglio|vorrei|penso di|sto pensando di)\s+(morire|uccidermi|ammazzarmi|farla finita(?!\s+con\b)|togliermi la vita|suicidarmi|farmi (del )?male)\b',
    r'\b(mi voglio|mi vorrei)\s+(uccidere|ammazzare)\b',
    r'\bpenso al suicidio\b',
    r'\bnon (voglio più vivere|ho più voglia di vivere|ce la faccio più a vivere)\b',
    r'\bla vita non ha (più )?senso\b',
    r'\bvorrei sparire per sempre\b',
    r'\bnon vedo (più )?(un motivo|nessun motivo)\s+(per|nel)\s+(vivere|andare avanti|continuare)\b',
    r'\b(voglio sapere|voglio informazioni su|come si fa a)\s+(come )?morire\b',
    r'\bautolesionis',
]]


def is_crisis_signal(user_input: str) -> bool:
    """True se il testo contiene un segnale di rischio autolesionismo/suicidio
    riconosciuto, indipendentemente da come lo classifica il modello ML."""
    return any(p.search(user_input) for p in _CRISIS_PATTERNS)
