"""
Rilevamento deterministico di segnali di rischio autolesionismo/suicidio,
indipendente dal classificatore ML.

crisis_support (knowledge/intents/crisis_support.yaml) è allenato come un
intent qualunque - ma un retrain che tocca SOLO altri dati di dominio (es.
aggiunta di esempi per un intent commerciale nuovo) può comunque spostare lo
spazio degli embedding fastText, che è condiviso e appreso su tutto il
corpus: osservato concretamente in sessione, aggiungere ~14 esempi ad altri
intent ha fatto perdere la classificazione corretta di "voglio farla finita"
e "non vedo più un motivo per andare avanti" (prima corrette al 98-100%),
mentre correggeva altre due frasi che nel retrain precedente erano fallite.
Un classificatore neurale su un corpus piccolo non garantisce stabilità tra
retrain per un intent specifico - inaccettabile quando l'intent è questo.
Stesso approccio già usato per conferma/annullamento (confirm_commands.py,
cancel_commands.py): un pattern-match deterministico non regredisce mai per
effetto di modifiche altrove nel dataset.

Va tenuto volutamente più stringente di quanto sembri necessario: falsi
positivi qui (scattare la risposta di crisi per una frustrazione qualunque)
sono un problema diverso ma reale, non solo "meglio prevenire che curare".
"""
import re

_PATTERNS = [
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
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def is_crisis_signal(user_input: str) -> bool:
    """True se il testo contiene un segnale di rischio autolesionismo/suicidio
    riconosciuto, indipendentemente da come lo classifica il modello ML."""
    return any(p.search(user_input) for p in _COMPILED)
