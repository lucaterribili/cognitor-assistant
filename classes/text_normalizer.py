"""
Normalizzazione del testo condivisa da training e inferenza.

Gli apostrofi (', ’, ‘, `, ´, ʼ) vengono sostituiti da uno spazio, un carattere per un carattere: "dell'UOC Cardiologia"
diventa "dell UOC Cardiologia". Senza elisioni il modello trova le entità e il testo ha le stesse parole in
addestramento e in uso; gli offset dei caratteri restano quelli del testo originale, quindi le annotazioni NER
(start/end) non vanno ricalcolate. Va applicata a ogni testo dell'utente e a ogni esempio di training, sempre.
"""
import re

_APOSTROPHES = re.compile(r"['’‘`´ʼ]")
_SPACES = re.compile(r"\s+")


def strip_apostrophes(text: str) -> str:
    """Apostrofi -> spazio, senza cambiare la lunghezza del testo."""
    return _APOSTROPHES.sub(" ", text)


def comparable(value) -> str:
    """Forma per confrontare valori (slot, `cases`): minuscolo, senza apostrofi, spazi singoli."""
    return _SPACES.sub(" ", strip_apostrophes(str(value)).lower()).strip()
