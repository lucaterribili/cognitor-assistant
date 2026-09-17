import os
from typing import Optional

import sentencepiece as spm

# Marcatore SentencePiece per "inizio di una nuova parola" (U+2581, LOWER ONE
# EIGHT BLOCK). Un pezzo che lo porta è l'inizio di una parola nel testo
# originale; un pezzo senza è la continuazione della parola precedente.
WORD_BOUNDARY_MARK = '▁'


class SentencePieceTokenizer:
    """
    Tokenizer a sotto-parola (BPE via SentencePiece): a differenza di
    SimpleTokenizer (split sugli spazi, un indice per parola intera),
    qualsiasi stringa - anche mai vista nel training, comprese sequenze di
    caratteri o simboli mai incontrati - si scompone sempre in pezzi noti del
    vocabolario, fino al singolo byte come fallback estremo (byte_fallback,
    vedi train_own_embedder.py). Nessuna parola risulta mai "fuori
    vocabolario": una forma come "preparano" mai vista nel training, se
    condivide caratteri con "prepara" (vista), si scompone in "prepara" + "no"
    e riusa lo stesso id/embedding già addestrato per "prepara".

    Stessa interfaccia minima di SimpleTokenizer (tokenize/__call__/__len__)
    più tokenize_with_ids, usato ovunque serva anche l'id oltre al pezzo
    (training del classifier, allineamento tag NER).
    """

    def __init__(self, model_path: Optional[str] = None):
        self._sp: Optional[spm.SentencePieceProcessor] = None
        if model_path and os.path.exists(model_path):
            self._sp = spm.SentencePieceProcessor(model_file=model_path)

    def __len__(self) -> int:
        return self._sp.vocab_size() if self._sp else 0

    def tokenize(self, text: str) -> list[str]:
        """
        Pezzi "di superficie" (senza il marcatore di inizio-parola): utili per
        la ricerca sequenziale di offset nel testo originale, esattamente come
        i token di SimpleTokenizer (vedi NERTagBuilder.align_tokens_to_bio).
        """
        tokens, _ = self.tokenize_with_ids(text)
        return tokens

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)

    def tokenize_with_ids(self, text: str) -> tuple[list[str], list[int]]:
        """
        Pezzi di superficie e i relativi id nel vocabolario, nello stesso
        ordine. Un'unica chiamata a encode (via id) e poi id_to_piece per
        ricavare i pezzi, cosà i due sono garantiti coerenti per costruzione
        (niente rischio di doppia tokenizzazione che diverge).
        """
        if not self._sp:
            return [], []
        ids = self._sp.encode(text.lower(), out_type=int)
        pieces = [self._sp.id_to_piece(i) for i in ids]
        tokens = [p.replace(WORD_BOUNDARY_MARK, '') for p in pieces]
        return tokens, ids

    def encode_ids(self, text: str) -> list[int]:
        if not self._sp:
            return []
        return self._sp.encode(text.lower(), out_type=int)
