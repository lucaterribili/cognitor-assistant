"""
Addestra un embedding di sotto-parola (skip-gram + negative sampling, stile
word2vec, sopra un vocabolario SentencePiece/BPE) sul corpus del dataset,
come alternativa a FastText.

L'algoritmo è gensim.models.Word2Vec (sg=1, negative sampling): stesso
principio della prima versione di questo modulo (skip-gram scritto a mano in
PyTorch), ma un'implementazione matura e testata al posto di scriverne una
da zero - meno codice da mantenere qui, nessun dubbio su correttezza di
negative sampling/finestra dinamica/init.

Stesso ruolo di train_fast_text.train_embedder() (fase di pre-training
separata, prima del classifier) e stesso formato di output (.vec + vocab.json)
- IntentClassifier.load_word_vectors() carica l'uno o l'altro senza saperne
la provenienza. Il vocabolario non è fatto di parole intere ma di
sotto-parole: qualsiasi stringa - anche mai vista nel training - si
scompone sempre in pezzi noti (fino al singolo byte come fallback estremo),
invece di risultare "fuori vocabolario". Vedi
classes/sentencepiece_tokenizer.py per il dettaglio del meccanismo e
classes/dataset_generator.py per dove viene consumato.
"""
import os
import json

import sentencepiece as spm
import numpy as np
from gensim.models import Word2Vec

from config import BASE_DIR

_SP_VOCAB_SIZE = 2000
_EMBED_DIM = 300
_WINDOW_SIZE = 5
_NEGATIVE_SAMPLES = 5
_EPOCHS = 25
_SEED = 42


def _train_sentencepiece(corpus_path: str) -> spm.SentencePieceProcessor:
    """
    Allena il modello SentencePiece (BPE) sul corpus e lo salva in
    .cognitor/sentencepiece.model - stesso path che
    classes/sentencepiece_tokenizer.py e agent/model_loader.py si aspettano.

    Il corpus viene lowercased in un file temporaneo prima del training: resta
    così coerente con il resto della pipeline, che lowercasa sempre il testo
    prima di tokenizzare (vedi SentencePieceTokenizer.tokenize_with_ids).
    """
    model_prefix = os.path.join(BASE_DIR, '.cognitor', 'sentencepiece')
    lower_corpus_path = os.path.join(BASE_DIR, '.cognitor', 'fast-text-lower.txt')

    with open(corpus_path, 'r', encoding='utf-8') as src, \
            open(lower_corpus_path, 'w', encoding='utf-8') as dst:
        for line in src:
            dst.write(line.lower())

    spm.SentencePieceTrainer.train(
        input=lower_corpus_path,
        model_prefix=model_prefix,
        vocab_size=_SP_VOCAB_SIZE,
        model_type='bpe',
        character_coverage=1.0,
        # Qualsiasi carattere/simbolo mai visto nel training (emoji, alfabeti
        # esteri, ...) si scompone in byte invece di diventare <unk>: nessun
        # input risulta mai completamente fuori vocabolario.
        byte_fallback=True,
    )

    print(f"✓ Modello SentencePiece salvato: {model_prefix}.model")
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")


def _encode_corpus_as_pieces(sp: spm.SentencePieceProcessor, corpus_path: str) -> list[list[str]]:
    """Frasi del corpus come sequenze di pezzi (stringhe, col marcatore di
    inizio-parola incluso) - l'input che si aspetta gensim.Word2Vec."""
    sequences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if line:
                sequences.append(sp.encode(line, out_type=str))
    return sequences


def train_own_embedder():
    """
    Allena l'embedder "nostro" (no FastText) sul corpus generato dalla
    pipeline. Stesso ruolo/output di train_fast_text.train_embedder():

    Salva:
    - .cognitor/sentencepiece.model: modello di tokenizzazione a sotto-parola
    - .cognitor/wordvectors.vec: matrice statica pezzo -> vettore
    - .cognitor/vocab.json: vocabolario puro (i pezzi SentencePiece, in ordine
      di id) per il tokenizer
    """
    VEC_PATH = os.path.join(BASE_DIR, '.cognitor', 'wordvectors.vec')
    VOCAB_PATH = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')

    corpus_path = os.path.join(BASE_DIR, '.cognitor', 'fast-text.txt')
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(BASE_DIR, 'knowledge', 'embeddings.txt')

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"Corpus non trovato! Esegui prima: python -m pipeline.intent_builder\n"
            f"Path cercati:\n"
            f"  - {os.path.join(BASE_DIR, '.cognitor', 'fast-text.txt')}\n"
            f"  - {os.path.join(BASE_DIR, 'knowledge', 'embeddings.txt')}"
        )

    print(f"Training embedder (gensim Word2Vec su sotto-parole SentencePiece, no FastText) con corpus: {corpus_path}")

    sp = _train_sentencepiece(corpus_path)
    vocab_size = sp.vocab_size()
    words = [sp.id_to_piece(i) for i in range(vocab_size)]

    sequences = _encode_corpus_as_pieces(sp, corpus_path)
    print(f"Vocabolario: {vocab_size} pezzi, {len(sequences)} frasi")

    print("Training embedder con gensim Word2Vec (skip-gram + negative sampling)...")
    w2v = Word2Vec(
        sentences=sequences,
        vector_size=_EMBED_DIM,
        window=_WINDOW_SIZE,
        sg=1,  # skip-gram (non CBOW)
        negative=_NEGATIVE_SAMPLES,
        min_count=1,  # ogni pezzo che compare almeno una volta ottiene un vettore
        epochs=_EPOCHS,
        seed=_SEED,
        # Thread singolo per riproducibilità a parità di dataset: come già
        # notato per FastText in train_fast_text.py, l'ordine di update tra
        # thread non è deterministico anche a seed fissato.
        workers=1,
    )

    # Il vocabolario di gensim (costruito dal corpus) e quello di SentencePiece
    # (fisso, con anche <unk>/<s>/</s> e i 256 byte di fallback) non
    # coincidono necessariamente: un pezzo SentencePiece che non compare mai
    # da solo nel corpus (tipicamente i byte di fallback, se il corpus non
    # forza mai la decomposizione byte-level) resta senza vettore da gensim -
    # gli si assegna un vettore casuale piccolo, come qualsiasi embedding non
    # addestrato, così la riga i-esima di wordvectors.vec corrisponde sempre
    # esattamente all'id i-esimo di SentencePiece.
    rng = np.random.default_rng(_SEED)
    embedding_matrix = np.empty((vocab_size, _EMBED_DIM), dtype=np.float32)
    missing = 0
    for i, word in enumerate(words):
        if word in w2v.wv:
            embedding_matrix[i] = w2v.wv[word]
        else:
            embedding_matrix[i] = rng.uniform(-0.5 / _EMBED_DIM, 0.5 / _EMBED_DIM, _EMBED_DIM)
            missing += 1

    if missing:
        print(f"⚠ {missing} pezzi del vocabolario SentencePiece non sono mai comparsi da soli nel corpus (vettore casuale)")

    with open(VEC_PATH, 'w') as f:
        f.write(f"{vocab_size} {_EMBED_DIM}\n")
        for word, vec in zip(words, embedding_matrix):
            f.write(word + " " + " ".join(str(x) for x in vec) + "\n")

    print(f"✓ Word vectors salvati: {VEC_PATH}")

    with open(VOCAB_PATH, 'w') as f:
        json.dump(words, f)

    print(f"✓ Vocabolario salvato: {VOCAB_PATH}")
    print(f"\nTraining embedder (no FastText) completato. {vocab_size} pezzi, {_EMBED_DIM} dimensioni.")
