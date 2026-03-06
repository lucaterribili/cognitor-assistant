import fasttext
import os
import json
from config import BASE_DIR

def train_embedder():
    """
    Allena il modello FastText per gli embeddings.
    Usa il corpus generato da DatasetGenerator (.cognitor/fast-text.txt).

    Salva tre file:
    - models/fasttext_model.bin: modello completo con subwords
    - .cognitor/wordvectors.vec: matrice statica parola → vettore
    - .cognitor/vocab.json: vocabolario puro per il tokenizer
    """
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
    VEC_PATH = os.path.join(BASE_DIR, '.cognitor', 'wordvectors.vec')
    VOCAB_PATH = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')

    # Usa il corpus generato dalla pipeline
    corpus_path = os.path.join(BASE_DIR, '.cognitor', 'fast-text.txt')

    # Fallback se non esiste (legacy)
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(BASE_DIR, 'knowledge', 'embeddings.txt')

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"Corpus non trovato! Esegui prima: python -m pipeline.intent_builder\n"
            f"Path cercati:\n"
            f"  - {os.path.join(BASE_DIR, '.cognitor', 'fast-text.txt')}\n"
            f"  - {os.path.join(BASE_DIR, 'knowledge', 'embeddings.txt')}"
        )

    print(f"Training FastText con corpus: {corpus_path}")

    model = fasttext.train_unsupervised(
        input=corpus_path,
        model='skipgram',
        dim=300,
        epoch=25,
        lr=0.1,
        minCount=1,
        wordNgrams=2,
        minn=2,
        maxn=5,
        ws=5
    )

    # Salva modello completo
    model.save_model(MODEL_PATH)
    print(f"✓ Modello FastText salvato: {MODEL_PATH}")
    print(f"  (Può essere eliminato dopo l'estrazione dei vectors)")

    # Estrai vocabolario
    words = model.get_words()
    dim = model.get_dimension()

    # Salva word vectors in formato standard
    with open(VEC_PATH, 'w') as f:
        f.write(f"{len(words)} {dim}\n")
        for w in words:
            vec = model.get_word_vector(w)
            f.write(w + " " + " ".join(str(x) for x in vec) + "\n")

    print(f"✓ Word vectors salvati: {VEC_PATH}")

    # Salva vocabolario JSON
    with open(VOCAB_PATH, 'w') as f:
        json.dump(words, f)

    print(f"✓ Vocabolario salvato: {VOCAB_PATH}")
    print(f"\nFastText training completato. {len(words)} parole, {dim} dimensioni.")
    print(f"\n💡 Nota: Il file {MODEL_PATH} può essere eliminato.")
    print(f"   Il sistema ora usa solo vocab.json e wordvectors.vec")
