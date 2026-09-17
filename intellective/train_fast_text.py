import fasttext
import os
import json
import re
from collections import Counter
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
        ws=5,
        # SGD asincrono multi-thread: anche con random.seed/torch.manual_seed fissi
        # altrove nella pipeline, l'ordine di update tra thread resta non
        # deterministico e produce embedding leggermente diversi ad ogni run (a
        # cascata: classificatore e dialogue policy addestrati sopra). thread=1
        # è più lento ma rende l'intera pipeline riproducibile a parità di dataset.
        thread=1
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


def build_vocab_only():
    """
    Costruisce vocab.json direttamente dal corpus, senza addestrare FastText.

    Usato quando config.USE_FASTTEXT è False: l'embedding non è pre-addestrato
    su un corpus separato, viene appreso end-to-end insieme al resto
    dell'Intent Classifier durante il suo training (vedi
    intellective/train_intent_classifier.py e IntentClassifier.__init__) - qui
    serve quindi solo il vocabolario (parola -> indice), non i vettori.

    Stessa tokenizzazione di SimpleTokenizer (lowercase, punteggiatura -> spazio,
    split su whitespace) per restare consistente con come il testo viene
    convertito in ID più avanti nella pipeline.

    Salva:
    - .cognitor/vocab.json: vocabolario puro per il tokenizer
    """
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

    print(f"Costruzione vocabolario (senza FastText) da: {corpus_path}")

    punctuation_pattern = re.compile(r'[^\w\s]')
    word_counts = Counter()
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned = punctuation_pattern.sub(' ', line.strip().lower())
            word_counts.update(cleaned.split())

    # Ordine deterministico: frequenza decrescente, alfabetico a parità
    words = [word for word, _ in sorted(word_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    with open(VOCAB_PATH, 'w') as f:
        json.dump(words, f)

    print(f"✓ Vocabolario salvato: {VOCAB_PATH} ({len(words)} parole)")
    return words
